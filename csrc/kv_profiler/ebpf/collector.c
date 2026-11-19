/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#define KVP_HOST 1
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>
#include "protocol.h"
#include "lifecycle.h"
#include "accounting.h"

struct target {
  u32 pid;
  u64 start, generation, device, inode;
  char path[1024];
  int pidfd, filefd;
};
struct plan {
  pid_t parent;
  u64 parent_start, dev, ino, netns, bound;
  long long offset;
  u32 count;
  int tcp, native;
  struct target targets[KVP_PROCESSES];
};
struct sent_row {
  struct kvp_call_key key;
  struct kvp_thread_key thread;
  int used, acked, command;
};
static struct sent_row sent[KVP_EPOCHS];
static struct bpf_link* links[4104];
static u32 nlinks;
static unsigned char return_link[4104];
static int failed;
static u64 monotonic_ns(void);
static long map_read_budget = 32768;
static int read_exhausted;
static int lookup(int descriptor, const void* key, void* value) {
  if (map_read_budget <= 0) {
    failed = read_exhausted = 1;
    return -E2BIG;
  }
  map_read_budget--;
  return bpf_map_lookup_elem(descriptor, key, value);
}
static int next_key(int descriptor, const void* key, void* next) {
  if (map_read_budget <= 0) {
    failed = read_exhausted = 1;
    return -E2BIG;
  }
  map_read_budget--;
  return bpf_map_get_next_key(descriptor, key, next);
}
static char output_queue[4096][KVP_RECORD_BYTES];
static u32 output_length[4096], output_head, output_tail, output_count;
static int flush_output(void) {
  while (output_count) {
    ssize_t n = write(STDOUT_FILENO, output_queue[output_head],
                      output_length[output_head]);
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return 0;
    if (n != output_length[output_head]) return -EIO;
    output_head = (output_head + 1) % 4096;
    output_count--;
  }
  return 0;
}
static int output(const char* format, ...) {
  char row[KVP_RECORD_BYTES];
  va_list args;
  va_start(args, format);
  int n = vsnprintf(row, sizeof(row), format, args);
  va_end(args);
  if (n < 0 || n >= KVP_RECORD_BYTES - 1) return -E2BIG;
  row[n++] = '\n';
  if (output_count == 4096) return -E2BIG;
  memcpy(output_queue[output_tail], row, n);
  output_length[output_tail] = n;
  output_tail = (output_tail + 1) % 4096;
  output_count++;
  return 0;
}
static int unavailable(const char* reason) {
  output(
      "{\"type\":\"status\",\"available\":false,\"reason\":\"%s\",\"exact\":"
      "false}",
      reason);
  return 2;
}
static u64 ticks(pid_t pid) {
  char path[64], data[8192];
  snprintf(path, sizeof(path), "/proc/%d/stat", pid);
  FILE* file = fopen(path, "re");
  if (!file) return 0;
  if (!fgets(data, sizeof(data), file)) {
    fclose(file);
    return 0;
  }
  fclose(file);
  char* p = strrchr(data, ')');
  if (!p) return 0;
  p += 2;
  for (int field = 3; field < 22; field++) {
    p = strchr(p, ' ');
    if (!p) return 0;
    p++;
  }
  return strtoull(p, 0, 10);
}
static int permitted(void) {
  FILE* f = fopen("/proc/self/status", "re");
  if (!f) return 0;
  char row[256];
  unsigned long long caps = 0;
  while (fgets(row, sizeof(row), f))
    if (sscanf(row, "CapEff: %llx", &caps) == 1) break;
  fclose(f);
  return !!(caps & (1ULL << 21)) ||
         ((caps & (1ULL << 38)) && (caps & (1ULL << 39)));
}
static int parse_plan(const char* path, struct plan* p) {
  struct stat st;
  if (stat(path, &st) || !S_ISREG(st.st_mode) || st.st_uid != geteuid() ||
      (st.st_mode & 0077))
    return -EINVAL;
  FILE* f = fopen(path, "re");
  if (!f) return -errno;
  char line[2048];
  u32 lines = 0;
  int have_parent = 0, have_ns = 0, have_bound = 0;
  while (fgets(line, sizeof(line), f)) {
    if (++lines > 260 || !strchr(line, '\n')) {
      fclose(f);
      return -E2BIG;
    }
    unsigned long long a, b, c, d, e;
    int used = 0;
    if (line[0] == 'P' && sscanf(line, "P %llu %llu", &a, &b) == 2 &&
        !have_parent) {
      if (!a || a > 0x7fffffff) {
        fclose(f);
        return -EINVAL;
      }
      p->parent = a;
      p->parent_start = b;
      have_parent = 1;
    } else if (line[0] == 'N' &&
               sscanf(line, "N %llu %llu %llu %lld %d %d", &a, &b, &c,
                      &p->offset, &p->tcp, &p->native) == 6 &&
               !have_ns) {
      p->dev = a;
      p->ino = b;
      p->netns = c;
      have_ns = 1;
      if (!a || !b || !c || (p->tcp != 0 && p->tcp != 1) ||
          (p->native != 0 && p->native != 1) || (!p->tcp && !p->native)) {
        fclose(f);
        return -EINVAL;
      }
    } else if (line[0] == 'B' && sscanf(line, "B %llu", &a) == 1 &&
               !have_bound) {
      p->bound = a;
      have_bound = 1;
    } else if (line[0] == 'T' &&
               sscanf(line, "T %llu %llu %llu %llu %llu %n", &a, &b, &c, &d, &e,
                      &used) == 5 &&
               used > 0) {
      if (!a || a > 0x7fffffff || !b || !c || !d || !e ||
          p->count == KVP_PROCESSES) {
        fclose(f);
        return -EINVAL;
      }
      struct target* t = &p->targets[p->count++];
      t->pid = a;
      t->start = b;
      t->generation = c;
      t->device = d;
      t->inode = e;
      t->pidfd = t->filefd = -1;
      size_t n = strcspn(line + used, "\r\n");
      if (!n || n >= sizeof(t->path) || line[used] != '/') {
        fclose(f);
        return -EINVAL;
      }
      memcpy(t->path, line + used, n);
      t->path[n] = 0;
      for (u32 i = 0; i + 1 < p->count; i++)
        if (p->targets[i].pid == a || p->targets[i].generation == c) {
          fclose(f);
          return -EINVAL;
        }
    } else {
      fclose(f);
      return -EINVAL;
    }
  }
  fclose(f);
  if (!have_parent || !have_ns || !have_bound || !p->count ||
      p->bound > KVP_MAP_BYTES || !p->bound)
    return -EINVAL;
  return 0;
}
static int map_contract(struct bpf_object* obj, int describe) {
  struct bpf_map* map;
  u32 count = 0, globals = 0;
  bpf_object__for_each_map(map, obj) {
    const char* name = bpf_map__name(map);
    u32 k = bpf_map__key_size(map), v = bpf_map__value_size(map),
        n = bpf_map__max_entries(map);
    enum bpf_map_type type = bpf_map__type(map);
    int known = 0;
#define CHECK_MAP(id, entries, key, value, kcap, vcap)                   \
  if (!strcmp(name, #id)) {                                              \
    known = 1;                                                           \
    if (type != BPF_MAP_TYPE_HASH || n != entries || k != sizeof(key) || \
        v != sizeof(value) || k > kcap || v > vcap)                      \
      return -EINVAL;                                                    \
  }
    KVP_MAPS(CHECK_MAP)
#undef CHECK_MAP
    if (!strcmp(name, "control")) {
      known = 1;
      if (type != BPF_MAP_TYPE_ARRAY || n != 16 || k != 4 ||
          v != sizeof(struct kvp_control))
        return -EINVAL;
    }
    if (!strcmp(name, "loss_counts")) {
      known = 1;
      if (type != BPF_MAP_TYPE_PERCPU_ARRAY || n != 1 || k != 4 || v != 256)
        return -EINVAL;
    }
    if (!known) {
      if (type != BPF_MAP_TYPE_ARRAY || name[0] != '.' || ++globals > 3 ||
          n != 1 || k > 4 || v > 16384)
        return -EINVAL;
    }
    if (bpf_map__map_flags(map) & BPF_F_NO_PREALLOC) return -EINVAL;
    if (++count > KVP_OBJECT_MAPS) return -E2BIG;
    if (describe && output("{\"type\":\"map\",\"name\":\"%s\",\"entries\":%u,"
                           "\"key\":%u,\"value\":%u}",
                           name, n, k, v))
      return -EIO;
  }
  return 0;
}
static int report_charge(enum kvp_budget_result result,
                         const struct kvp_charge* charge) {
  if (result == KVP_BUDGET_UNAVAILABLE)
    return unavailable("kernel_map_accounting_unavailable");
  if (output("{\"type\":\"resource\",\"outcome\":\"%s\","
             "\"kernel_map_bytes\":%llu,\"bound_bytes\":%llu,"
             "\"last_map_bytes\":%llu,\"allocation_overflow\":%s}",
             result == KVP_BUDGET_FAIL ? "FAIL" : "PASS", charge->observed,
             charge->bound, charge->last, charge->overflow ? "true" : "false"))
    return 3;
  if (result == KVP_BUDGET_FAIL) {
    output(
        "{\"type\":\"status\",\"available\":false,\"outcome\":\"FAIL\","
        "\"reason\":\"kernel_map_budget_exceeded\",\"exact\":false}");
    return 3;
  }
  return 0;
}
static int memory_charge(struct bpf_object* obj, u64 bound) {
  struct kvp_charge charge = {.bound = bound};
  struct bpf_map* map;
  bpf_object__for_each_map(map, obj) {
    int descriptor = bpf_map__fd(map);
    if (descriptor < 0) continue;
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fdinfo/%d", descriptor);
    FILE* file = fopen(path, "re");
    enum kvp_budget_result result = kvp_charge_fdinfo(&charge, file);
    if (file) fclose(file);
    if (result != KVP_BUDGET_OK) return report_charge(result, &charge);
  }
  return report_charge(charge.observed ? KVP_BUDGET_OK : KVP_BUDGET_UNAVAILABLE,
                       &charge);
}
static int fd(struct bpf_object* obj, const char* name) {
  struct bpf_map* m = bpf_object__find_map_by_name(obj, name);
  return m ? bpf_map__fd(m) : -1;
}
static int attach_api(struct bpf_object* obj, struct target* t) {
  if (!t->pid || t->pid > 0x7fffffff) return -EINVAL;
  t->pidfd = syscall(SYS_pidfd_open, t->pid, 0);
  if (t->pidfd < 0 || ticks(t->pid) != t->start) return -ESRCH;
  t->filefd = open(t->path, O_RDONLY | O_CLOEXEC);
  struct stat st;
  if (t->filefd < 0 || fstat(t->filefd, &st) ||
      (st.st_ino != t->inode || st.st_dev != t->device))
    return -EINVAL;
  char path[64];
  snprintf(path, sizeof(path), "/proc/self/fd/%d", t->filefd);
  const unsigned long offset = 0x48900;
  const char* names[] = {"api_enter", "api_exit"};
  for (u32 i = 0; i < 2; i++) {
    LIBBPF_OPTS(bpf_uprobe_multi_opts, opts, .offsets = &offset, .cnt = 1,
                .retprobe = (i == 1));
    struct bpf_program* prog = bpf_object__find_program_by_name(obj, names[i]);
    if (!prog || nlinks == 4096) return -EINVAL;
    struct bpf_link* link = bpf_program__attach_uprobe_multi(
        prog, (pid_t)t->pid, path, NULL, &opts);
    int error = libbpf_get_error(link);
    if (error) return error;
    return_link[nlinks] = (i == 1);
    links[nlinks++] = link;
  }
  return ticks(t->pid) == t->start ? 0 : -ESRCH;
}
static int send_epoch(int map, const struct kvp_call_key* key,
                      const struct kvp_epoch* row) {
  struct kvp_epoch again;
  if (lookup(map, key, &again) || memcmp(&again, row, sizeof(again))) return 0;
  for (u32 i = 0; i < KVP_EPOCHS; i++)
    if (sent[i].used && !memcmp(&sent[i].key, key, sizeof(*key))) return 0;
  for (u32 i = 0; i < KVP_EPOCHS; i++)
    if (!sent[i].used) {
      if (output("{\"type\":\"api\",\"generation\":%llu,\"epoch\":%llu,\"pid\":"
                 "%u,\"tid\":%u,\"thread_ns\":%llu,\"begin_ns\":%llu,\"end_"
                 "ns\":%llu,\"status\":%"
                 "d,\"exact\":false}",
                 (unsigned long long)key->generation,
                 (unsigned long long)key->epoch, row->thread.tgid,
                 row->thread.tid, (unsigned long long)row->thread.start,
                 (unsigned long long)row->begin_ns,
                 (unsigned long long)row->end_ns, row->status))
        return -EIO;
      sent[i] =
          (struct sent_row){.key = *key, .thread = row->thread, .used = 1};
      return 0;
    }
  return -E2BIG;
}
static int flush_epochs(struct bpf_object* obj) {
  int map = fd(obj, "api_epochs");
  struct kvp_call_key key, next, *prior = NULL;
  u32 rows = 0;
  while (!next_key(map, prior, &next)) {
    if (++rows > 32768) return -E2BIG;
    struct kvp_epoch value;
    if (!lookup(map, &next, &value) && value.complete)
      if (send_epoch(map, &next, &value)) return -EIO;
    key = next;
    prior = &key;
  }
  return 0;
}
static void commands(struct bpf_object* obj) {
  int map = fd(obj, "control");
  for (u32 slot = 1; slot < 16; slot++) {
    struct kvp_control old;
    if (lookup(map, &slot, &old)) continue;
    if (old.command == 2) {
      for (u32 i = 0; i < KVP_EPOCHS; i++)
        if (sent[i].used && sent[i].command == (int)slot) sent[i].used = 0;
      struct kvp_control empty = {0};
      if (bpf_map_update_elem(map, &slot, &empty, BPF_ANY)) {
        failed = 1;
        return;
      }
    } else if (old.command)
      continue;
    for (u32 i = 0; i < KVP_EPOCHS; i++)
      if (sent[i].used && sent[i].acked && !sent[i].command) {
        struct kvp_control command = {.retire = sent[i].thread,
                                      .netns = sent[i].key.epoch};
        if (bpf_map_update_elem(map, &slot, &command, BPF_ANY)) {
          failed = 1;
          return;
        }
        command.command = 3;
        if (bpf_map_update_elem(map, &slot, &command, BPF_ANY)) {
          failed = 1;
          return;
        }
        sent[i].command = slot;
        break;
      }
  }
}
static int counters(struct bpf_object* obj) {
  int cpus = libbpf_num_possible_cpus();
  if (cpus < 1 || cpus > 4096) return -EINVAL;
  struct kvp_loss_counts* values = calloc(cpus, sizeof(*values));
  if (!values) return -ENOMEM;
  u32 zero = 0;
  u64 begin = monotonic_ns();
  int error = lookup(fd(obj, "loss_counts"), &zero, values);
  u64 end = monotonic_ns();
  for (u32 i = 0; i <= STAT_UNKNOWN_RX_LENGTH && !error; i++) {
    u64 sum = 0;
    for (int cpu = 0; cpu < cpus; cpu++) {
      if (~0ULL - sum < values[cpu].count[i]) error = -EOVERFLOW;
      sum += values[cpu].count[i];
    }
    if (sum && i < LOSS_COUNT) failed = 1;
    if (output("{\"type\":\"%s\",\"counter\":%u,\"value\":%llu,\"begin_ns\":%"
               "llu,\"end_ns\":%llu}",
               i < LOSS_COUNT ? "loss" : "tcp_namespace", i,
               (unsigned long long)sum, (unsigned long long)begin,
               (unsigned long long)end))
      error = -EIO;
  }
  free(values);
  return error;
}
static int sockets(struct bpf_object* obj) {
  int map = fd(obj, "socket_counts");
  struct kvp_socket_key key, next, *prior = NULL;
  u32 rows = 0;
  while (!next_key(map, prior, &next)) {
    if (++rows > 1024) return -E2BIG;
    struct kvp_socket row;
    u64 begin = monotonic_ns();
    if (!lookup(map, &next, &row)) {
      u64 end = monotonic_ns();
      const char* names[] = {"rx_events", "skb_len_bytes", "states",
                             "retrans",   "reset_tx",      "reset_rx"};
      u64 values[] = {row.rx_events,   row.skb_len_bytes, row.states,
                      row.retransmits, row.send_resets,   row.recv_resets};
      for (u32 i = 0; i < 6; i++)
        if (output("{\"type\":\"tcp\",\"cookie\":%llu,\"counter\":\"%s\","
                   "\"value\":%llu,\"begin_ns\":%llu,\"end_ns\":%llu,\"exact\":"
                   "false}",
                   (unsigned long long)next.cookie, names[i],
                   (unsigned long long)values[i], (unsigned long long)begin,
                   (unsigned long long)end))
          return -EIO;
    }
    key = next;
    prior = &key;
  }
  return 0;
}
static u64 monotonic_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts)) return 0;
  return (u64)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
static int expired(struct kvp_epoch* row, u64 now) {
  return !row->complete &&
         (now < row->begin_ns || now - row->begin_ns > 30000000000ULL);
}
static int watchdog(struct bpf_object* obj, long long offset) {
  u64 clock = monotonic_ns(), now;
  if (offset >= 0) {
    if (clock < (u64)offset) return -EINVAL;
    now = clock - offset;
  } else {
    if (clock > ~0ULL - ((u64)(-(offset + 1)) + 1)) return -EINVAL;
    now = clock + ((u64)(-(offset + 1)) + 1);
  }
  int map = fd(obj, "api_epochs");
  struct kvp_call_key key, next, *prior = NULL;
  u32 count = 0;
  while (!next_key(map, prior, &next)) {
    if (++count > 32768) return -E2BIG;
    struct kvp_epoch row;
    if (!lookup(map, &next, &row) && expired(&row, now)) return -ETIMEDOUT;
    key = next;
    prior = &key;
  }
  return 0;
}
static void retire(struct bpf_object* obj) {
  int map = fd(obj, "threads"), ctl = fd(obj, "control");
  struct kvp_thread_key key, next, *prior = NULL;
  u32 count = 0;
  while (!next_key(map, prior, &next)) {
    if (++count > 2048) {
      failed = 1;
      return;
    }
    struct kvp_thread row;
    if (!lookup(map, &next, &row) && !row.pending_epochs) {
      char path[128];
      snprintf(path, sizeof(path), "/proc/%u/task/%u", next.tgid, next.tid);
      struct stat st;
      if (stat(path, &st) && errno == ENOENT) {
        for (u32 slot = 1; slot < 16; slot++) {
          struct kvp_control command;
          if (lookup(ctl, &slot, &command) || command.command) continue;
          command = (struct kvp_control){.retire = next};
          if (bpf_map_update_elem(ctl, &slot, &command, BPF_ANY)) {
            failed = 1;
            return;
          }
          command.command = 1;
          if (bpf_map_update_elem(ctl, &slot, &command, BPF_ANY)) {
            failed = 1;
            return;
          }
          break;
        }
      }
    }
    key = next;
    prior = &key;
  }
}
static void parse_command(char* line, int* stop) {
  unsigned long long gen, epoch;
  if (!strcmp(line, "stop\n")) {
    *stop = 1;
    return;
  }
  if (sscanf(line, "ack %llu %llu", &gen, &epoch) == 2) {
    for (u32 i = 0; i < KVP_EPOCHS; i++)
      if (sent[i].used && sent[i].key.generation == gen &&
          sent[i].key.epoch == epoch) {
        sent[i].acked = 1;
        return;
      }
  }
  failed = 1;
}
static void detach(int returns) {
  for (u32 i = 0; i < nlinks; i++)
    if (links[i] && return_link[i] == returns) {
      bpf_link__destroy(links[i]);
      links[i] = NULL;
    }
}
static void finish(struct bpf_object* obj) {
  u64 start = monotonic_ns(), end = start + 1500000000ULL,
      grace = start + 500000000ULL;
  map_read_budget = 32768;
  detach(0);
  char line[128];
  while (monotonic_ns() < grace) {
    struct kvp_call_key key;
    if (next_key(fd(obj, "active_calls"), NULL, &key)) break;
    if (flush_epochs(obj)) {
      failed = 1;
      break;
    }
    struct pollfd input = {.fd = STDIN_FILENO, .events = POLLIN};
    if (poll(&input, 1, 10) > 0 && fgets(line, sizeof(line), stdin)) {
      int stop = 0;
      parse_command(line, &stop);
    }
  }
  struct kvp_call_key active;
  if (!next_key(fd(obj, "active_calls"), NULL, &active)) failed = 1;
  detach(1);
  if (flush_epochs(obj) || counters(obj) || sockets(obj)) failed = 1;
  /* Detach is implemented, but not qualified as a target-kernel quiescence
   * barrier. */
  while (monotonic_ns() < end) {
    int pending = 0;
    for (u32 i = 0; i < KVP_EPOCHS; i++)
      if (sent[i].used && !sent[i].acked) {
        pending = 1;
        break;
      }
    if (flush_output()) {
      failed = 1;
      break;
    }
    if (!pending && !output_count) break;
    struct pollfd input = {.fd = STDIN_FILENO, .events = POLLIN | POLLHUP};
    if (poll(&input, 1, 10) > 0) {
      if (!fgets(line, sizeof(line), stdin)) {
        failed = 1;
        break;
      }
      int stop = 0;
      parse_command(line, &stop);
    }
  }
  for (u32 i = 0; i < KVP_EPOCHS; i++)
    if (sent[i].used && !sent[i].acked) failed = 1;
}
static int loaded_contract(struct bpf_object* obj) {
  struct bpf_program* prog;
  size_t count = 0;
  bpf_object__for_each_program(prog, obj) {
    int descriptor = bpf_program__fd(prog);
    if (descriptor < 0) continue;
    struct bpf_prog_info info = {0};
    u32 size = sizeof(info);
    if (bpf_obj_get_info_by_fd(descriptor, &info, &size)) return -EINVAL;
    size_t insns = info.xlated_prog_len / 8;
    if (info.xlated_prog_len % 8 || !insns || insns > 16384) return -EINVAL;
    count += insns;
    if (count > 262144 || !(fcntl(descriptor, F_GETFD) & FD_CLOEXEC))
      return -EINVAL;
  }
  return 0;
}
static int run(struct bpf_object* obj, struct plan* p) {
  if (!permitted()) return unavailable("missing_effective_privilege");
  struct rlimit limit;
  if (getrlimit(RLIMIT_NOFILE, &limit) ||
      p->count * 4ULL + 64 > limit.rlim_cur || p->count * 4ULL + 64 > 8192)
    return unavailable("fd_budget");
  struct stat pidns, netns;
  if (stat("/proc/self/ns/pid", &pidns) || stat("/proc/self/ns/net", &netns) ||
      pidns.st_dev != p->dev || pidns.st_ino != p->ino ||
      netns.st_ino != p->netns)
    return unavailable("namespace_identity");
  if (kvp_parent_guard(p->parent) || ticks(p->parent) != p->parent_start)
    return unavailable("parent_identity");
  int cpus = libbpf_num_possible_cpus();
  if (cpus < 1 || cpus > 4096) return unavailable("possible_cpu_limit");
  struct bpf_program* prog;
  struct bpf_map* map;
  bpf_object__for_each_program(prog, obj) {
    bpf_program__set_autoattach(prog, false);
    if ((!p->tcp && !strncmp(bpf_program__name(prog), "tcp_", 4)) ||
        (!p->native && !strncmp(bpf_program__name(prog), "api_", 4)))
      bpf_program__set_autoload(prog, false);
  }
  bpf_object__for_each_map(map, obj) {
    const char* name = bpf_map__name(map);
    if (!strcmp(name, "legacy_lists") || !strcmp(name, "ext_txns") ||
        !strcmp(name, "qp_counts"))
      bpf_map__set_autocreate(map, false);
  }
  if (libbpf_set_memlock_rlim(0)) return unavailable("memlock_policy");
  if (bpf_object__load(obj))
    return unavailable("kernel_program_or_helper_unavailable");
  if (loaded_contract(obj)) return unavailable("loaded_program_contract");
  int accounting = memory_charge(obj, p->bound);
  if (accounting) return accounting;
  u32 zero = 0;
  struct kvp_control cfg = {
      .netns = p->netns, .pid_dev = p->dev, .pid_ino = p->ino};
  if (bpf_map_update_elem(fd(obj, "control"), &zero, &cfg, BPF_ANY))
    return unavailable("control_map");
  for (u32 i = 0; i < p->count; i++) {
    struct kvp_process process = {.generation = p->targets[i].generation};
    if (bpf_map_update_elem(fd(obj, "processes"), &p->targets[i].pid, &process,
                            BPF_NOEXIST) ||
        (p->native && attach_api(obj, &p->targets[i])))
      return unavailable("process_scoped_uprobe_unavailable");
  }
  if (p->tcp)
    bpf_object__for_each_program(
        prog, obj) if (!strncmp(bpf_program__name(prog), "tcp_", 4)) {
      if (nlinks >= 4104) return unavailable("link_limit");
      struct bpf_link* link = bpf_program__attach_trace(prog);
      if (libbpf_get_error(link))
        return unavailable("typed_tcp_hook_unavailable");
      links[nlinks++] = link;
    }
  cfg.enabled = 1;
  if (bpf_map_update_elem(fd(obj, "control"), &zero, &cfg, BPF_ANY))
    return unavailable("enable");
  if (output("{\"type\":\"ready\",\"available\":true,\"exact\":false,\"miss_"
             "visibility\":\"unverified\"}"))
    return 2;
  setvbuf(stdin, NULL, _IONBF, 0);
  char command[128];
  int stopping = 0;
  u64 next_tick = monotonic_ns();
  while (!stopping && !failed) {
    u64 now = monotonic_ns();
    int timeout = now >= next_tick ? 0 : (int)((next_tick - now) / 1000000ULL);
    if (flush_output()) {
      failed = 1;
      break;
    }
    struct pollfd inputs[2] = {
        {.fd = STDIN_FILENO, .events = POLLIN | POLLHUP},
        {.fd = STDOUT_FILENO, .events = output_count ? POLLOUT : 0}};
    int ready = poll(inputs, 2, timeout);
    if (ready < 0) {
      failed = 1;
      break;
    }
    if (inputs[1].revents & POLLOUT)
      if (flush_output()) {
        failed = 1;
        break;
      }
    if (inputs[0].revents) {
      if (!fgets(command, sizeof(command), stdin)) {
        failed = 1;
        break;
      }
      parse_command(command, &stopping);
    }
    now = monotonic_ns();
    int sample_due = now >= next_tick;
    if (sample_due) map_read_budget = 32768;
    commands(obj);
    if (!sample_due) continue;
    next_tick = now + 1000000000ULL;
    if (flush_epochs(obj) || counters(obj) || sockets(obj) ||
        watchdog(obj, p->offset)) {
      failed = 1;
      break;
    }
    retire(obj);
    for (u32 i = 0; i < p->count; i++)
      if (ticks(p->targets[i].pid) != p->targets[i].start) {
        failed = 1;
        break;
      }
    if (lookup(fd(obj, "control"), &zero, &cfg) || cfg.command) {
      failed = 1;
      break;
    }
  }
  finish(obj);
  if (read_exhausted) {
    output(
        "{\"type\":\"status\",\"available\":false,\"reason\":\"map_read_"
        "budget\",\"exact\":false}");
    dprintf(STDERR_FILENO, "map read budget exhausted\n");
  }
  /* No quiescence/loss-free claim before the separately assigned kernel tests.
   */
  output(
      "{\"type\":\"end\",\"complete\":false,\"reason\":\"live_quiescence_and_"
      "miss_visibility_unqualified\",\"failed\":%s}",
      failed ? "true" : "false");
  return 2;
}
int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    unavailable("usage");
    flush_output();
    return 2;
  }
  int inspect_mode = !strcmp(argv[1], "--inspect");
  struct plan* plan = calloc(1, sizeof(*plan));
  if (!plan) return 2;
  const char* object = argv[argc - 1];
  if (!inspect_mode) {
    if (strcmp(argv[1], "--arm") || argc != 4 || parse_plan(argv[2], plan)) {
      free(plan);
      unavailable("invalid_plan");
      flush_output();
      return 2;
    }
    if (!permitted() || kvp_parent_guard(plan->parent) ||
        ticks(plan->parent) != plan->parent_start) {
      free(plan);
      unavailable("privilege_or_parent");
      flush_output();
      return 2;
    }
  }
  fcntl(STDOUT_FILENO, F_SETFL, fcntl(STDOUT_FILENO, F_GETFL) | O_NONBLOCK);
  struct bpf_object* obj = bpf_object__open_file(object, NULL);
  if (libbpf_get_error(obj)) {
    free(plan);
    unavailable("object_open");
    flush_output();
    return 2;
  }
  struct bpf_program* prog;
  bpf_object__for_each_program(prog, obj)
      bpf_program__set_autoattach(prog, false);
  int result;
  size_t instructions = 0;
  int instruction_error = 0;
  bpf_object__for_each_program(prog, obj) {
    size_t count = bpf_program__insn_cnt(prog);
    instructions += count;
    if (count > 16384 || instructions > 262144) instruction_error = 1;
  }
  if (instruction_error)
    result = unavailable("instruction_budget");
  else if (map_contract(obj, inspect_mode))
    result = unavailable("map_contract");
  else if (inspect_mode) {
    bpf_object__for_each_program(prog, obj)
        output("{\"type\":\"program\",\"name\":\"%s\",\"autoattach\":false}",
               bpf_program__name(prog));
    result = 0;
  } else
    result = run(obj, plan);
  for (u32 i = 0; i < nlinks; i++)
    if (links[i]) bpf_link__destroy(links[i]);
  bpf_object__close(obj);
  for (u32 i = 0; i < plan->count; i++) {
    if (plan->targets[i].pidfd >= 0) close(plan->targets[i].pidfd);
    if (plan->targets[i].filefd >= 0) close(plan->targets[i].filefd);
  }
  if (flush_output()) result = 2;
  free(plan);
  return result;
}
