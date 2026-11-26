/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#define KVP_HOST 1
#include <arpa/inet.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "lifecycle.h"
#include "protocol.h"

struct plan {
  pid_t parent;
  u64 start;
  struct kvp_control control;
};
static char queue[KVP_QUEUE][KVP_RECORD_BYTES];
static size_t lengths[KVP_QUEUE], head, count, written;
static u64 output_drops, malformed;
static int output_broken;
static u64 monotonic_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts)) return 0;
  return (u64)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
static int flush(void) {
  while (count) {
    ssize_t n = write(1, queue[head] + written, lengths[head] - written);
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return 0;
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) {
      output_broken = 1;
      return -EIO;
    }
    written += n;
    if (written == lengths[head]) {
      written = 0;
      head = (head + 1) % KVP_QUEUE;
      count--;
    }
  }
  return 0;
}
static int output(int event, const char *format, ...) {
  /* Keep one queue slot independent of event pressure for status/terminal. */
  if (count >= KVP_QUEUE - (event ? 1 : 0)) {
    if (event) output_drops++;
    return -ENOSPC;
  }
  size_t tail = (head + count) % KVP_QUEUE;
  va_list args;
  va_start(args, format);
  int n = vsnprintf(queue[tail], KVP_RECORD_BYTES, format, args);
  va_end(args);
  if (n < 0 || n >= KVP_RECORD_BYTES - 1) {
    if (event) output_drops++;
    return -E2BIG;
  }
  queue[tail][n++] = '\n';
  lengths[tail] = n;
  count++;
  return 0;
}
static const char *kinds[] = {"tcp_retransmit_skb", "tcp_send_reset",
                              "tcp_receive_reset", "inet_sock_set_state"};
static const char *fields[] = {"srtt_us",       "snd_cwnd",    "snd_ssthresh",
                               "packets_out",   "retrans_out", "lost_out",
                               "total_retrans", "mss_cache"};
static int event_row(void *ctx, void *data, size_t size) {
  (void)ctx;
  const struct kvp_event *e = data;
  if (size != sizeof(*e) || e->schema != KVP_SCHEMA || e->size != size ||
      e->kind >= KVP_EVENTS || (e->family != AF_INET && e->family != AF_INET6) ||
      !e->cookie || !e->admission ||
      (e->available | e->read_errors) > 255 || (e->available & e->read_errors)) {
    malformed++;
  } else {
    char local[INET6_ADDRSTRLEN], peer[INET6_ADDRSTRLEN], snapshot[1024];
    size_t used = 0;
    if (!inet_ntop(e->family, e->local_address, local, sizeof(local)) ||
        !inet_ntop(e->family, e->peer_address, peer, sizeof(peer))) {
      malformed++;
    } else {
      for (u32 i = 0; i < KVP_FIELDS; i++) {
        char value[24];
        if (e->available & (1U << i))
          snprintf(value, sizeof(value), "%u", e->fields[i]);
        else
          strcpy(value, "null");
        int n = snprintf(snapshot + used, sizeof(snapshot) - used,
                         "%s\"%s\":{\"value\":%s,\"state\":\"%s\"}", i ? "," : "",
                         fields[i], value,
                         e->available & (1U << i)     ? "available"
                         : e->read_errors & (1U << i) ? "read_error"
                                                      : "unavailable");
        if (n < 0 || (size_t)n >= sizeof(snapshot) - used) {
          malformed++;
          return 0;
        }
        used += n;
      }
      char old[16] = "null", next[16] = "null";
      if (e->kind == KVP_STATE) {
        snprintf(old, sizeof(old), "%u", e->old_state);
        snprintf(next, sizeof(next), "%u", e->new_state);
      }
      output(1,
             "{\"type\":\"tcp_event\",\"schema\":3,\"timestamp_ns\":%llu,"
             "\"netns\":%llu,\"event\":\"%s\",\"family\":%u,\"local_ip\":\"%s\","
             "\"peer_ip\":\"%s\",\"local_port\":%u,\"peer_port\":%u,"
             "\"cookie\":%llu,\"admission\":%llu,\"old_state\":%s,\"new_state\":%s,"
             "\"snapshot\":{%s},\"srtt_encoding\":\"usec_shift_3\",\"exact\":false}",
             (unsigned long long)e->timestamp_ns, (unsigned long long)e->netns,
             kinds[e->kind], e->family, local, peer, e->local_port, e->peer_port,
             (unsigned long long)e->cookie, (unsigned long long)e->admission, old,
             next, snapshot);
    }
  }
  return 0;
}
static int status(const char *operation, const char *hook, int error) {
  if (error < 0) error = -error;
  /* Full libbpf/verifier diagnostics are retained on stderr, untruncated here. */
  dprintf(2, "%s %s: %s (%d)\n", operation, hook, strerror(error), error);
  output(0,
         "{\"type\":\"status\",\"available\":false,\"operation\":\"%s\","
         "\"hook\":\"%s\",\"errno\":%d,\"reason\":\"%s\",\"exact\":false}",
         operation, hook, error, operation);
  return 2;
}
static u64 ticks(pid_t pid) {
  char path[64], raw[8192];
  snprintf(path, sizeof(path), "/proc/%d/stat", pid);
  FILE *f = fopen(path, "re");
  if (!f) return 0;
  char *result = fgets(raw, sizeof(raw), f);
  fclose(f);
  char *end = result ? strrchr(raw, ')') : NULL;
  if (!end) return 0;
  char *save, *word = strtok_r(end + 2, " ", &save);
  for (u32 i = 0; word && i < 19; i++) word = strtok_r(NULL, " ", &save);
  return word ? strtoull(word, NULL, 10) : 0;
}
static int permitted(void) {
  FILE *f = fopen("/proc/self/status", "re");
  if (!f) return 0;
  char line[256];
  unsigned long long caps = 0;
  while (fgets(line, sizeof(line), f))
    if (sscanf(line, "CapEff: %llx", &caps) == 1) break;
  fclose(f);
  return !!((caps & (1ULL << 21)) || ((caps & (1ULL << 38)) && (caps & (1ULL << 39))));
}
static int parse_plan(const char *path, struct plan *p) {
  FILE *f = fopen(path, "re");
  if (!f) return -errno;
  char line[1024], extra;
  unsigned int pid;
  unsigned long long start, netns;
  int ok = fgets(line, sizeof(line), f) &&
           sscanf(line, "P %u %llu %c", &pid, &start, &extra) == 2;
  ok = ok && pid && pid <= 0x7fffffff && start;
  if (ok) {
    p->parent = pid;
    p->start = start;
  }
  ok = ok && fgets(line, sizeof(line), f) &&
       sscanf(line, "N %llu %c", &netns, &extra) == 1 && netns;
  if (ok) p->control.netns = netns;
  for (u32 side = 0; ok && side < 2; side++) {
    ok =
        fgets(line, sizeof(line), f) && line[0] == (side ? 'R' : 'L') && line[1] == ' ';
    if (!ok) break;
    char *save, *token = strtok_r(line + 2, " \n", &save);
    u32 n = 0;
    while (token) {
      char *end;
      unsigned long port = strtoul(token, &end, 10);
      if (*end || !port || port > 65535 || n >= KVP_PORTS) {
        ok = 0;
        break;
      }
      (side ? p->control.peer_ports : p->control.local_ports)[n++] = port;
      token = strtok_r(NULL, " \n", &save);
    }
    if (side)
      p->control.peer_count = n;
    else
      p->control.local_count = n;
  }
  if (fgets(line, sizeof(line), f)) ok = 0;
  fclose(f);
  return ok ? 0 : -EINVAL;
}
static int contract(struct bpf_object *obj, int inspect) {
  struct bpf_map *map;
  unsigned int maps = 0, mask = 0;
  bpf_object__for_each_map(map, obj) {
    const char *name = bpf_map__name(map);
    unsigned int bit = !strcmp(name, "events")        ? 1
                       : !strcmp(name, "control")     ? 2
                       : !strcmp(name, "loss_counts") ? 4
                       : !strcmp(name, "allowed")     ? 8
                       : !strcmp(name, "rejected")    ? 16
                                                      : 0;
    unsigned int n = bpf_map__max_entries(map), k = bpf_map__key_size(map),
                 v = bpf_map__value_size(map), type = bpf_map__type(map);
    if (!bit || (mask & bit)) return -EINVAL;
    int bad = bit == 1   ? (type != BPF_MAP_TYPE_RINGBUF || n != KVP_RING_BYTES || k || v)
              : bit == 8 ? (type != BPF_MAP_TYPE_HASH || n != KVP_ALLOWED || k != 8 ||
                            v != sizeof(struct kvp_admission))
              : bit == 16
                  ? (type != BPF_MAP_TYPE_LRU_HASH || n != KVP_REJECTED || k != 8 ||
                     v != sizeof(struct kvp_rejected))
                  : (type != BPF_MAP_TYPE_ARRAY || n != 1 || k != 4 ||
                     v != (bit == 2 ? sizeof(struct kvp_control)
                                    : sizeof(struct kvp_loss_counts)));
    if (bad) return -EINVAL;
    mask |= bit;
    maps++;
    if (inspect)
      output(
          0,
          "{\"type\":\"map\",\"name\":\"%s\",\"entries\":%u,\"key\":%u,\"value\":%u}",
          name, n, k, v);
  }
  struct bpf_program *prog;
  unsigned int programs = 0, total = 0, seen = 0;
  const char *sections[] = {"tp_btf/tcp_retransmit_skb", "tp_btf/tcp_send_reset",
                            "tp_btf/tcp_receive_reset", "tp_btf/inet_sock_set_state"};
  bpf_object__for_each_program(prog, obj) {
    unsigned int bit = 0;
    for (u32 i = 0; i < 4; i++)
      if (!strcmp(bpf_program__section_name(prog), sections[i])) bit = 1U << i;
    size_t n = bpf_program__insn_cnt(prog);
    if (!bit || (seen & bit) || !n || n > KVP_INSNS) return -EINVAL;
    seen |= bit;
    programs++;
    total += n;
    bpf_program__set_autoattach(prog, false);
    if (inspect)
      output(0,
             "{\"type\":\"program\",\"name\":\"%s\",\"section\":\"%s\","
             "\"instructions\":%zu,\"autoattach\":false}",
             bpf_program__name(prog), bpf_program__section_name(prog), n);
  }
  return maps == 5 && mask == 31 && programs == 4 && seen == 15 && total <= 65536
             ? 0
             : -EINVAL;
}
static int map_fd(struct bpf_object *obj, const char *name) {
  return bpf_map__fd(bpf_object__find_map_by_name(obj, name));
}
static void map_diagnostics(struct bpf_object *obj) {
  struct bpf_map *map;
  bpf_object__for_each_map(map, obj) {
    char path[64], line[256];
    unsigned long long bytes = 0;
    int reported = 0;
    snprintf(path, sizeof(path), "/proc/self/fdinfo/%d", bpf_map__fd(map));
    FILE *f = fopen(path, "re");
    if (f) {
      while (fgets(line, sizeof(line), f))
        if (sscanf(line, "memlock: %llu", &bytes) == 1) {
          reported = 1;
          break;
        }
      fclose(f);
    }
    output(0,
           "{\"type\":\"resource\",\"map\":\"%s\",\"reported\":%s,\"fdinfo_memlock\":%"
           "llu,\"diagnostic_only\":true}",
           bpf_map__name(map), reported ? "true" : "false", bytes);
  }
}
static int losses(struct bpf_object *obj) {
  const char *names[] = {
      "global_null",           "global_scope_read", "global_nonfull",
      "excluded_namespace",    "excluded_protocol", "excluded_port",
      "excluded_family",       "excluded_unowned",  "rejected_untracked",
      "scoped_ring_drop",      "scoped_tuple_read", "counter_wrap"};
  _Static_assert(sizeof(names) / sizeof(*names) == STAT_COUNT, "loss names");
  struct kvp_loss_counts value;
  u32 zero = 0;
  if (bpf_map_lookup_elem(map_fd(obj, "loss_counts"), &zero, &value)) return -errno;
  for (u32 i = 0; i < STAT_COUNT; i++)
    if (output(0,
               "{\"type\":\"loss\",\"scope\":\"%s\",\"counter\":\"%s\",\"value\":%llu}",
               i <= GLOBAL_NONFULL       ? "global_unknown"
               : i <= EXCLUDED_UNOWNED   ? "filter_exclusion"
               : i == REJECTED_UNTRACKED ? "filter_accounting"
               : i == STAT_WRAP          ? "counter_health"
                                         : "endpoint_namespace",
               names[i], (unsigned long long)value.count[i]))
      return -ENOSPC;
  return 0;
}
/* Userspace's ownership commands, one per line on stdin, each acknowledged on
   stdout with its sequence number and when it was applied (CLOCK_MONOTONIC,
   before and after the map operation):
     A <seq> <cookie> <admission>  admit (replacing any entry of the cookie)
     D <seq> <cookie> <admission>  delete, only if the cookie is still under that
                                   admission (a stale delete changes nothing)
     R <seq> <cookie>              read back the cookie's entry
     stop                          detach and end
   An admit, and a delete that removes the entry, harvest the cookie's
   rejection counter (read and deleted at once, so each rejection is in exactly
   one harvest or the terminal dump): an admit's after its update (once no
   rejection can start), a delete's before it (while none can).
   A malformed command fails the collector. */
/* 0, or the errno of a harvest that failed (the counter, if any, kept) */
static int harvest(struct bpf_object *obj, u64 cookie, struct kvp_rejected *seen) {
  u64 key = cookie;
  memset(seen, 0, sizeof(*seen));
  if (!bpf_map_lookup_and_delete_elem(map_fd(obj, "rejected"), &key, seen)) return 0;
  int error = errno == ENOENT ? 0 : errno;
  memset(seen, 0, sizeof(*seen));
  return error;
}
static char line_buffer[256];
static size_t line_used;
static int command(struct bpf_object *obj, char *line) {
  unsigned long long seq, cookie, admission;
  char extra;
  int allowed = map_fd(obj, "allowed");
  u64 before = monotonic_ns();
  if (sscanf(line, "A %llu %llu %llu %c", &seq, &cookie, &admission, &extra) == 3 &&
      cookie && admission) {
    u64 key = cookie;
    struct kvp_admission value = {.admission = admission};
    int error = bpf_map_update_elem(allowed, &key, &value, BPF_ANY) ? errno : 0;
    u64 applied = monotonic_ns();
    struct kvp_rejected seen = {0};
    int harvested = error ? 0 : harvest(obj, cookie, &seen);
    return output(0,
                  "{\"type\":\"ack\",\"op\":\"A\",\"seq\":%llu,\"cookie\":%llu,"
                  "\"admission\":%llu,\"ok\":%s,\"errno\":%d,\"before_ns\":%llu,"
                  "\"applied_ns\":%llu,\"rejected\":%llu,"
                  "\"rejected_first_ns\":%llu,\"rejected_errno\":%d}",
                  seq, cookie, admission, error ? "false" : "true", error,
                  (unsigned long long)before, (unsigned long long)applied,
                  (unsigned long long)seen.count, (unsigned long long)seen.first_ns,
                  harvested);
  }
  if (sscanf(line, "D %llu %llu %llu %c", &seq, &cookie, &admission, &extra) == 3 &&
      cookie && admission) {
    u64 key = cookie;
    struct kvp_admission current;
    struct kvp_rejected seen = {0};
    int error = 0, stale = 0, harvested = 0, matched = 0;
    if (bpf_map_lookup_elem(allowed, &key, &current)) {
      if (errno == ENOENT)
        stale = 1; /* absent: nothing to delete */
      else
        error = errno; /* the entry could not be read: nothing is known */
    } else if (current.admission != admission)
      stale = 1; /* a newer admission of the cookie: kept */
    else {
      matched = 1;
      harvested = harvest(obj, cookie, &seen);
      if (bpf_map_delete_elem(allowed, &key)) error = errno;
    }
    return output(0,
                  "{\"type\":\"ack\",\"op\":\"D\",\"seq\":%llu,\"cookie\":%llu,"
                  "\"admission\":%llu,\"ok\":%s,\"errno\":%d,\"stale\":%s,"
                  "\"before_ns\":%llu,\"applied_ns\":%llu,\"rejected\":%llu,"
                  "\"rejected_first_ns\":%llu,\"rejected_errno\":%d,"
                  "\"harvested\":%s}",
                  seq, cookie, admission, error ? "false" : "true", error,
                  stale ? "true" : "false", (unsigned long long)before,
                  (unsigned long long)monotonic_ns(), (unsigned long long)seen.count,
                  (unsigned long long)seen.first_ns, harvested, matched ? "true" : "false");
  }
  if (sscanf(line, "R %llu %llu %c", &seq, &cookie, &extra) == 2 && cookie) {
    u64 key = cookie;
    struct kvp_admission current;
    int error = bpf_map_lookup_elem(allowed, &key, &current) ? errno : 0;
    char value[24] = "null";
    if (!error) snprintf(value, sizeof(value), "%llu", (unsigned long long)current.admission);
    return output(0,
                  "{\"type\":\"ack\",\"op\":\"R\",\"seq\":%llu,\"cookie\":%llu,"
                  "\"ok\":%s,\"errno\":%d,\"admission\":%s,\"applied_ns\":%llu}",
                  seq, cookie, error && error != ENOENT ? "false" : "true",
                  error == ENOENT ? 0 : error, value, (unsigned long long)monotonic_ns());
  }
  return -EINVAL;
}
/* Read what stdin holds: 1 on stop, -1 on a malformed command or EOF. */
static int commands(struct bpf_object *obj) {
  ssize_t size = read(0, line_buffer + line_used, sizeof(line_buffer) - 1 - line_used);
  if (size < 0 && (errno == EAGAIN || errno == EINTR)) return 0;
  if (size <= 0) return -1;
  line_used += size;
  line_buffer[line_used] = 0;
  char *start = line_buffer, *end;
  while ((end = memchr(start, '\n', line_used - (start - line_buffer)))) {
    *end = 0;
    if (!strcmp(start, "stop")) return 1;
    if (command(obj, start)) return -1;
    start = end + 1;
  }
  line_used -= start - line_buffer;
  memmove(line_buffer, start, line_used);
  return line_used >= sizeof(line_buffer) - 1 ? -1 : 0;
}
/* Every per-cookie rejection counter, at the end (bounded by the map), with the
   records the output queue admits; the exact total is the loss counter. */
static int rejected_complete;
static int rejected_dump(struct bpf_object *obj, u64 deadline) {
  int fd = map_fd(obj, "rejected");
  u64 key, next, rows = 0;
  int first = 1, unreadable = 0;
  rejected_complete = 0;
  /* twice the map: a key evicted meanwhile restarts the walk */
  while (rows < 2 * KVP_REJECTED && monotonic_ns() < deadline) {
    if (bpf_map_get_next_key(fd, first ? NULL : &key, &next)) {
      rejected_complete = errno == ENOENT && !unreadable;
      break;
    }
    first = 0;
    key = next;
    /* hooks may still be updating it: read until two reads agree (each hook
       sets its earliest time before its count, so equal reads are
       consistent), else the row says it is not */
    struct kvp_rejected value, again;
    int consistent = 0, gone = 0;
    if (bpf_map_lookup_elem(fd, &key, &value)) {
      if (errno != ENOENT) unreadable = 1; /* evicted meanwhile is not */
      continue;
    }
    for (int i = 0; i < 8 && !consistent && !gone; i++) {
      if (bpf_map_lookup_elem(fd, &key, &again)) {
        if (errno != ENOENT) unreadable = 1;
        gone = 1;
      } else if (!memcmp(&value, &again, sizeof(value)))
        consistent = 1;
      else
        value = again;
    }
    if (gone) continue;
    while (count >= KVP_QUEUE - 1 && monotonic_ns() < deadline) {
      if (flush()) return -EIO;
      struct pollfd out = {1, POLLOUT, 0};
      poll(&out, 1, 5);
    }
    if (output(0,
               "{\"type\":\"rejected\",\"cookie\":%llu,\"count\":%llu,"
               "\"first_ns\":%llu,\"last_ns\":%llu,\"consistent\":%s}",
               (unsigned long long)key, (unsigned long long)value.count,
               (unsigned long long)value.first_ns, (unsigned long long)value.last_ns,
               consistent ? "true" : "false"))
      return -ENOSPC;
    rows++;
  }
  return 0;
}
static int run(struct bpf_object *obj, struct plan *p) {
  if (!permitted()) return status("missing_effective_privilege", "", EPERM);
  if (kvp_parent_guard(p->parent) || ticks(p->parent) != p->start)
    return status("parent_identity", "", ESRCH);
  struct stat ns;
  if (stat("/proc/self/ns/net", &ns) || ns.st_ino != p->control.netns)
    return status("namespace_identity", "", EINVAL);
  /* Disable libbpf's automatic rlimit adjustment; never raise host policy. */
  libbpf_set_memlock_rlim(0);
  int error = bpf_object__load(obj);
  if (error) return status("kernel_load", "", error);
  map_diagnostics(obj);
  u32 zero = 0;
  if (bpf_map_update_elem(map_fd(obj, "control"), &zero, &p->control, BPF_ANY))
    return status("control", "", errno);
  struct ring_buffer *ring =
      ring_buffer__new(map_fd(obj, "events"), event_row, NULL, NULL);
  if (!ring) return status("ring_open", "", errno);
  struct bpf_link *links[KVP_EVENTS] = {0};
  u32 n = 0;
  struct bpf_program *prog;
  int rc = 2, failed = 0;
  bpf_object__for_each_program(prog, obj) {
    struct bpf_link *link = bpf_program__attach_trace(prog);
    error = libbpf_get_error(link);
    if (error) {
      status("kernel_attach", bpf_program__section_name(prog), error);
      goto cleanup;
    }
    links[n++] = link;
  }
  p->control.enabled = 1;
  if (bpf_map_update_elem(map_fd(obj, "control"), &zero, &p->control, BPF_ANY)) {
    status("enable", "", errno);
    goto cleanup;
  }
  output(0,
         "{\"type\":\"ready\",\"available\":true,\"exact\":false,\"coverage\":"
         "\"partial\"}");
  u64 next = monotonic_ns() + 1000000000ULL;
  while (!failed) {
    struct pollfd fds[] = {{0, POLLIN | POLLHUP, 0},
                           {ring_buffer__epoll_fd(ring), POLLIN, 0}};
    if (poll(fds, 2, 20) < 0 && errno != EINTR) {
      failed = 1;
      break;
    }
    if (fds[0].revents) {
      int result = commands(obj);
      if (result > 0) break;
      if (result < 0) {
        failed = 1;
        break;
      }
    }
    if (ring_buffer__consume_n(ring, 256) < 0 || flush()) {
      failed = 1;
      break;
    }
    u64 now = monotonic_ns();
    if (now >= next) {
      if (losses(obj)) {
        failed = 1;
        break;
      }
      next = now + 1000000000ULL;
    }
  }
  p->control.enabled = 0;
  if (bpf_map_update_elem(map_fd(obj, "control"), &zero, &p->control, BPF_ANY))
    failed = 1;
cleanup:
  for (u32 i = 0; i < n; i++) {
    error = bpf_link__destroy(links[i]);
    if (error) {
      status("kernel_detach", "", error);
      failed = 1;
    }
    links[i] = NULL;
  }
  /* Detach is not a claimed producer-quiescence barrier. Preserve partial status. */
  u64 deadline = monotonic_ns() + 1000000000ULL;
  while (monotonic_ns() < deadline) {
    int consumed = ring_buffer__consume_n(ring, 256);
    if (consumed < 0 || flush()) {
      failed = 1;
      break;
    }
    if (!consumed) break;
  }
  /* Make room for independent terminal/loss records, bounded by the same deadline. */
  while (count && monotonic_ns() < deadline && !flush()) {
    struct pollfd out = {1, POLLOUT, 0};
    poll(&out, 1, 5);
  }
  if (losses(obj)) failed = 1;
  if (rejected_dump(obj, deadline)) failed = 1;
  if (output(0,
             "{\"type\":\"end\",\"complete\":false,\"reason\":\"producer_quiescence_"
             "and_miss_visibility_unverified\","
             "\"coverage\":\"partial\",\"output_drops\":%llu,\"malformed\":%llu,"
             "\"rejected_complete\":%s,\"failed\":%s}",
             (unsigned long long)output_drops, (unsigned long long)malformed,
             rejected_complete ? "true" : "false",
             failed || output_broken ? "true" : "false"))
    failed = 1;
  while (count && monotonic_ns() < deadline && !flush()) {
    struct pollfd out = {1, POLLOUT, 0};
    poll(&out, 1, 5);
  }
  if (count || failed || output_broken || malformed) rc = 3;
  ring_buffer__free(ring);
  return rc;
}
#ifndef KVP_TEST
int main(int argc, char **argv) {
  signal(SIGPIPE, SIG_IGN);
  int inspect = argc == 3 && !strcmp(argv[1], "--inspect");
  struct plan p = {0};
  if (!inspect) {
    if (argc != 4 || strcmp(argv[1], "--arm") || parse_plan(argv[2], &p))
      return status("invalid_plan", "", EINVAL), flush(), 2;
    if (!permitted() || kvp_parent_guard(p.parent) || ticks(p.parent) != p.start)
      return status("privilege_or_parent", "", EPERM), flush(), 2;
  }
  fcntl(1, F_SETFL, fcntl(1, F_GETFL) | O_NONBLOCK);
  struct bpf_object *obj = bpf_object__open_file(argv[argc - 1], NULL);
  int error = libbpf_get_error(obj), rc;
  if (error) return status("object_open", "", error), flush(), 2;
  if (contract(obj, inspect))
    rc = status("artifact_contract", "", EINVAL);
  else
    rc = inspect ? 0 : run(obj, &p);
  bpf_object__close(obj);
  if (flush() || count) rc = 3;
  return rc;
}
#endif
