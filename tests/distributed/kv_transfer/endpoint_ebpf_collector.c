/* SPDX-License-Identifier: Apache-2.0 */
/* Kernel/libbpf boundary emulation with a real unopened-to-kernel BPF object. */
#define _GNU_SOURCE
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <pthread.h>
#include <stdio.h>
static int fixture_load(struct bpf_object *);
static struct bpf_link *fixture_attach(const struct bpf_program *);
static int fixture_destroy(struct bpf_link *);
static int fixture_map_fd(const struct bpf_map *);
static int fixture_update(int, const void *, const void *, __u64);
static int fixture_lookup(int, const void *, void *);
static int fixture_delete(int, const void *);
static int fixture_next_key(int, const void *, void *);
static int fixture_lookup_and_delete(int, const void *, void *);
static struct ring_buffer *fixture_ring_new(int, ring_buffer_sample_fn, void *,
                                            const struct ring_buffer_opts *);
static int fixture_consume(struct ring_buffer *);
static int fixture_consume_n(struct ring_buffer *, size_t);
static int fixture_ring_fd(const struct ring_buffer *);
static void fixture_ring_free(struct ring_buffer *);
static FILE *fixture_fopen(const char *, const char *);
#define bpf_object__load fixture_load
#define bpf_program__attach_trace fixture_attach
#define bpf_link__destroy fixture_destroy
#define bpf_map__fd fixture_map_fd
#define bpf_map_update_elem fixture_update
#define bpf_map_lookup_elem fixture_lookup
#define bpf_map_delete_elem fixture_delete
#define bpf_map_get_next_key fixture_next_key
#define bpf_map_lookup_and_delete_elem fixture_lookup_and_delete
#define ring_buffer__new fixture_ring_new
#define ring_buffer__consume fixture_consume
#define ring_buffer__consume_n fixture_consume_n
#define ring_buffer__epoll_fd fixture_ring_fd
#define ring_buffer__free fixture_ring_free
#define fopen fixture_fopen
#define KVP_TEST 1
#include "csrc/kv_profiler/ebpf/collector.c"
#undef bpf_object__load
#undef bpf_program__attach_trace
#undef bpf_link__destroy
#undef bpf_map__fd
#undef bpf_map_update_elem
#undef bpf_map_lookup_elem
#undef bpf_map_delete_elem
#undef bpf_map_get_next_key
#undef bpf_map_lookup_and_delete_elem
#undef ring_buffer__new
#undef ring_buffer__consume
#undef ring_buffer__consume_n
#undef ring_buffer__epoll_fd
#undef ring_buffer__free
#undef fopen
/* Exact build-dependency source, supplied by the unprivileged test command. */
#include KVP_LIBBPF_RING_SOURCE
#include <assert.h>
static unsigned int attached, destroyed, rings, frees, attempt, failure_at;
static int load_error, destroy_error;
static uint64_t last_enabled;
static unsigned long consumer_pos, producer_pos;
static unsigned int seeded_records, active_batches, stop_checks, largest_batch;
static int stop_writer = -1;
static unsigned int output_events(void) {
  struct stat st;
  assert(!fstat(1, &st));
  char *text = calloc(1, st.st_size + 1);
  assert(text);
  assert(pread(1, text, st.st_size, 0) == st.st_size);
  unsigned int n = 0;
  for (char *p = text; (p = strstr(p, "\"type\":\"tcp_event\"")); p++) n++;
  free(text);
  return n;
}
static const char *fdinfo = "map_type: 27\n";
static int fixture_load(struct bpf_object *o) {
  (void)o;
  return load_error;
}
static struct bpf_link *fixture_attach(const struct bpf_program *p) {
  (void)p;
  if (++attempt == failure_at) return (struct bpf_link *)(long)-ENOENT;
  attached++;
  return (struct bpf_link *)malloc(1);
}
static int fixture_destroy(struct bpf_link *link) {
  free(link);
  destroyed++;
  return destroy_error;
}
/* One descriptor per map; the two hash maps are small emulated tables. */
enum { FD_EVENTS = 771, FD_CONTROL, FD_LOSS, FD_ALLOWED, FD_REJECTED };
static uint64_t owned_keys[8], owned_values[8], rejected_keys[KVP_REJECTED];
static struct kvp_rejected rejected_values[KVP_REJECTED];
static uint64_t rejected_unreadable, rejected_unstable;
static unsigned int owned_count, rejected_count;
static int delete_error, update_error, lookup_error;
static int fixture_map_fd(const struct bpf_map *map) {
  const char *name = bpf_map__name(map);
  return !strcmp(name, "events")     ? FD_EVENTS
         : !strcmp(name, "control")  ? FD_CONTROL
         : !strcmp(name, "allowed")  ? FD_ALLOWED
         : !strcmp(name, "rejected") ? FD_REJECTED
                                     : FD_LOSS;
}
static int owned_index(uint64_t key) {
  for (unsigned int i = 0; i < owned_count; i++)
    if (owned_keys[i] == key) return i;
  return -1;
}
static int fixture_update(int fd, const void *k, const void *v, __u64 flags) {
  (void)flags;
  if (fd == FD_ALLOWED) {
    if (update_error) return errno = update_error, -1;
    uint64_t key = *(const uint64_t *)k;
    int i = owned_index(key);
    if (i < 0) {
      i = owned_count++;
      owned_keys[i] = key;
    }
    owned_values[i] = ((const struct kvp_admission *)v)->admission;
    return 0;
  }
  last_enabled = ((const struct kvp_control *)v)->enabled;
  return 0;
}
static int fixture_lookup(int fd, const void *k, void *v) {
  if (fd == FD_ALLOWED) {
    if (lookup_error) return errno = lookup_error, -1;
    int i = owned_index(*(const uint64_t *)k);
    if (i < 0) return errno = ENOENT, -1;
    ((struct kvp_admission *)v)->admission = owned_values[i];
    return 0;
  }
  if (fd == FD_REJECTED) {
    if (rejected_unreadable && *(const uint64_t *)k == rejected_unreadable)
      return errno = EIO, -1;
    for (unsigned int i = 0; i < rejected_count; i++)
      if (rejected_keys[i] == *(const uint64_t *)k) {
        /* a hook still counting: every read differs */
        if (rejected_keys[i] == rejected_unstable) rejected_values[i].count++;
        *(struct kvp_rejected *)v = rejected_values[i];
        return 0;
      }
    return errno = ENOENT, -1;
  }
  (void)k;
  memset(v, 0, sizeof(struct kvp_loss_counts));
  return 0;
}
static int fixture_delete(int fd, const void *k) {
  assert(fd == FD_ALLOWED);
  if (delete_error) return errno = delete_error, -1;
  int i = owned_index(*(const uint64_t *)k);
  if (i < 0) return errno = ENOENT, -1;
  owned_keys[i] = owned_keys[--owned_count];
  owned_values[i] = owned_values[owned_count];
  return 0;
}
static int fixture_lookup_and_delete(int fd, const void *k, void *v) {
  assert(fd == FD_REJECTED);
  for (unsigned int i = 0; i < rejected_count; i++)
    if (rejected_keys[i] == *(const uint64_t *)k) {
      *(struct kvp_rejected *)v = rejected_values[i];
      rejected_keys[i] = rejected_keys[--rejected_count];
      rejected_values[i] = rejected_values[rejected_count];
      return 0;
    }
  return errno = ENOENT, -1;
}
static int fixture_next_key(int fd, const void *k, void *next) {
  assert(fd == FD_REJECTED);
  unsigned int i = 0;
  if (k) {
    while (i < rejected_count && rejected_keys[i] != *(const uint64_t *)k) i++;
    i++;
  }
  if (i >= rejected_count) return errno = ENOENT, -1;
  *(uint64_t *)next = rejected_keys[i];
  return 0;
}
static struct ring_buffer *fixture_ring_new(int fd, ring_buffer_sample_fn cb, void *ctx,
                                            const struct ring_buffer_opts *opts) {
  (void)fd;
  (void)opts;
  rings++;
  struct ring_buffer *rb = calloc(1, sizeof(*rb));
  assert(rb);
  rb->rings = calloc(1, sizeof(*rb->rings));
  assert(rb->rings);
  struct ring *ring = calloc(1, sizeof(*ring));
  assert(ring);
  rb->rings[0] = ring;
  rb->ring_cnt = 1;
  ring->data = calloc(1, KVP_RING_BYTES);
  assert(ring->data);
  consumer_pos = 0;
  unsigned long stride = sizeof(struct ringbuf_hdr) + sizeof(struct kvp_event);
  producer_pos = seeded_records * stride;
  assert(producer_pos < KVP_RING_BYTES);
  ring->consumer_pos = &consumer_pos;
  ring->producer_pos = &producer_pos;
  ring->mask = KVP_RING_BYTES - 1;
  ring->sample_cb = cb;
  ring->ctx = ctx;
  struct kvp_event event = {.schema = 3,
                            .size = 256,
                            .cookie = 42,
                            .admission = 7,
                            .kind = KVP_STATE,
                            .family = AF_INET,
                            .timestamp_ns = 123,
                            .netns = 99,
                            .old_state = 1,
                            .new_state = 7};
  for (unsigned int i = 0; i < seeded_records; i++) {
    struct ringbuf_hdr *header = (void *)((char *)ring->data + i * stride);
    header->len = sizeof(event);
    memcpy(header + 1, &event, sizeof(event));
  }
  return rb;
}
static int checked_drain(struct ring_buffer *r, size_t n, int bounded) {
  if (seeded_records && !last_enabled && !stop_checks++) {
    /* The stop command was handled before draining the remaining records. */
    assert(active_batches == 1 && destroyed == 4);
    assert(output_events() == 256);
  }
  int consumed = bounded ? ring_buffer__consume_n(r, n) : ring_buffer__consume(r);
  assert(consumed >= 0);
  if (seeded_records)
    fprintf(stderr, "drain: active=%d bounded=%d consumed=%d\n", !!last_enabled,
            bounded, consumed);
  if ((unsigned int)consumed > largest_batch) largest_batch = consumed;
  assert(consumed <= 256);
  if (seeded_records && last_enabled) {
    assert(++active_batches == 1 && consumed == 256);
    assert(write(stop_writer, "stop\n", 5) == 5);
  }
  return consumed;
}
static int __attribute__((unused)) fixture_consume(struct ring_buffer *r) {
  return checked_drain(r, 0, 0);
}
static int __attribute__((unused)) fixture_consume_n(struct ring_buffer *r, size_t n) {
  assert(n == 256);
  return checked_drain(r, n, 1);
}
static int fixture_ring_fd(const struct ring_buffer *r) {
  (void)r;
  return -1;
}
static void fixture_ring_free(struct ring_buffer *r) {
  frees++;
  free(r->rings[0]->data);
  free(r->rings[0]);
  free(r->rings);
  free(r);
}
static FILE *fixture_fopen(const char *path, const char *mode) {
  if (!strcmp(path, "/proc/self/status")) {
    const char *caps = "CapEff: 0000000000200000\n";
    return fmemopen((void *)caps, strlen(caps), "r");
  }
  if (!strncmp(path, "/proc/self/fdinfo/", 18))
    return fmemopen((void *)fdinfo, strlen(fdinfo), "r");
  return fopen(path, mode);
}
static void reset(void) {
  memset(queue, 0, sizeof(queue));
  head = count = written = output_drops = malformed = output_broken = 0;
  attempt = attached = destroyed = rings = frees = 0;
  load_error = failure_at = destroy_error = 0;
  seeded_records = active_batches = stop_checks = largest_batch = 0;
  owned_count = rejected_count = 0;
  rejected_unreadable = rejected_unstable = 0;
  delete_error = update_error = lookup_error = 0;
  line_used = 0;
}
/* Run with ``input`` on stdin and the output captured: the output's text. */
static char *run_with_input(struct bpf_object *obj, struct plan *p, const char *input,
                            int *rc) {
  int pipe_fds[2];
  assert(!pipe(pipe_fds));
  assert(write(pipe_fds[1], input, strlen(input)) == (ssize_t)strlen(input));
  close(pipe_fds[1]);
  assert(dup2(pipe_fds[0], 0) == 0);
  close(pipe_fds[0]);
  int saved = dup(1);
  FILE *captured = tmpfile();
  assert(saved >= 0 && captured && dup2(fileno(captured), 1) == 1);
  *rc = run(obj, p);
  flush();
  assert(dup2(saved, 1) == 1);
  close(saved);
  long size = ftell(captured);
  if (size <= 0) size = lseek(fileno(captured), 0, SEEK_END);
  char *text = calloc(1, size + 1);
  assert(text && pread(fileno(captured), text, size, 0) == size);
  fclose(captured);
  return text;
}
/* A reader of stdout that starts late: what it read, once the pipe ends. */
struct late_reader {
  int fd;
  unsigned int delay_us;
  char *text;
  size_t used;
};
static void *read_late(void *arg) {
  struct late_reader *r = arg;
  usleep(r->delay_us);
  int flags = fcntl(r->fd, F_GETFL);
  fcntl(r->fd, F_SETFL, flags & ~O_NONBLOCK);
  size_t cap = 1 << 20;
  r->text = malloc(cap);
  ssize_t n;
  while ((n = read(r->fd, r->text + r->used, cap - 1 - r->used)) > 0) r->used += n;
  r->text[r->used] = 0;
  return NULL;
}
/* Commands fed as the collector takes them (a pipe of any size). */
struct feeder {
  int fd;
  unsigned int commands;
};
static void *feed(void *arg) {
  struct feeder *f = arg;
  char line[64];
  /* the shortest command: one buffer holds the most, so the queue can come
     within one slot of full */
  for (unsigned int i = 1; i <= f->commands; i++) {
    int n = snprintf(line, sizeof(line), "R 1 4\n");
    assert(write(f->fd, line, n) == n);
  }
  assert(write(f->fd, "stop\n", 5) == 5);
  close(f->fd);
  return NULL;
}
static void test_commands_wait_for_output_room(struct bpf_object *obj, struct plan *p) {
  /* More commands than the output queue holds, through 4 KiB pipes, while
     nothing reads stdout for 1.4 s (past a periodic counter report): the
     collector holds the commands back (backpressure on stdin) and defers the
     report instead of failing, and once stdout is read every command is
     acknowledged, the counters reported. */
  reset();
  enum { COMMANDS = 5000 };
  int in[2], out[2];
  assert(!pipe(in) && !pipe(out));
  fcntl(in[1], 1031 /* F_SETPIPE_SZ */, 4096);
  fcntl(out[1], 1031 /* F_SETPIPE_SZ */, 4096);
  fcntl(out[1], F_SETFL, fcntl(out[1], F_GETFL) | O_NONBLOCK);
  int saved_in = dup(0), saved_out = dup(1);
  assert(dup2(in[0], 0) == 0 && dup2(out[1], 1) == 1);
  close(in[0]);
  close(out[1]);
  struct late_reader reader = {.fd = out[0], .delay_us = 1400000};
  struct feeder feeder = {.fd = in[1], .commands = COMMANDS};
  pthread_t reading, feeding;
  /* the pipe full, and output queued within ten slots of full: the periodic
     report falls due while it has no room for its rows */
  unsigned int piped = 0;
  while (write(1, "{\"type\":\"pad\"}\n", 15) == 15) piped++;
  for (unsigned int i = 0; i < KVP_QUEUE - 10; i++)
    assert(!output(0, "{\"type\":\"pad\",\"i\":%u}", i));
  assert(!pthread_create(&reading, NULL, read_late, &reader));
  assert(!pthread_create(&feeding, NULL, feed, &feeder));
  int rc = run(obj, p);
  while (count && !flush()) usleep(1000);
  assert(dup2(saved_out, 1) == 1 && dup2(saved_in, 0) == 0);
  close(saved_out);
  close(saved_in);
  assert(!pthread_join(feeding, NULL) && !pthread_join(reading, NULL));
  close(out[0]);
  unsigned int acks = 0, reports = 0;
  for (char *at = reader.text; (at = strstr(at, "\"op\":\"R\"")); at++) acks++;
  for (char *at = reader.text; (at = strstr(at, "\"counter\":\"counter_wrap\""));
       at++)
    reports++;
  unsigned int pads = 0;
  for (char *at = reader.text; (at = strstr(at, "\"type\":\"pad\"")); at++) pads++;
  assert(pads == piped + KVP_QUEUE - 10);
  assert(rc == 2 && acks == COMMANDS && reports >= 2); /* a periodic and the final */
  assert(strstr(reader.text, "\"failed\":false"));
  free(reader.text);
  puts("PASS: commands and a periodic report wait for output room: backpressure on "
       "stdin through small pipes, a stall past a report, no failed acknowledgement");
}
static void test_ownership_commands(struct bpf_object *obj, struct plan *p) {
  /* Admit (with the cookie's rejections so far), a stale delete across a
     replacement, a real delete, readbacks, a failed delete, the terminal
     per-cookie rejection dump; acknowledgements carry their sequence. */
  reset();
  rejected_keys[0] = 42;
  rejected_values[0] = (struct kvp_rejected){.count = 3, .first_ns = 1, .last_ns = 2};
  rejected_keys[1] = 900;
  rejected_values[1] = (struct kvp_rejected){.count = 5, .first_ns = 3, .last_ns = 4};
  rejected_count = 2;
  int rc;
  char *out = run_with_input(obj, p,
                             "A 1 42 7\nA 2 43 8\nA 3 42 9\nD 4 42 7\nD 5 43 8\n"
                             "R 6 42\nR 7 43\nstop\n",
                             &rc);
  assert(rc == 2);
  assert(strstr(out, "\"op\":\"A\",\"seq\":1,\"cookie\":42,\"admission\":7,\"ok\":true"));
  /* the admit harvests (reads and deletes) the cookie's counter; its second
     admit finds none; a matched delete harvests too, a stale one does not */
  assert(strstr(out, "\"seq\":1,\"cookie\":42,\"admission\":7,\"ok\":true") &&
         strstr(out, "\"rejected\":3,\"rejected_first_ns\":1,\"rejected_errno\":0}"));
  assert(strstr(out, "\"stale\":true,") && strstr(out, "\"harvested\":false}") &&
         strstr(out, "\"harvested\":true}"));
  assert(!strstr(out, "{\"type\":\"rejected\",\"cookie\":42,"));
  /* 42 is under admission 9 now: deleting admission 7 is stale, keeps it */
  assert(strstr(out, "\"op\":\"D\",\"seq\":4,\"cookie\":42,\"admission\":7,\"ok\":true,"
                     "\"errno\":0,\"stale\":true"));
  assert(strstr(out, "\"op\":\"D\",\"seq\":5,\"cookie\":43,\"admission\":8,\"ok\":true,"
                     "\"errno\":0,\"stale\":false"));
  assert(strstr(out, "\"op\":\"R\",\"seq\":6,\"cookie\":42,\"ok\":true,\"errno\":0,"
                     "\"admission\":9"));
  assert(strstr(out, "\"op\":\"R\",\"seq\":7,\"cookie\":43,\"ok\":true,\"errno\":0,"
                     "\"admission\":null"));
  assert(strstr(out, "{\"type\":\"rejected\",\"cookie\":900,\"count\":5,\"first_ns\":3,"
                     "\"last_ns\":4,\"consistent\":true}"));
  assert(strstr(out, "\"rejected_complete\":true,\"failed\":false"));
  free(out);
  /* a delete that really fails is acknowledged as failed, with its errno */
  reset();
  out = run_with_input(obj, p, "A 1 42 7\nstop\n", &rc);
  free(out);
  delete_error = EIO;
  line_used = 0;
  out = run_with_input(obj, p, "D 2 42 7\nstop\n", &rc);
  assert(strstr(out, "\"op\":\"D\",\"seq\":2,\"cookie\":42,\"admission\":7,\"ok\":false,"
                     "\"errno\":5"));
  free(out);
  /* an entry that cannot be read is no stale delete: failed, with its errno */
  delete_error = 0;
  lookup_error = EPERM;
  line_used = 0;
  out = run_with_input(obj, p, "D 3 42 7\nstop\n", &rc);
  assert(strstr(out, "\"op\":\"D\",\"seq\":3,\"cookie\":42,\"admission\":7,\"ok\":false,"
                     "\"errno\":1,\"stale\":false"));
  free(out);
  /* a malformed command fails the collector */
  reset();
  out = run_with_input(obj, p, "A one 42 7\n", &rc);
  assert(rc == 3 && strstr(out, "\"failed\":true"));
  free(out);
  /* the dump is complete at the map's end, a full map's too, and not when an
     entry could not be read */
  for (int unreadable = 0; unreadable < 2; unreadable++) {
    reset();
    rejected_count = KVP_REJECTED;
    for (unsigned int i = 0; i < KVP_REJECTED; i++) {
      rejected_keys[i] = 1000 + i;
      rejected_values[i] = (struct kvp_rejected){.count = 1, .first_ns = 1, .last_ns = 1};
    }
    rejected_unreadable = unreadable ? 1002 : 0;
    out = run_with_input(obj, p, "stop\n", &rc);
    unsigned int rows = 0;
    for (char *at = out; (at = strstr(at, "{\"type\":\"rejected\"")); at++) rows++;
    assert(rows == KVP_REJECTED - unreadable);
    assert(strstr(out, unreadable ? "\"rejected_complete\":false" : "\"rejected_complete\":true"));
    free(out);
  }
  /* an entry still changing under the dump's reads is reported inconsistent */
  reset();
  rejected_keys[0] = 77;
  rejected_values[0] = (struct kvp_rejected){.count = 1, .first_ns = 1, .last_ns = 1};
  rejected_count = 1;
  rejected_unstable = 77;
  out = run_with_input(obj, p, "stop\n", &rc);
  assert(strstr(out, "\"cookie\":77,") && strstr(out, "\"consistent\":false}"));
  free(out);
  puts("PASS: sequenced admit/delete/readback acknowledgements, stale and failed "
       "deletes, an unreadable entry's delete failed, harvests at admit and delete, full and "
       "unreadable dumps, inconsistent entries, terminal per-cookie rejection counters, malformed command");
}
int main(int argc, char **argv) {
  assert(argc == 2);
  struct bpf_object *obj = bpf_object__open_file(argv[1], NULL);
  assert(!libbpf_get_error(obj) && !contract(obj, 0));
  struct plan p = {.parent = getppid(), .start = ticks(getppid())};
  struct stat ns;
  assert(!stat("/proc/self/ns/net", &ns));
  p.control.netns = ns.st_ino;
  FILE *plan_file = tmpfile();
  assert(plan_file);
  fprintf(plan_file, "P %d %llu\nN %llu\nL \nR 1234\n", p.parent,
          (unsigned long long)p.start, (unsigned long long)p.control.netns);
  fflush(plan_file);
  char plan_path[64];
  snprintf(plan_path, sizeof(plan_path), "/proc/self/fd/%d", fileno(plan_file));
  struct plan parsed = {0};
  assert(!parse_plan(plan_path, &parsed));
  assert(parsed.control.local_count == 0 && parsed.control.peer_ports[0] == 1234);
  fclose(plan_file);
  /* address lines are no part of the plan any more, nor anything after it */
  const char *bad[] = {"LA 10.0.0.1\nRA \n", "extra\n"};
  for (unsigned int i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
    plan_file = tmpfile();
    assert(plan_file);
    fprintf(plan_file, "P %d %llu\nN %llu\nL \nR \n%s", p.parent,
            (unsigned long long)p.start, (unsigned long long)p.control.netns, bad[i]);
    fflush(plan_file);
    snprintf(plan_path, sizeof(plan_path), "/proc/self/fd/%d", fileno(plan_file));
    struct plan refused = {0};
    assert(parse_plan(plan_path, &refused) == -EINVAL);
    fclose(plan_file);
  }
  reset();
  load_error = -EPERM;
  assert(run(obj, &p) == 2 && !attached && !rings);
  assert(strstr(queue[0], "kernel_load") && strstr(queue[0], "\"errno\":1"));
  for (failure_at = 1; failure_at <= 4;) {
    unsigned int fail = failure_at;
    reset();
    failure_at = fail;
    assert(run(obj, &p) == 2 && attached == fail - 1 && destroyed == attached);
    assert(rings == frees && strstr(queue[5], "kernel_attach"));
    failure_at = fail + 1;
  }
  for (int broken = 0; broken < 2; broken++) {
    reset();
    destroy_error = broken ? -EIO : 0;
    int input[2];
    assert(!pipe(input));
    assert(write(input[1], "stop\n", 5) == 5);
    close(input[1]);
    assert(dup2(input[0], 0) == 0);
    close(input[0]);
    assert(run(obj, &p) == (broken ? 3 : 2));
    assert(attached == 4 && destroyed == 4 && rings == frees && !last_enabled);
    int terminal = 0;
    for (unsigned int i = 0; i < KVP_QUEUE; i++)
      if (strstr(queue[i], "producer_quiescence_and_miss_visibility_unverified")) {
        terminal = 1;
        assert(strstr(queue[i], "\"complete\":false"));
        assert(strstr(queue[i], broken ? "\"failed\":true" : "\"failed\":false"));
      }
    assert(terminal);
  }
  /* Exercise actual production run/cleanup loops and the pinned consumer. */
  reset();
  seeded_records = 1024;
  int commands[2];
  assert(!pipe(commands));
  stop_writer = commands[1];
  assert(dup2(commands[0], 0) == 0);
  close(commands[0]);
  int saved_stdout = dup(1);
  assert(saved_stdout >= 0);
  FILE *captured = tmpfile();
  assert(captured);
  assert(dup2(fileno(captured), 1) == 1);
  assert(run(obj, &p) == 2);
  assert(active_batches == 1 && stop_checks && largest_batch == 256);
  assert(output_events() == 1024 && !output_drops && !malformed);
  assert(consumer_pos == producer_pos && rings == frees && attached == destroyed);
  assert(dup2(saved_stdout, 1) == 1);
  close(saved_stdout);
  fclose(captured);
  close(stop_writer);
  stop_writer = -1;
  puts(
      "PASS: exact pinned libbpf consumed 1024 events in <=256-record batches; "
      "256 events flushed and stop checked before shutdown drained the other 768");
  test_ownership_commands(obj, &p);
  test_commands_wait_for_output_room(obj, &p);
  reset();
  struct kvp_event e = {.schema = 3,
                        .size = 256,
                        .cookie = 42,
                        .admission = 7,
                        .kind = KVP_STATE,
                        .family = 2,
                        .timestamp_ns = 123,
                        .netns = 99,
                        .old_state = 1,
                        .new_state = 7,
                        .available = 255};
  e.local_address[0] = 127;
  e.local_address[3] = 1;
  e.peer_address[0] = 127;
  e.peer_address[3] = 2;
  for (unsigned int i = 0; i < 8; i++) e.fields[i] = i + 10;
  event_row(NULL, &e, sizeof(e));
  assert(strstr(queue[0], "\"cookie\":42,\"admission\":7,\"old_state\":1,\"new_state\":7") &&
         strstr(queue[0], "\"value\":10"));
  assert(strstr(queue[0], "usec_shift_3") && lengths[0] < KVP_RECORD_BYTES);
  for (unsigned int i = 0; i < KVP_QUEUE + 10; i++) event_row(NULL, &e, sizeof(e));
  assert(count == KVP_QUEUE - 1 && output_drops == 12);
  assert(!output(0, "{\"type\":\"end\",\"complete\":false}"));
  assert(count == KVP_QUEUE);
  reset();
  map_diagnostics(obj);
  assert(count == 5 && strstr(queue[0], "\"reported\":false"));
  reset();
  fdinfo = "memlock: 99999999999\n";
  map_diagnostics(obj);
  assert(count == 5 && strstr(queue[0], "99999999999") && !strstr(queue[0], "FAIL"));
  bpf_object__close(obj);
  puts(
      "PASS: real object; emulated kernel load/attach errors; partial-link cleanup; "
      "event serializer/queue loss; diagnostic-only memory; no BPF load/attach");
}
