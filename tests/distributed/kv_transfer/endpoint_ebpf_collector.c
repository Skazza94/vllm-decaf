/* SPDX-License-Identifier: Apache-2.0 */
/* Kernel/libbpf boundary emulation with a real unopened-to-kernel BPF object. */
#define _GNU_SOURCE
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <stdio.h>
static int fixture_load(struct bpf_object *);
static struct bpf_link *fixture_attach(const struct bpf_program *);
static int fixture_destroy(struct bpf_link *);
static int fixture_map_fd(const struct bpf_map *);
static int fixture_update(int, const void *, const void *, __u64);
static int fixture_lookup(int, const void *, void *);
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
static int fixture_map_fd(const struct bpf_map *map) {
  (void)map;
  return 777;
}
static int fixture_update(int fd, const void *k, const void *v, __u64 flags) {
  (void)fd;
  (void)k;
  last_enabled = ((const struct kvp_control *)v)->enabled;
  (void)flags;
  return 0;
}
static int fixture_lookup(int fd, const void *k, void *v) {
  (void)fd;
  (void)k;
  memset(v, 0, sizeof(struct kvp_loss_counts));
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
  struct kvp_event event = {.schema = 2,
                            .size = 256,
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
  fprintf(plan_file,
          "P %d %llu\nN %llu\nL \nR 1234\nLA 10.0.0.1 ::ffff:10.0.0.2\nRA fd00::5\n",
          p.parent, (unsigned long long)p.start, (unsigned long long)p.control.netns);
  fflush(plan_file);
  char plan_path[64];
  snprintf(plan_path, sizeof(plan_path), "/proc/self/fd/%d", fileno(plan_file));
  struct plan parsed = {0};
  assert(!parse_plan(plan_path, &parsed));
  assert(parsed.control.local_count == 0 && parsed.control.peer_ports[0] == 1234);
  /* IPv4-mapped IPv6 is kept as IPv4; IPv6 in network order */
  assert(parsed.control.local_v4_count == 2 && !parsed.control.local_v6_count);
  assert(parsed.control.local_v4[0] == htonl(0x0a000001) &&
         parsed.control.local_v4[1] == htonl(0x0a000002));
  assert(!parsed.control.peer_v4_count && parsed.control.peer_v6_count == 1 &&
         parsed.control.peer_v6[0][0] == htonl(0xfd000000) &&
         parsed.control.peer_v6[0][3] == htonl(5));
  fclose(plan_file);
  /* a malformed address, nine addresses, or a plan without its address lines */
  const char *bad[] = {"LA 10.0.0.300\nRA \n",
                       "LA 1.1.1.1 1.1.1.2 1.1.1.3 1.1.1.4 1.1.1.5 1.1.1.6 1.1.1.7 1.1.1.8 "
                       "1.1.1.9\nRA \n",
                       ""};
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
    assert(rings == frees && strstr(queue[3], "kernel_attach"));
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
  reset();
  struct kvp_event e = {.schema = 2,
                        .size = 256,
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
  assert(strstr(queue[0], "\"old_state\":1,\"new_state\":7") &&
         strstr(queue[0], "\"value\":10"));
  assert(strstr(queue[0], "usec_shift_3") && lengths[0] < KVP_RECORD_BYTES);
  for (unsigned int i = 0; i < KVP_QUEUE + 10; i++) event_row(NULL, &e, sizeof(e));
  assert(count == KVP_QUEUE - 1 && output_drops == 12);
  assert(!output(0, "{\"type\":\"end\",\"complete\":false}"));
  assert(count == KVP_QUEUE);
  reset();
  map_diagnostics(obj);
  assert(count == 3 && strstr(queue[0], "\"reported\":false"));
  reset();
  fdinfo = "memlock: 99999999999\n";
  map_diagnostics(obj);
  assert(count == 3 && strstr(queue[0], "99999999999") && !strstr(queue[0], "FAIL"));
  bpf_object__close(obj);
  puts(
      "PASS: real object; emulated kernel load/attach errors; partial-link cleanup; "
      "event serializer/queue loss; diagnostic-only memory; no BPF load/attach");
}
