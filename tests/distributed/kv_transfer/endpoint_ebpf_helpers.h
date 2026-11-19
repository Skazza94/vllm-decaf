/* SPDX-License-Identifier: Apache-2.0 */
#ifndef ENDPOINT_EBPF_HELPERS_H
#define ENDPOINT_EBPF_HELPERS_H
/* External map/ring/CO-RE helpers only. Not a verifier or kernel test. */
#include <arpa/inet.h>
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "csrc/kv_profiler/ebpf/core_types.h"
#include "csrc/kv_profiler/ebpf/protocol.h"
#define SEC(name)
#define __uint(name, value) int(*name)[value]
#define __type(name, type) type *name
#ifndef __always_inline
#define __always_inline inline __attribute__((always_inline))
#endif
#define __noinline __attribute__((noinline))
#define BPF_PROG(name, ...) name(__VA_ARGS__)
#define bpf_ntohs(x) ntohs(x)
static struct kvp_control test_control;
static struct kvp_loss_counts test_loss;
static struct kvp_event test_records[4096];
static unsigned int test_count, test_reserved, test_discards;
static pthread_mutex_t test_mutex = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local unsigned int missing_fields, full_reads;
static _Thread_local const char *read_fault;
static const char *test_fields[] = {"srtt_us",       "snd_cwnd",    "snd_ssthresh",
                                    "packets_out",   "retrans_out", "lost_out",
                                    "total_retrans", "mss_cache"};
static int test_exists(const char *expression) {
  for (unsigned int i = 0; i < KVP_FIELDS; i++)
    if (strstr(expression, test_fields[i])) return !(missing_fields & (1U << i));
  return 1;
}
#define bpf_core_field_exists(field) test_exists(#field)
#define bpf_core_field_size(field) sizeof(field)
static int test_read(void *dst, const void *from, size_t size, const char *field) {
  if (!strcmp(field, "sk_protocol") || strstr(field, "skc_cookie")) full_reads++;
  for (unsigned int i = 0; i < KVP_FIELDS; i++)
    if (!strcmp(field, test_fields[i])) full_reads++;
  if (read_fault && !strcmp(field, read_fault)) return -1;
  for (unsigned int i = 0; i < KVP_FIELDS; i++)
    if (!strcmp(field, test_fields[i])) {
      u32 value = 100 + i;
      memcpy(dst, &value, size);
      return 0;
    }
  memcpy(dst, from, size);
  return 0;
}
#define BPF_CORE_READ_INTO(dst, p, field) \
  test_read(dst, &(p)->field, sizeof(*(dst)), #field)
static void *bpf_map_lookup_elem(void *map, const void *key);
static void *bpf_ringbuf_reserve(void *map, unsigned int bytes, unsigned long flags) {
  (void)map;
  (void)flags;
  assert(bytes == sizeof(struct kvp_event));
  pthread_mutex_lock(&test_mutex);
  int full = test_reserved + 264 > KVP_RING_BYTES;
  if (!full) test_reserved += 264;
  pthread_mutex_unlock(&test_mutex);
  return full ? NULL : calloc(1, bytes);
}
static void bpf_ringbuf_submit(void *event, unsigned long flags) {
  (void)flags;
  pthread_mutex_lock(&test_mutex);
  assert(test_count < 4096);
  test_records[test_count++] = *(struct kvp_event *)event;
  pthread_mutex_unlock(&test_mutex);
  free(event);
}
static void bpf_ringbuf_discard(void *event, unsigned long flags) {
  (void)flags;
  pthread_mutex_lock(&test_mutex);
  test_reserved -= 264;
  test_discards++;
  pthread_mutex_unlock(&test_mutex);
  free(event);
}
static u64 bpf_ktime_get_ns(void) {
  static u64 sequence;
  return __sync_add_and_fetch(&sequence, 1);
}
#endif
