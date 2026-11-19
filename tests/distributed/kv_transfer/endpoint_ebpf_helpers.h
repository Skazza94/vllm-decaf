/* SPDX-License-Identifier: Apache-2.0 */
#ifndef ENDPOINT_EBPF_HELPERS_H
#define ENDPOINT_EBPF_HELPERS_H
/* External BPF map/helper service emulator. Not a verifier/kernel test. */
#include <pthread.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "csrc/kv_profiler/ebpf/protocol.h"
#define SEC(name)
#define __uint(name, value) int(*name)[value]
#define __type(name, type) type* name
#ifndef __always_inline
  #define __always_inline inline __attribute__((always_inline))
#endif
#define __noinline __attribute__((noinline))
#define BPF_MAP_TYPE_HASH 1
#define BPF_MAP_TYPE_ARRAY 2
#define BPF_MAP_TYPE_PERCPU_ARRAY 6
#define BPF_PROG(name, ...) name(__VA_ARGS__)
#define BPF_CORE_READ(p, field) ((p)->field)
#define PT_REGS_PARM2(p) ((p)->si)
#define PT_REGS_RC(p) ((p)->ax)
struct pt_regs {
  u64 di, si, ax;
};
struct task_struct {
  u64 start_boottime;
};
struct net {
  struct {
    u32 inum;
  } ns;
};
struct sock {
  struct {
    struct {
      struct net* net;
    } skc_net;
    struct {
      u64 counter;
    } skc_cookie;
  } __sk_common;
};
struct sk_buff {
  u32 len;
};
struct bpf_pidns_info {
  u32 pid, tgid;
};
struct fixture_context {
  u32 pid, tid;
  u64 start, clock;
  int namespace_error;
};
static _Thread_local struct fixture_context fixture;
static _Thread_local struct task_struct fixture_task;
struct emulated_map {
  void* id;
  size_t key_bytes, value_bytes;
  u32 capacity, count;
  unsigned char *keys, *values, *used;
};
static struct emulated_map emulated_maps[16];
static u32 emulated_count;
static pthread_mutex_t maps_mutex = PTHREAD_MUTEX_INITIALIZER;
static void register_map(void* id, size_t k, size_t v, u32 n, int array) {
  struct emulated_map* m = &emulated_maps[emulated_count++];
  *m = (struct emulated_map){
      .id = id, .key_bytes = k, .value_bytes = v, .capacity = n};
  m->keys = calloc(n, k);
  m->values = calloc(n, ((v + 7) / 8) * 8);
  m->used = calloc(n, 1);
  if (!m->keys || !m->values || !m->used) abort();
  if (array)
    for (u32 i = 0; i < n; i++) {
      m->used[i] = 1;
      memcpy(m->keys + i * k, &i, k);
      m->count++;
    }
}
static struct emulated_map* find_map(void* id) {
  for (u32 i = 0; i < emulated_count; i++)
    if (emulated_maps[i].id == id) return &emulated_maps[i];
  abort();
}
static void* lookup_slot(struct emulated_map* m, const void* key) {
  for (u32 i = 0; i < m->capacity; i++)
    if (m->used[i] && !memcmp(m->keys + i * m->key_bytes, key, m->key_bytes))
      return m->values + i * ((m->value_bytes + 7) / 8) * 8;
  return 0;
}
static void (*lookup_observer)(void*, const void*, void*);
static void* bpf_map_lookup_elem(void* id, const void* key) {
  pthread_mutex_lock(&maps_mutex);
  void* v = lookup_slot(find_map(id), key);
  pthread_mutex_unlock(&maps_mutex);
  if (lookup_observer) lookup_observer(id, key, v);
  return v;
}
static int bpf_map_update_elem(void* id, const void* key, const void* value,
                               u64 flags) {
  pthread_mutex_lock(&maps_mutex);
  struct emulated_map* m = find_map(id);
  void* old = lookup_slot(m, key);
  int error = 0;
  if (old) {
    if (flags == KVP_NOEXIST)
      error = -EEXIST;
    else
      memcpy(old, value, m->value_bytes);
  } else {
    u32 i = 0;
    for (; i < m->capacity; i++)
      if (!m->used[i]) break;
    if (i == m->capacity)
      error = -E2BIG;
    else {
      memcpy(m->keys + i * m->key_bytes, key, m->key_bytes);
      memcpy(m->values + i * ((m->value_bytes + 7) / 8) * 8, value,
             m->value_bytes);
      m->used[i] = 1;
      m->count++;
    }
  }
  pthread_mutex_unlock(&maps_mutex);
  return error;
}
static int bpf_map_delete_elem(void* id, const void* key) {
  pthread_mutex_lock(&maps_mutex);
  struct emulated_map* m = find_map(id);
  int rc = -ENOENT;
  for (u32 i = 0; i < m->capacity; i++)
    if (m->used[i] && !memcmp(m->keys + i * m->key_bytes, key, m->key_bytes)) {
      m->used[i] = 0;
      m->count--;
      rc = 0;
      break;
    }
  pthread_mutex_unlock(&maps_mutex);
  return rc;
}
static int bpf_get_ns_current_pid_tgid(u64 dev, u64 ino,
                                       struct bpf_pidns_info* ids, u32 size) {
  if (fixture.namespace_error || dev != 11 || ino != 22 || size != sizeof(*ids))
    return -EINVAL;
  *ids = (struct bpf_pidns_info){.pid = fixture.tid, .tgid = fixture.pid};
  return 0;
}
static void* bpf_get_current_task_btf(void) {
  fixture_task.start_boottime = fixture.start;
  return &fixture_task;
}
static u64 bpf_ktime_get_ns(void) { return ++fixture.clock; }
static void reset_emulator(void) {
  for (u32 i = 0; i < emulated_count; i++) {
    free(emulated_maps[i].keys);
    free(emulated_maps[i].values);
    free(emulated_maps[i].used);
  }
  emulated_count = 0;
}
#endif
