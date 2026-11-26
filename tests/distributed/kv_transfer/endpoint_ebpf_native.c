/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#define KVP_HOST 1
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

#include "csrc/kv_profiler/ebpf/lifecycle.h"
#include "endpoint_ebpf_helpers.h"
/* Watch a counter's increment: whether its entry's earliest time already
   holds the counted event's then (a read that sees the count sees it). */
static struct kvp_rejected *watched;
static u64 watched_time;
static int ordered = -1;
static void before_add(void *at) {
  if (watched && at == &watched->count) ordered = watched->first_ns <= watched_time;
}
#define __sync_fetch_and_add(p, n) \
  (before_add((void *)(p)), __atomic_fetch_add((p), (n), __ATOMIC_SEQ_CST))
#include "csrc/kv_profiler/ebpf/observe.bpf.c"
#undef __sync_fetch_and_add
/* Small emulations of the hash maps: owned cookies and the rejection LRU. */
static u64 allowed_keys[16], rejected_keys[4];
static struct kvp_admission allowed_values[16];
static struct kvp_rejected rejected_values[4];
static unsigned int allowed_count, rejected_count, rejected_evictions;
static int insert_race; /* the next LRU insertion loses to a concurrent one */
static void *bpf_map_lookup_elem(void *map, const void *key) {
  if (map == &allowed || map == &rejected) {
    u64 cookie = *(const u64 *)key;
    int owned = map == &allowed;
    unsigned int n = owned ? allowed_count : rejected_count;
    for (unsigned int i = 0; i < n; i++)
      if ((owned ? allowed_keys : rejected_keys)[i] == cookie)
        return owned ? (void *)&allowed_values[i] : (void *)&rejected_values[i];
    return NULL;
  }
  assert(*(const u32 *)key == 0);
  if (map == &control) return &test_control;
  assert(map == &loss_counts);
  return &test_loss;
}
static long bpf_map_update_elem(void *map, const void *key, const void *value,
                                unsigned long long flags) {
  assert(map == &rejected && flags == BPF_NOEXIST);
  if (insert_race) {
    insert_race = 0;
    return -17; /* EEXIST, the entry gone again before the lookup */
  }
  if (bpf_map_lookup_elem(map, key)) return -17;
  unsigned int slot = rejected_count;
  if (slot == 4) { /* full: the LRU evicts its oldest */
    memmove(rejected_keys, rejected_keys + 1, 3 * sizeof(*rejected_keys));
    memmove(rejected_values, rejected_values + 1, 3 * sizeof(*rejected_values));
    slot = 3;
    rejected_evictions++;
  } else {
    rejected_count++;
  }
  rejected_keys[slot] = *(const u64 *)key;
  rejected_values[slot] = *(const struct kvp_rejected *)value;
  return 0;
}
static void own(u64 cookie, u64 admission) {
  allowed_keys[allowed_count] = cookie;
  allowed_values[allowed_count++].admission = admission;
}
struct socket_fixture {
  struct sock sk;
  struct tcp_sock tcp;
};
/* The production cast uses first-member embedding; the emulator maps TCP reads. */
static struct net test_net = {.ns.inum = 99};
static struct sock make_socket(void) {
  struct sock sk = {0};
  sk.__sk_common.skc_state = 1;
  sk.__sk_common.skc_net.net = &test_net;
  sk.__sk_common.skc_family = 2;
  sk.__sk_common.skc_num = 1234;
  sk.__sk_common.skc_dport = htons(4321);
  sk.__sk_common.skc_cookie.counter = 42;
  sk.__sk_common.skc_rcv_saddr = htonl(0x7f000001);
  sk.__sk_common.skc_daddr = htonl(0x7f000002);
  sk.sk_protocol = 6;
  return sk;
}
static void init(void) {
  test_control = (struct kvp_control){.enabled = 1, .netns = 99};
  memset(&test_loss, 0, sizeof(test_loss));
  test_count = test_reserved = full_reads = missing_fields = 0;
  allowed_count = rejected_count = rejected_evictions = 0;
  insert_race = 0;
  read_fault = NULL;
  own(42, 7); /* the fixture socket is owned unless a test says otherwise */
}
static void test_scope_and_full_socket(void) {
  init();
  tcp_reset_tx(NULL);
  assert(test_loss.count[GLOBAL_NULL] == 1 && !test_count);
  struct sock sk = make_socket();
  for (unsigned int i = 0; i < 2; i++) {
    sk.__sk_common.skc_state = i ? TCP_NEW_SYN_RECV : TCP_TIME_WAIT;
    tcp_reset_tx(&sk);
  }
  assert(test_loss.count[GLOBAL_NONFULL] == 2 && full_reads == 0 && !test_count);
  sk = make_socket();
  read_fault = "__sk_common.skc_state";
  tcp_reset_tx(&sk);
  assert(test_loss.count[GLOBAL_SCOPE_READ] == 1 && full_reads == 0);
  read_fault = NULL;
  test_control.netns = 100;
  tcp_reset_tx(&sk);
  assert(test_loss.count[EXCLUDED_NAMESPACE] == 1 && !test_count);
  test_control.netns = 99;
  sk.sk_protocol = 17;
  tcp_state(&sk, 1, 7);
  assert(test_loss.count[EXCLUDED_PROTOCOL] == 1);
  sk.sk_protocol = 6;
  test_control.local_count = 1;
  test_control.local_ports[0] = 999;
  tcp_reset_rx(&sk);
  assert(test_loss.count[EXCLUDED_PORT] == 1);
  test_control.local_ports[0] = 1234;
  test_control.peer_count = 1;
  test_control.peer_ports[0] = 4321;
  tcp_state(&sk, 1, 7);
  assert(test_count == 1 && test_records[0].old_state == 1 &&
         test_records[0].new_state == 7);
  assert(test_records[0].timestamp_ns && test_records[0].cookie == 42);
  assert(test_records[0].admission == 7);
  for (unsigned int i = 0; i < KVP_FIELDS; i++)
    assert(test_records[0].fields[i] == 100 + i);
  assert(test_records[0].local_port == 1234 && test_records[0].peer_port == 4321);
}
static void test_optional_fields_and_reads(void) {
  struct sock sk = make_socket();
  for (unsigned int field = 0; field < KVP_FIELDS; field++) {
    init();
    missing_fields = 1U << field;
    tcp_retrans(&sk);
    assert(test_count == 1 && test_records[0].available == (255U ^ (1U << field)));
    assert(test_records[0].read_errors == 0);
    init();
    read_fault = test_fields[field];
    tcp_retrans(&sk);
    assert(test_count == 1 && test_records[0].read_errors == (1U << field));
  }
  init();
  read_fault = "__sk_common.skc_rcv_saddr";
  tcp_reset_tx(&sk);
  /* the tuple is read before the ring: nothing reserved */
  assert(!test_count && !test_reserved &&
         test_loss.count[SCOPED_TUPLE_READ] == 1);
  init();
  sk.__sk_common.skc_family = 10;
  sk.__sk_common.skc_v6_rcv_saddr.s6_addr[15] = 1;
  sk.__sk_common.skc_v6_daddr.s6_addr[15] = 2;
  tcp_reset_rx(&sk);
  assert(test_count == 1 && test_records[0].cookie == 42);
  assert(test_records[0].family == 10 && test_records[0].peer_address[15] == 2);
}
static void test_ownership_filter(void) {
  /* Only an owned cookie is recorded, with the admission it matched; every
     other eligible event is one EXCLUDED_UNOWNED and, best effort, a count in
     its cookie's LRU entry; an insertion lost to a race is untracked (in the
     total, in no cookie); an evicted counter leaves the total exact. */
  struct sock sk = make_socket();
  init();
  sk.__sk_common.skc_cookie.counter = 500;
  tcp_retrans(&sk);
  tcp_retrans(&sk);
  assert(!test_count && !test_reserved && test_loss.count[EXCLUDED_UNOWNED] == 2);
  struct kvp_rejected *seen = bpf_map_lookup_elem(&rejected, &(u64){500});
  assert(seen && seen->count == 2 && seen->first_ns < seen->last_ns);
  insert_race = 1;
  sk.__sk_common.skc_cookie.counter = 501;
  tcp_retrans(&sk);
  assert(test_loss.count[EXCLUDED_UNOWNED] == 3 && test_loss.count[REJECTED_UNTRACKED] == 1);
  assert(!bpf_map_lookup_elem(&rejected, &(u64){501}));
  for (u64 cookie = 600; cookie < 605; cookie++) {
    sk.__sk_common.skc_cookie.counter = cookie;
    tcp_retrans(&sk);
  }
  assert(rejected_evictions == 2 && test_loss.count[EXCLUDED_UNOWNED] == 8);
  /* owned after its rejections: recorded under its admission */
  own(500, 9);
  sk.__sk_common.skc_cookie.counter = 500;
  tcp_state(&sk, 1, 4);
  assert(test_count == 1 && test_records[0].cookie == 500 &&
         test_records[0].admission == 9);
  /* empty ownership matches nothing */
  init();
  allowed_count = 0;
  sk.__sk_common.skc_cookie.counter = 42;
  tcp_retrans(&sk);
  assert(!test_count && test_loss.count[EXCLUDED_UNOWNED] == 1);
}
/* The counter keeps its entry's earliest event time, whatever the order and
   concurrency of the increments: a late-counted early event lowers it. */
static struct kvp_rejected shared_entry;
static void *late_events(void *base) {
  for (u64 i = 0; i < 5000; i++) earliest(&shared_entry, (u64)(uintptr_t)base + 7 * i);
  return NULL;
}
static void test_earliest_rejection_time(void) {
  struct kvp_rejected entry = {.count = 1, .first_ns = 100, .last_ns = 100};
  earliest(&entry, 120);
  assert(entry.first_ns == 100);
  earliest(&entry, 40);
  assert(entry.first_ns == 40);
  entry.first_ns |= KVP_FIRST_UNKNOWN;
  earliest(&entry, 1);
  assert(entry.first_ns == (40 | KVP_FIRST_UNKNOWN));
  /* an early event counted late: its time is in before its count */
  init();
  count_rejected(700, 1000);
  watched = bpf_map_lookup_elem(&rejected, &(u64){700});
  watched_time = 5;
  count_rejected(700, 5);
  assert(ordered == 1 && watched->count == 2 && watched->first_ns == 5);
  watched = NULL;
  shared_entry = (struct kvp_rejected){.count = 1, .first_ns = 1000000};
  pthread_t threads[4];
  for (uintptr_t i = 0; i < 4; i++)
    assert(!pthread_create(&threads[i], NULL, late_events, (void *)(1000 + i)));
  for (unsigned int i = 0; i < 4; i++) assert(!pthread_join(threads[i], NULL));
  assert(shared_entry.first_ns == 1000 || shared_entry.first_ns & KVP_FIRST_UNKNOWN);
}
static void *flood(void *unused) {
  (void)unused;
  struct sock sk = make_socket();
  for (unsigned int i = 0; i < 2000; i++) tcp_retrans(&sk);
  return NULL;
}
static void test_concurrent_ring_loss(void) {
  init();
  pthread_t threads[4];
  for (unsigned int i = 0; i < 4; i++)
    assert(!pthread_create(&threads[i], NULL, flood, NULL));
  for (unsigned int i = 0; i < 4; i++) assert(!pthread_join(threads[i], NULL));
  assert(test_count == KVP_RING_BYTES / 264);
  assert(test_loss.count[SCOPED_RING_DROP] + test_count == 8000);
  assert(!test_loss.count[GLOBAL_NULL] && !test_loss.count[GLOBAL_SCOPE_READ]);
}
static void test_real_parent_guard(void) {
  assert(kvp_parent_guard(0) < 0);
  int ready[2];
  assert(!pipe(ready));
  pid_t pid = fork();
  assert(pid >= 0);
  if (!pid) {
    close(ready[0]);
    assert(!kvp_parent_guard(getppid()));
    char yes = 'y';
    assert(write(ready[1], &yes, 1) == 1);
    _exit(0);
  }
  close(ready[1]);
  char yes;
  assert(read(ready[0], &yes, 1) == 1 && yes == 'y');
  close(ready[0]);
  int status;
  assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status));
}
static void test_parent_death(void) {
  assert(kvp_parent_guard(getppid() + 1) == -ESRCH);
  assert(!prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0));
  int ready[2], gate[2];
  assert(!pipe(ready) && !pipe(gate));
  pid_t broker = fork();
  assert(broker >= 0);
  if (!broker) {
    pid_t parent = getpid(), child = fork();
    assert(child >= 0);
    if (!child) {
      assert(!kvp_parent_guard(parent));
      pid_t me = getpid();
      assert(write(ready[1], &me, sizeof(me)) == sizeof(me));
      for (;;) pause();
    }
    char byte;
    assert(read(gate[0], &byte, 1) == 1);
    _exit(0);
  }
  pid_t child;
  assert(read(ready[0], &child, sizeof(child)) == sizeof(child));
  assert(write(gate[1], "x", 1) == 1);
  int status;
  assert(waitpid(broker, &status, 0) == broker);
  assert(waitpid(child, &status, 0) == child && WIFSIGNALED(status) &&
         WTERMSIG(status) == SIGKILL);
  close(ready[0]);
  close(ready[1]);
  close(gate[0]);
  close(gate[1]);
}

int main(void) {
  test_scope_and_full_socket();
  test_optional_fields_and_reads();
  test_ownership_filter();
  test_earliest_rejection_time();
  test_concurrent_ring_loss();
  test_real_parent_guard();
  test_parent_death();
  puts(
      "PASS: TCP events/full-socket/filter/ownership/earliest-rejection/field/loss contracts; "
      "external BPF helpers "
      "emulated; parent death uses real OS; no BPF load/attach");
}
