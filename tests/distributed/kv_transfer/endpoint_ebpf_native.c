/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#define KVP_HOST 1
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

#include "csrc/kv_profiler/ebpf/lifecycle.h"
#include "csrc/kv_profiler/ebpf/observe.bpf.c"
static void *bpf_map_lookup_elem(void *map, const void *key) {
  assert(*(const u32 *)key == 0);
  if (map == &control) return &test_control;
  assert(map == &loss_counts);
  return &test_loss;
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
  read_fault = NULL;
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
  sk.__sk_common.skc_cookie.counter = 0;
  sk.__sk_common.skc_family = 10;
  sk.__sk_common.skc_v6_rcv_saddr.s6_addr[15] = 1;
  sk.__sk_common.skc_v6_daddr.s6_addr[15] = 2;
  tcp_reset_rx(&sk);
  assert(test_count == 1 && !test_records[0].cookie_available);
  assert(test_records[0].family == 10 && test_records[0].peer_address[15] == 2);
}
static void test_address_filter(void) {
  /* Empty lists admit any address; otherwise both ends must be listed. IPv4
     lists also match IPv4-mapped IPv6 sockets; IPv6 lists match exactly. */
  struct sock sk = make_socket();
  init();
  test_control.local_v4_count = test_control.peer_v4_count = 1;
  test_control.local_v4[0] = htonl(0x7f000001);
  test_control.peer_v4[0] = htonl(0x7f000003);
  tcp_retrans(&sk);
  assert(!test_count && !test_reserved && test_loss.count[EXCLUDED_ADDRESS] == 1);
  test_control.peer_v4[0] = htonl(0x7f000002);
  tcp_retrans(&sk);
  assert(test_count == 1 && test_records[0].peer_address[3] == 2);
  sk.__sk_common.skc_family = 10;
  memset(&sk.__sk_common.skc_v6_rcv_saddr, 0, 16);
  memset(&sk.__sk_common.skc_v6_daddr, 0, 16);
  sk.__sk_common.skc_v6_rcv_saddr.s6_addr[10] = 0xff;
  sk.__sk_common.skc_v6_rcv_saddr.s6_addr[11] = 0xff;
  memcpy(&sk.__sk_common.skc_v6_rcv_saddr.s6_addr[12], &test_control.local_v4[0], 4);
  sk.__sk_common.skc_v6_daddr.s6_addr[10] = 0xff;
  sk.__sk_common.skc_v6_daddr.s6_addr[11] = 0xff;
  memcpy(&sk.__sk_common.skc_v6_daddr.s6_addr[12], &test_control.peer_v4[0], 4);
  tcp_retrans(&sk);
  assert(test_count == 2 && test_records[1].family == 10);
  sk.__sk_common.skc_v6_daddr.s6_addr[10] = 0;
  sk.__sk_common.skc_v6_daddr.s6_addr[11] = 0;
  tcp_retrans(&sk);
  assert(test_count == 2 && test_loss.count[EXCLUDED_ADDRESS] == 2);
  test_control.peer_v4_count = 0;
  test_control.peer_v6_count = 1;
  memcpy(test_control.peer_v6[0], &sk.__sk_common.skc_v6_daddr, 16);
  tcp_retrans(&sk);
  assert(test_count == 3);
  test_control.peer_v6[0][3] ^= 1;
  tcp_retrans(&sk);
  assert(test_count == 3 && test_loss.count[EXCLUDED_ADDRESS] == 3);
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
  test_address_filter();
  test_concurrent_ring_loss();
  test_real_parent_guard();
  test_parent_death();
  puts(
      "PASS: TCP events/full-socket/filter/address/field/loss contracts; external BPF helpers "
      "emulated; parent death uses real OS; no BPF load/attach");
}
