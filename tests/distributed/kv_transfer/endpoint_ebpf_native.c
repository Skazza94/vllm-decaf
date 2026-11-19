/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#define KVP_HOST 1
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>
#include "csrc/kv_profiler/ebpf/observe.bpf.c"
#include "csrc/kv_profiler/ebpf/parsers.h"
#include "csrc/kv_profiler/ebpf/lifecycle.h"
#include "csrc/kv_profiler/ebpf/accounting.h"
static void init(void) {
  reset_emulator();
#define REG(name, n, key, value, kc, vc) \
  register_map(&name, sizeof(key), sizeof(value), n, 0);
  KVP_MAPS(REG)
#undef REG
  register_map(&control, 4, sizeof(struct kvp_control), 16, 1);
  register_map(&loss_counts, 4, sizeof(struct kvp_loss_counts), 1, 1);
  fixture = (struct fixture_context){
      .pid = 1234, .tid = 1, .start = 100, .clock = 100};
  u32 zero = 0, pid = 1234;
  struct kvp_control cfg = {
      .enabled = 1, .netns = 99, .pid_dev = 11, .pid_ino = 22};
  struct kvp_process p = {.generation = 77};
  assert(!bpf_map_update_elem(&control, &zero, &cfg, 0));
  assert(!bpf_map_update_elem(&processes, &pid, &p, 1));
}
static u64 losses(u32 kind) {
  u32 zero = 0;
  struct kvp_loss_counts* v = bpf_map_lookup_elem(&loss_counts, &zero);
  return v->count[kind];
}
static struct kvp_epoch* epoch(u64 n) {
  struct kvp_call_key key = {77, n};
  return bpf_map_lookup_elem(&api_epochs, &key);
}
static void test_api(void) {
  init();
  struct pt_regs ctx = {.di = 0xbad, .si = 0xcafe, .ax = (u64)-7};
  api_enter(&ctx);
  assert(find_map(&threads)->count == 1);
  assert(epoch(1)->handle == 0xcafe && !epoch(1)->complete);
  api_exit(&ctx);
  assert(epoch(1)->complete && epoch(1)->status == -7);
  assert(epoch(1)->end_ns > epoch(1)->begin_ns);
  assert(find_map(&active_calls)->count == 0);
  api_enter(&ctx);
  api_exit(&ctx);
  assert(epoch(2)->complete);
  api_exit(&ctx);
  assert(losses(LOSS_RETURN) == 1);
  init();
  fixture.namespace_error = 1;
  api_enter(&ctx);
  assert(losses(LOSS_NAMESPACE) == 1 && !find_map(&threads)->count);
  init();
  fixture.start = 0;
  api_enter(&ctx);
  assert(losses(LOSS_IDENTITY) == 1);
  init();
  for (int i = 0; i < 5; i++) api_enter(&ctx);
  assert(losses(LOSS_DEPTH) == 1);
  for (int i = 0; i < 5; i++) api_exit(&ctx);
  for (int i = 1; i <= 4; i++) assert(epoch(i)->complete);
  assert(!losses(LOSS_RETURN));
  init();
  u32 pid = 1234;
  struct kvp_process* p = bpf_map_lookup_elem(&processes, &pid);
  p->sequence = ULLONG_MAX;
  api_enter(&ctx);
  assert(losses(LOSS_WRAP) == 1 && !epoch(0));
  init();
  for (u32 i = 0; i < 129; i++) {
    fixture.tid = i + 1;
    fixture.start = i + 100;
    api_enter(&ctx);
    api_exit(&ctx);
  }
  assert(find_map(&threads)->count == 128 && losses(LOSS_THREADS) == 1);
  init();
  find_map(&api_epochs)->capacity = 1;
  api_enter(&ctx);
  api_exit(&ctx);
  api_enter(&ctx);
  api_exit(&ctx);
  assert(losses(LOSS_MAP) == 1 && find_map(&api_epochs)->count == 1);
}
static void test_missing_return(void) {
  init();
  struct pt_regs ctx = {.si = 123};
  api_enter(&ctx);
  assert(epoch(1) && !epoch(1)->complete);
  assert(find_map(&active_calls)->count == 1);
  struct kvp_thread_key old = epoch(1)->thread;
  fixture.start++;
  api_exit(&ctx);
  assert(losses(LOSS_RETURN) == 1 && !epoch(1)->complete);
  u32 slot = 1;
  struct kvp_control command = {.command = 1, .retire = old};
  bpf_map_update_elem(&control, &slot, &command, 0);
  retire_threads();
  assert(find_map(&threads)->count == 1);
  fixture.start = old.start;
  api_exit(&ctx);
  assert(epoch(1)->complete && find_map(&active_calls)->count == 0);
}
static void test_retirement(void) {
  init();
  struct pt_regs ctx = {.si = 123};
  api_enter(&ctx);
  api_exit(&ctx);
  struct kvp_thread_key key = epoch(1)->thread;
  u32 slot = 1;
  struct kvp_control command = {.command = 1, .retire = key};
  bpf_map_update_elem(&control, &slot, &command, 0);
  retire_threads();
  assert(find_map(&threads)->count ==
         1); /* Unacknowledged rows pin identity. */
  command = (struct kvp_control){.command = 3, .retire = key, .netns = 1};
  bpf_map_update_elem(&control, &slot, &command, 0);
  retire_threads();
  assert(!epoch(1));
  command = (struct kvp_control){.command = 1, .retire = key};
  bpf_map_update_elem(&control, &slot, &command, 0);
  retire_threads();
  assert(!find_map(&threads)->count);
  fixture.start++;
  api_enter(&ctx);
  api_exit(&ctx);
  assert(epoch(2)->thread.start != key.start);
}
static void test_ack_retry_and_slot_reuse(void) {
  init();
  struct pt_regs ctx = {.si = 123};
  api_enter(&ctx);
  struct kvp_thread_key key = epoch(1)->thread;
  u32 slot = 1;
  struct kvp_control command = {.command = 3, .retire = key, .netns = 1};
  bpf_map_update_elem(&control, &slot, &command, 0);
  retire_threads();
  struct kvp_control* state = bpf_map_lookup_elem(&control, &slot);
  struct kvp_thread* owner = bpf_map_lookup_elem(&threads, &key);
  assert(state->command == 3 && owner->pending_epochs == 1 && epoch(1));
  api_exit(&ctx);
  retire_threads();
  assert(state->command == 2 && !epoch(1) && !owner->pending_epochs);
  api_enter(&ctx);
  api_exit(&ctx);
  command.netns = 2;
  bpf_map_update_elem(&control, &slot, &command, 0);
  retire_threads();
  assert(state->command == 2 && !epoch(2) && !owner->pending_epochs);
  retire_threads();
  assert(!owner->pending_epochs && !losses(LOSS_STATE));
}
static pthread_mutex_t ack_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ack_cond = PTHREAD_COND_INITIALIZER;
static int ack_entered, ack_release;
static void ack_lookup(void* id, const void* key, void* row) {
  if (id != &api_epochs || ((struct kvp_call_key*)key)->epoch != 1 || !row)
    return;
  pthread_mutex_lock(&ack_mutex);
  if (!ack_entered) {
    ack_entered = 1;
    pthread_cond_broadcast(&ack_cond);
    while (!ack_release) pthread_cond_wait(&ack_cond, &ack_mutex);
  }
  pthread_mutex_unlock(&ack_mutex);
}
static void* concurrent_ack(void* unused) {
  (void)unused;
  retire_threads();
  pthread_mutex_lock(&ack_mutex);
  ack_release = 1;
  pthread_cond_broadcast(&ack_cond);
  pthread_mutex_unlock(&ack_mutex);
  return NULL;
}
static void test_concurrent_ack(void) {
  init();
  struct pt_regs ctx = {.si = 123};
  api_enter(&ctx);
  api_exit(&ctx);
  api_enter(&ctx);
  api_exit(&ctx);
  struct kvp_thread_key key = epoch(1)->thread;
  struct kvp_thread* owner = bpf_map_lookup_elem(&threads, &key);
  assert(owner->pending_epochs == 2);
  u32 slot = 1;
  struct kvp_control command = {.command = 3, .retire = key, .netns = 1};
  bpf_map_update_elem(&control, &slot, &command, 0);
  ack_entered = ack_release = 0;
  lookup_observer = ack_lookup;
  pthread_t a, b;
  assert(!pthread_create(&a, NULL, concurrent_ack, NULL));
  pthread_mutex_lock(&ack_mutex);
  while (!ack_entered) pthread_cond_wait(&ack_cond, &ack_mutex);
  pthread_mutex_unlock(&ack_mutex);
  assert(!pthread_create(&b, NULL, concurrent_ack, NULL));
  assert(!pthread_join(a, NULL) && !pthread_join(b, NULL));
  lookup_observer = NULL;
  assert(!epoch(1) && epoch(2) && owner->pending_epochs == 1);
  command = (struct kvp_control){.command = 1, .retire = key};
  bpf_map_update_elem(&control, &slot, &command, 0);
  retire_threads();
  assert(bpf_map_lookup_elem(&threads, &key) == owner && epoch(2));
  command = (struct kvp_control){.command = 3, .retire = key, .netns = 2};
  bpf_map_update_elem(&control, &slot, &command, 0);
  retire_threads();
  assert(!epoch(2) && !owner->pending_epochs);
  command = (struct kvp_control){.command = 1, .retire = key};
  bpf_map_update_elem(&control, &slot, &command, 0);
  retire_threads();
  assert(!bpf_map_lookup_elem(&threads, &key));
  assert(!losses(LOSS_STATE));
}
static void* calls(void* number) {
  u32 n = (u32)(uintptr_t)number;
  fixture = (struct fixture_context){
      .pid = 1234, .tid = n + 1, .start = n + 100, .clock = 100};
  struct pt_regs ctx = {.si = 44};
  for (int i = 0; i < 1000; i++) {
    api_enter(&ctx);
    api_exit(&ctx);
  }
  return 0;
}
static void test_concurrency(void) {
  init();
  pthread_t threads_[8];
  for (u32 i = 0; i < 8; i++)
    assert(!pthread_create(&threads_[i], 0, calls, (void*)(uintptr_t)i));
  for (int i = 0; i < 8; i++) assert(!pthread_join(threads_[i], 0));
  assert(find_map(&api_epochs)->count == 8000 &&
         !find_map(&active_calls)->count);
  for (int i = 1; i <= 8000; i++)
    assert(epoch(i) && epoch(i)->complete && epoch(i)->status == 0);
  for (int i = 0; i < LOSS_COUNT; i++) assert(!losses(i));
}
static int safe_read(void* out, u64 address, u32 size) {
  struct iovec local = {out, size}, remote = {(void*)(uintptr_t)address, size};
  return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == size ? 0 : -1;
}
static void test_parsers(void) {
  struct kvp_batch b = {};
  u64 count, bytes;
  assert(!kvp_parse_list(&b, 0, safe_read) &&
         !kvp_accepted(&b, 0, 0, &count, &bytes) && count == 0);
  struct kvp_sge sges[9];
  for (int i = 0; i < 9; i++) sges[i] = (struct kvp_sge){.length = 3};
  struct kvp_wr wr[65];
  for (int i = 0; i < 65; i++)
    wr[i] =
        (struct kvp_wr){.next = (u64)(uintptr_t)(i + 1 < 65 ? &wr[i + 1] : 0),
                        .sges = (u64)(uintptr_t)sges,
                        .count = 8,
                        .opcode = 1};
  assert(kvp_parse_list(&b, (u64)(uintptr_t)wr, safe_read) == LOSS_LIST &&
         !b.valid);
  wr[63].next = 0;
  assert(!kvp_parse_list(&b, (u64)(uintptr_t)wr, safe_read));
  assert(!kvp_accepted(&b, 0, 0, &count, &bytes) && count == 64 &&
         bytes == 1536);
  assert(!kvp_accepted(&b, 5, (u64)(uintptr_t)&wr[7], &count, &bytes) &&
         count == 7 && bytes == 168);
  assert(kvp_accepted(&b, 5, 99, &count, &bytes) == LOSS_BAD_WR && count == 0);
  wr[0].count = 9;
  assert(kvp_parse_list(&b, (u64)(uintptr_t)wr, safe_read) == LOSS_SGE);
  wr[0].count = 8;
  wr[0].next = (u64)(uintptr_t)&wr[0];
  assert(kvp_parse_list(&b, (u64)(uintptr_t)wr, safe_read) == LOSS_LIST);
  assert(kvp_parse_list(&b, 1, safe_read) == LOSS_READ);
  b = (struct kvp_batch){};
  assert(!kvp_tx_start(&b, 1, 2, 3));
  assert(kvp_tx_start(&b, 1, 2, 3) == LOSS_TXN);
  assert(!kvp_tx_op(&b, 1, 2, 5));
  assert(!kvp_tx_sges(&b, 1, 2, 8, (u64)(uintptr_t)sges, safe_read) &&
         b.op[0].bytes == 24);
  assert(!kvp_tx_sges(&b, 1, 2, 1, (u64)(uintptr_t)sges, safe_read) &&
         b.op[0].bytes == 3);
  assert(kvp_tx_op(&b, 99, 2, 5) == LOSS_TXN);
  assert(!kvp_tx_end(&b, 1, 2, 1) && b.operations == 0);
  b = (struct kvp_batch){};
  assert(!kvp_tx_start(&b, 1, 2, 3));
  for (int i = 0; i < 64; i++) assert(!kvp_tx_op(&b, 1, 2, 5));
  assert(kvp_tx_op(&b, 1, 2, 5) == LOSS_OPERATION && !b.valid);
  b = (struct kvp_batch){};
  assert(!kvp_tx_start(&b, 1, 2, 3));
  assert(!kvp_tx_op(&b, 1, 2, 5));
  assert(kvp_tx_sges(&b, 1, 2, 1, 1, safe_read) == LOSS_READ);
  b = (struct kvp_batch){};
  assert(!kvp_tx_start(&b, 1, 2, 3));
  for (int i = 1; i < 1024; i++) assert(!kvp_tx_tick(&b, 1, 2));
  assert(kvp_tx_tick(&b, 1, 2) == LOSS_BUILDER);
}
static void test_tcp(void) {
  init();
  struct net net = {.ns = {99}};
  struct sock sk = {.__sk_common = {.skc_net = {&net}, .skc_cookie = {45}}};
  struct sk_buff skb = {.len = 100};
  fixture.pid = 999;
  tcp_rx(&sk, &skb);
  tcp_retrans(&sk);
  tcp_state(&sk);
  tcp_reset_tx(&sk);
  tcp_reset_rx(&sk);
  struct kvp_socket_key key = {99, 45};
  struct kvp_socket* v = bpf_map_lookup_elem(&socket_counts, &key);
  assert(v && v->rx_events == 1 && v->skb_len_bytes == 100 &&
         v->retransmits == 1 && v->states == 1 && v->send_resets == 1 &&
         v->recv_resets == 1);
  net.ns.inum = 44;
  tcp_rx(&sk, &skb);
  assert(v->rx_events == 1);
  net.ns.inum = 99;
  sk.__sk_common.skc_cookie.counter = 0;
  tcp_rx(&sk, &skb);
  assert(losses(STAT_UNKNOWN_SOCKET) == 1 && losses(LOSS_IDENTITY) == 0);
  assert(losses(STAT_UNKNOWN_RX_LENGTH) == 100);
  u32 zero = 0;
  struct kvp_loss_counts* loss_ = bpf_map_lookup_elem(&loss_counts, &zero);
  loss_->count[LOSS_NAMESPACE] = ULLONG_MAX;
  fixture.namespace_error = 1;
  api_enter(&(struct pt_regs){});
  assert(settings()->command & (1ULL << LOSS_WRAP));
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
static enum kvp_budget_result charge_text(struct kvp_charge* charge,
                                          const char* text) {
  FILE* f = fmemopen((void*)text, strlen(text), "r");
  assert(f);
  enum kvp_budget_result result = kvp_charge_fdinfo(charge, f);
  fclose(f);
  return result;
}
static void test_map_accounting(void) {
  struct kvp_charge charge = {.bound = 300};
  assert(charge_text(&charge, "memlock: 100\n") == KVP_BUDGET_OK);
  assert(charge_text(&charge, "memlock: 200\n") == KVP_BUDGET_OK);
  assert(charge.observed == 300);
  assert(charge_text(&charge, "memlock: 1\n") == KVP_BUDGET_FAIL);
  assert(charge.observed == 301 && charge.last == 1 && charge.bound == 300);
  charge = (struct kvp_charge){.bound = 300};
  assert(charge_text(&charge, "memlock: 301\n") == KVP_BUDGET_FAIL);
  assert(charge.observed == 301 && charge.last == 301);
  charge = (struct kvp_charge){.bound = KVP_MAP_BYTES + 1ULL};
  assert(charge_text(&charge, "memlock: 16777217\n") == KVP_BUDGET_FAIL);
  charge = (struct kvp_charge){.bound = 300};
  assert(charge_text(&charge, "map_type: 1\n") == KVP_BUDGET_UNAVAILABLE);
  assert(charge_text(&charge, "memlock: -1\n") == KVP_BUDGET_UNAVAILABLE);
  assert(charge_text(&charge, "memlock: 0\n") == KVP_BUDGET_UNAVAILABLE);
  assert(kvp_charge_fdinfo(&charge, NULL) == KVP_BUDGET_UNAVAILABLE);
  charge = (struct kvp_charge){.observed = ~0ULL - 1, .bound = KVP_MAP_BYTES};
  assert(charge_text(&charge, "memlock: 3\n") == KVP_BUDGET_FAIL);
  assert(charge.overflow && charge.last == 3 && charge.observed == ~0ULL - 1);
}
int main(void) {
  test_map_accounting();
  test_api();
  test_retirement();
  test_concurrent_ack();
  test_ack_retry_and_slot_reuse();
  test_missing_return();
  test_concurrency();
  test_parsers();
  test_tcp();
  test_parent_death();
  reset_emulator();
  puts(
      "PASS: native code contracts; external BPF helpers emulated; parent "
      "death uses real OS; no BPF load/attach");
  return 0;
}
