/* SPDX-License-Identifier: (BSD-2-Clause OR GPL-2.0-only) */
#ifdef KVP_HOST
  #include "endpoint_ebpf_helpers.h"
#else
  #include "vmlinux.h"
  #include <bpf/bpf_helpers.h>
  #include <bpf/bpf_core_read.h>
  #include <bpf/bpf_tracing.h>
#endif
#include "protocol.h"
#define DECLARE_MAP(name, count, keytype, valuetype, kc, vc) \
  struct {                                                   \
    __uint(type, BPF_MAP_TYPE_HASH);                         \
    __uint(max_entries, count);                              \
    __type(key, keytype);                                    \
    __type(value, valuetype);                                \
  } name SEC(".maps");
KVP_MAPS(DECLARE_MAP)
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 16);
  __type(key, u32);
  __type(value, struct kvp_control);
} control SEC(".maps");
struct {
  __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, struct kvp_loss_counts);
} loss_counts SEC(".maps");
static __always_inline struct kvp_control* settings(void) {
  u32 zero = 0;
  return bpf_map_lookup_elem(&control, &zero);
}
static __always_inline void add_count(u64* counter, u64 n) {
  u64 old = __sync_fetch_and_add(counter, n);
  if (old > ~0ULL - n) {
    struct kvp_control* cfg = settings();
    if (cfg) __sync_fetch_and_or(&cfg->command, 1ULL << LOSS_WRAP);
  }
}
static __always_inline void loss(u32 kind) {
  u32 zero = 0;
  struct kvp_loss_counts* v = bpf_map_lookup_elem(&loss_counts, &zero);
  if (v && kind < 32) add_count(&v->count[kind], 1);
}
static __noinline u64 next_epoch(struct kvp_process* process) {
  for (u32 i = 0; i < 32; i++) {
    u64 old = process->sequence;
    if (old == ~0ULL) {
      loss(LOSS_WRAP);
      return 0;
    }
    if (__sync_val_compare_and_swap(&process->sequence, old, old + 1) == old)
      return old + 1;
  }
  loss(LOSS_STATE);
  return 0;
}
static __noinline void retire_threads(void) {
  for (u32 slot = 1; slot < 16; slot++) {
    struct kvp_control* cmd = bpf_map_lookup_elem(&control, &slot);
    if (!cmd) continue;
    u64 operation = cmd->command;
    if (operation != 1 && operation != 3) continue;
    if (__sync_val_compare_and_swap(&cmd->command, operation, 4) != operation)
      continue;
    struct kvp_thread_key key = cmd->retire;
    if (operation == 3) {
      struct kvp_call_key epoch = {.generation = key.generation,
                                   .epoch = cmd->netns};
      struct kvp_epoch* row = bpf_map_lookup_elem(&api_epochs, &epoch);
      if (row && !row->complete) {
        __sync_lock_test_and_set(&cmd->command, 3);
        continue;
      }
      if (row && !bpf_map_delete_elem(&api_epochs, &epoch)) {
        struct kvp_thread* owner = bpf_map_lookup_elem(&threads, &key);
        if (!owner || !owner->pending_epochs)
          loss(LOSS_STATE);
        else
          __sync_fetch_and_sub(&owner->pending_epochs, 1);
      }
      __sync_lock_test_and_set(&cmd->command, 2);
      continue;
    }
    struct kvp_frames* frame = bpf_map_lookup_elem(&call_frames, &key);
    if (frame && (frame->depth || frame->suppressed)) {
      __sync_lock_test_and_set(&cmd->command, 1);
      continue;
    }
    struct kvp_thread* thread = bpf_map_lookup_elem(&threads, &key);
    if (thread) {
      if (thread->pending_epochs) {
        __sync_lock_test_and_set(&cmd->command, 1);
        continue;
      }
      u32 pid = key.tgid;
      struct kvp_process* p = bpf_map_lookup_elem(&processes, &pid);
      if (!p || p->generation != key.generation) {
        loss(LOSS_TARGET);
        __sync_lock_test_and_set(&cmd->command, 2);
        continue;
      }
      bpf_map_delete_elem(&call_frames, &key);
      if (!bpf_map_delete_elem(&threads, &key))
        __sync_fetch_and_sub(&p->thread_slots, 1);
    }
    __sync_lock_test_and_set(&cmd->command, 2);
  }
}
static __noinline int identify(struct kvp_thread_key* key) {
  struct kvp_control* cfg = settings();
  if (!cfg || !cfg->enabled) return -1;
  struct bpf_pidns_info ids = {};
  if (bpf_get_ns_current_pid_tgid(cfg->pid_dev, cfg->pid_ino, &ids,
                                  sizeof(ids))) {
    loss(LOSS_NAMESPACE);
    return -1;
  }
  u32 pid = ids.tgid;
  struct kvp_process* p = bpf_map_lookup_elem(&processes, &pid);
  if (!p) return -1;
  struct task_struct* task = (void*)bpf_get_current_task_btf();
  u64 start = BPF_CORE_READ(task, start_boottime);
  if (!start) {
    loss(LOSS_IDENTITY);
    return -1;
  }
  *key = (struct kvp_thread_key){
      .generation = p->generation, .start = start, .tgid = pid, .tid = ids.pid};
  return 0;
}
static __noinline struct kvp_frames* admit(struct kvp_thread_key* key) {
  struct kvp_thread* t = bpf_map_lookup_elem(&threads, key);
  if (!t) {
    u32 pid = key->tgid;
    struct kvp_process* p = bpf_map_lookup_elem(&processes, &pid);
    if (!p) return 0;
    u32 before = __sync_fetch_and_add(&p->thread_slots, 1);
    if (before >= KVP_PER_PROCESS_THREADS) {
      __sync_fetch_and_sub(&p->thread_slots, 1);
      loss(LOSS_THREADS);
      return 0;
    }
    struct kvp_thread initial = {.first_ns = bpf_ktime_get_ns()};
    int error = bpf_map_update_elem(&threads, key, &initial, KVP_NOEXIST);
    if (error) {
      __sync_fetch_and_sub(&p->thread_slots, 1);
      t = bpf_map_lookup_elem(&threads, key);
      if (!t) {
        loss(LOSS_THREADS);
        return 0;
      }
    }
  }
  struct kvp_frames* frames = bpf_map_lookup_elem(&call_frames, key);
  if (!frames) {
    struct kvp_frames initial = {};
    bpf_map_update_elem(&call_frames, key, &initial, KVP_NOEXIST);
    frames = bpf_map_lookup_elem(&call_frames, key);
    if (!frames) loss(LOSS_MAP);
  }
  return frames;
}
static __noinline int reserve_epoch(struct kvp_call_key* key,
                                    struct kvp_thread_key* thread, u64 handle) {
  struct kvp_epoch row = {
      .begin_ns = bpf_ktime_get_ns(), .handle = handle, .thread = *thread};
  if (bpf_map_update_elem(&api_epochs, key, &row, KVP_NOEXIST)) {
    loss(LOSS_MAP);
    return -1;
  }
  return 0;
}
static __noinline int reserve_call(struct kvp_call_key* key,
                                   struct kvp_thread_key* thread, u64 handle) {
  struct kvp_call row = {.thread = *thread, .handle = handle};
  if (bpf_map_update_elem(&active_calls, key, &row, KVP_NOEXIST)) {
    bpf_map_delete_elem(&api_epochs, key);
    loss(LOSS_MAP);
    return -1;
  }
  return 0;
}
SEC("uprobe.multi") int api_enter(struct pt_regs* ctx) {
  struct kvp_thread_key thread = {};
  if (identify(&thread)) return 0;
  retire_threads();
  struct kvp_frames* frame = admit(&thread);
  if (!frame) return 0;
  if (frame->depth >= KVP_DEPTH || frame->suppressed) {
    __sync_fetch_and_add(&frame->suppressed, 1);
    loss(LOSS_DEPTH);
    return 0;
  }
  u32 index = frame->depth++;
  frame->frame[index].epoch = 0;
  u32 pid = thread.tgid;
  struct kvp_process* p = bpf_map_lookup_elem(&processes, &pid);
  if (!p) {
    loss(LOSS_TARGET);
    return 0;
  }
  u64 epoch = next_epoch(p);
  if (!epoch) return 0;
  struct kvp_call_key key = {.generation = p->generation, .epoch = epoch};
  /* C++ this is argument 1; argument 2 is the public SDK handle. */
  u64 handle = PT_REGS_PARM2(ctx);
  if (reserve_epoch(&key, &thread, handle) ||
      reserve_call(&key, &thread, handle))
    return 0;
  struct kvp_thread* owner = bpf_map_lookup_elem(&threads, &thread);
  if (!owner) {
    loss(LOSS_STATE);
    return 0;
  }
  add_count(&owner->pending_epochs, 1);
  add_count(&owner->calls, 1);
  frame->frame[index].epoch = epoch;
  return 0;
}
SEC("uretprobe.multi") int api_exit(struct pt_regs* ctx) {
  struct kvp_thread_key thread = {};
  if (identify(&thread)) return 0;
  struct kvp_frames* frame = bpf_map_lookup_elem(&call_frames, &thread);
  if (!frame) {
    loss(LOSS_RETURN);
    return 0;
  }
  if (frame->suppressed) {
    __sync_fetch_and_sub(&frame->suppressed, 1);
    return 0;
  }
  if (!frame->depth || frame->depth > KVP_DEPTH) {
    loss(LOSS_RETURN);
    return 0;
  }
  u32 index = --frame->depth;
  u64 epoch = frame->frame[index].epoch;
  frame->frame[index].epoch = 0;
  if (!epoch) {
    loss(LOSS_RETURN);
    return 0;
  }
  struct kvp_call_key key = {.generation = thread.generation, .epoch = epoch};
  struct kvp_call* call = bpf_map_lookup_elem(&active_calls, &key);
  struct kvp_epoch* row = bpf_map_lookup_elem(&api_epochs, &key);
  if (!call || !row || row->complete) {
    loss(LOSS_RETURN);
    return 0;
  }
  row->end_ns = bpf_ktime_get_ns();
  row->status = (s32)PT_REGS_RC(ctx);
  __sync_lock_test_and_set(&row->complete, 1);
  bpf_map_delete_elem(&active_calls, &key);
  return 0;
}
static __noinline int socket_event(struct sock* sk, struct sk_buff* skb,
                                   u32 kind) {
  struct kvp_control* cfg = settings();
  if (!cfg || !cfg->enabled || !sk) return 0;
  struct net* net = BPF_CORE_READ(sk, __sk_common.skc_net.net);
  if (!net || BPF_CORE_READ(net, ns.inum) != cfg->netns) return 0;
  u64 cookie = BPF_CORE_READ(sk, __sk_common.skc_cookie.counter);
  if (!cookie) {
    loss(STAT_UNKNOWN_SOCKET);
    if (!kind && skb) {
      u32 zero = 0;
      struct kvp_loss_counts* stats = bpf_map_lookup_elem(&loss_counts, &zero);
      if (stats)
        add_count(&stats->count[STAT_UNKNOWN_RX_LENGTH],
                  BPF_CORE_READ(skb, len));
    }
    return 0;
  }
  struct kvp_socket_key key = {.netns = cfg->netns, .cookie = cookie};
  struct kvp_socket* row = bpf_map_lookup_elem(&socket_counts, &key);
  if (!row) {
    struct kvp_socket zero = {};
    bpf_map_update_elem(&socket_counts, &key, &zero, KVP_NOEXIST);
    row = bpf_map_lookup_elem(&socket_counts, &key);
    if (!row) {
      loss(LOSS_MAP);
      return 0;
    }
  }
  switch (kind) {
    case 0:
      add_count(&row->rx_events, 1);
      if (skb) add_count(&row->skb_len_bytes, BPF_CORE_READ(skb, len));
      break;
    case 1:
      add_count(&row->states, 1);
      break;
    case 2:
      add_count(&row->retransmits, 1);
      break;
    case 3:
      add_count(&row->send_resets, 1);
      break;
    case 4:
      add_count(&row->recv_resets, 1);
      break;
    default:
      loss(LOSS_UNSUPPORTED);
  }
  return 0;
}
SEC("tp_btf/tcp_probe")
int BPF_PROG(tcp_rx, struct sock* sk, struct sk_buff* skb) {
  return socket_event(sk, skb, 0);
}
SEC("tp_btf/inet_sock_set_state") int BPF_PROG(tcp_state, struct sock* sk) {
  return socket_event(sk, 0, 1);
}
SEC("tp_btf/tcp_retransmit_skb") int BPF_PROG(tcp_retrans, struct sock* sk) {
  return socket_event(sk, 0, 2);
}
SEC("tp_btf/tcp_send_reset") int BPF_PROG(tcp_reset_tx, struct sock* sk) {
  return socket_event(sk, 0, 3);
}
SEC("tp_btf/tcp_receive_reset") int BPF_PROG(tcp_reset_rx, struct sock* sk) {
  return socket_event(sk, 0, 4);
}
char LICENSE[] SEC("license") = "Dual BSD/GPL";
