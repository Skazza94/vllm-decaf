/* SPDX-License-Identifier: (BSD-2-Clause OR GPL-2.0-only) */
#ifndef KVP_HOST
#include "core_types.h"
#endif
#ifdef KVP_HOST
#include "endpoint_ebpf_helpers.h"
#else
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_endian.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#endif
#include "protocol.h"
struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, KVP_RING_BYTES);
} events SEC(".maps");
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, struct kvp_control);
} control SEC(".maps");
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, struct kvp_loss_counts);
} loss_counts SEC(".maps");
static __always_inline void count_stat(u32 kind) {
  u32 zero = 0;
  struct kvp_loss_counts *counts = bpf_map_lookup_elem(&loss_counts, &zero);
  if (counts && kind < STAT_COUNT) {
    u64 old = __sync_fetch_and_add(&counts->count[kind], 1);
    if (old == ~0ULL) __sync_fetch_and_or(&counts->count[STAT_WRAP], 1);
  }
}
static __always_inline int port_allowed(u16 port, u16 *ports, u32 count) {
  if (!count) return 1;
  for (u32 i = 0; i < KVP_PORTS; i++)
    if (i < count && ports[i] == port) return 1;
  return 0;
}
static __noinline int socket_event(struct sock *sk, u32 kind, u32 old, u32 next) {
  u64 timestamp = bpf_ktime_get_ns();
  u32 zero = 0;
  struct kvp_control *cfg = bpf_map_lookup_elem(&control, &zero);
  if (!cfg || !cfg->enabled) return 0;
  if (!sk) {
    count_stat(GLOBAL_NULL);
    return 0;
  }
  u8 state = 0;
  if (BPF_CORE_READ_INTO(&state, sk, __sk_common.skc_state)) {
    count_stat(GLOBAL_SCOPE_READ);
    return 0;
  }
  /* sk_fullsock(): request/time-wait objects are not full struct sock objects. */
  if (state == TCP_TIME_WAIT || state == TCP_NEW_SYN_RECV) {
    count_stat(GLOBAL_NONFULL);
    return 0;
  }
  struct net *net = 0;
  u32 netns = 0;
  if (BPF_CORE_READ_INTO(&net, sk, __sk_common.skc_net.net) || !net ||
      BPF_CORE_READ_INTO(&netns, net, ns.inum)) {
    count_stat(GLOBAL_SCOPE_READ);
    return 0;
  }
  if (netns != cfg->netns) {
    count_stat(EXCLUDED_NAMESPACE);
    return 0;
  }
  u16 protocol = 0, family = 0, local = 0, peer = 0;
  if (BPF_CORE_READ_INTO(&protocol, sk, sk_protocol)) {
    count_stat(SCOPED_TUPLE_READ);
    return 0;
  }
  if (protocol != 6) {
    count_stat(EXCLUDED_PROTOCOL);
    return 0;
  }
  if (BPF_CORE_READ_INTO(&family, sk, __sk_common.skc_family) ||
      BPF_CORE_READ_INTO(&local, sk, __sk_common.skc_num) ||
      BPF_CORE_READ_INTO(&peer, sk, __sk_common.skc_dport)) {
    count_stat(SCOPED_TUPLE_READ);
    return 0;
  }
  if (family != 2 && family != 10) {
    count_stat(EXCLUDED_FAMILY);
    return 0;
  }
  peer = bpf_ntohs(peer);
  if (!port_allowed(local, cfg->local_ports, cfg->local_count) ||
      !port_allowed(peer, cfg->peer_ports, cfg->peer_count)) {
    count_stat(EXCLUDED_PORT);
    return 0;
  }
  struct kvp_event *row = bpf_ringbuf_reserve(&events, sizeof(*row), 0);
  if (!row) {
    count_stat(SCOPED_RING_DROP);
    return 0;
  }
  __builtin_memset(row, 0, sizeof(*row));
  row->timestamp_ns = timestamp;
  row->netns = netns;
  row->schema = KVP_SCHEMA;
  row->size = sizeof(*row);
  row->kind = kind;
  row->family = family;
  row->local_port = local;
  row->peer_port = peer;
  row->old_state = old;
  row->new_state = next;
  int error;
  if (family == 2) {
    error =
        BPF_CORE_READ_INTO((u32 *)row->local_address, sk, __sk_common.skc_rcv_saddr);
    error |= BPF_CORE_READ_INTO((u32 *)row->peer_address, sk, __sk_common.skc_daddr);
  } else {
    error = BPF_CORE_READ_INTO(&row->local_address, sk, __sk_common.skc_v6_rcv_saddr);
    error |= BPF_CORE_READ_INTO(&row->peer_address, sk, __sk_common.skc_v6_daddr);
  }
  if (error) {
    count_stat(SCOPED_TUPLE_READ);
    count_stat(SCOPED_RECORD_DISCARD);
    bpf_ringbuf_discard(row, 0);
    return 0;
  }
  if (bpf_core_field_exists(sk->__sk_common.skc_cookie.counter) &&
      !BPF_CORE_READ_INTO(&row->cookie, sk, __sk_common.skc_cookie.counter))
    row->cookie_available = row->cookie != 0;
  struct tcp_sock *tp = (void *)sk;
#define SNAP(index, field)                                                         \
  do {                                                                             \
    if (bpf_core_field_exists(tp->field) && bpf_core_field_size(tp->field) == 4) { \
      if (BPF_CORE_READ_INTO(&row->fields[index], tp, field))                      \
        row->read_errors |= 1U << index;                                           \
      else                                                                         \
        row->available |= 1U << index;                                             \
    }                                                                              \
  } while (0)
  SNAP(0, srtt_us);
  SNAP(1, snd_cwnd);
  SNAP(2, snd_ssthresh);
  SNAP(3, packets_out);
  SNAP(4, retrans_out);
  SNAP(5, lost_out);
  SNAP(6, total_retrans);
  SNAP(7, mss_cache);
#undef SNAP
  bpf_ringbuf_submit(row, 0);
  return 0;
}
SEC("tp_btf/tcp_retransmit_skb")
int BPF_PROG(tcp_retrans, struct sock *sk) {
  return socket_event(sk, KVP_RETRANS, 0, 0);
}
SEC("tp_btf/tcp_send_reset")
int BPF_PROG(tcp_reset_tx, struct sock *sk) {
  return socket_event(sk, KVP_RESET_TX, 0, 0);
}
SEC("tp_btf/tcp_receive_reset")
int BPF_PROG(tcp_reset_rx, struct sock *sk) {
  return socket_event(sk, KVP_RESET_RX, 0, 0);
}
SEC("tp_btf/inet_sock_set_state")
int BPF_PROG(tcp_state, struct sock *sk, int old, int next) {
  return socket_event(sk, KVP_STATE, old, next);
}
char LICENSE[] SEC("license") = "Dual BSD/GPL";
