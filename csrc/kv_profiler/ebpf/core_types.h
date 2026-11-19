/* SPDX-License-Identifier: (BSD-2-Clause OR GPL-2.0-only) */
#ifndef KVP_CORE_TYPES_H
#define KVP_CORE_TYPES_H
#include <linux/bpf.h>
/* Minimal CO-RE declarations. No target offsets, padding or struct sizes. */
#ifndef KVP_HOST
#pragma clang attribute push(__attribute__((preserve_access_index)), apply_to = record)
#endif
struct ns_common {
  unsigned int inum;
};
struct net {
  struct ns_common ns;
};
typedef struct {
  struct net *net;
} possible_net_t;
typedef struct {
  long long counter;
} atomic64_t;
#ifndef KVP_HOST
struct in6_addr {
  union {
    __u8 u6_addr8[16];
    __u32 u6_addr32[4];
  } in6_u;
};
#endif
struct sock_common {
  __u32 skc_daddr, skc_rcv_saddr;
  __u16 skc_dport, skc_num, skc_family;
  unsigned char skc_state;
  possible_net_t skc_net;
  struct in6_addr skc_v6_daddr, skc_v6_rcv_saddr;
  atomic64_t skc_cookie;
};
struct sock {
  struct sock_common __sk_common;
  __u16 sk_protocol;
};
struct tcp_sock {
  __u32 srtt_us, snd_cwnd, snd_ssthresh, packets_out;
  __u32 retrans_out, lost_out, total_retrans, mss_cache;
};
#ifndef KVP_HOST
#pragma clang attribute pop
#endif
/* Stable TCP state ABI, as used by the kernel full-socket predicate. */
enum { TCP_TIME_WAIT = 6, TCP_NEW_SYN_RECV = 12 };
#endif
