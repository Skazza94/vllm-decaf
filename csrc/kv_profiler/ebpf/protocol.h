/* SPDX-License-Identifier: Apache-2.0 */
#ifndef KVP_EBPF_PROTOCOL_H
#define KVP_EBPF_PROTOCOL_H
#ifdef KVP_HOST
#include <stdint.h>
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
#else
typedef __u64 u64;
typedef __u32 u32;
typedef __u16 u16;
typedef __u8 u8;
#endif
#define KVP_SCHEMA 3
#define KVP_RING_BYTES 1048576
#define KVP_PORTS 32
/* Owned sockets in force (cookie -> admission), and the per-cookie rejection
   counters (best effort: an LRU; the exact total is a loss counter). */
#define KVP_ALLOWED 65536
#define KVP_REJECTED 4096
#define KVP_RECORD_BYTES 2048
#define KVP_QUEUE 4096
#define KVP_EVENTS 4
#define KVP_FIELDS 8
#define KVP_INSNS 16384
/* These are protocol constants, not kernel layouts. */
enum kvp_kind { KVP_RETRANS, KVP_RESET_TX, KVP_RESET_RX, KVP_STATE };
enum kvp_stat {
  GLOBAL_NULL,
  GLOBAL_SCOPE_READ,
  GLOBAL_NONFULL,
  EXCLUDED_NAMESPACE,
  EXCLUDED_PROTOCOL,
  EXCLUDED_PORT,
  EXCLUDED_FAMILY,
  /* eligible events (namespace, protocol, family and ports passed) whose socket
     is not an owned one: every rejection, counted once */
  EXCLUDED_UNOWNED,
  /* rejections the per-cookie counters could not record (an LRU insertion lost
     to a concurrent one): part of the total, never of a cookie's count */
  REJECTED_UNTRACKED,
  SCOPED_RING_DROP,
  SCOPED_TUPLE_READ,
  STAT_WRAP,
  STAT_COUNT
};
struct kvp_control {
  u64 enabled, netns;
  u32 local_count, peer_count;
  u16 local_ports[KVP_PORTS], peer_ports[KVP_PORTS];
  u8 reserved[360];
};
/* An owned socket's entry in force: the admission it was matched under. */
struct kvp_admission {
  u64 admission;
};
/* A rejected cookie's counter: its events rejected, the first and last. */
/* A cookie's rejections since its counter was last harvested: their count,
   the earliest event time (KVP_FIRST_UNKNOWN set when contention left it
   unknown) and the latest written. */
struct kvp_rejected {
  u64 count, first_ns, last_ns;
};
#define KVP_FIRST_UNKNOWN (1ULL << 63)
struct kvp_loss_counts {
  u64 count[32];
};
struct kvp_event {
  u64 timestamp_ns, netns, cookie, admission;
  u32 schema, size, kind, family;
  u16 local_port, peer_port;
  u8 local_address[16], peer_address[16];
  u32 old_state, new_state;
  u32 fields[KVP_FIELDS];
  u32 available, read_errors;
  u8 reserved[124];
};
_Static_assert(sizeof(struct kvp_control) == 512, "control capacity");
_Static_assert(sizeof(struct kvp_loss_counts) == 256, "statistics capacity");
_Static_assert(sizeof(struct kvp_event) == 256, "binary event capacity");
_Static_assert(sizeof(struct kvp_admission) == 8, "admission value");
_Static_assert(sizeof(struct kvp_rejected) == 24, "rejection value");
#endif
