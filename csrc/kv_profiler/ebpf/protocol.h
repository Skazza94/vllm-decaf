/* SPDX-License-Identifier: Apache-2.0 */
#ifndef KVP_EBPF_PROTOCOL_H
#define KVP_EBPF_PROTOCOL_H
#ifdef KVP_HOST
  #include <stdint.h>
typedef uint64_t u64;
typedef uint32_t u32;
typedef int32_t s32;
#else
typedef __u64 u64;
typedef __u32 u32;
typedef __s32 s32;
#endif
#define KVP_PROCESSES 256
#define KVP_THREADS 2048
#define KVP_PER_PROCESS_THREADS 128
#define KVP_DEPTH 4
#define KVP_CALLS 4096
#define KVP_EPOCHS 16384
#define KVP_LISTS 256
#define KVP_TRANSACTIONS 1024
#define KVP_WR 64
#define KVP_SGE 8
#define KVP_BUILDERS 1024
#define KVP_MAP_BYTES 16777216ULL
#define KVP_OBJECT_MAPS 16
#define KVP_API_CAPACITY 16384
#define KVP_RECORD_BYTES 256
#define KVP_NOEXIST 1

enum kvp_loss {
  LOSS_NAMESPACE,
  LOSS_IDENTITY,
  LOSS_THREADS,
  LOSS_DEPTH,
  LOSS_MAP,
  LOSS_RETURN,
  LOSS_STATE,
  LOSS_WRAP,
  LOSS_READ,
  LOSS_LIST,
  LOSS_SGE,
  LOSS_BAD_WR,
  LOSS_TXN,
  LOSS_OPERATION,
  LOSS_BUILDER,
  LOSS_UNSUPPORTED,
  LOSS_QUIESCENCE,
  LOSS_QUEUE,
  LOSS_CLOCK,
  LOSS_TARGET,
  LOSS_COUNT
};
enum { STAT_UNKNOWN_SOCKET = 20, STAT_UNKNOWN_RX_LENGTH = 21 };
struct kvp_thread_key {
  u64 generation, start;
  u32 tgid, tid;
};
struct kvp_call_key {
  u64 generation, epoch;
};
struct kvp_process {
  u64 generation, sequence;
  u32 thread_slots, reserved;
};
struct kvp_thread {
  u64 first_ns, calls, pending_epochs;
};
struct kvp_frame {
  u64 epoch;
};
struct kvp_frames {
  u32 depth, suppressed;
  struct kvp_frame frame[KVP_DEPTH];
};
struct kvp_call {
  struct kvp_thread_key thread;
  u64 handle;
};
struct kvp_epoch {
  u64 begin_ns, end_ns, handle;
  struct kvp_thread_key thread;
  s32 status;
  u32 complete;
};
struct kvp_socket_key {
  u64 netns, cookie;
};
struct kvp_socket {
  u64 rx_events, skb_len_bytes, states, retransmits, send_resets, recv_resets;
};
struct kvp_control {
  u64 enabled, netns, pid_dev, pid_ino;
  struct kvp_thread_key retire;
  u64 command;
};
/* Control is <=64 bytes: use separate configuration/retirement entries. */
struct kvp_loss_counts {
  u64 count[32];
};
struct kvp_operation {
  u64 token, bytes;
  u32 opcode, flags, sges, reserved;
};
struct kvp_batch {
  u64 owner, qp, generation, builders, operations, valid;
  u64 reserved[10];
  struct kvp_operation op[KVP_WR];
};
#define KVP_MAPS(X)                                                           \
  X(processes, KVP_PROCESSES, u32, struct kvp_process, 32, 128)               \
  X(threads, KVP_THREADS, struct kvp_thread_key, struct kvp_thread, 32, 96)   \
  X(call_frames, KVP_THREADS, struct kvp_thread_key, struct kvp_frames, 32,   \
    128)                                                                      \
  X(active_calls, KVP_CALLS, struct kvp_call_key, struct kvp_call, 48, 96)    \
  X(api_epochs, KVP_EPOCHS, struct kvp_call_key, struct kvp_epoch, 48, 128)   \
  X(legacy_lists, KVP_LISTS, struct kvp_call_key, struct kvp_batch, 48, 2176) \
  X(ext_txns, KVP_TRANSACTIONS, struct kvp_call_key, struct kvp_batch, 48,    \
    2176)                                                                     \
  X(qp_counts, 4096, struct kvp_socket_key, struct kvp_socket, 48, 128)       \
  X(socket_counts, 1024, struct kvp_socket_key, struct kvp_socket, 48, 128)
#define KVP_CHECK(name, count, key, value, kc, vc)       \
  _Static_assert(sizeof(key) <= kc, #name " key bound"); \
  _Static_assert(sizeof(value) <= vc, #name " value bound");
KVP_MAPS(KVP_CHECK)
_Static_assert(sizeof(struct kvp_control) <= 64, "control bound");
_Static_assert(sizeof(struct kvp_loss_counts) == 256, "per-cpu bound");
#endif
