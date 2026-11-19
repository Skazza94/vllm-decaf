/* SPDX-License-Identifier: Apache-2.0 */
#ifndef KVP_PARSERS_H
#define KVP_PARSERS_H
#include "protocol.h"
/* Owned fixture ABI, never a claim about an installed provider's layout. */
struct kvp_sge {
  u64 address;
  u32 length, key;
};
struct kvp_wr {
  u64 next, token, sges;
  u32 count, opcode, flags, reserved;
};
typedef int (*kvp_read_fn)(void*, u64, u32);
static int kvp_parse_list(struct kvp_batch* batch, u64 wr, kvp_read_fn read) {
  batch->operations = 0;
  batch->valid = 0;
  for (u32 i = 0; i < KVP_WR; i++) {
    if (!wr) {
      batch->valid = 1;
      return 0;
    }
    for (u32 j = 0; j < i; j++)
      if (batch->op[j].token == wr) return LOSS_LIST;
    struct kvp_wr row;
    if (read(&row, wr, sizeof(row))) return LOSS_READ;
    if (row.count > KVP_SGE) return LOSS_SGE;
    struct kvp_operation* op = &batch->op[i];
    *op = (struct kvp_operation){.token = wr,
                                 .opcode = row.opcode,
                                 .flags = row.flags,
                                 .sges = row.count};
    for (u32 s = 0; s < KVP_SGE; s++) {
      if (s >= row.count) break;
      struct kvp_sge sg;
      if (row.sges > ~0ULL - s * sizeof(sg) ||
          read(&sg, row.sges + s * sizeof(sg), sizeof(sg)))
        return LOSS_READ;
      op->bytes += sg.length;
    }
    batch->operations++;
    wr = row.next;
  }
  if (wr) return LOSS_LIST;
  batch->valid = 1;
  return 0;
}
static int kvp_accepted(const struct kvp_batch* b, int rc, u64 bad, u64* ops,
                        u64* bytes) {
  *ops = *bytes = 0;
  if (!b->valid) return LOSS_BAD_WR;
  if (rc && !bad) return LOSS_BAD_WR;
  for (u32 i = 0; i < KVP_WR; i++) {
    if (i >= b->operations) break;
    if (rc && b->op[i].token == bad) return 0;
    (*ops)++;
    *bytes += b->op[i].bytes;
  }
  if (rc) {
    *ops = *bytes = 0;
    return LOSS_BAD_WR;
  }
  return 0;
}
static int kvp_tx_start(struct kvp_batch* b, u64 owner, u64 qp,
                        u64 generation) {
  if (b->valid) return LOSS_TXN;
  *b = (struct kvp_batch){.owner = owner,
                          .qp = qp,
                          .generation = generation,
                          .valid = 1,
                          .builders = 1};
  return 0;
}
static int kvp_tx_tick(struct kvp_batch* b, u64 owner, u64 qp) {
  if (!b->valid || b->owner != owner || b->qp != qp) return LOSS_TXN;
  if (++b->builders > KVP_BUILDERS) {
    b->valid = 0;
    return LOSS_BUILDER;
  }
  return 0;
}
static int kvp_tx_op(struct kvp_batch* b, u64 owner, u64 qp, u32 opcode) {
  int error = kvp_tx_tick(b, owner, qp);
  if (error) return error;
  if (b->operations == KVP_WR) {
    b->valid = 0;
    return LOSS_OPERATION;
  }
  b->op[b->operations++] = (struct kvp_operation){.opcode = opcode};
  return 0;
}
static int kvp_tx_sges(struct kvp_batch* b, u64 owner, u64 qp, u32 count,
                       u64 sges, kvp_read_fn read) {
  int error = kvp_tx_tick(b, owner, qp);
  if (error) return error;
  if (!b->operations) return LOSS_TXN;
  if (count > KVP_SGE) {
    b->valid = 0;
    return LOSS_SGE;
  }
  struct kvp_operation* op = &b->op[b->operations - 1];
  op->bytes = 0;
  op->sges = count;
  for (u32 i = 0; i < count; i++) {
    struct kvp_sge sg;
    if (sges > ~0ULL - i * sizeof(sg) ||
        read(&sg, sges + i * sizeof(sg), sizeof(sg))) {
      b->valid = 0;
      return LOSS_READ;
    }
    op->bytes += sg.length;
  }
  return 0; /* This fixture's setter replaces, rather than appends. */
}
static int kvp_tx_end(struct kvp_batch* b, u64 owner, u64 qp, int abort) {
  int error = kvp_tx_tick(b, owner, qp);
  if (error) return error;
  b->valid = 0;
  if (abort) b->operations = 0;
  return 0;
}
#endif
