// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstddef>
#include <cstdint>

struct KvprofIdentity {
  const char* data;
  uint32_t length;
  uint32_t encoding;
};

struct KvprofEvent {
  uint64_t kind, context;
  KvprofIdentity identities[3];
  uint64_t handle;
  uint64_t values[7];
  uint64_t transport, status, subject;
  uint32_t valid;
  bool has_handle;
};

// The mapping must remain open until all producers have stopped.
// Zero means published; nonzero means dropped. Never waits for the consumer.
extern "C" __attribute__((visibility("default"))) int kvprof_record(
    void* mapping, const KvprofEvent* event);
