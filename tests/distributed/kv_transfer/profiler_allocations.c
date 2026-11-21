// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

extern void* __libc_malloc(size_t);
extern void* __libc_calloc(size_t, size_t);
extern void* __libc_realloc(void*, size_t);
static _Thread_local bool active;
static _Thread_local uint64_t calls;

void kvprof_alloc_begin(void) {
  calls = 0;
  active = true;
}
uint64_t kvprof_alloc_end(void) {
  active = false;
  return calls;
}
void* malloc(size_t n) {
  if (active) ++calls;
  return __libc_malloc(n);
}
void* calloc(size_t n, size_t s) {
  if (active) ++calls;
  return __libc_calloc(n, s);
}
void* realloc(void* p, size_t n) {
  if (active) ++calls;
  return __libc_realloc(p, n);
}
