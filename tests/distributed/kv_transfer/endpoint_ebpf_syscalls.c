/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Preserve the variadic ABI, including stack args, for unrelated syscalls. */
static void* kvp_real_syscall __attribute__((used));
__attribute__((constructor)) static void resolve_syscall(void) {
  kvp_real_syscall = dlsym(RTLD_NEXT, "syscall");
  if (!kvp_real_syscall) _exit(98);
}
static long kvp_deny(long number) __attribute__((used));
static long kvp_deny(long number) {
  const char* path = getenv("KVP_SYSCALL_LOG");
  if (path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (fd >= 0) {
      char row[64];
      int n = snprintf(row, sizeof(row), "syscall %ld PROBE\n", number);
      if (n > 0) write(fd, row, n);
      close(fd);
    }
  }
  errno = EPERM;
  return -1;
}
#define KVP_STRING_(x) #x
#define KVP_STRING(x) KVP_STRING_(x)
#if !defined(__x86_64__)
  #error "The pinned passive-inspection fixture is x86-64 only"
#endif
__attribute__((naked)) long syscall(long number, ...) {
  __asm__ volatile("cmp $" KVP_STRING(SYS_bpf) ", %rdi\n"
                     "je kvp_deny\n"
                     "cmp $" KVP_STRING(SYS_perf_event_open) ", %rdi\n"
                     "je kvp_deny\n"
                     "jmp *kvp_real_syscall(%rip)\n");
}
