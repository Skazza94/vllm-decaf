/* SPDX-License-Identifier: Apache-2.0 */
#ifndef KVP_LIFECYCLE_H
#define KVP_LIFECYCLE_H
#include <errno.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <unistd.h>
/* Call before opening any BPF object, maps, programs or links. */
static int kvp_parent_guard(pid_t parent) {
  if (parent <= 0 || getppid() != parent) return -ESRCH;
  if (prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0)) return -errno;
  if (getppid() != parent) return -ESRCH;
  return 0;
}
#endif
