/* SPDX-License-Identifier: Apache-2.0 */
#ifndef KVP_ACCOUNTING_H
#define KVP_ACCOUNTING_H
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* No map creation/attachment: an fdinfo reader shared with unprivileged tests.
 */
enum kvp_budget_result {
  KVP_BUDGET_OK,
  KVP_BUDGET_UNAVAILABLE,
  KVP_BUDGET_FAIL
};
struct kvp_charge {
  unsigned long long observed, bound, last;
  int overflow;
};
static enum kvp_budget_result kvp_charge_fdinfo(struct kvp_charge* charge,
                                                FILE* file) {
  if (!file) return KVP_BUDGET_UNAVAILABLE;
  char line[256];
  while (fgets(line, sizeof(line), file)) {
    if (strncmp(line, "memlock:", 8)) continue;
    char* start = line + 8;
    while (isspace((unsigned char)*start)) start++;
    if (!isdigit((unsigned char)*start)) return KVP_BUDGET_UNAVAILABLE;
    char* end;
    errno = 0;
    unsigned long long bytes = strtoull(start, &end, 10);
    if (errno || !bytes) return KVP_BUDGET_UNAVAILABLE;
    while (isspace((unsigned char)*end)) end++;
    if (*end) return KVP_BUDGET_UNAVAILABLE;
    charge->last = bytes;
    if (~0ULL - charge->observed < bytes) {
      charge->overflow = 1;
      return KVP_BUDGET_FAIL;
    }
    charge->observed += bytes;
    if (charge->observed > charge->bound || charge->observed > KVP_MAP_BYTES)
      return KVP_BUDGET_FAIL;
    return KVP_BUDGET_OK;
  }
  return KVP_BUDGET_UNAVAILABLE;
}
#endif
