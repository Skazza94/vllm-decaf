/* SPDX-License-Identifier: Apache-2.0 */
#define main kvp_collector_entry
#include "csrc/kv_profiler/ebpf/collector.c"
#undef main
#include <assert.h>
int main(int argc, char** argv) {
  assert(argc == 3);
  struct kvp_charge charge = {.bound = strtoull(argv[1], NULL, 10)};
  FILE* file = fopen(argv[2], "re");
  enum kvp_budget_result result = kvp_charge_fdinfo(&charge, file);
  if (file) fclose(file);
  int decision = report_charge(result, &charge);
  if (!decision)
    output("{\"type\":\"status\",\"reason\":\"fixture_gate_passed\"}");
  assert(nlinks == 0);
  assert(!flush_output());
  return decision;
}
