# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run by absolute file path to avoid importing vLLM or Torch in the observer."""

if __name__ == "__main__":
    from endpoint_probes.cli import main

    raise SystemExit(main())
