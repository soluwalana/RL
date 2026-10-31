#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Sandboxed counterpart of run_nemo_gym_single_node_sanity_tests.sh.
# Runs the live OpenSandbox host test plus SandboxedGymActor sanity.
# Requires LIVE_OPENSANDBOX=1 and the OPENSANDBOX_* placement env vars.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
cd "$PROJECT_ROOT"

if [[ "${LIVE_OPENSANDBOX:-}" != "1" ]]; then
  echo "[ERROR] Set LIVE_OPENSANDBOX=1 and OpenSandbox placement env vars first" >&2
  exit 1
fi

uv sync --all-groups --extra nemo_gym || true

uv run ray stop --force || true
uv run python -c "import ray; ray.shutdown()" || true

# Host provider live functionality (create/wait/proxy/egress/destroy).
./tests/run_unit.sh unit/environments/test_sandboxed_gym_host_live.py::test_live_host_functionality

# Ray SandboxedGymActor → OpenSandbox host (real Gym runtime) → postprocess.
./tests/run_unit.sh unit/environments/test_sandboxed_nemo_gym.py::test_sandboxed_nemo_gym_sanity
