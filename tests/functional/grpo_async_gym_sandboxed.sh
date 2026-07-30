#!/bin/bash
# Sandboxed counterpart of grpo_async_gym.sh.
# Requires OpenSandbox placement env (see lib/sandboxed_gym_hydra_overrides.sh).

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
# shellcheck source=lib/sandboxed_gym_hydra_overrides.sh
source "$SCRIPT_DIR/lib/sandboxed_gym_hydra_overrides.sh"

mapfile -t SANDBOXED_OVERRIDES < <(sandboxed_gym_hydra_overrides)

exec bash "$SCRIPT_DIR/grpo_async_gym.sh" "${SANDBOXED_OVERRIDES[@]}" "$@"
