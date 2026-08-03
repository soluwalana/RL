#!/bin/sh
# Start gym_host_runtime inside the training image with a writable Gym tree.
#
# Usage:
#   gym_host.sh [venv] [git_root] [writable_gym_src] [runtime]
#
# Defaults match nemo_rl.environments.sandbox.host.entrypoint.
set -eu

venv=${1:-/opt/ray_venvs/nemo_rl.environments.sandbox.nemo_gym_actor.SandboxedGymActor}
root=${2:-/opt/nemo-rl}
gym_rw=${3:-/tmp/gym-src/Gym}
runtime=${4:-$root/nemo_rl/environments/sandbox/gym_host_runtime.py}

# Copy Gym to a writable path only because /opt/gym_venvs is not writable by
# uid 1000 under OpenSandbox. Remove this once that directory is writable.
if [ ! -d "$gym_rw/nemo_gym" ]; then
    mkdir -p "$(dirname "$gym_rw")"
    cp -a "$root/3rdparty/Gym-workspace/Gym" "$gym_rw"
fi

export PYTHONPATH="$gym_rw:$root${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"
echo "gym-host: starting gym_host_runtime (gym src $gym_rw)" >&2
exec "$venv/bin/python" "$runtime"
