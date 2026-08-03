# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default entrypoint for the sandboxed Gym host inside the training image.

The published ``nmp-rl-training`` / ``nemo-rl`` images bake Gym under
``/opt/nemo-rl/3rdparty/Gym-workspace/Gym`` as an editable install, but that tree
is root-owned while OpenSandbox pods run as uid 1000. Gym derives its working
dir from ``nemo_gym.__file__`` and writes ``cache/nemo_gym.egg-info`` when
building per-app venvs, so the host must run against a writable Gym copy.

``/opt/nemo_rl_venv`` also lacks the ``nemo_gym[sandbox]`` extra, and
``uv sync --extra nemo_gym`` cannot complete from the image's offline uv cache
(missing daytona/socketio wheels). The image does bake a per-Ray-worker venv for
``SandboxedGymActor`` that already has those packages.

The host process is started by :file:`gym_host.sh` (same directory).
"""

from __future__ import annotations

import os

NEMO_RL_IMAGE_GIT_ROOT = "/opt/nemo-rl"
SANDBOXED_GYM_ACTOR_VENV = (
    "/opt/ray_venvs/nemo_rl.environments.sandbox.nemo_gym_actor.SandboxedGymActor"
)
# Writable by uid 1000 in the image; /opt/uv_cache is root-owned and read-only.
DEFAULT_GYM_UV_CACHE_DIR = "/home/ubuntu/.cache/uv"
DEFAULT_GYM_UV_VENV_DIR = "/opt/gym_venvs"
DEFAULT_GYM_WRITABLE_SRC = "/tmp/gym-src/Gym"

_GYM_HOST_SCRIPT_NAME = "gym_host.sh"
_GYM_HOST_SCRIPT_RELPATH = (
    f"nemo_rl/environments/sandbox/host/{_GYM_HOST_SCRIPT_NAME}"
)


def gym_uv_cache_dir() -> str:
    """Writable uv cache for Gym servers inside the job sandbox."""
    return os.environ.get("SANDBOXED_GYM_UV_CACHE_DIR", DEFAULT_GYM_UV_CACHE_DIR)


def gym_uv_venv_dir() -> str:
    """Writable root for Gym per-app venvs inside the job sandbox."""
    return (
        os.environ.get("NEMO_GYM_VENV_DIR")
        or os.environ.get("SANDBOXED_GYM_UV_VENV_DIR")
        or DEFAULT_GYM_UV_VENV_DIR
    )


def gym_writable_src_dir() -> str:
    """Writable Gym checkout used so editable installs can write egg-info."""
    return os.environ.get("SANDBOXED_GYM_SRC_DIR", DEFAULT_GYM_WRITABLE_SRC)


def gym_host_script_path(*, git_root: str | None = None) -> str:
    """Absolute path to ``gym_host.sh`` inside the training image."""
    root = git_root or NEMO_RL_IMAGE_GIT_ROOT
    return f"{root}/{_GYM_HOST_SCRIPT_RELPATH}"


def default_gym_host_entrypoint(
    *,
    venv: str | None = None,
    git_root: str | None = None,
    writable_gym_src: str | None = None,
) -> list[str]:
    """Shell entrypoint that starts ``gym_host_runtime`` in the training image."""
    root = git_root or NEMO_RL_IMAGE_GIT_ROOT
    return [
        "/bin/sh",
        gym_host_script_path(git_root=root),
        venv or SANDBOXED_GYM_ACTOR_VENV,
        root,
        writable_gym_src or gym_writable_src_dir(),
    ]
