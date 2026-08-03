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

from pathlib import Path

from nemo_rl.environments.sandbox.host.entrypoint import (
    DEFAULT_GYM_WRITABLE_SRC,
    NEMO_RL_IMAGE_GIT_ROOT,
    SANDBOXED_GYM_ACTOR_VENV,
    default_gym_host_entrypoint,
    gym_host_script_path,
)
from nemo_rl.environments.sandbox import gym_host_runtime as runtime


def test_default_gym_host_entrypoint_uses_sandboxed_actor_venv():
    entrypoint = default_gym_host_entrypoint()
    assert entrypoint == [
        "/bin/sh",
        gym_host_script_path(),
        SANDBOXED_GYM_ACTOR_VENV,
        NEMO_RL_IMAGE_GIT_ROOT,
        DEFAULT_GYM_WRITABLE_SRC,
    ]
    script = Path(__file__).resolve().parents[3] / (
        "nemo_rl/environments/sandbox/host/gym_host.sh"
    )
    text = script.read_text(encoding="utf-8")
    assert "cp -a" in text
    assert "gym_host_runtime.py" in text
    assert SANDBOXED_GYM_ACTOR_VENV in text


def test_ensure_writable_uv_dirs_fills_defaults(tmp_path, monkeypatch):
    cache = tmp_path / "uv-cache"
    venvs = tmp_path / "gym-venvs"
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("NEMO_GYM_VENV_DIR", raising=False)
    monkeypatch.setenv("SANDBOXED_GYM_UV_CACHE_DIR", str(cache))
    monkeypatch.setenv("SANDBOXED_GYM_UV_VENV_DIR", str(venvs))
    cfg: dict = {}
    runtime._ensure_writable_uv_dirs(cfg)
    assert cfg["uv_cache_dir"] == str(cache)
    assert cfg["uv_venv_dir"] == str(venvs)
    assert cache.is_dir()
    assert venvs.is_dir()


def test_ensure_writable_uv_dirs_preserves_config(tmp_path, monkeypatch):
    cache = tmp_path / "from-config-cache"
    venvs = tmp_path / "from-config-venvs"
    monkeypatch.setenv("SANDBOXED_GYM_UV_CACHE_DIR", str(tmp_path / "ignored"))
    cfg = {"uv_cache_dir": str(cache), "uv_venv_dir": str(venvs)}
    runtime._ensure_writable_uv_dirs(cfg)
    assert cfg["uv_cache_dir"] == str(cache)
    assert cfg["uv_venv_dir"] == str(venvs)


def test_opensandbox_host_provider_defaults_skip_health_check():
    from nemo_rl.environments.sandbox.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(connection={"domain": "x", "api_key": "k"})
    assert provider._create_options["skip_health_check"] is True


def test_opensandbox_host_provider_honors_explicit_skip_health_check():
    from nemo_rl.environments.sandbox.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(
        connection={"domain": "x", "api_key": "k"},
        create={"skip_health_check": False},
    )
    assert provider._create_options["skip_health_check"] is False
