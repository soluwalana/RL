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

import asyncio
import json
import threading
from http.server import HTTPServer
from unittest.mock import MagicMock

import pytest

from nemo_rl.environments.sandbox import gym_host_runtime as runtime


class _FakeRolloutHelper:
    """Stands in for ``RolloutCollectionHelper``, including its completion ordering.

    The real ``run_examples`` returns ``tqdm.asyncio.tqdm.as_completed``, so rows come
    back in the order they finish. Modelling that here is what keeps a fake from
    certifying a pairing that only works when nothing overtakes anything else.
    """

    def __init__(self, latency_by_rowidx: dict | None = None):
        self.latency_by_rowidx = latency_by_rowidx or {}

    def run_examples(self, examples, head_server_config=None):
        async def _one(row):
            await asyncio.sleep(self.latency_by_rowidx.get(row.get("_rowidx"), 0))
            return row, {
                "response": {"output": []},
                "reward": 0.0,
                "answered_rowidx": row.get("_rowidx"),
            }

        return asyncio.as_completed([_one(row) for row in examples])


@pytest.fixture
def ready_server():
    runtime._READY = True
    runtime._RUN_HELPER = MagicMock()
    runtime._HEAD_SERVER_CONFIG = MagicMock()
    runtime._ROLLOUT_HELPER = _FakeRolloutHelper()
    runtime.Handler.max_request_bytes = 1024
    runtime.Handler.max_response_bytes = 4096

    server = HTTPServer(("127.0.0.1", 0), runtime.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        runtime._READY = False
        runtime._HEAD_SERVER_CONFIG = None
        runtime._ROLLOUT_HELPER = None


def test_health_not_ready():
    runtime._READY = False
    server = HTTPServer(("127.0.0.1", 0), runtime.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import urllib.error
        import urllib.request

        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
        assert exc.value.code == 503
        body = json.loads(exc.value.read().decode())
        assert body["status"] == "starting"
    finally:
        server.shutdown()
        server.server_close()


def test_health_ready(ready_server):
    import urllib.request

    with urllib.request.urlopen(f"{ready_server}/health", timeout=5) as resp:
        assert resp.status == 200
        assert json.loads(resp.read().decode()) == {"status": "ready"}


def test_rollouts_run_returns_results(ready_server):
    import urllib.request

    payload = json.dumps(
        {"examples": [{"agent_ref": {"name": "a"}, "id": 1, "_rowidx": 0}]}
    ).encode()
    req = urllib.request.Request(
        f"{ready_server}/rollouts/run",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
    assert len(body["results"]) == 1
    rowidx, result = body["results"][0]
    assert rowidx == 0
    assert result["reward"] == 0.0


def test_rollouts_run_rejects_oversize_request(ready_server):
    import urllib.error
    import urllib.request

    payload = b"x" * 2048
    req = urllib.request.Request(
        f"{ready_server}/rollouts/run",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        },
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 413
    err = json.loads(exc.value.read().decode())
    assert err["error"]["code"] == "payload_too_large"


def test_run_rollouts_sync_collects():
    helper = _FakeRolloutHelper()
    results = runtime.run_rollouts_sync(
        [{"agent_ref": {"name": "x"}, "_rowidx": 0}],
        MagicMock(),
        helper,
    )
    assert len(results) == 1


def test_run_rollouts_sync_tags_results_with_their_own_rowidx():
    """Results must carry the row they were produced for, not their arrival position.

    Gym completes rows out of order, so an untagged list pairs a prompt with whichever
    result happened to finish in its slot -- silently training one prompt's tokens
    under another prompt's group.
    """
    examples = [{"agent_ref": {"name": "x"}, "_rowidx": i} for i in range(8)]
    # Row 0 finishes last, row 7 first: completion order is the exact reverse.
    helper = _FakeRolloutHelper({i: (8 - i) * 0.01 for i in range(8)})

    results = runtime.run_rollouts_sync(examples, MagicMock(), helper)

    assert [rowidx for rowidx, _ in results] != list(range(8)), (
        "fake did not overtake; the test is not exercising completion ordering"
    )
    for rowidx, result in results:
        assert rowidx == result["answered_rowidx"]


def test_run_rollouts_sync_rejects_row_without_rowidx():
    """Pairing is by tag, so an untaggable row must fail loudly rather than by position."""
    with pytest.raises(RuntimeError, match="_rowidx"):
        runtime.run_rollouts_sync(
            [{"agent_ref": {"name": "x"}}], MagicMock(), _FakeRolloutHelper()
        )


def test_run_rollouts_sync_reuses_one_live_event_loop():
    """Successive batches must share a loop that is still running afterwards.

    NeMo-Gym memoizes a global aiohttp session bound to the loop that first created it,
    so a per-request loop leaves every batch after the first driving a closed loop.
    """
    helper = _FakeRolloutHelper()
    loops = []

    class _LoopCapturingHelper:
        def run_examples(self, examples, head_server_config=None):
            loops.append(asyncio.get_running_loop())
            return helper.run_examples(examples, head_server_config)

    capturing = _LoopCapturingHelper()
    for _ in range(3):
        runtime.run_rollouts_sync(
            [{"agent_ref": {"name": "x"}, "_rowidx": 0}], MagicMock(), capturing
        )

    assert len(loops) == 3
    assert len({id(loop) for loop in loops}) == 1
    assert not loops[0].is_closed()
    assert loops[0].is_running()


def test_run_rollouts_sync_supports_concurrent_callers():
    """Threaded handlers submit to the shared loop, so chunked POSTs can overlap."""
    started = threading.Barrier(3, timeout=10)

    class _BarrierHelper:
        def run_examples(self, examples, head_server_config=None):
            async def _one(row):
                # Blocks until all three callers are in flight on the same loop.
                await asyncio.to_thread(started.wait)
                return row, {"response": {"output": []}, "reward": 0.0}

            return [_one(row) for row in examples]

    helper = _BarrierHelper()
    results: list[int] = []

    def _call():
        out = runtime.run_rollouts_sync(
            [{"agent_ref": {"name": "x"}, "_rowidx": 0}], MagicMock(), helper
        )
        results.append(len(out))

    threads = [threading.Thread(target=_call) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert results == [1, 1, 1]


def test_apply_uv_dirs_sets_config_keys_in_container(monkeypatch):
    """Gym reads the CONFIG keys, not the env vars - the env alone gets overwritten."""
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", "/opt/gym_venvs")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        runtime.subprocess, "check_output", lambda *a, **k: "/opt/uv_cache\n"
    )

    config: dict = {}
    runtime._apply_uv_dirs(config)

    assert config[runtime.UV_CACHE_DIR_KEY] == "/opt/uv_cache"
    assert config[runtime.UV_VENV_DIR_KEY] == "/opt/gym_venvs"


def test_apply_uv_dirs_noop_outside_container(monkeypatch):
    monkeypatch.delenv("NRL_CONTAINER", raising=False)
    monkeypatch.delenv("NEMO_GYM_VENV_DIR", raising=False)

    config: dict = {}
    runtime._apply_uv_dirs(config)

    assert config == {}


def test_apply_uv_dirs_does_not_override_explicit_config(monkeypatch):
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", "/opt/gym_venvs")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        runtime.subprocess, "check_output", lambda *a, **k: "/opt/uv_cache\n"
    )

    config = {runtime.UV_CACHE_DIR_KEY: "/custom/cache"}
    runtime._apply_uv_dirs(config)

    assert config[runtime.UV_CACHE_DIR_KEY] == "/custom/cache"
    assert config[runtime.UV_VENV_DIR_KEY] == "/opt/gym_venvs"


def test_uv_cache_dir_prefers_the_configured_env_var(monkeypatch):
    """`uv cache dir` exits non-zero when the CWD's pyproject pins a conflicting
    [tool.uv] required-version - true in the nemo-platform image - so an explicit
    UV_CACHE_DIR must win without shelling out at all."""
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("UV_CACHE_DIR", "/home/ubuntu/.cache/uv")

    def _never(*a, **k):
        raise AssertionError("uv should not be invoked when UV_CACHE_DIR is set")

    monkeypatch.setattr(runtime.subprocess, "check_output", _never)

    assert runtime._uv_cache_dir() == "/home/ubuntu/.cache/uv"


def test_uv_cache_dir_returns_none_when_uv_unavailable(monkeypatch):
    """No env var and no usable `uv`: let Gym pick its own rather than crash."""
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)

    def _boom(*a, **k):
        raise FileNotFoundError("uv")

    monkeypatch.setattr(runtime.subprocess, "check_output", _boom)

    assert runtime._uv_cache_dir() is None


def test_environment_path_reads_the_bootstrap_env(tmp_path, monkeypatch):
    """The trusted actor injects the sandbox-side staging path as NMP_ENVIRONMENT_PATH."""
    monkeypatch.setenv(runtime.ENVIRONMENT_PATH_ENV_KEY, "/job/environment")
    assert runtime._environment_path() == "/job/environment"


@pytest.mark.parametrize("value", ["", "   "])
def test_environment_path_none_when_blank(value, monkeypatch):
    """Standalone runs and bundled-config_paths jobs leave it unset; both are valid."""
    monkeypatch.setenv(runtime.ENVIRONMENT_PATH_ENV_KEY, value)
    assert runtime._environment_path() is None


def test_environment_path_none_when_unset(monkeypatch):
    monkeypatch.delenv(runtime.ENVIRONMENT_PATH_ENV_KEY, raising=False)
    assert runtime._environment_path() is None


def test_bootstrap_registers_the_search_root_before_importing_gym(monkeypatch):
    """Ordering contract: the root must be on the search path before nemo_gym is imported.

    Gym's ``_augment_sys_path()`` runs at import time and folds ``NEMO_GYM_EXTRA_ROOTS``
    into ``sys.path``, so a root registered afterwards never reaches this process's import
    path. Forcing the Gym import to fail proves the registration already happened by then,
    without needing Gym installed.
    """
    import sys

    registered = []
    monkeypatch.setattr(
        runtime,
        "register_environment_search_root",
        lambda root: registered.append(root),
    )
    monkeypatch.setattr(
        runtime,
        "install_environment_wheels",
        lambda *a, **k: pytest.fail("wheels must not be installed before start()"),
    )
    monkeypatch.setenv(runtime.ENVIRONMENT_PATH_ENV_KEY, "/job/environment")
    # sys.modules[name] = None makes `import name` raise ImportError (CPython contract).
    monkeypatch.setitem(sys.modules, "nemo_gym.cli.env", None)

    with pytest.raises(ImportError):
        runtime.bootstrap_gym_host()

    assert registered == ["/job/environment"]
