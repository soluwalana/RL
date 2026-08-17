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

"""In-sandbox Gym host HTTP runtime (``GET /health``, ``POST /rollouts/run``).

Started inside the OpenSandbox job image via ``RunHelper`` + ``RolloutCollectionHelper``.
Reads ``NMP_GYM_GLOBAL_CONFIG`` from bootstrap env (same JSON as colocated Gym, minus Ray GCS).

Run as a script by ``host/gym_host.sh``, which puts the NeMo-RL image root on ``PYTHONPATH``
-- so ``nemo_rl`` is importable here, but keep what is imported cheap and free of the
``nemo_rl.environments.sandbox`` package ``__init__`` (fastapi, broker HTTP app). The
environment-package helpers live in :mod:`nemo_rl.environments.gym_env_package`, which is
stdlib-only for exactly that reason and is shared with the colocated actor.
"""

import asyncio
import json
import os
import socket
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from nemo_rl.environments.gym_env_package import (
    UV_VENV_DIR_KEY,
    configure_environment_wheelhouse,
    install_environment_wheels,
    isolate_uv_from_ambient_project,
    register_environment_search_root,
)

GYM_GLOBAL_CONFIG_ENV_KEY = "NMP_GYM_GLOBAL_CONFIG"
ENVIRONMENT_PATH_ENV_KEY = "NMP_ENVIRONMENT_PATH"
UV_CACHE_DIR_KEY = "uv_cache_dir"
# Mirrors DEFAULT_GYM_PORT_RANGE_{LOW,HIGH} in nemo_rl.distributed.virtual_cluster.
DEFAULT_GYM_PORT_RANGE_LOW = 5000
DEFAULT_GYM_PORT_RANGE_HIGH = 5999

_DEFAULT_HTTP_PORT = 8080
_READY: bool = False
_RUN_HELPER: Any = None
_HEAD_SERVER_CONFIG: Any = None
_ROLLOUT_HELPER: Any = None
_EVENT_LOOP: asyncio.AbstractEventLoop | None = None
_EVENT_LOOP_LOCK = threading.Lock()
# Bounded so a deeply recursive failure cannot produce an oversized error response.
_TRACEBACK_FRAMES = 20
_MAX_TRACEBACK_CHARS = 8_000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _runtime_error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _load_global_config_dict() -> dict[str, Any]:
    raw = os.environ.get(GYM_GLOBAL_CONFIG_ENV_KEY, "").strip()
    if not raw:
        raise RuntimeError(f"{GYM_GLOBAL_CONFIG_ENV_KEY} is not set")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{GYM_GLOBAL_CONFIG_ENV_KEY} must be a JSON object")
    return parsed


def _free_port_in_range(low: int, high: int) -> int:
    for port in range(low, high + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
            except OSError:
                continue
            sock.listen(1)
            return port
    raise RuntimeError(f"no free port in range [{low}, {high}]")


def _allocate_head_server_port(global_config: dict[str, Any]) -> int:
    from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME

    low = int(global_config.get("port_range_low", DEFAULT_GYM_PORT_RANGE_LOW))
    high = int(global_config.get("port_range_high", DEFAULT_GYM_PORT_RANGE_HIGH))
    port = _free_port_in_range(low, high)
    global_config[HEAD_SERVER_KEY_NAME] = {"host": "0.0.0.0", "port": port}
    return port


def _create_rollout_helper() -> Any:
    from nemo_gym.rollout_collection import RolloutCollectionHelper

    return RolloutCollectionHelper()


def _uv_cache_dir() -> str | None:
    """Cache dir uv resolves to here, or None to let Gym pick its own.

    Deliberately not shared with ``nemo_rl.environments.nemo_gym.get_nemo_gym_uv_cache_dir``:
    that one pins ``uv``'s project discovery to the NeMo-RL checkout, which is the wrong
    answer inside the sandbox, where the working directory is the image's own tree.
    """
    if not os.environ.get("NRL_CONTAINER"):
        return None
    # Prefer the explicit env var. The container image sets it, and it sidesteps
    # `uv cache dir`, which exits non-zero whenever the working directory's
    # pyproject.toml pins a [tool.uv] required-version that disagrees with the uv on
    # PATH - true in the nemo-platform image, whose WORKDIR is the platform workspace.
    configured = os.environ.get("UV_CACHE_DIR")
    if configured:
        return configured
    try:
        resolved = subprocess.check_output(["uv", "cache", "dir"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return resolved or None


def _apply_uv_dirs(global_config: dict[str, Any]) -> None:
    """Point Gym at the image-baked uv cache / venv dirs.

    The colocated path does this in ``NemoGym._spinup``, but the sandboxed path returns
    before it, so the host applies it here from the sandbox's own environment. These must
    land in the CONFIG, not just the environment: Gym overwrites ``UV_CACHE_DIR`` from the
    config key, so without it Gym falls back to ``<Gym>/cache/uv`` in the read-only image
    tree and every per-app server dies with EACCES.
    """
    cache_dir = _uv_cache_dir()
    if cache_dir:
        global_config.setdefault(UV_CACHE_DIR_KEY, cache_dir)
    venv_dir = os.environ.get("NEMO_GYM_VENV_DIR")
    if venv_dir:
        global_config.setdefault(UV_VENV_DIR_KEY, venv_dir)


def _environment_path() -> str | None:
    """Staging path of the environment package inside the sandbox, or None.

    The trusted actor injects it as bootstrap env; the colocated path reads the same value
    off the Gym config instead. Both then hand it to the shared helpers in
    :mod:`nemo_rl.environments.gym_env_package`.
    """
    return os.environ.get(ENVIRONMENT_PATH_ENV_KEY, "").strip() or None


def bootstrap_gym_host() -> tuple[Any, Any, Any]:
    """Start Gym servers and return (RunHelper, head_server_config, RolloutCollectionHelper)."""
    # Registered before nemo_gym is imported below: Gym's _augment_sys_path() folds the
    # extra roots into sys.path at import time, and server directories are resolved during
    # RunHelper.start.
    environment_path = _environment_path()
    register_environment_search_root(environment_path)
    # Before start(): Gym composes the per-server install itself, so uv's environment
    # is the only way the wheelhouse reaches it, and the only way to keep the platform
    # workspace at WORKDIR from imposing its pins on the environment's venvs.
    isolate_uv_from_ambient_project()
    configure_environment_wheelhouse(environment_path)

    from nemo_gym.cli.env import RunHelper
    from nemo_gym.global_config import GlobalConfigDictParserConfig
    from nemo_gym.server_utils import BaseServerConfig
    from omegaconf import DictConfig

    global_config = _load_global_config_dict()
    _apply_uv_dirs(global_config)
    head_port = _allocate_head_server_port(global_config)

    run_helper = RunHelper()
    run_helper.start(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=DictConfig(global_config),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
        )
    )
    # After start(): the per-server venvs do not exist until it returns.
    install_environment_wheels(global_config, environment_path)
    head_server_config = BaseServerConfig(host="127.0.0.1", port=head_port)
    rollout_helper = _create_rollout_helper()
    return run_helper, head_server_config, rollout_helper


async def _collect_rollout_results(
    examples: list[dict],
    head_server_config: Any,
    rollout_helper: Any,
) -> list[list]:
    """Run ``examples`` and return ``[[_rowidx, result], ...]``.

    Tagged rather than ordered: Gym yields through ``asyncio.as_completed``, so results
    arrive in completion order and the caller pairs them by tag.
    """
    results: list[list] = []
    for task in rollout_helper.run_examples(
        examples=examples, head_server_config=head_server_config
    ):
        row, nemo_gym_result = await task
        if "_rowidx" not in row:
            raise RuntimeError(
                "NeMo-Gym row is missing _rowidx; results cannot be paired with prompts"
            )
        results.append([row["_rowidx"], nemo_gym_result])
    return results


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    """Return the process-wide event loop that every rollout request runs on.

    One loop per process, not one per request: the shared HTTP client binds to the loop
    that created it, so a per-request loop would leave it pointing at a closed one.
    """
    global _EVENT_LOOP
    with _EVENT_LOOP_LOCK:
        if _EVENT_LOOP is None:
            _EVENT_LOOP = asyncio.new_event_loop()
            threading.Thread(
                target=_EVENT_LOOP.run_forever,
                name="gym-host-event-loop",
                daemon=True,
            ).start()
        return _EVENT_LOOP


def run_rollouts_sync(
    examples: list[dict],
    head_server_config: Any,
    rollout_helper: Any,
) -> list[dict]:
    # Handler threads hand work to the shared loop and block on the result, so several
    # concurrent /rollouts/run calls interleave on one loop rather than one per thread.
    return asyncio.run_coroutine_threadsafe(
        _collect_rollout_results(examples, head_server_config, rollout_helper),
        _ensure_event_loop(),
    ).result()


class Handler(BaseHTTPRequestHandler):
    max_request_bytes: int = 268_435_456
    max_response_bytes: int = 268_435_456

    def do_GET(self) -> None:
        if not self.path.startswith("/health"):
            self.send_response(404)
            self.end_headers()
            return
        if not _READY:
            body = json.dumps({"status": "starting"}).encode("utf-8")
            self.send_response(503)
        else:
            body = json.dumps({"status": "ready"}).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if not self.path.startswith("/rollouts/run"):
            self.send_response(404)
            self.end_headers()
            return
        if not _READY or _HEAD_SERVER_CONFIG is None or _ROLLOUT_HELPER is None:
            self._send_json(
                503, _runtime_error("bootstrap_failed", "Gym host not ready")
            )
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length > self.max_request_bytes:
            self._send_json(
                413,
                _runtime_error(
                    "payload_too_large",
                    f"request body {length} exceeds max {self.max_request_bytes}",
                ),
            )
            return

        raw = self.rfile.read(length)
        try:
            request = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json(
                400,
                _runtime_error("internal", "invalid JSON body"),
            )
            return

        examples = request.get("examples")
        if not isinstance(examples, list):
            self._send_json(
                400,
                _runtime_error("internal", "examples must be a list"),
            )
            return

        # The only signal this process emits during a rollout: log_message is silenced
        # and both Gym servers filter their own 200s.
        print(f"gym-host: rollouts/run <- {len(examples)} example(s)", flush=True)
        started = time.monotonic()
        try:
            results = run_rollouts_sync(examples, _HEAD_SERVER_CONFIG, _ROLLOUT_HELPER)
        except Exception as exc:
            # Returned to the caller: this process's stdout is not surfaced to the job.
            detail = traceback.format_exc(limit=_TRACEBACK_FRAMES)
            print(f"gym-host: rollouts/run failed: {detail}", flush=True)
            self._send_json(
                500,
                _runtime_error(
                    "internal",
                    f"{type(exc).__name__}: {exc}\n{detail[-_MAX_TRACEBACK_CHARS:]}",
                ),
            )
            return

        print(
            f"gym-host: rollouts/run -> {len(results)} result(s) in "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )
        envelope = {
            "results": results,
            "job_id": os.environ.get("NMP_JOB_ID", ""),
            "environment_path": os.environ.get("NMP_ENVIRONMENT_PATH", ""),
            "work_path": os.environ.get("NMP_WORK_PATH", ""),
        }
        body = json.dumps(envelope).encode("utf-8")
        if len(body) > self.max_response_bytes:
            self._send_json(
                413,
                _runtime_error(
                    "payload_too_large",
                    f"response body {len(body)} exceeds max {self.max_response_bytes}",
                ),
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    global _READY, _RUN_HELPER, _HEAD_SERVER_CONFIG, _ROLLOUT_HELPER

    Handler.max_request_bytes = _env_int(
        "NMP_MAX_REQUEST_BYTES", Handler.max_request_bytes
    )
    Handler.max_response_bytes = _env_int(
        "NMP_MAX_RESPONSE_BYTES", Handler.max_response_bytes
    )

    _ensure_event_loop()
    _RUN_HELPER, _HEAD_SERVER_CONFIG, _ROLLOUT_HELPER = bootstrap_gym_host()
    _READY = True

    port = _env_int("NMP_RUNTIME_HTTP_PORT", _DEFAULT_HTTP_PORT)
    # Threaded so chunked rollouts can overlap and /health stays answerable mid-batch.
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
