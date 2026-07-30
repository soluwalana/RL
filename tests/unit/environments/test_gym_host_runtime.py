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

import json
import threading
from http.server import HTTPServer
from unittest.mock import MagicMock

import pytest

from nemo_rl.environments.sandbox import gym_host_runtime as runtime


class _FakeRolloutHelper:
    def run_examples(self, examples, head_server_config=None):
        async def _one(row):
            return row, {"response": {"output": []}, "reward": 0.0}

        return [_one(row) for row in examples]


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

    payload = json.dumps({"examples": [{"agent_ref": {"name": "a"}, "id": 1}]}).encode()
    req = urllib.request.Request(
        f"{ready_server}/rollouts/run",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
    assert len(body["results"]) == 1
    assert body["results"][0]["reward"] == 0.0


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
        [{"agent_ref": {"name": "x"}}],
        MagicMock(),
        helper,
    )
    assert len(results) == 1
