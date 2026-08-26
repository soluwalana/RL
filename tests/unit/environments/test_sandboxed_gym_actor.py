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

"""Unit tests for SandboxedGymActor rollout HTTP and factory selection."""

import io
import json
import threading
import time
import urllib.error
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.environments.sandbox.config import BrokerEndpoint
from nemo_rl.environments.sandbox.host.models import GymHostHandle


def _sandbox_block():
    return {
        "image": "runtime:dev",
        "network_policy": {"egress_allow": []},
        "environment_pvc_claim": "env-pvc",
        "workspace_pvc_claim": "work-pvc",
        "max_request_bytes": 256,
        "max_response_bytes": 256,
        "rollout_timeout_s": 5.0,
    }


def _actor_cfg():
    return {
        "model_name": "meta-llama/Llama-3.1-8B",
        "base_urls": ["http://vllm-0.svc.cluster.local:8000/v1"],
        "initial_global_config_dict": {
            "config_paths": ["/job/environment/env.yaml"],
        },
        "sandboxed": {
            "sandboxed": True,
            "sandbox": _sandbox_block(),
            "job_id": "job-1",
        },
    }


def _actor_class():
    from nemo_rl.environments.sandbox.nemo_gym_actor import SandboxedGymActor

    return SandboxedGymActor.__ray_metadata__.modified_class


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeHostProvider:
    def __init__(self):
        self.created = []
        self.ready = []
        self.destroyed = []

    async def create_host(self, spec):
        self.created.append(spec)
        return GymHostHandle(
            host_id="host-1",
            health_url="http://host.svc/health",
            rollout_url="http://host.svc/rollouts/run",
            provider=None,
        )

    async def wait_ready(self, handle, timeout_s):
        self.ready.append((handle.host_id, timeout_s))

    async def destroy_host(self, handle):
        self.destroyed.append(handle.host_id)


def test_post_rollouts_enforces_request_byte_limit():
    actor = _actor_class().__new__(_actor_class())
    actor._host_handle = GymHostHandle(
        host_id="host-1",
        health_url="http://host.svc/health",
        rollout_url="http://host.svc/rollouts/run",
    )
    actor._max_request_bytes = 32
    actor._max_response_bytes = 1024
    actor._rollout_timeout_s = 1.0

    with pytest.raises(ValueError, match="max_request_bytes"):
        actor._post_rollouts([{"payload": "x" * 64}])


def test_post_rollouts_enforces_response_byte_limit(monkeypatch):
    actor = _actor_class().__new__(_actor_class())
    actor._host_handle = GymHostHandle(
        host_id="host-1",
        health_url="http://host.svc/health",
        rollout_url="http://host.svc/rollouts/run",
    )
    actor._max_request_bytes = 1024
    actor._max_response_bytes = 8
    actor._rollout_timeout_s = 1.0

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _FakeResponse(b'{"results":[1,2,3,4,5,6,7,8,9]}'),
    )
    with pytest.raises(ValueError, match="max_response_bytes"):
        actor._post_rollouts([{"ok": True}])


def test_post_rollouts_accepts_results_envelope(monkeypatch):
    actor = _actor_class().__new__(_actor_class())
    actor._host_handle = GymHostHandle(
        host_id="host-1",
        health_url="http://host.svc/health",
        rollout_url="http://host.svc/rollouts/run",
    )
    actor._max_request_bytes = 1024
    actor._max_response_bytes = 1024
    actor._rollout_timeout_s = 1.0

    payload = json.dumps({"results": [{"reward": 1.0}]}).encode("utf-8")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _FakeResponse(payload),
    )
    assert actor._post_rollouts([{"ok": True}]) == [{"reward": 1.0}]


def _chunking_actor(chunk_size: int, max_in_flight: int, max_attempts: int = 1):
    actor = _actor_class().__new__(_actor_class())
    actor._host_handle = GymHostHandle(
        host_id="host-1",
        health_url="http://host.svc/health",
        rollout_url="http://host.svc/rollouts/run",
    )
    actor._max_request_bytes = 1024
    actor._max_response_bytes = 1024
    actor._rollout_timeout_s = 1.0
    actor._rollout_chunk_size = chunk_size
    actor._rollout_max_in_flight = max_in_flight
    actor._rollout_max_attempts = max_attempts
    actor._rollout_retry_backoff_s = 0.0
    return actor


def _http_error(code: int, body: bytes):
    return urllib.error.HTTPError(
        url="http://host.svc/rollouts/run",
        code=code,
        msg="err",
        hdrs=None,
        fp=io.BytesIO(body),
    )


def test_post_rollouts_classifies_proxy_timeout_as_terminal(monkeypatch):
    """OpenSandbox's flat ``{"code","message"}`` is the proxy giving up, not a blip.

    ``httpx.ReadTimeout`` stringifies to an empty message, which is exactly the 180s
    cutoff: retrying it re-runs generation for the same wall time and fails the same way.
    """
    from nemo_rl.environments.sandbox.nemo_gym_actor import RolloutTransportError

    actor = _chunking_actor(chunk_size=8, max_in_flight=8)
    ticks = iter((0.0, 180.0))
    monkeypatch.setattr(
        "nemo_rl.environments.sandbox.nemo_gym_actor.time.monotonic",
        lambda: next(ticks),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(
            _http_error(500, b'{"code":"GENERAL::UNKNOWN_ERROR","message":""}')
        ),
    )

    with pytest.raises(RolloutTransportError) as exc:
        actor._post_rollouts([{"ok": True}])

    assert exc.value.retryable is False
    assert exc.value.origin == "proxy"
    message = str(exc.value)
    assert "GENERAL::UNKNOWN_ERROR" in message
    assert "http://host.svc/rollouts/run" in message
    assert "rollout_chunk_size" in message, "must say which knob to turn"


def test_post_rollouts_classifies_unstructured_5xx_as_retryable(monkeypatch):
    """HTML or an empty body is a dropped hop, not a decision the proxy already made."""
    from nemo_rl.environments.sandbox.nemo_gym_actor import RolloutTransportError

    actor = _chunking_actor(chunk_size=8, max_in_flight=8)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(
            _http_error(502, b"<html>Bad Gateway</html>")
        ),
    )

    with pytest.raises(RolloutTransportError) as exc:
        actor._post_rollouts([{"ok": True}])

    assert exc.value.retryable is True
    assert exc.value.origin == "proxy"


def test_sandbox_reported_error_ignores_the_proxy_envelope():
    """The host's nested envelope and the proxy's flat one must not be interchangeable."""
    from nemo_rl.environments.sandbox.nemo_gym_actor import (
        _proxy_reported_error,
        _sandbox_reported_error,
    )

    host = '{"error": {"code": "internal", "message": "KeyError"}}'
    proxy = '{"code":"GENERAL::UNKNOWN_ERROR","message":""}'

    assert _sandbox_reported_error(host) == "internal: KeyError"
    assert _proxy_reported_error(host) is None
    assert _sandbox_reported_error(proxy) is None
    assert _proxy_reported_error(proxy) == "GENERAL::UNKNOWN_ERROR:"


def test_post_rollouts_classifies_environment_failure_as_terminal(monkeypatch):
    """The host's own {"error": ...} envelope means the environment raised. Retrying it
    just burns generation time, and the message must carry the environment's own text."""
    from nemo_rl.environments.sandbox.nemo_gym_actor import RolloutTransportError

    actor = _chunking_actor(chunk_size=8, max_in_flight=8)
    body = json.dumps(
        {"error": {"code": "internal", "message": "KeyError: 'expected_answer'"}}
    ).encode()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(500, body)),
    )

    with pytest.raises(RolloutTransportError) as exc:
        actor._post_rollouts([{"ok": True}])

    assert exc.value.retryable is False
    assert exc.value.origin == "sandbox"
    assert "KeyError: 'expected_answer'" in str(exc.value)


@pytest.mark.asyncio
async def test_post_rollouts_chunked_retries_transport_failures(monkeypatch):
    """One flaky chunk must not fail the step; 65 chunks make that likely, not rare."""
    actor = _chunking_actor(chunk_size=1, max_in_flight=4, max_attempts=3)
    from nemo_rl.environments.sandbox.nemo_gym_actor import RolloutTransportError

    attempts: dict[int, int] = {}

    def _fake_post(chunk):
        idx = chunk[0]["_rowidx"]
        attempts[idx] = attempts.get(idx, 0) + 1
        if idx == 1 and attempts[idx] < 3:
            raise RolloutTransportError("cut off", retryable=True, origin="proxy")
        return [(idx, {"reward": float(idx)})]

    monkeypatch.setattr(actor, "_post_rollouts", _fake_post)

    results = await actor._post_rollouts_chunked([{"_rowidx": i} for i in range(3)])

    assert sorted(rowidx for rowidx, _ in results) == [0, 1, 2]
    assert attempts == {0: 1, 1: 3, 2: 1}


@pytest.mark.asyncio
async def test_post_rollouts_chunked_does_not_retry_environment_failures(monkeypatch):
    """A deterministic environment error retried 3x per chunk wastes a whole batch."""
    actor = _chunking_actor(chunk_size=1, max_in_flight=4, max_attempts=3)
    from nemo_rl.environments.sandbox.nemo_gym_actor import RolloutTransportError

    calls = 0

    def _fake_post(chunk):
        nonlocal calls
        calls += 1
        raise RolloutTransportError("env raised", retryable=False, origin="sandbox")

    monkeypatch.setattr(actor, "_post_rollouts", _fake_post)

    with pytest.raises(RolloutTransportError) as exc:
        await actor._post_rollouts_chunked([{"_rowidx": 0}])

    assert calls == 1
    assert "failed after 1 attempt(s)" in str(exc.value)


@pytest.mark.asyncio
async def test_post_rollouts_chunked_does_not_retry_proxy_timeout(monkeypatch):
    """A proxy timeout retried 3x is the same 180s, three times, then the step still fails."""
    actor = _chunking_actor(chunk_size=1, max_in_flight=4, max_attempts=3)
    from nemo_rl.environments.sandbox.nemo_gym_actor import RolloutTransportError

    calls = 0

    def _fake_open(*a, **k):
        nonlocal calls
        calls += 1
        raise _http_error(500, b'{"code":"GENERAL::UNKNOWN_ERROR","message":""}')

    monkeypatch.setattr("urllib.request.urlopen", _fake_open)

    with pytest.raises(RolloutTransportError) as exc:
        await actor._post_rollouts_chunked([{"_rowidx": 0}])

    assert calls == 1
    assert exc.value.retryable is False
    assert "GENERAL::UNKNOWN_ERROR" in str(exc.value)


@pytest.mark.asyncio
async def test_post_rollouts_chunked_reports_how_many_chunks_failed(monkeypatch):
    """The batch-level error has to say the scale, not just surface one chunk's message."""
    actor = _chunking_actor(chunk_size=1, max_in_flight=4, max_attempts=1)
    from nemo_rl.environments.sandbox.nemo_gym_actor import RolloutTransportError

    def _fake_post(chunk):
        if chunk[0]["_rowidx"] % 2:
            raise RolloutTransportError("cut off", retryable=True, origin="proxy")
        return [(chunk[0]["_rowidx"], {"reward": 0.0})]

    monkeypatch.setattr(actor, "_post_rollouts", _fake_post)

    with pytest.raises(RolloutTransportError) as exc:
        await actor._post_rollouts_chunked([{"_rowidx": i} for i in range(4)])

    message = str(exc.value)
    assert "2 of 4 rollout chunk(s) failed" in message
    assert "batch of 4 example(s)" in message


@pytest.mark.asyncio
async def test_post_rollouts_chunked_splits_and_preserves_order(monkeypatch):
    """Chunking is a transport detail: results must come back in batch order."""
    actor = _chunking_actor(chunk_size=3, max_in_flight=8)
    seen: list[list[dict]] = []

    def _fake_post(chunk):
        seen.append(chunk)
        return [{"idx": row["idx"]} for row in chunk]

    monkeypatch.setattr(actor, "_post_rollouts", _fake_post)

    examples = [{"idx": i} for i in range(7)]
    results = await actor._post_rollouts_chunked(examples)

    # Chunks are dispatched concurrently, so completion order is not defined -- only the
    # partition and the order of the reassembled results are.
    assert sorted(len(chunk) for chunk in seen) == [1, 3, 3]
    assert sorted([row["idx"] for row in chunk] for chunk in seen) == [
        [0, 1, 2],
        [3, 4, 5],
        [6],
    ]
    assert [row["idx"] for row in results] == list(range(7))


@pytest.mark.asyncio
async def test_post_rollouts_chunked_bounds_in_flight(monkeypatch):
    """Chunks run concurrently, but never more than max_in_flight at once."""
    actor = _chunking_actor(chunk_size=1, max_in_flight=2)
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def _fake_post(chunk):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            time.sleep(0.02)
            return list(chunk)
        finally:
            with lock:
                in_flight -= 1

    monkeypatch.setattr(actor, "_post_rollouts", _fake_post)

    results = await actor._post_rollouts_chunked([{"idx": i} for i in range(6)])

    assert len(results) == 6
    assert peak > 1, "chunks should overlap rather than run strictly serially"
    assert peak <= 2


def test_reward_summary_flags_all_zero_rewards():
    """An all-zero batch is the common misconfiguration; it must be visible in logs."""
    from nemo_rl.environments.sandbox.nemo_gym_actor import _reward_summary

    assert "nonzero=0/3" in _reward_summary(
        [{"reward": 0.0}, {"reward": 0.0}, {"reward": 0.0}]
    )
    summary = _reward_summary([{"reward": 1.0}, {"reward": 0.0}])
    assert "nonzero=1/2" in summary
    assert "mean=0.500" in summary
    # The host may send (row, result) pairs rather than bare results.
    assert "nonzero=1/1" in _reward_summary([({"meta": True}, {"reward": 1.0})])
    # Absent rewards must not raise.
    assert _reward_summary([{"no_reward": True}]) == "reward=n/a"


def test_rowidx_span_renders_chunk_range():
    from nemo_rl.environments.sandbox.nemo_gym_actor import _rowidx_span

    assert _rowidx_span([{"_rowidx": 8}, {"_rowidx": 15}]) == "8-15"
    assert _rowidx_span([{}]) == "?"


def test_post_rollouts_error_reports_url_and_elapsed(monkeypatch):
    """A proxy failure must name the URL and how long it was held, not just the code."""
    actor = _chunking_actor(chunk_size=8, max_in_flight=8)

    def _raise(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="http://host.svc/rollouts/run",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b'{"code":"GENERAL::UNKNOWN_ERROR"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    with pytest.raises(RuntimeError) as exc:
        actor._post_rollouts([{"ok": True}])

    message = str(exc.value)
    assert "http://host.svc/rollouts/run" in message
    assert "1 example(s)" in message
    assert "HTTP 500" in message


@pytest.mark.asyncio
async def test_run_rollouts_posts_then_postprocesses(monkeypatch):
    actor = _actor_class().__new__(_actor_class())
    actor.cfg = {"use_fastokens": False}
    actor._host_handle = GymHostHandle(
        host_id="host-1",
        health_url="http://host.svc/health",
        rollout_url="http://host.svc/rollouts/run",
    )
    actor._max_request_bytes = 1024
    actor._max_response_bytes = 1024
    actor._rollout_timeout_s = 1.0
    actor._rollout_chunk_size = 8
    actor._rollout_max_in_flight = 8
    actor._postprocess_cfg = {}

    monkeypatch.setattr(
        actor,
        "_post_rollouts",
        lambda examples: [(7, {"reward": 0.5})],
    )
    monkeypatch.setattr(
        actor,
        "_postprocess",
        lambda result, tokenizer: {"post": result["reward"]},
    )
    monkeypatch.setattr(
        "nemo_rl.environments.sandbox.nemo_gym_actor._has_nan_generation_logprobs",
        lambda result: False,
    )
    monkeypatch.setattr(
        "nemo_rl.utils.fastokens.maybe_patch_fastokens",
        lambda enabled: None,
    )

    examples = [
        {
            "_rowidx": 7,
            "agent_ref": {"name": "agent_a"},
        }
    ]
    streamed = [
        item
        async for item in actor.run_rollouts(
            examples, tokenizer=object(), timer_prefix="t"
        )
    ]
    assert streamed == [(7, {"post": 0.5}, streamed[0][2])]
    assert streamed[0][2] is not None
    assert "t/await_results" in streamed[0][2]


def _pairing_actor(**overrides):
    actor = _actor_class().__new__(_actor_class())
    actor.cfg = {"use_fastokens": False}
    actor._host_handle = GymHostHandle(
        host_id="host-1",
        health_url="http://host.svc/health",
        rollout_url="http://host.svc/rollouts/run",
    )
    actor._max_request_bytes = 1024
    actor._max_response_bytes = 1024
    actor._rollout_timeout_s = 1.0
    actor._rollout_chunk_size = 8
    actor._rollout_max_in_flight = 8
    actor._postprocess_cfg = {}
    for key, value in overrides.items():
        setattr(actor, key, value)
    return actor


def _patch_rollout_postprocess(monkeypatch, actor):
    monkeypatch.setattr(
        actor, "_postprocess", lambda result, tokenizer: {"post": result["reward"]}
    )
    monkeypatch.setattr(
        "nemo_rl.environments.sandbox.nemo_gym_actor._has_nan_generation_logprobs",
        lambda result: False,
    )
    monkeypatch.setattr(
        "nemo_rl.utils.fastokens.maybe_patch_fastokens", lambda enabled: None
    )


@pytest.mark.asyncio
async def test_run_rollouts_pairs_by_rowidx_not_arrival_order(monkeypatch):
    """A result must reach the prompt it was generated for, whatever order it lands in.

    Gym runs a request's rows concurrently and returns them through as_completed, and
    chunks are POSTed concurrently on top of that. Pairing on position would attach one
    prompt's tokens and reward to another prompt's row -- silently, since the row set is
    still a bijection and no guard downstream can notice.
    """
    actor = _pairing_actor()
    _patch_rollout_postprocess(monkeypatch, actor)

    examples = [{"_rowidx": i, "agent_ref": {"name": "agent_a"}} for i in range(4)]
    # Exactly reversed: every result lands in a slot belonging to a different prompt.
    monkeypatch.setattr(
        actor,
        "_post_rollouts",
        lambda chunk: [
            (row["_rowidx"], {"reward": float(row["_rowidx"])})
            for row in reversed(chunk)
        ],
    )

    streamed = [
        item
        async for item in actor.run_rollouts(
            examples, tokenizer=object(), timer_prefix="t"
        )
    ]

    assert [(rowidx, result["post"]) for rowidx, result, _ in streamed] == [
        (0, 0.0),
        (1, 1.0),
        (2, 2.0),
        (3, 3.0),
    ]


@pytest.mark.asyncio
async def test_run_rollouts_rejects_results_that_do_not_cover_the_batch(monkeypatch):
    """A host that drops or invents a row must fail, not shift every later pairing."""
    actor = _pairing_actor()
    _patch_rollout_postprocess(monkeypatch, actor)

    examples = [{"_rowidx": i, "agent_ref": {"name": "agent_a"}} for i in range(3)]
    monkeypatch.setattr(
        actor,
        "_post_rollouts",
        lambda chunk: [
            (0, {"reward": 0.0}),
            (0, {"reward": 1.0}),
            (9, {"reward": 2.0}),
        ],
    )

    with pytest.raises(RuntimeError, match="duplicate _rowidx"):
        [
            item
            async for item in actor.run_rollouts(
                examples, tokenizer=object(), timer_prefix="t"
            )
        ]


@pytest.mark.asyncio
async def test_run_rollouts_rejects_untagged_results(monkeypatch):
    """An untagged list is the shape that silently mispaired; it must not be accepted."""
    actor = _pairing_actor()
    _patch_rollout_postprocess(monkeypatch, actor)

    examples = [{"_rowidx": 0, "agent_ref": {"name": "agent_a"}}]
    monkeypatch.setattr(actor, "_post_rollouts", lambda chunk: [{"reward": 0.0}])

    with pytest.raises(RuntimeError, match="untagged result"):
        [
            item
            async for item in actor.run_rollouts(
                examples, tokenizer=object(), timer_prefix="t"
            )
        ]


def test_spinup_starts_broker_and_creates_host(monkeypatch):
    fake_provider = _FakeHostProvider()
    endpoint = BrokerEndpoint(
        url="http://broker.svc.cluster.local:51234",
        host="127.0.0.1",
        port=51234,
        token="tok",
    )
    broker_actor = MagicMock()

    monkeypatch.setattr(
        "nemo_rl.environments.sandbox.nemo_gym_actor.start_episode_broker",
        lambda cfg, node_id=None: (broker_actor, endpoint),
    )
    monkeypatch.setattr(
        "nemo_rl.environments.sandbox.nemo_gym_actor.get_host_provider",
        lambda name, options=None: fake_provider,
    )
    monkeypatch.setattr(
        "ray.get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "node-1"),
    )

    actor = _actor_class().__new__(_actor_class())
    actor.__init__(_actor_cfg())
    actor._spinup()

    assert len(fake_provider.created) == 1
    assert fake_provider.ready == [("host-1", pytest.approx(15 * 60))]
    assert actor._host_handle.host_id == "host-1"
    assert actor._broker_actor is broker_actor
    hosts = [rule.host for rule in fake_provider.created[0].egress_allow]
    assert "broker.svc.cluster.local" in hosts
    assert "vllm-0.svc.cluster.local" in hosts


def test_shutdown_destroys_host_then_broker(monkeypatch):
    fake_provider = _FakeHostProvider()
    broker_actor = MagicMock()
    broker_actor.shutdown.remote.return_value = "ok"

    monkeypatch.setattr("ray.get", lambda ref: ref)

    actor = _actor_class().__new__(_actor_class())
    actor._host_provider = fake_provider
    actor._host_handle = GymHostHandle(
        host_id="host-1",
        health_url="http://host.svc/health",
        rollout_url="http://host.svc/rollouts/run",
    )
    actor._broker_actor = broker_actor
    actor._broker_endpoint = object()

    actor.shutdown()

    assert fake_provider.destroyed == ["host-1"]
    assert actor._host_handle is None
    broker_actor.shutdown.remote.assert_called_once()
    assert actor._broker_actor is None


def test_spinup_nemo_gym_actor_selects_sandboxed_path(monkeypatch):
    created = {}

    class _FakeRemote:
        def remote(self, cfg):
            created["cfg"] = cfg
            return SimpleNamespace(_spinup=SimpleNamespace(remote=lambda: "spin"))

    class _FakeActor:
        @staticmethod
        def options(**opts):
            created["opts"] = opts
            return _FakeRemote()

    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.get_actor_python_env",
        lambda fqn: "/bin/python",
    )
    monkeypatch.setattr("nemo_rl.environments.nemo_gym.ray.get", lambda ref: ref)
    monkeypatch.setattr(
        "nemo_rl.environments.sandbox.nemo_gym_actor.SandboxedGymActor",
        _FakeActor,
    )
    # Import happens inside the branch; patch the module attribute after import by
    # intercepting the local import path via sys.modules once loaded.
    import nemo_rl.environments.sandbox.nemo_gym_actor as actor_mod

    monkeypatch.setattr(actor_mod, "SandboxedGymActor", _FakeActor)

    from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor

    env_configs = {
        "nemo_gym": {
            "sandboxed": True,
            "host_provider": "opensandbox",
            "config_paths": ["resources_servers/math/configs/math.yaml"],
            "sandbox": _sandbox_block(),
            "job_id": "job-42",
            "invalid_tool_call_patterns": ["bad"],
            "thinking_tags": ["think"],
            "num_gpu_nodes": 0,
        }
    }

    handle = spinup_nemo_gym_actor(
        env_configs,
        base_urls=["http://vllm.svc:8000/v1"],
        model_name="model-x",
        enable_router_replay=False,
        routed_experts_dtype="int16",
        use_fastokens=False,
    )
    assert handle is not None
    assert created["cfg"]["model_name"] == "model-x"
    assert created["cfg"]["initial_global_config_dict"]["config_paths"] == [
        "resources_servers/math/configs/math.yaml"
    ]
    assert "sandboxed" not in created["cfg"]["initial_global_config_dict"]
    assert created["cfg"]["sandboxed"]["job_id"] == "job-42"
    assert created["cfg"]["invalid_tool_call_patterns"] == ["bad"]


def test_spinup_nemo_gym_actor_keeps_colocated_when_not_sandboxed(monkeypatch):
    created = {}

    class _FakeRemote:
        def remote(self, cfg):
            created["cfg"] = cfg
            return SimpleNamespace(_spinup=SimpleNamespace(remote=lambda: "spin"))

    class _FakeNemoGym:
        @staticmethod
        def options(**opts):
            created["opts"] = opts
            return _FakeRemote()

    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.get_actor_python_env",
        lambda fqn: "/bin/python",
    )
    monkeypatch.setattr("nemo_rl.environments.nemo_gym.ray.get", lambda ref: ref)
    monkeypatch.setattr("nemo_rl.environments.nemo_gym.NemoGym", _FakeNemoGym)
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.get_nemo_gym_uv_cache_dir",
        lambda: None,
    )
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.get_nemo_gym_venv_dir",
        lambda: None,
    )

    from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor

    handle = spinup_nemo_gym_actor(
        {
            "nemo_gym": {
                "sandboxed": False,
                "config_paths": ["resources_servers/math/configs/math.yaml"],
            }
        },
        base_urls=["http://vllm.svc:8000/v1"],
        model_name="model-x",
        enable_router_replay=True,
        routed_experts_dtype="int16",
        use_fastokens=False,
    )
    assert handle is not None
    assert created["cfg"]["require_routed_experts"] is True
    assert created["cfg"]["initial_global_config_dict"]["config_paths"] == [
        "resources_servers/math/configs/math.yaml"
    ]
    assert "sandboxed" not in created["cfg"]


def test_spinup_nemo_gym_actor_threads_environment_path_to_the_colocated_actor(
    monkeypatch,
):
    """Colocated runs need the environment package for the same two reasons mode B does.

    ``environment_path`` is popped off the Gym config (it is a NeMo-RL key, not a Gym one)
    and has to land on the actor config instead. Dropping it -- which is what happened
    before -- leaves a colocated run with no search root for a native-v1 tree and no
    wheel closure for a wheels-v1 one, and the platform driver deliberately installs
    neither, so nothing else would put them there.
    """
    created = {}

    class _FakeRemote:
        def remote(self, cfg):
            created["cfg"] = cfg
            return SimpleNamespace(_spinup=SimpleNamespace(remote=lambda: "spin"))

    class _FakeNemoGym:
        @staticmethod
        def options(**opts):
            return _FakeRemote()

    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.get_actor_python_env",
        lambda fqn: "/bin/python",
    )
    monkeypatch.setattr("nemo_rl.environments.nemo_gym.ray.get", lambda ref: ref)
    monkeypatch.setattr("nemo_rl.environments.nemo_gym.NemoGym", _FakeNemoGym)
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.get_nemo_gym_uv_cache_dir",
        lambda: None,
    )
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.get_nemo_gym_venv_dir",
        lambda: "/opt/gym_venvs",
    )

    from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor

    spinup_nemo_gym_actor(
        {
            "nemo_gym": {
                "sandboxed": False,
                "environment_path": "/job/storage/environment",
                "config_paths": ["/job/storage/environment/configs/agent.yaml"],
            }
        },
        base_urls=["http://vllm.svc:8000/v1"],
        model_name="model-x",
        enable_router_replay=False,
        routed_experts_dtype="int16",
        use_fastokens=False,
    )

    assert created["cfg"]["environment_path"] == "/job/storage/environment"
    # Popped, not forwarded: Gym's global config has no such key, and RunHelper walks
    # every non-reserved top-level key looking for server blocks.
    assert "environment_path" not in created["cfg"]["initial_global_config_dict"]
    # The venv root has to reach Gym's config too, or install_environment_wheels cannot
    # locate the per-server venvs and raises.
    assert (
        created["cfg"]["initial_global_config_dict"]["uv_venv_dir"] == "/opt/gym_venvs"
    )
