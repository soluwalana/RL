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

import json
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
    actor._postprocess_cfg = {}

    monkeypatch.setattr(
        actor,
        "_post_rollouts",
        lambda examples: [({"meta": True}, {"reward": 0.5})],
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
        async for item in actor.run_rollouts(examples, tokenizer=object(), timer_prefix="t")
    ]
    assert streamed == [(7, {"post": 0.5}, streamed[0][2])]
    assert streamed[0][2] is not None
    assert "t/await_results" in streamed[0][2]


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
