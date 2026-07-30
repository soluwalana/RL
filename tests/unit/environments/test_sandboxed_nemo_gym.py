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

"""Live SandboxedGymActor e2e (Ray actor + OpenSandbox job host).

Sandboxed counterpart of ``test_nemo_gym.py::test_nemo_gym_sanity``. Gated behind
``LIVE_OPENSANDBOX=1`` plus the same OpenSandbox env vars as the host live test.

Trusted actor uses ``PY_EXECUTABLES.NEMO_GYM``
(``uv run --locked --extra nemo_gym --directory {git_root}``). The job host
image (``NMP_RL_TRAINING_IMAGE`` / ``NEMO_RL_BASE_IMAGE``) starts Gym the same
way from the image's NeMo-RL tree + ``3rdparty`` Gym; vLLM stays outside on
reachable ``base_urls``.

Host-provider plumbing tests may still use the slim stub runtime; actor sanity uses
``nmp-rl-training`` (or ``OPENSANDBOX_LIVE_RUNTIME_IMAGE``) with ``real_gym_entrypoint``
(``uv run --extra nemo_gym`` + injected ``gym_host_runtime``) — no stub tokens.
"""

import json
import os
import time
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
import ray
from transformers import AutoTokenizer

from nemo_rl.environments.sandbox.nemo_gym_actor import SandboxedGymActor
from sandboxed_gym_live_common import (
    DEFAULT_POLICY_BASE_URL,
    DEFAULT_POLICY_MODEL_NAME,
    READY_TIMEOUT_S,
    build_live_target,
    create_ephemeral_pvcs,
    live_actor_py_executable,
    live_opensandbox_enabled,
    port_forward_server,
    sandboxed_env_block,
)

pytestmark = [
    pytest.mark.nemo_gym,
    pytest.mark.skipif(
        not live_opensandbox_enabled(),
        reason="Set LIVE_OPENSANDBOX=1 to run sandboxed Gym actor live tests",
    ),
]


@pytest.fixture
def live_target():
    return build_live_target()


@pytest.fixture
def port_forward(live_target) -> Iterator[tuple[str, str]]:
    yield from port_forward_server(live_target)


@pytest.fixture
def ephemeral_pvcs(live_target) -> Iterator[tuple[str, str]]:
    yield from create_ephemeral_pvcs(live_target)


@pytest.fixture
def sandboxed_tokenizer():
    # Match the nmp-temp1 vLLM deployment (Qwen/Qwen3.5-2B).
    return AutoTokenizer.from_pretrained(
        os.environ.get("SANDBOXED_GYM_TOKENIZER", "Qwen/Qwen3.5-2B"),
        trust_remote_code=True,
    )


@pytest.fixture
def sandboxed_sanity_inputs():
    fpath = Path(__file__).parent / "nemo_gym_test_data/test_nemo_gym_sanity.json"
    with open(fpath) as f:
        data = json.load(f)
    examples = deepcopy(data["input"])
    for idx, example in enumerate(examples):
        example["_rowidx"] = idx
        example.setdefault(
            "agent_ref",
            {"name": "example_multi_step_simple_agent", "type": "responses_api_agents"},
        )
        # Force parallel_tool_calls to True to avoid vLLM dropping tool calls from responses in Qwen 3.5 models.
        example["responses_create_params"]["parallel_tool_calls"] = True
    return examples


@pytest.fixture
def sandboxed_gym_actor(port_forward, ephemeral_pvcs):
    domain, api_key = port_forward
    env_claim, work_claim = ephemeral_pvcs
    use_stub = os.environ.get("SANDBOXED_GYM_LIVE_USE_STUB", "0") == "1"
    sandboxed = sandboxed_env_block(
        domain, api_key, env_claim, work_claim, with_stub_entrypoint=use_stub
    )
    if use_stub:
        initial_global_config_dict = {
            "config_paths": sandboxed.pop("config_paths"),
        }
    else:
        initial_global_config_dict = sandboxed.pop("initial_global_config_dict")
    cfg = {
        "model_name": DEFAULT_POLICY_MODEL_NAME,
        "base_urls": [DEFAULT_POLICY_BASE_URL],
        "initial_global_config_dict": initial_global_config_dict,
        "sandboxed": sandboxed,
        "use_fastokens": False,
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    py_exec = live_actor_py_executable()
    env = SandboxedGymActor.options(
        runtime_env={"py_executable": py_exec},
    ).remote(cfg)
    ray.get(env._spinup.remote(), timeout=READY_TIMEOUT_S)
    try:
        yield env
    finally:
        try:
            ray.get(env.shutdown.remote(), timeout=120)
        except Exception:
            pass
        ray.kill(env)
        time.sleep(0.1)


@pytest.mark.nemo_gym
def test_sandboxed_nemo_gym_sanity(
    sandboxed_gym_actor,
    sandboxed_sanity_inputs,
    sandboxed_tokenizer,
):
    """Sandboxed mirror of ``test_nemo_gym_sanity`` (real Gym host, live OpenSandbox)."""
    actual = [None] * len(sandboxed_sanity_inputs)
    for result_ref in sandboxed_gym_actor.run_rollouts.options(
        num_returns="streaming"
    ).remote(sandboxed_sanity_inputs, sandboxed_tokenizer, "timing/sandboxed"):
        rowidx, result, _ = ray.get(result_ref)
        actual[rowidx] = result

    assert all(row is not None for row in actual)
    for row in actual:
        assert "message_log" in row
        assert "input_message_log" in row
        assert "full_result" in row
        # Real Gym assistant turn (not stub-* placeholder text).
        roles = [m["role"] for m in row["message_log"]]
        assert "assistant" in roles
        assistant = next(m for m in row["message_log"] if m["role"] == "assistant")
        assert len(assistant["token_ids"]) > 0
        assert len(assistant["generation_logprobs"]) == len(assistant["token_ids"])
        content = assistant.get("content") or ""
        assert not str(content).startswith("stub-")
