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

"""NeMo-RL adapter over the ``nemo-sandboxed-gym`` package.

The broker, the job-host orchestration and the rollout transport all live in
``sandboxed_gym``. What stays here is what that package deliberately does not know
about: NeMo-RL's Gym config dialect, the training knobs injected into the Gym global
config, the ``_rowidx`` row identity RL joins on, and the Gym-to-NeMo-RL result
postprocessing behind :class:`~nemo_rl.environments.interfaces.EnvironmentInterface`.
"""

import asyncio
import logging
import os
from collections import Counter
from collections.abc import AsyncGenerator, Mapping
from typing import Any, NotRequired

import ray
from sandboxed_gym.host.entrypoint import default_gym_host_entrypoint
from sandboxed_gym.host.models import (
    NemoGymSandboxedConfig,
    uv_env_passthrough,
)
from sandboxed_gym.orchestrator import (
    SandboxedGymOrchestrator,
    SandboxedGymSession,
    install_termination_cleanup,
)
from sandboxed_gym.ray.broker_actor import RayEpisodeBroker
from sandboxed_gym.runtime.gym_host_runtime import SG_EXAMPLE_ID
from sandboxed_gym.serve_config import SandboxedGymServeConfig
from transformers import PreTrainedTokenizerBase

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GYM_PORT_RANGE_HIGH,
    DEFAULT_GYM_PORT_RANGE_LOW,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import (
    NemoGym,
    NemoGymConfig,
    _has_nan_generation_logprobs,
)
from nemo_rl.utils.timer import Timer


LOGGER = logging.getLogger(__name__)
# Opted in explicitly: the root logger is configured without a level, so INFO records
# here would otherwise be dropped and a rollout would emit no progress at all.
LOGGER.setLevel(logging.INFO)

SANDBOXED_GYM_ACTOR_FQN = (
    "nemo_rl.environments.sandbox.nemo_gym_actor.SandboxedGymActor"
)

NEMO_RL_IMAGE_GIT_ROOT = "/opt/nemo-rl"
# Ray names a worker venv after the actor FQN, so renaming the actor without moving this
# path leaves the sandbox executing from a directory that does not exist.
SANDBOXED_GYM_ACTOR_VENV = f"/opt/ray_venvs/{SANDBOXED_GYM_ACTOR_FQN}"


class SandboxedGymActorConfig(NemoGymConfig):
    """``NemoGymConfig`` plus platform-injected job-sandbox placement fields.

    GRPO job definitions keep the same Gym config shape users already write
    (``config_paths``, agent/resources knobs, etc.). nemo-platform inserts the
    ``sandboxed`` block from the environment at compile time; NeMo-RL peels those
    keys off and forwards the remainder as ``initial_global_config_dict``.
    Sandboxing changes where the Gym tree runs, not the user-facing config dialect.
    """

    sandboxed: NotRequired[dict[str, Any]]


def build_sandbox_global_config(cfg: SandboxedGymActorConfig) -> dict[str, Any]:
    """Build the Gym global config the job sandbox should start with.

    Deliberately only the training-time injections ``NemoGym._spinup`` makes for the
    colocated tree. Sandbox-local infrastructure defaults are the package's to add.
    """
    global_config = dict(cfg.get("initial_global_config_dict") or {})
    # NeMo-RL-only training knob that the Gym servers reject.
    global_config.pop("effort_levels", None)

    global_config["policy_model_name"] = cfg["model_name"]
    global_config["policy_api_key"] = "dummy_key"
    global_config["policy_base_url"] = cfg["base_urls"]

    global_config["port_range_low"] = cfg.get(
        "port_range_low", DEFAULT_GYM_PORT_RANGE_LOW
    )
    global_config["port_range_high"] = cfg.get(
        "port_range_high", DEFAULT_GYM_PORT_RANGE_HIGH
    )
    return global_config


def nemo_rl_gym_host_entrypoint() -> list[str]:
    """Entrypoint that starts the Gym host inside the NeMo-RL training image.

    The package defaults ``venv`` to its own actor's Ray venv, which does not exist in this
    image. Resolving the script and runtime paths against this process's install is correct
    only because the actor and the sandbox run the same image.
    """
    return default_gym_host_entrypoint(
        venv=os.environ.get("SANDBOXED_GYM_VENV") or SANDBOXED_GYM_ACTOR_VENV,
        git_root=NEMO_RL_IMAGE_GIT_ROOT,
    )


def build_serve_config(
    cfg: SandboxedGymActorConfig, sandboxed: NemoGymSandboxedConfig
) -> SandboxedGymServeConfig:
    """Translate the NeMo-RL actor config into the package's serve config."""
    sandbox = sandboxed.sandbox
    assert sandbox is not None

    if not sandbox.entrypoint:
        # The package leaves this unset so a runtime image's own CMD starts the host; the
        # training image has no such CMD.
        sandbox = sandbox.model_copy(
            update={"entrypoint": nemo_rl_gym_host_entrypoint()}
        )

    return SandboxedGymServeConfig(
        job_id=sandboxed.job_id,
        host_provider=sandboxed.host_provider,
        environment_path=sandboxed.environment_path,
        environment_offline=sandboxed.environment_offline,
        sandbox=sandbox,
        episode_broker=dict(sandboxed.episode_broker),
        gym_global_config=build_sandbox_global_config(cfg),
        host_env=uv_env_passthrough(),
        policy_base_urls=tuple(str(url) for url in (cfg.get("base_urls") or []) if url),
    )


def _tag_examples(examples: list[dict]) -> list[dict]:
    """Copy each row's ``_rowidx`` onto the package's caller-owned join key.

    Not ``_ng_task_index``: that identifies a *prompt group* here, so all ``num_generations``
    rows of a group share one value while their ``_rowidx`` differ. Joining on it would see
    duplicates, and overwriting it would change what Gym groups reward metrics by.

    Copies rather than mutates, so the caller's rows keep the identity NeMo-RL reads later.
    """
    tagged = []
    for row in examples:
        if "_rowidx" not in row:
            raise RuntimeError(
                "NeMo-Gym row is missing _rowidx; results cannot be paired with prompts"
            )
        tagged.append({**row, SG_EXAMPLE_ID: row["_rowidx"]})
    return tagged


def _index_results_by_rowidx(results: list, examples: list[dict]) -> dict[Any, Any]:
    """Map each result's join key back to its ``_rowidx``, rejecting anything but an exact cover.

    The host runs examples concurrently and returns them tagged rather than ordered, so
    results are paired by tag. Validated rather than trusted: a mispairing would attribute
    one prompt's tokens and reward to another and raise nothing.
    """
    by_rowidx: dict[Any, Any] = {}
    for result in results:
        if not isinstance(result, Mapping) or result.get(SG_EXAMPLE_ID) is None:
            raise RuntimeError(
                f"rollout host returned an untagged result; expected {SG_EXAMPLE_ID} on "
                f"every result, got {type(result).__name__}"
            )
        rowidx = result[SG_EXAMPLE_ID]
        if rowidx in by_rowidx:
            raise RuntimeError(
                f"rollout host returned duplicate {SG_EXAMPLE_ID} {rowidx}"
            )
        by_rowidx[rowidx] = result

    expected = {row["_rowidx"] for row in examples}
    if by_rowidx.keys() != expected:
        missing = sorted(expected - by_rowidx.keys())
        unexpected = sorted(by_rowidx.keys() - expected)
        raise RuntimeError(
            f"rollout host result rows do not cover the batch; "
            f"missing={missing} unexpected={unexpected}"
        )
    return by_rowidx


def _reward_summary(results: list) -> str:
    """Summarize a batch's rewards, including how many were non-zero.

    The non-zero count distinguishes an all-zero batch, which a mean alone can hide.
    """
    rewards = []
    for result in results:
        if isinstance(result, Mapping) and "reward" in result:
            try:
                rewards.append(float(result["reward"]))
            except (TypeError, ValueError):
                continue
    if not rewards:
        return "reward=n/a"
    nonzero = sum(1 for reward in rewards if reward != 0.0)
    return (
        f"reward mean={sum(rewards) / len(rewards):.3f} "
        f"min={min(rewards):.3f} max={max(rewards):.3f} nonzero={nonzero}/{len(rewards)}"
    )


# Deliberately no ``max_restarts``: the host handle lives only in this process, so a
# restarted actor comes back unable to name the sandbox its predecessor created.
# ``_spinup`` then provisions a second one and the first survives to its ttl_s, doubling
# the pods a job holds. A crash should fail the job instead. Restart support needs
# label-based reconciliation of the JOB_ID_METADATA_KEY the host spec already stamps.
@ray.remote  # pragma: no cover
class SandboxedGymActor(EnvironmentInterface):
    """Trusted proxy that runs Gym rollouts inside an isolated job sandbox."""

    def __init__(self, cfg: SandboxedGymActorConfig) -> None:
        self.cfg = cfg
        self._session: SandboxedGymSession | None = None
        self._postprocess_cfg = {
            "invalid_tool_call_patterns": cfg.get("invalid_tool_call_patterns"),
            "thinking_tags": cfg.get("thinking_tags"),
            "require_routed_experts": cfg.get("require_routed_experts", False),
            "routed_experts_dtype": cfg.get("routed_experts_dtype", "int16"),
        }

    def _spinup(self) -> None:
        """Start the episode broker and provision the job Gym host."""
        sandboxed = self.cfg.get("sandboxed")
        if isinstance(sandboxed, Mapping):
            sandboxed = NemoGymSandboxedConfig.model_validate(sandboxed)
        if sandboxed is None or not sandboxed.sandboxed or sandboxed.sandbox is None:
            raise ValueError("SandboxedGymActor requires env.nemo_gym.sandboxed=true")

        serve_cfg = build_serve_config(self.cfg, sandboxed)
        self._session = SandboxedGymOrchestrator().start(
            serve_cfg,
            # Pinned to this actor's node so the hop stays local; the Gym host reaches it over
            # HTTP regardless, and is never given a Ray handle.
            broker=RayEpisodeBroker(
                serve_cfg.broker_config(),
                node_id=ray.get_runtime_context().get_node_id(),
            ),
        )
        # After start(), so a spinup that failed leaves nothing registered to destroy.
        install_termination_cleanup(self.shutdown)

    def _postprocess(
        self, nemo_gym_result: dict, tokenizer: PreTrainedTokenizerBase
    ) -> dict:
        # ``NemoGym`` is a Ray actor class; postprocess helpers live on the
        # underlying Python class.
        nemo_gym_cls = NemoGym.__ray_metadata__.modified_class
        helper = nemo_gym_cls.__new__(nemo_gym_cls)
        helper.cfg = self._postprocess_cfg
        return helper._postprocess_nemo_gym_to_nemo_rl_result(
            nemo_gym_result, tokenizer
        )

    async def run_rollouts(
        self,
        nemo_gym_examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        timer_prefix: str,
    ) -> AsyncGenerator[tuple[int, dict, dict | None], None]:
        """POST examples to the job host and stream postprocessed results."""
        if not nemo_gym_examples:
            raise ValueError("NeMo-Gym rollout batch must not be empty")
        if self._session is None:
            raise RuntimeError("SandboxedGymActor._spinup has not completed")

        from nemo_rl.utils.fastokens import maybe_patch_fastokens

        maybe_patch_fastokens(bool(self.cfg.get("use_fastokens")))

        timer = Timer()
        counts_left = Counter(row["agent_ref"]["name"] for row in nemo_gym_examples)

        # Before the POST, not after: a batch that stalls in the sandbox still names itself.
        LOGGER.info(
            "rollout batch: POST %d example(s) in chunks of %d -> %s",
            len(nemo_gym_examples),
            self._session.cfg.sandbox.rollout_chunk_size,
            self._session.host.rollout_url,
        )
        timer.start("_run_rollouts_total")
        with timer.time(label=f"{timer_prefix}/await_results"):
            # Off-thread: the session's transport is synchronous, and awaiting it inline
            # would block the event loop Ray runs this async actor on.
            results = await asyncio.to_thread(
                self._session.run_rollouts, _tag_examples(nemo_gym_examples)
            )
        LOGGER.info(
            "rollout batch: %d result(s) for %d example(s) in %.1fs | %s",
            len(results),
            len(nemo_gym_examples),
            timer.get_timing_metrics("sum").get(f"{timer_prefix}/await_results", 0.0),
            _reward_summary(results),
        )

        if len(results) != len(nemo_gym_examples):
            raise RuntimeError(
                f"rollout host returned {len(results)} results for "
                f"{len(nemo_gym_examples)} examples"
            )

        results_by_rowidx = _index_results_by_rowidx(results, nemo_gym_examples)

        for index, nemo_gym_row in enumerate(nemo_gym_examples):
            nemo_gym_result = results_by_rowidx[nemo_gym_row["_rowidx"]]

            with timer.time(label=f"{timer_prefix}/postprocess_results"):
                nemo_rl_result = self._postprocess(nemo_gym_result, tokenizer)
                if _has_nan_generation_logprobs(nemo_rl_result):
                    raise RuntimeError("Generation logprobs contain NaN")

            timing_metrics = None
            if index + 1 == len(nemo_gym_examples):
                timer.stop("_run_rollouts_total")
                timing_metrics = timer.get_timing_metrics("sum")
                total_time = timing_metrics.pop("_run_rollouts_total")
                timing_metrics[f"{timer_prefix}/postprocess_results_pct"] = (
                    100
                    * timing_metrics[f"{timer_prefix}/postprocess_results"]
                    / total_time
                )

            agent_name = nemo_gym_row["agent_ref"]["name"]
            counts_left[agent_name] -= 1
            if counts_left[agent_name] <= 0:
                counts_left.pop(agent_name)

            yield nemo_gym_row["_rowidx"], nemo_rl_result, timing_metrics

    def shutdown(self) -> None:
        """Destroy the job host, then stop the episode broker."""
        if self._session is not None:
            self._session.shutdown()
            self._session = None

    def step(self, message_log_batch, metadata):
        raise NotImplementedError

    def global_post_process_and_metrics(self, batch):
        raise NotImplementedError
