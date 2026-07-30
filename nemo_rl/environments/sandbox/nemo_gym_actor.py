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

"""Trusted Ray proxy that runs NeMo-Gym inside a job-level sandbox."""

import asyncio
import concurrent.futures
import json
import logging
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import AsyncGenerator, Coroutine, Mapping
from typing import Any, NotRequired, TypeVar
from urllib.parse import urlparse

import ray
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
from nemo_rl.environments.sandbox.broker_actor import start_episode_broker
from nemo_rl.environments.sandbox.config import EpisodeBrokerConfig
from nemo_rl.environments.sandbox.host.models import (
    GymHostEgressRule,
    GymHostSpec,
    GymHostVolumeMount,
    NemoGymSandboxedConfig,
    SandboxConfig,
    build_bootstrap_env,
)
from nemo_rl.environments.sandbox.gym_host_runtime import GYM_GLOBAL_CONFIG_ENV_KEY
from nemo_rl.environments.sandbox.host.provider import get_host_provider
from nemo_rl.utils.timer import Timer


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


def _run_coro_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` from sync Ray methods on an async actor.

    ``SandboxedGymActor.run_rollouts`` is async, so Ray installs a running event
    loop on the actor. Sync methods like ``_spinup`` / ``shutdown`` must not call
    ``asyncio.run`` on that thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


SANDBOXED_GYM_ACTOR_FQN = (
    "nemo_rl.environments.sandbox.nemo_gym_actor.SandboxedGymActor"
)

# Bootstrap reads the Gym global config from GYM_GLOBAL_CONFIG_ENV_KEY (defined in
# gym_host_runtime, which stays nemo_rl-import-free) instead of a shared filesystem:
# the runtime image has no access to the training pod's config.


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

    Starts from the user/job Gym dict (``initial_global_config_dict``), then
    applies the same training-time injections ``NemoGym._spinup`` makes for the
    colocated tree. Trust-boundary differences: the training Ray GCS address is
    never sent, and Gym servers bind to sandbox loopback instead of a routable
    node IP. The head server entry is left to the runtime bootstrap, which owns
    port selection inside the sandbox.
    """
    global_config = dict(cfg.get("initial_global_config_dict") or {})
    # NeMo-RL-only training knob that the Gym servers reject.
    global_config.pop("effort_levels", None)
    # The sandbox never joins the training Ray cluster, even if a user config
    # carries an address.
    global_config.pop("ray_head_node_address", None)

    global_config["policy_model_name"] = cfg["model_name"]
    global_config["policy_api_key"] = "dummy_key"
    global_config["policy_base_url"] = cfg["base_urls"]
    global_config.setdefault("default_host", "127.0.0.1")

    global_config["port_range_low"] = cfg.get(
        "port_range_low", DEFAULT_GYM_PORT_RANGE_LOW
    )
    global_config["port_range_high"] = cfg.get(
        "port_range_high", DEFAULT_GYM_PORT_RANGE_HIGH
    )

    global_config.setdefault("global_aiohttp_connector_limit_per_host", 16_384)
    global_config.setdefault("global_aiohttp_connector_limit", 65_536)
    return global_config


def collect_gym_host_egress_allows(
    *,
    configured: list[GymHostEgressRule],
    broker_host: str,
    broker_port: int,
    base_urls: list[str | None],
) -> tuple[GymHostEgressRule, ...]:
    """Collect and deduplicate trusted job-host whitelist endpoints."""
    rules = list(configured)
    rules.append(GymHostEgressRule(host=broker_host, port=broker_port))
    for base_url in base_urls:
        if not base_url:
            continue
        parsed = urlparse(str(base_url))
        if not parsed.hostname:
            continue
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        rules.append(GymHostEgressRule(host=parsed.hostname, port=port))
    deduped: dict[tuple[str, int], GymHostEgressRule] = {}
    for rule in rules:
        deduped.setdefault((rule.host, rule.port), rule)
    return tuple(deduped.values())


def _gym_host_spec_from_config(
    cfg: SandboxedGymActorConfig,
    sandboxed: NemoGymSandboxedConfig,
    broker_url: str,
    broker_token: str,
    broker_host: str,
    broker_port: int,
) -> GymHostSpec:
    sandbox = sandboxed.sandbox
    assert sandbox is not None
    job_id = sandboxed.job_id or "nemo-rl-job"

    dataset_path = None
    dataset_mount = None
    if sandbox.dataset_pvc_claim:
        dataset_path = sandbox.dataset_mount_path
        dataset_mount = GymHostVolumeMount(
            pvc_claim=sandbox.dataset_pvc_claim,
            sub_path=sandbox.dataset_sub_path,
            mount_path=sandbox.dataset_mount_path,
            read_only=True,
        )

    bootstrap_env = build_bootstrap_env(
        job_id,
        sandboxed.environment_path or sandbox.env_mount_path,
        sandbox.work_mount_path,
        broker_url,
        broker_token,
        sandbox.max_request_bytes,
        sandbox.max_response_bytes,
        dataset_path=dataset_path,
        extra={
            GYM_GLOBAL_CONFIG_ENV_KEY: json.dumps(
                build_sandbox_global_config(cfg), sort_keys=True
            )
        },
    )

    egress_allow = collect_gym_host_egress_allows(
        configured=sandbox.network_policy.egress_allow,
        broker_host=broker_host,
        broker_port=broker_port,
        base_urls=list(cfg.get("base_urls") or []),
    )

    return GymHostSpec(
        job_id=job_id,
        runtime_image=sandbox.image,
        environment_mount=GymHostVolumeMount(
            pvc_claim=sandbox.environment_pvc_claim,
            sub_path=sandbox.environment_sub_path,
            mount_path=sandbox.env_mount_path,
            read_only=True,
        ),
        dataset_mount=dataset_mount,
        workspace_mount=GymHostVolumeMount(
            pvc_claim=sandbox.workspace_pvc_claim,
            sub_path=sandbox.workspace_sub_path,
            mount_path=sandbox.work_mount_path,
            read_only=False,
        ),
        egress_allow=egress_allow,
        bootstrap_env=bootstrap_env,
        max_request_bytes=sandbox.max_request_bytes,
        max_response_bytes=sandbox.max_response_bytes,
        ttl_s=sandbox.ttl_s,
        ready_timeout_s=sandbox.ready_timeout_s,
        resources=sandbox.resources,
        runtime_http_port=sandbox.runtime_http_port,
        allow_internet=sandbox.allow_internet,
        public_dns_allow=sandbox.network_policy.public_dns_allow,
        resolver_addresses=sandbox.network_policy.resolver_addresses,
        entrypoint=tuple(sandbox.entrypoint) if sandbox.entrypoint else None,
    )


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class SandboxedGymActor(EnvironmentInterface):
    """Trusted proxy that runs Gym rollouts inside an isolated job sandbox."""

    def __init__(self, cfg: SandboxedGymActorConfig) -> None:
        self.cfg = cfg
        self._host_handle = None
        self._host_provider = None
        self._broker_actor = None
        self._broker_endpoint = None
        self._rollout_timeout_s = 30 * 60.0
        self._max_request_bytes = 268_435_456
        self._max_response_bytes = 268_435_456
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

        sandbox: SandboxConfig = sandboxed.sandbox
        self._rollout_timeout_s = float(sandbox.rollout_timeout_s)
        self._max_request_bytes = sandbox.max_request_bytes
        self._max_response_bytes = sandbox.max_response_bytes

        job_id = sandboxed.job_id or "nemo-rl-job"
        broker_cfg = EpisodeBrokerConfig.model_validate(
            {"job_id": job_id, **dict(sandboxed.episode_broker)}
        )
        self._broker_actor, self._broker_endpoint = start_episode_broker(
            broker_cfg,
            node_id=ray.get_runtime_context().get_node_id(),
        )

        broker_host = self._broker_endpoint.host
        # Prefer hostname for egress allowlists when the URL carries one.
        parsed = urlparse(self._broker_endpoint.url)
        if parsed.hostname and not parsed.hostname.replace(".", "").isdigit():
            broker_host = parsed.hostname

        host_spec = _gym_host_spec_from_config(
            self.cfg,
            sandboxed,
            self._broker_endpoint.url,
            self._broker_endpoint.token,
            broker_host,
            self._broker_endpoint.port,
        )

        self._host_provider = get_host_provider(
            sandboxed.host_provider,
            sandbox.host_provider_options,
        )
        self._host_handle = _run_coro_sync(self._host_provider.create_host(host_spec))
        try:
            _run_coro_sync(
                self._host_provider.wait_ready(
                    self._host_handle, sandbox.ready_timeout_s
                )
            )
        except Exception:
            _run_coro_sync(self._host_provider.destroy_host(self._host_handle))
            self._host_handle = None
            raise

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

    def _post_rollouts(self, examples: list[dict]) -> list:
        assert self._host_handle is not None
        body = json.dumps({"examples": examples}).encode("utf-8")
        if len(body) > self._max_request_bytes:
            raise ValueError(
                f"rollout request exceeds max_request_bytes "
                f"({len(body)} > {self._max_request_bytes})"
            )
        request = urllib.request.Request(
            self._host_handle.rollout_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **self._host_handle.headers,
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._rollout_timeout_s
            ) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"rollout POST failed with HTTP {exc.code}: {error_body}"
            ) from exc
        if len(payload) > self._max_response_bytes:
            raise ValueError(
                f"rollout response exceeds max_response_bytes "
                f"({len(payload)} > {self._max_response_bytes})"
            )
        decoded = json.loads(payload.decode("utf-8"))
        if isinstance(decoded, dict) and "error" in decoded:
            raise RuntimeError(f"rollout runtime error: {decoded['error']}")
        if isinstance(decoded, dict) and "results" in decoded:
            return list(decoded["results"])
        if isinstance(decoded, list):
            return decoded
        raise RuntimeError(f"unexpected rollout response shape: {type(decoded)}")

    async def run_rollouts(
        self,
        nemo_gym_examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        timer_prefix: str,
    ) -> AsyncGenerator[tuple[int, dict, dict | None], None]:
        """POST examples to the job host and stream postprocessed results."""
        if not nemo_gym_examples:
            raise ValueError("NeMo-Gym rollout batch must not be empty")
        if self._host_handle is None:
            raise RuntimeError("SandboxedGymActor._spinup has not completed")

        from nemo_rl.utils.fastokens import maybe_patch_fastokens

        maybe_patch_fastokens(bool(self.cfg.get("use_fastokens")))

        timer = Timer()
        counts_left = Counter(row["agent_ref"]["name"] for row in nemo_gym_examples)

        timer.start("_run_rollouts_total")
        with timer.time(label=f"{timer_prefix}/await_results"):
            results = await asyncio.to_thread(self._post_rollouts, nemo_gym_examples)

        if len(results) != len(nemo_gym_examples):
            raise RuntimeError(
                f"rollout host returned {len(results)} results for "
                f"{len(nemo_gym_examples)} examples"
            )

        for index, (nemo_gym_row, pair) in enumerate(
            zip(nemo_gym_examples, results, strict=True)
        ):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                _, nemo_gym_result = pair
            else:
                nemo_gym_result = pair

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
        if self._host_provider is not None and self._host_handle is not None:
            try:
                _run_coro_sync(self._host_provider.destroy_host(self._host_handle))
            except Exception:
                LOGGER.exception("Failed to destroy sandboxed Gym host")
            self._host_handle = None

        if self._broker_actor is not None:
            try:
                ray.get(self._broker_actor.shutdown.remote())
            except Exception:
                LOGGER.exception("Failed to shut down episode broker")
            self._broker_actor = None
            self._broker_endpoint = None

    def step(self, message_log_batch, metadata):
        raise NotImplementedError

    def global_post_process_and_metrics(self, batch):
        raise NotImplementedError
