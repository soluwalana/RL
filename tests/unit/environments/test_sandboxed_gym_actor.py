"""Unit tests for the SandboxedGymActor adapter over the nemo-sandboxed-gym package.

Broker, transport and host provisioning are the package's; what is exercised here is the
NeMo-RL side of the seam -- config translation, the ``_rowidx`` join, and the streaming
postprocess.
"""

from types import SimpleNamespace

from sandboxed_gym.ray.broker_actor import RayEpisodeBroker

import pytest
from sandboxed_gym.host.entrypoint import (
    DEFAULT_GYM_WRITABLE_SRC,
    gym_host_script_path,
    gym_host_runtime_path,
)


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


def _sandboxed_config(**overrides):
    from sandboxed_gym.host.models import NemoGymSandboxedConfig

    return NemoGymSandboxedConfig.model_validate(
        {**_actor_cfg()["sandboxed"], **overrides}
    )


class _FakeSession:
    def __init__(self, results):
        self._results = results
        self.posted = []
        self.shutdowns = 0
        self.cfg = SimpleNamespace(sandbox=SimpleNamespace(rollout_chunk_size=8))
        self.host = SimpleNamespace(rollout_url="http://host.svc/rollouts/run")

    def run_rollouts(self, examples):
        self.posted.append(examples)
        return self._results(examples) if callable(self._results) else self._results

    def shutdown(self):
        self.shutdowns += 1


def test_build_sandbox_global_config_injects_policy_and_drops_training_only_keys():
    from nemo_rl.environments.sandbox.nemo_gym_actor import build_sandbox_global_config

    cfg = _actor_cfg()
    cfg["initial_global_config_dict"] = {
        "config_paths": ["/job/environment/env.yaml"],
        "effort_levels": {"high": 1},
    }

    global_config = build_sandbox_global_config(cfg)

    assert global_config["policy_model_name"] == "meta-llama/Llama-3.1-8B"
    assert global_config["policy_base_url"] == cfg["base_urls"]
    assert global_config["policy_api_key"] == "dummy_key"
    assert global_config["config_paths"] == ["/job/environment/env.yaml"]
    # Gym's servers reject this NeMo-RL-only knob.
    assert "effort_levels" not in global_config


def test_build_sandbox_global_config_honors_configured_port_range():
    from nemo_rl.environments.sandbox.nemo_gym_actor import build_sandbox_global_config

    cfg = _actor_cfg()
    cfg["port_range_low"] = 41000
    cfg["port_range_high"] = 41100

    global_config = build_sandbox_global_config(cfg)

    assert global_config["port_range_low"] == 41000
    assert global_config["port_range_high"] == 41100


def test_build_serve_config_names_the_training_image_entrypoint():
    """The package defaults to the runtime image's CMD; the training image has none."""
    from nemo_rl.environments.sandbox.nemo_gym_actor import (
        NEMO_RL_IMAGE_GIT_ROOT,
        SANDBOXED_GYM_ACTOR_VENV,
        build_serve_config,
        nemo_rl_gym_host_entrypoint,
    )

    serve_cfg = build_serve_config(_actor_cfg(), _sandboxed_config())

    assert serve_cfg.sandbox.entrypoint == nemo_rl_gym_host_entrypoint()
    assert serve_cfg.sandbox.entrypoint == [
        "/bin/sh",
        gym_host_script_path(git_root=NEMO_RL_IMAGE_GIT_ROOT),
        SANDBOXED_GYM_ACTOR_VENV,
        NEMO_RL_IMAGE_GIT_ROOT,
        DEFAULT_GYM_WRITABLE_SRC,
        gym_host_runtime_path(git_root=NEMO_RL_IMAGE_GIT_ROOT),
    ]


def test_sandboxed_gym_actor_venv_tracks_the_actor_fqn():
    """Ray names the worker venv after the FQN, and the sandbox runs out of that path."""
    from nemo_rl.environments.sandbox.nemo_gym_actor import (
        SANDBOXED_GYM_ACTOR_FQN,
        SANDBOXED_GYM_ACTOR_VENV,
    )

    assert SANDBOXED_GYM_ACTOR_VENV.endswith(f"/{SANDBOXED_GYM_ACTOR_FQN}")


def test_build_serve_config_preserves_a_configured_entrypoint():
    from nemo_rl.environments.sandbox.nemo_gym_actor import build_serve_config

    sandbox = {**_sandbox_block(), "entrypoint": ["/bin/sh", "-c", "custom"]}
    serve_cfg = build_serve_config(_actor_cfg(), _sandboxed_config(sandbox=sandbox))

    assert serve_cfg.sandbox.entrypoint == ["/bin/sh", "-c", "custom"]


def test_build_serve_config_forwards_policy_base_urls():
    """They become the host's egress allowlist; a dropped one makes the sandbox unable to generate."""
    from nemo_rl.environments.sandbox.nemo_gym_actor import build_serve_config

    serve_cfg = build_serve_config(_actor_cfg(), _sandboxed_config())

    assert serve_cfg.policy_base_urls == ("http://vllm-0.svc.cluster.local:8000/v1",)
    assert serve_cfg.job_id == "job-1"


def test_tag_examples_stamps_rowidx_without_touching_the_task_index():
    """A prompt group shares one _ng_task_index, so the join key has to be a separate field."""
    from sandboxed_gym.runtime.gym_host_runtime import SG_EXAMPLE_ID

    from nemo_rl.environments.sandbox.nemo_gym_actor import _tag_examples

    group = [{"_rowidx": 3, "_ng_task_index": 7}, {"_rowidx": 4, "_ng_task_index": 7}]
    tagged = _tag_examples(group)

    assert [row[SG_EXAMPLE_ID] for row in tagged] == [3, 4]
    assert [row["_ng_task_index"] for row in tagged] == [7, 7]
    # The caller's own rows keep the identity NeMo-RL reads after the rollout.
    assert SG_EXAMPLE_ID not in group[0]


def test_tag_examples_rejects_a_row_without_rowidx():
    from nemo_rl.environments.sandbox.nemo_gym_actor import _tag_examples

    with pytest.raises(RuntimeError, match="missing _rowidx"):
        _tag_examples([{"agent_ref": {"name": "agent_a"}}])


def test_reward_summary_flags_all_zero_rewards():
    from nemo_rl.environments.sandbox.nemo_gym_actor import _reward_summary

    assert "nonzero=0/2" in _reward_summary([{"reward": 0.0}, {"reward": 0.0}])
    assert "nonzero=1/2" in _reward_summary([{"reward": 0.0}, {"reward": 1.0}])
    assert _reward_summary([{"no_reward": 1}]) == "reward=n/a"


def _rollout_actor(session):
    actor = _actor_class().__new__(_actor_class())
    actor.cfg = {"use_fastokens": False}
    actor._session = session
    actor._postprocess_cfg = {}
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


async def _stream(actor, examples):
    return [
        item
        async for item in actor.run_rollouts(
            examples, tokenizer=object(), timer_prefix="t"
        )
    ]


@pytest.mark.asyncio
async def test_run_rollouts_posts_then_postprocesses(monkeypatch):
    from sandboxed_gym.runtime.gym_host_runtime import SG_EXAMPLE_ID

    session = _FakeSession([{SG_EXAMPLE_ID: 7, "reward": 0.5}])
    actor = _rollout_actor(session)
    _patch_rollout_postprocess(monkeypatch, actor)

    examples = [{"_rowidx": 7, "agent_ref": {"name": "agent_a"}}]
    streamed = await _stream(actor, examples)

    assert streamed == [(7, {"post": 0.5}, streamed[0][2])]
    assert "t/await_results" in streamed[0][2]
    # The row went out tagged, which is what let the result come back joinable.
    assert session.posted[0][0][SG_EXAMPLE_ID] == 7


@pytest.mark.asyncio
async def test_run_rollouts_pairs_by_rowidx_not_arrival_order(monkeypatch):
    """A result must reach the prompt it was generated for, whatever order it lands in.

    Gym runs a request's rows concurrently and returns them through as_completed, and
    chunks are POSTed concurrently on top of that. Pairing on position would attach one
    prompt's tokens and reward to another prompt's row -- silently, since the row set is
    still a bijection and no guard downstream can notice.
    """
    from sandboxed_gym.runtime.gym_host_runtime import SG_EXAMPLE_ID

    # Exactly reversed: every result lands in a slot belonging to a different prompt.
    session = _FakeSession(
        lambda examples: [
            {SG_EXAMPLE_ID: row[SG_EXAMPLE_ID], "reward": float(row[SG_EXAMPLE_ID])}
            for row in reversed(examples)
        ]
    )
    actor = _rollout_actor(session)
    _patch_rollout_postprocess(monkeypatch, actor)

    examples = [{"_rowidx": i, "agent_ref": {"name": "agent_a"}} for i in range(4)]
    streamed = await _stream(actor, examples)

    assert [(rowidx, result["post"]) for rowidx, result, _ in streamed] == [
        (0, 0.0),
        (1, 1.0),
        (2, 2.0),
        (3, 3.0),
    ]


@pytest.mark.asyncio
async def test_run_rollouts_rejects_duplicate_task_indices(monkeypatch):
    """A host that drops or invents a row must fail, not shift every later pairing."""
    from sandboxed_gym.runtime.gym_host_runtime import SG_EXAMPLE_ID

    session = _FakeSession(
        [
            {SG_EXAMPLE_ID: 0, "reward": 0.0},
            {SG_EXAMPLE_ID: 0, "reward": 1.0},
            {SG_EXAMPLE_ID: 9, "reward": 2.0},
        ]
    )
    actor = _rollout_actor(session)
    _patch_rollout_postprocess(monkeypatch, actor)

    examples = [{"_rowidx": i, "agent_ref": {"name": "agent_a"}} for i in range(3)]

    with pytest.raises(RuntimeError, match="duplicate _sg_example_id"):
        await _stream(actor, examples)


@pytest.mark.asyncio
async def test_run_rollouts_rejects_results_that_do_not_cover_the_batch(monkeypatch):
    from sandboxed_gym.runtime.gym_host_runtime import SG_EXAMPLE_ID

    session = _FakeSession(
        [{SG_EXAMPLE_ID: 0, "reward": 0.0}, {SG_EXAMPLE_ID: 9, "reward": 1.0}]
    )
    actor = _rollout_actor(session)
    _patch_rollout_postprocess(monkeypatch, actor)

    examples = [{"_rowidx": i, "agent_ref": {"name": "agent_a"}} for i in range(2)]

    with pytest.raises(RuntimeError, match="do not cover the batch"):
        await _stream(actor, examples)


@pytest.mark.asyncio
async def test_run_rollouts_rejects_untagged_results(monkeypatch):
    """An untagged list is the shape that silently mispaired; it must not be accepted."""
    session = _FakeSession([{"reward": 0.0}])
    actor = _rollout_actor(session)
    _patch_rollout_postprocess(monkeypatch, actor)

    examples = [{"_rowidx": 0, "agent_ref": {"name": "agent_a"}}]

    with pytest.raises(RuntimeError, match="untagged result"):
        await _stream(actor, examples)


def _patch_spinup(monkeypatch, started, cleanups, session):
    monkeypatch.setattr(
        "nemo_rl.environments.sandbox.nemo_gym_actor.SandboxedGymOrchestrator",
        lambda: SimpleNamespace(
            start=lambda cfg, *, broker=None: (
                started.update(cfg=cfg, broker=broker),
                session,
            )[1]
        ),
    )
    monkeypatch.setattr(
        "nemo_rl.environments.sandbox.nemo_gym_actor.install_termination_cleanup",
        cleanups.append,
    )
    monkeypatch.setattr(
        "nemo_rl.environments.sandbox.nemo_gym_actor.ray.get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "node-1"),
    )


def test_spinup_starts_a_session_and_registers_cleanup(monkeypatch):
    session = _FakeSession([])
    started: dict = {}
    cleanups: list = []
    _patch_spinup(monkeypatch, started, cleanups, session)

    actor = _actor_class().__new__(_actor_class())
    actor.__init__(_actor_cfg())
    actor._spinup()

    assert actor._session is session
    assert started["cfg"].job_id == "job-1"
    assert started["cfg"].sandbox.image == "runtime:dev"
    assert cleanups == [actor.shutdown]


def test_spinup_hosts_the_broker_in_its_own_ray_actor(monkeypatch):
    """The broker serves every episode `exec`; sharing this process's GIL is what we avoid.

    Asserts a broker is supplied at all -- the orchestrator's default is in-process, so passing
    nothing is a silent revert to it rather than an error anyone would notice.
    """
    started: dict = {}
    _patch_spinup(monkeypatch, started, [], _FakeSession([]))

    actor = _actor_class().__new__(_actor_class())
    actor.__init__(_actor_cfg())
    actor._spinup()

    assert isinstance(started["broker"], RayEpisodeBroker)
    # Pinned to this actor's node, so the broker hop stays local.
    assert started["broker"]._node_id == "node-1"


def test_spinup_rejects_a_config_that_is_not_sandboxed():
    actor = _actor_class().__new__(_actor_class())
    cfg = _actor_cfg()
    cfg["sandboxed"] = {"sandboxed": False}
    actor.__init__(cfg)

    with pytest.raises(ValueError, match="sandboxed=true"):
        actor._spinup()


def test_build_serve_config_honors_the_venv_override(monkeypatch):
    """The sandbox runs whatever venv this names; a silently ignored override is unrunnable."""
    from nemo_rl.environments.sandbox.nemo_gym_actor import build_serve_config

    monkeypatch.setenv("SANDBOXED_GYM_VENV", "/opt/ray_venvs/custom")
    serve_cfg = build_serve_config(_actor_cfg(), _sandboxed_config())

    assert serve_cfg.sandbox.entrypoint[2] == "/opt/ray_venvs/custom"


def test_shutdown_closes_the_session_once():
    session = _FakeSession([])
    actor = _actor_class().__new__(_actor_class())
    actor._session = session

    actor.shutdown()
    actor.shutdown()

    assert session.shutdowns == 1
    assert actor._session is None


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
            "environment_offline": True,
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
    # The sandboxed block is rebuilt from a fixed set of keys rather than forwarded, so a field
    # the platform sets but the rebuild omits does not fail -- it silently takes the model
    # default. environment_offline went that way: the compiled config said true, the sandbox got
    # NMP_ENVIRONMENT_OFFLINE=0, and a wheels-v1 job reached for an index it was denied
    # (nvbug 6716627).
    assert created["cfg"]["sandboxed"]["environment_offline"] is True
    # Consumed here, so it must not travel on as a Gym config key either.
    assert "environment_offline" not in created["cfg"]["initial_global_config_dict"]


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
                "environment_offline": True,
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
    # NemoGymConfig declares environment_offline and _spinup reads it, but the constructor
    # never passed it -- so the colocated path defaulted to online too.
    assert created["cfg"]["environment_offline"] is True
    assert "environment_offline" not in created["cfg"]["initial_global_config_dict"]


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


def test_the_rebuild_forwards_every_field_the_package_declares():
    """`spinup_nemo_gym_actor` rebuilds the sandboxed block from a hand-written key list.

    A field the package declares and that list omits does not fail -- it silently takes the
    model default, and the job runs with a setting the platform did not ask for. That has
    happened twice: `environment_path`, then `environment_offline` (nvbug 6716627). This fails
    when the package grows a field, which is the moment the list needs updating.
    """
    import inspect

    from sandboxed_gym.host.models import NemoGymSandboxedConfig

    from nemo_rl.environments import nemo_gym

    source = inspect.getsource(nemo_gym.spinup_nemo_gym_actor)
    rebuild = source[source.index("NemoGymSandboxedConfig.model_validate") :]

    missing = [
        field
        for field in NemoGymSandboxedConfig.model_fields
        if f'"{field}"' not in rebuild
    ]
    assert missing == [], (
        f"the sandboxed block rebuild omits {missing}; those fields will silently take the "
        f"model default instead of what the platform set"
    )
