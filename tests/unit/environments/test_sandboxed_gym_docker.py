"""End-to-end rollout round trip against a real container, without a GPU or a cluster.

Covers the seam the wheel adoption changed: the NeMo-RL config dialect is translated into the
package's serve config, a host is provisioned, examples are POSTed, and results are joined back
to prompts. The `docker` host provider stands in for OpenSandbox -- same runtime image contract,
same bootstrap environment, same HTTP surface -- so everything except the isolation boundary is
the real code path. Isolation is not asserted here and cannot be: that provider records the
egress allowlist rather than enforcing it, and bind-mounts host directories.

The Gym host is the same stub the live OpenSandbox tests use, running on a plain Python image,
so no Gym tree, model or GPU is involved.
"""

import shutil
import subprocess

import pytest

from nemo_rl.environments.sandbox.nemo_gym_actor import (
    _index_results_by_rowidx,
    _tag_examples,
    build_serve_config,
)

from sandboxed_gym_live_common import stub_entrypoint


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="needs a working docker daemon"
)

STUB_IMAGE = "python:3.12-slim"
NUM_GENERATIONS = 2


def _examples() -> list[dict]:
    """Two prompts, two rollouts each: one `_ng_task_index` per pair, `_rowidx` unique per row.

    The shape that made a task-index join wrong. Joining on `_ng_task_index` here sees each value
    twice; joining on position survives only while the host answers in order.
    """
    rows = []
    for task_index in range(2):
        for _ in range(NUM_GENERATIONS):
            rows.append(
                {
                    "_rowidx": len(rows),
                    "_ng_task_index": task_index,
                    "agent_ref": {
                        "name": "example_multi_step_simple_agent",
                        "type": "responses_api_agents",
                    },
                    "responses_create_params": {
                        "input": [{"role": "user", "content": f"row {len(rows)}"}]
                    },
                }
            )
    return rows


def _serve_config(tmp_path):
    from sandboxed_gym.host.models import NemoGymSandboxedConfig

    sandboxed = NemoGymSandboxedConfig.model_validate(
        {
            "sandboxed": True,
            "host_provider": "docker",
            "job_id": "docker-e2e",
            # No cluster and no OpenSandbox credential here. The episode tier is unused by this
            # stub anyway -- only an environment asking for nested sandboxes ever reaches it.
            "episode_broker": {
                "backend": "memory",
                "allow_insecure_memory_backend": True,
            },
            "sandbox": {
                "image": STUB_IMAGE,
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env",
                "workspace_pvc_claim": "work",
                "ready_timeout_s": 120.0,
                "rollout_timeout_s": 120.0,
                "entrypoint": stub_entrypoint(),
                "host_provider_options": {"root_dir": str(tmp_path / "hostroot")},
            },
        }
    )
    actor_cfg = {
        "model_name": "stub-model",
        "base_urls": ["http://127.0.0.1:9/v1"],
        "initial_global_config_dict": {},
    }
    return build_serve_config(actor_cfg, sandboxed)


@pytest.fixture
def session(tmp_path):
    from sandboxed_gym.orchestrator import SandboxedGymOrchestrator

    started = SandboxedGymOrchestrator().start(_serve_config(tmp_path))
    try:
        yield started
    finally:
        started.shutdown()


def test_a_rollout_batch_round_trips_through_a_real_host(session):
    """The whole trusted path: provision, POST, and pair every result with its prompt."""
    examples = _examples()

    results = session.run_rollouts(_tag_examples(examples))

    assert len(results) == len(examples)
    by_rowidx = _index_results_by_rowidx(results, examples)
    assert sorted(by_rowidx) == [row["_rowidx"] for row in examples]
    # Each result reached the prompt it was generated for, not the one in its slot.
    for row in examples:
        echoed = by_rowidx[row["_rowidx"]]["responses_create_params"]["input"][0]
        assert echoed["content"] == f"row {row['_rowidx']}"


def test_the_host_leaves_the_task_index_alone(session):
    """`_ng_task_index` is Gym's; carrying our own id must not disturb it."""
    examples = _examples()

    results = session.run_rollouts(_tag_examples(examples))

    by_rowidx = _index_results_by_rowidx(results, examples)
    for row in examples:
        assert by_rowidx[row["_rowidx"]]["_ng_task_index"] == row["_ng_task_index"]
