"""Boot the real Gym host from the training image, and run rollouts through it, without a GPU.

Complements ``test_sandboxed_gym_docker.py``, which uses a stub host: this one runs the
*packaged* ``gym_host.sh`` and ``gym_host_runtime`` out of the installed ``sandboxed-gym``
wheel, so it covers what the stub cannot -- that the entrypoint the adapter builds resolves
inside the image, and that the host's own row-identity handling is what tags the results.

Generation is served by a local stub standing in for vLLM (see ``sandboxed_gym_stub_model``).
No GPU is involved: in a sandboxed rollout the policy is reached over HTTP, and only training
needs local devices.

Requires an image built from this branch::

    docker buildx build --build-context nemo-rl=<RL checkout> \\
        -f docker/rl/Dockerfile.nmp-rl-base --tag nmp-rl-training:aalgo-583 --load .

Point ``SANDBOXED_GYM_DOCKER_IMAGE`` elsewhere to use a different tag.
"""

import os
import pathlib
import shutil
import socket
import subprocess

import pytest

from nemo_rl.environments.sandbox.nemo_gym_actor import (
    NEMO_RL_IMAGE_GIT_ROOT,
    SANDBOXED_GYM_ACTOR_VENV,
    _index_results_by_rowidx,
    _tag_examples,
    build_serve_config,
)

from sandboxed_gym_live_common import colocated_parity_global_config_dict
from sandboxed_gym_stub_model import serve as serve_stub_model

IMAGE = os.environ.get("SANDBOXED_GYM_DOCKER_IMAGE", "nmp-rl-training:aalgo-583")
# The Gym host runs in a container; this is how it reaches a server on the Docker host.
HOST_GATEWAY = os.environ.get("SANDBOXED_GYM_HOST_GATEWAY", "host.docker.internal")


def _image_present() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    not _image_present(), reason=f"needs a local {IMAGE} image"
)


@pytest.fixture
def stub_model():
    """A vLLM-shaped policy on a free port, reachable from the host container."""
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", 0))
        port = probe.getsockname()[1]
    server = serve_stub_model(port)
    try:
        yield f"http://{HOST_GATEWAY}:{port}/v1"
    finally:
        server.shutdown()


def _real_host_entrypoint() -> list[str]:
    """The adapter's entrypoint, with argv[0] the docker provider will not discard.

    ``DockerGymHostProvider`` drops an entrypoint whose first token is ``/bin/sh``, assuming the
    image starts the runtime from its own CMD. This image has no CMD -- it is the NeMo-RL training
    image, and the launcher script is exactly what has to run -- so the same shell is named in a
    way that survives that filter.
    """
    from sandboxed_gym.host.entrypoint import (
        gym_host_runtime_path,
        gym_host_script_path,
        gym_writable_src_dir,
    )

    return [
        "sh",
        gym_host_script_path(git_root=NEMO_RL_IMAGE_GIT_ROOT),
        SANDBOXED_GYM_ACTOR_VENV,
        NEMO_RL_IMAGE_GIT_ROOT,
        gym_writable_src_dir(),
        gym_host_runtime_path(git_root=NEMO_RL_IMAGE_GIT_ROOT),
    ]


def _serve_config(tmp_path, policy_base_url: str):
    from sandboxed_gym.host.models import NemoGymSandboxedConfig

    sandboxed = NemoGymSandboxedConfig.model_validate(
        {
            "sandboxed": True,
            "host_provider": "docker",
            "job_id": "real-host-e2e",
            # Nested episode sandboxes need a cluster; this environment asks for none.
            "episode_broker": {
                "backend": "memory",
                "allow_insecure_memory_backend": True,
            },
            "sandbox": {
                "image": IMAGE,
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env",
                "workspace_pvc_claim": "work",
                # Gym starts a Ray cluster and several uvicorn servers before it answers.
                "ready_timeout_s": 900.0,
                "rollout_timeout_s": 900.0,
                "entrypoint": _real_host_entrypoint(),
                "host_provider_options": {"root_dir": str(tmp_path / "hostroot")},
            },
        }
    )
    actor_cfg = {
        "model_name": "stub-model",
        "base_urls": [policy_base_url],
        "initial_global_config_dict": colocated_parity_global_config_dict(),
    }
    serve = build_serve_config(actor_cfg, sandboxed)
    # The image runs as uid 1000 while HOME is the build user's home at mode 0750, so uv's Python
    # discovery fails with EACCES the moment Gym starts its per-app servers and the host never
    # becomes ready. Redirect HOME somewhere the runtime uid can write.
    return serve.model_copy(
        update={"host_env": {**serve.host_env, "HOME": "/tmp/gym-home"}}
    )


@pytest.fixture
def session(tmp_path, stub_model):
    from sandboxed_gym.orchestrator import SandboxedGymOrchestrator

    started = SandboxedGymOrchestrator().start(_serve_config(tmp_path, stub_model))
    try:
        yield started
    finally:
        started.shutdown()


def test_the_packaged_entrypoint_brings_the_real_host_up(session):
    """Tier 2a: the adapter's entrypoint resolves and the packaged runtime answers.

    Reaching this point means ``gym_host.sh`` and ``gym_host_runtime.py`` were found inside the
    image at the paths the adapter names -- which come from the installed wheel, not a source
    tree -- and that Gym itself started. ``start`` polls ``/health`` until ready, so an assertion
    on the handle is enough; a host that never came up raises during setup.
    """
    assert session.host.rollout_url.endswith("/rollouts/run")


GYM_EXAMPLES = (
    "3rdparty/Gym-workspace/Gym/resources_servers/example_multi_step/data/example.jsonl"
)


def _examples() -> list[dict]:
    """Two rollouts of one prompt, built from the environment's own dataset rows.

    Hand-made rows do not work: ``example_multi_step`` declares ``id``,
    ``expected_synonyms``, ``expected_synonym_values``, ``minefield_label`` and
    ``minefield_label_value`` on its run/verify request, so a row missing them is rejected by
    the resources server's ``/verify`` long after the rollout looks like it is working.

    Both rollouts share one ``_ng_task_index`` with distinct ``_rowidx`` -- the shape a
    task-index join cannot separate, which is the whole point of the caller-owned key.
    """
    import json

    root = pathlib.Path(__file__).resolve().parents[3]
    row = json.loads((root / GYM_EXAMPLES).read_text().splitlines()[0])
    return [
        {
            **row,
            "_rowidx": index,
            "_ng_task_index": 0,
            "agent_ref": {
                "name": "example_multi_step_simple_agent",
                "type": "responses_api_agents",
            },
        }
        for index in range(2)
    ]


def test_real_gym_rollouts_pair_back_to_their_prompts(session):
    """Tier 2b: real Gym results, tagged by the real host, joined by the adapter."""
    examples = _examples()

    results = session.run_rollouts(_tag_examples(examples))

    by_rowidx = _index_results_by_rowidx(results, examples)
    assert sorted(by_rowidx) == [0, 1]
    for row in examples:
        # Tagged by the host runtime in the image, not by anything in this process.
        assert by_rowidx[row["_rowidx"]]["_ng_task_index"] == 0
