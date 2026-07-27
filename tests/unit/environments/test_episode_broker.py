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

"""Tests for the trusted episode provisioning broker.

Exercises the HTTP app directly rather than through Ray: placement is the actor's job, and every
control worth testing here lives in the app, the sanitizer, or the episode registry.
"""

import base64

import pytest
from fastapi.testclient import TestClient
from nemo_gym.sandbox.broker import BROKER_AUTH_HEADER, BROKER_PROTOCOL_VERSION
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxStatus

from nemo_rl.environments.sandbox.backends.base import (
    EpisodeBackendError,
    SanitizedEpisodeSpec,
    UnsupportedEpisodeOperationError,
)
from nemo_rl.environments.sandbox.backends.memory import InMemoryEpisodeBackend
from nemo_rl.environments.sandbox.backends.registry import build_backend
from nemo_rl.environments.sandbox.config import JOB_ID_METADATA_KEY, EpisodeBrokerConfig
from nemo_rl.environments.sandbox.http_app import build_broker_app, close_all_episodes


pytestmark = pytest.mark.nemo_gym

TOKEN = "test-broker-token"
APPROVED_IMAGE = "registry.example.com/swe/grader:1.0"
B64_HELLO = base64.b64encode(b"hello").decode()


class RecordingBackend(InMemoryEpisodeBackend):
    """In-memory backend that records specs and can be told to fail."""

    def __init__(
        self, *, fail_on: str | None = None, unsupported_user: bool = False
    ) -> None:
        super().__init__()
        self.specs: list[SanitizedEpisodeSpec] = []
        self.closed: list[str] = []
        self._fail_on = fail_on
        self._unsupported_user = unsupported_user

    async def create(self, spec: SanitizedEpisodeSpec) -> str:
        if self._fail_on == "create":
            raise EpisodeBackendError("backend exploded: cluster-internal-host:9999")
        self.specs.append(spec)
        return await super().create(spec)

    async def exec(
        self, backend_id, command, *, cwd=None, env=None, timeout_s=None, user=None
    ):
        if self._unsupported_user and user is not None:
            raise UnsupportedEpisodeOperationError(
                "this backend cannot exec as a specific user"
            )
        self.last_exec = {
            "command": command,
            "cwd": cwd,
            "env": env,
            "timeout_s": timeout_s,
            "user": user,
        }
        return SandboxExecResult(stdout="ok", stderr=None, return_code=0)

    async def upload_file(self, backend_id, path, content):
        if self._fail_on == "upload":
            raise EpisodeBackendError("upload failed")
        await super().upload_file(backend_id, path, content)

    async def close(self, backend_id: str) -> None:
        self.closed.append(backend_id)
        await super().close(backend_id)


def make_config(**overrides) -> EpisodeBrokerConfig:
    settings = {
        "job_id": "job-1",
        "backend": "memory",
        "allow_insecure_memory_backend": True,
        "approved_images": (APPROVED_IMAGE,),
    }
    settings.update(overrides)
    return EpisodeBrokerConfig(**settings)


@pytest.fixture
def backend() -> RecordingBackend:
    return RecordingBackend()


@pytest.fixture
def client(backend) -> TestClient:
    app = build_broker_app(backend=backend, config=make_config(), token=TOKEN)
    with TestClient(app) as test_client:
        test_client.headers.update({BROKER_AUTH_HEADER: TOKEN})
        yield test_client


def create_episode(client: TestClient, **overrides) -> str:
    payload = {"image": APPROVED_IMAGE}
    payload.update(overrides)
    response = client.post("/episodes", json=payload)
    assert response.status_code == 201, response.text
    return response.json()["episode_id"]


# --------------------------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------------------------


ROUTES = [
    ("get", "/health", {}),
    ("post", "/episodes", {"json": {"image": APPROVED_IMAGE}}),
    ("get", "/episodes/ep_x", {}),
    ("post", "/episodes/ep_x/exec", {"json": {"command": "ls"}}),
    (
        "put",
        "/episodes/ep_x/files",
        {"json": {"path": "/work/f.txt", "content_b64": B64_HELLO}},
    ),
    ("get", "/episodes/ep_x/files", {"params": {"path": "/work/f.txt"}}),
    ("delete", "/episodes/ep_x", {}),
]


@pytest.mark.parametrize("method, path, request_kwargs", ROUTES)
def test_every_route_requires_the_token(backend, method, path, request_kwargs):
    app = build_broker_app(backend=backend, config=make_config(), token=TOKEN)
    with TestClient(app) as unauthenticated:
        response = unauthenticated.request(method.upper(), path, **request_kwargs)

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_wrong_token_is_rejected(backend):
    app = build_broker_app(backend=backend, config=make_config(), token=TOKEN)
    with TestClient(app) as client:
        response = client.get("/health", headers={BROKER_AUTH_HEADER: "not-the-token"})

    assert response.status_code == 401


def test_health_reports_job_and_protocol_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "job_id": "job-1",
        "protocol_version": BROKER_PROTOCOL_VERSION,
    }


# --------------------------------------------------------------------------------------------
# Sanitization: the escalation levers must be unreachable
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        # A docker socket bind mount, the escalation the broker exists to prevent.
        (
            "volumes",
            [
                {
                    "name": "sock",
                    "host": {"path": "/var/run/docker.sock"},
                    "mountPath": "/sock",
                }
            ],
        ),
        # Another tenant's PVC.
        (
            "volumes",
            [
                {
                    "name": "steal",
                    "pvc": {"claimName": "neighbour-tenant-pvc"},
                    "mountPath": "/steal",
                }
            ],
        ),
        # CAP_SYS_ADMIN with unconfined seccomp and AppArmor.
        ("extensions", {"bootstrap.execd.isolation": "enable"}),
        ("platform", {"os": "linux", "arch": "amd64"}),
        ("host_network", True),
        ("privileged", True),
    ],
)
def test_escalation_fields_are_not_expressible(client, field, value):
    response = client.post("/episodes", json={"image": APPROVED_IMAGE, field: value})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_snapshot_id_cannot_bypass_the_image_allowlist(client):
    # Creating from a snapshot needs no image, so a snapshot reference would sidestep the
    # approved-image check entirely if it were forwardable.
    response = client.post(
        "/episodes", json={"image": APPROVED_IMAGE, "snapshot_id": "snap-123"}
    )

    assert response.status_code == 422


def test_provider_options_are_rejected_by_default(client):
    response = client.post(
        "/episodes",
        json={"image": APPROVED_IMAGE, "provider_options": {"snapshot_id": "s"}},
    )

    body = response.json()
    assert response.status_code == 400
    assert body["code"] == "field_not_allowed"
    # Naming the offending key is what lets a rejected environment be debugged.
    assert "snapshot_id" in body["error"]


def test_allowed_provider_option_keys_pass(backend):
    config = make_config(allowed_provider_option_keys=("skip_health_check",))
    app = build_broker_app(backend=backend, config=config, token=TOKEN)
    with TestClient(app) as client:
        client.headers.update({BROKER_AUTH_HEADER: TOKEN})
        response = client.post(
            "/episodes",
            json={
                "image": APPROVED_IMAGE,
                "provider_options": {"skip_health_check": True},
            },
        )

    assert response.status_code == 201
    # Permitted or not, the option never reaches the backend spec in this phase.
    assert not hasattr(backend.specs[0], "provider_options")


def test_unapproved_image_is_refused(client):
    response = client.post(
        "/episodes", json={"image": "docker.io/attacker/evil:latest"}
    )

    assert response.status_code == 403
    assert response.json()["code"] == "image_not_approved"


def test_image_allowlist_fails_closed_when_unconfigured(backend):
    app = build_broker_app(
        backend=backend, config=make_config(approved_images=()), token=TOKEN
    )
    with TestClient(app) as client:
        client.headers.update({BROKER_AUTH_HEADER: TOKEN})
        response = client.post("/episodes", json={"image": APPROVED_IMAGE})

    assert response.status_code == 403


def test_image_prefix_grant_does_not_leak_to_sibling_repositories(backend):
    config = make_config(
        approved_images=(), approved_image_prefixes=("registry.example.com/swe/",)
    )
    app = build_broker_app(backend=backend, config=config, token=TOKEN)
    with TestClient(app) as client:
        client.headers.update({BROKER_AUTH_HEADER: TOKEN})

        assert (
            client.post(
                "/episodes", json={"image": "registry.example.com/swe/x:1"}
            ).status_code
            == 201
        )
        # "registry.example.com/swe-attacker/..." must not match a grant on ".../swe/".
        refused = client.post(
            "/episodes", json={"image": "registry.example.com/swe-attacker/x:1"}
        )

    assert refused.status_code == 403


def test_image_prefixes_must_end_at_a_path_boundary():
    with pytest.raises(ValueError, match="must end with"):
        make_config(approved_image_prefixes=("registry.example.com/swe",))


def test_broker_owned_metadata_keys_are_refused(client):
    response = client.post(
        "/episodes",
        json={
            "image": APPROVED_IMAGE,
            "metadata": {JOB_ID_METADATA_KEY: "some-other-job"},
        },
    )

    assert response.status_code == 400
    assert response.json()["code"] == "field_not_allowed"


def test_job_id_is_stamped_into_backend_metadata(client, backend):
    create_episode(client, metadata={"suite": "swe"})

    spec = backend.specs[0]
    assert spec.metadata[JOB_ID_METADATA_KEY] == "job-1"
    assert spec.metadata["suite"] == "swe"
    assert spec.job_id == "job-1"


def test_mounts_and_egress_default_closed(client, backend):
    create_episode(client)

    spec = backend.specs[0]
    assert spec.mounts == ()
    # Sent explicitly so a backend cannot fall back to its own allow-all default.
    assert spec.egress.default_action == "deny"
    assert spec.egress.rules == ()


# --------------------------------------------------------------------------------------------
# Resource, lifetime, and concurrency limits
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resources",
    [
        {"cpu": 1024},
        {"memory_mib": 1024 * 1024},
        {"disk_gib": 10_000},
        {"gpu": 1},
    ],
)
def test_resource_requests_beyond_the_cap_are_refused(client, resources):
    response = client.post(
        "/episodes", json={"image": APPROVED_IMAGE, "resources": resources}
    )

    assert response.status_code == 400
    assert response.json()["code"] == "quota_exceeded"


def test_gpu_requests_are_refused_by_default(client, backend):
    # max_gpu defaults to 0: episode graders are CPU work unless a deployment says otherwise.
    assert (
        client.post(
            "/episodes", json={"image": APPROVED_IMAGE, "resources": {"gpu": 1}}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/episodes", json={"image": APPROVED_IMAGE, "resources": {"gpu": 0}}
        ).status_code
        == 201
    )


def test_ttl_is_defaulted_and_clamped(client, backend):
    create_episode(client)
    assert backend.specs[0].ttl_s == 300.0

    create_episode(client, ttl_s=10_000_000)
    assert backend.specs[1].ttl_s == 3600.0


def test_exec_timeout_is_defaulted_and_clamped(client, backend):
    episode_id = create_episode(client)

    client.post(f"/episodes/{episode_id}/exec", json={"command": "true"})
    assert backend.last_exec["timeout_s"] == 180.0

    client.post(
        f"/episodes/{episode_id}/exec",
        json={"command": "true", "timeout_s": 10_000_000},
    )
    assert backend.last_exec["timeout_s"] == 1800.0


def test_concurrent_episode_budget_is_enforced(backend):
    app = build_broker_app(
        backend=backend, config=make_config(max_concurrent_episodes=2), token=TOKEN
    )
    with TestClient(app) as client:
        client.headers.update({BROKER_AUTH_HEADER: TOKEN})
        first = create_episode(client)
        create_episode(client)
        refused = client.post("/episodes", json={"image": APPROVED_IMAGE})

        assert refused.status_code == 429
        assert refused.json()["code"] == "quota_exceeded"

        # Closing one frees its slot.
        assert client.delete(f"/episodes/{first}").status_code == 200
        assert (
            client.post("/episodes", json={"image": APPROVED_IMAGE}).status_code == 201
        )


def test_oversized_request_body_is_refused(backend):
    app = build_broker_app(
        backend=backend, config=make_config(max_request_bytes=1024), token=TOKEN
    )
    with TestClient(app) as client:
        client.headers.update({BROKER_AUTH_HEADER: TOKEN})
        response = client.post(
            "/episodes",
            json={
                "image": APPROVED_IMAGE,
                "files_b64": {"/work/big": base64.b64encode(b"x" * 4096).decode()},
            },
        )

    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"


# --------------------------------------------------------------------------------------------
# Episode ownership
# --------------------------------------------------------------------------------------------


def test_episode_handles_are_opaque(client, backend):
    episode_id = create_episode(client)

    assert episode_id.startswith("ep_")
    # The caller never learns a backend-native id, so a leaked handle is useless against a backend.
    assert episode_id not in backend._episodes


@pytest.mark.parametrize(
    "method, path, request_kwargs",
    [
        (method, path.replace("ep_x", "ep_unknown"), kwargs)
        for method, path, kwargs in ROUTES
        if "ep_x" in path
    ],
)
def test_unknown_episode_handles_are_not_found(client, method, path, request_kwargs):
    response = client.request(method.upper(), path, **request_kwargs)

    assert response.status_code == 404
    assert response.json()["code"] == "episode_not_found"


def test_a_second_brokers_handle_is_not_accepted(backend):
    other_backend = RecordingBackend()
    other_app = build_broker_app(
        backend=other_backend, config=make_config(job_id="job-2"), token="other-token"
    )
    app = build_broker_app(backend=backend, config=make_config(), token=TOKEN)

    with TestClient(other_app) as other_client, TestClient(app) as client:
        other_client.headers.update({BROKER_AUTH_HEADER: "other-token"})
        client.headers.update({BROKER_AUTH_HEADER: TOKEN})
        foreign_id = create_episode(other_client)

        response = client.get(f"/episodes/{foreign_id}")

    assert response.status_code == 404


# --------------------------------------------------------------------------------------------
# Episode operations
# --------------------------------------------------------------------------------------------


def test_status_round_trip(client):
    episode_id = create_episode(client)

    response = client.get(f"/episodes/{episode_id}")

    assert response.status_code == 200
    assert response.json() == {"status": SandboxStatus.RUNNING.value}


def test_exec_forwards_the_call_and_returns_the_result(client, backend):
    episode_id = create_episode(client)

    response = client.post(
        f"/episodes/{episode_id}/exec",
        json={
            "command": "pytest -q",
            "cwd": "/workspace",
            "env": {"CI": "1"},
            "user": "root",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "stdout": "ok",
        "stderr": None,
        "return_code": 0,
        "error_type": None,
    }
    assert backend.last_exec["command"] == "pytest -q"
    assert backend.last_exec["cwd"] == "/workspace"
    assert backend.last_exec["env"] == {"CI": "1"}
    assert backend.last_exec["user"] == "root"


def test_exec_defaults_to_no_explicit_user(client, backend):
    episode_id = create_episode(client)

    client.post(f"/episodes/{episode_id}/exec", json={"command": "ls"})

    assert backend.last_exec["user"] is None


def test_backend_that_cannot_honour_a_user_fails_fast():
    strict_backend = RecordingBackend(unsupported_user=True)
    app = build_broker_app(backend=strict_backend, config=make_config(), token=TOKEN)

    with TestClient(app) as client:
        client.headers.update({BROKER_AUTH_HEADER: TOKEN})
        episode_id = create_episode(client)
        response = client.post(
            f"/episodes/{episode_id}/exec", json={"command": "ls", "user": "root"}
        )

    assert response.status_code == 501
    assert response.json()["code"] == "unsupported_operation"
    # The environment is told what is unsupported rather than grading under different conditions.
    assert "cannot exec as a specific user" in response.json()["error"]


def test_file_upload_and_download_round_trip(client):
    episode_id = create_episode(client)

    upload = client.put(
        f"/episodes/{episode_id}/files",
        json={"path": "/work/f.txt", "content_b64": B64_HELLO},
    )
    assert upload.status_code == 204

    download = client.get(
        f"/episodes/{episode_id}/files", params={"path": "/work/f.txt"}
    )
    assert download.status_code == 200
    assert base64.b64decode(download.json()["content_b64"]) == b"hello"


def test_files_staged_at_create_are_written(client, backend):
    episode_id = create_episode(client, files_b64={"/work/seed.txt": B64_HELLO})

    download = client.get(
        f"/episodes/{episode_id}/files", params={"path": "/work/seed.txt"}
    )

    assert base64.b64decode(download.json()["content_b64"]) == b"hello"


@pytest.mark.parametrize("bad_path", ["relative", "/work/../../etc/shadow"])
def test_traversal_paths_are_refused(client, bad_path):
    episode_id = create_episode(client)

    upload = client.put(
        f"/episodes/{episode_id}/files",
        json={"path": bad_path, "content_b64": B64_HELLO},
    )
    download = client.get(f"/episodes/{episode_id}/files", params={"path": bad_path})

    assert upload.status_code == 422
    assert download.status_code == 400


def test_create_failure_releases_the_reserved_slot():
    failing_backend = RecordingBackend(fail_on="create")
    app = build_broker_app(
        backend=failing_backend,
        config=make_config(max_concurrent_episodes=1),
        token=TOKEN,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({BROKER_AUTH_HEADER: TOKEN})
        failed = client.post("/episodes", json={"image": APPROVED_IMAGE})

        assert failed.status_code == 502
        # A failed create must not permanently consume the job's episode budget.
        failing_backend._fail_on = None
        assert (
            client.post("/episodes", json={"image": APPROVED_IMAGE}).status_code == 201
        )


def test_backend_failures_do_not_leak_internals(client):
    failing_backend = RecordingBackend(fail_on="create")
    app = build_broker_app(backend=failing_backend, config=make_config(), token=TOKEN)

    with TestClient(app, raise_server_exceptions=False) as failing_client:
        failing_client.headers.update({BROKER_AUTH_HEADER: TOKEN})
        response = failing_client.post("/episodes", json={"image": APPROVED_IMAGE})

    assert response.status_code == 502
    assert response.json()["code"] == "backend_error"
    # The backend's own message named an internal host; only a generic message crosses the boundary.
    assert "cluster-internal-host" not in response.text


def test_file_staging_failure_tears_the_episode_down():
    failing_backend = RecordingBackend(fail_on="upload")
    app = build_broker_app(backend=failing_backend, config=make_config(), token=TOKEN)

    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({BROKER_AUTH_HEADER: TOKEN})
        response = client.post(
            "/episodes",
            json={"image": APPROVED_IMAGE, "files_b64": {"/w/f": B64_HELLO}},
        )

    assert response.status_code == 502
    # A half-built episode is not left running.
    assert failing_backend.closed


# --------------------------------------------------------------------------------------------
# Teardown and backend selection
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_all_episodes_reaps_everything(backend):
    app = build_broker_app(backend=backend, config=make_config(), token=TOKEN)
    with TestClient(app) as client:
        client.headers.update({BROKER_AUTH_HEADER: TOKEN})
        create_episode(client)
        create_episode(client)

    await close_all_episodes(app)

    assert len(backend.closed) == 2
    assert await app.state.episodes.size() == 0


def test_memory_backend_requires_an_explicit_insecure_opt_in():
    with pytest.raises(ValueError, match="allow_insecure_memory_backend"):
        build_backend(EpisodeBrokerConfig(job_id="job-1", backend="memory"))

    backend = build_backend(
        EpisodeBrokerConfig(
            job_id="job-1", backend="memory", allow_insecure_memory_backend=True
        )
    )
    assert backend.name == "memory"


def test_opensandbox_backend_is_not_wired_yet():
    with pytest.raises(NotImplementedError, match="OpenSandbox episode backend"):
        build_backend(EpisodeBrokerConfig(job_id="job-1", backend="opensandbox"))
