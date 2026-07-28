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

"""Tests for the OpenSandbox episode backend.

NeMo-Gym's provider is stubbed: what is under test is the part the broker owns -- how the create
request is built from a sanitized spec, and that the egress policy is both sent and verified.
"""

from pathlib import Path
from typing import Any

import pytest
from nemo_gym.sandbox.broker import EpisodeResources
from nemo_gym.sandbox.providers import opensandbox as gym_opensandbox
from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)

from nemo_rl.environments.sandbox.backends.base import (
    EpisodeBackendError,
    PlatformMount,
    SanitizedEpisodeSpec,
    UnsupportedEpisodeOperationError,
)
from nemo_rl.environments.sandbox.egress import (
    DEFAULT_CLUSTER_DENY_TARGETS,
    EpisodeEgressPolicy,
    EpisodeEgressRule,
    build_egress_policy,
)


pytestmark = pytest.mark.nemo_gym


class FakeRawSandbox:
    """Stands in for the SDK sandbox object carried on a provider handle."""

    def __init__(self, policy: dict[str, Any] | None) -> None:
        self._policy = policy

    async def get_egress_policy(self) -> Any:
        if self._policy is None:
            raise AssertionError("egress policy was not requested")
        rules = [
            type("Rule", (), {"action": rule["action"], "target": rule["target"]})()
            for rule in self._policy["egress"]
        ]
        return type(
            "Policy",
            (),
            {"default_action": self._policy["defaultAction"], "egress": rules},
        )()


class FakeProvider:
    """Records what the backend asks the provider to do."""

    name = "opensandbox"
    instances: list["FakeProvider"] = []

    def __init__(
        self, *, connection=None, create=None, probe=None, operations=None
    ) -> None:
        self.connection = connection
        self.create_config = create
        self.probe = probe
        self.operations = operations
        self.created_specs: list[SandboxSpec] = []
        self.closed: list[str] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.files: dict[str, bytes] = {}
        self.applied_policy: dict[str, Any] | None = (
            create.get("network_policy") if create else None
        )
        self.aclosed = False
        FakeProvider.instances.append(self)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.created_specs.append(spec)
        return SandboxHandle(
            sandbox_id=f"sandbox-{len(self.created_specs)}",
            provider_name=self.name,
            raw=FakeRawSandbox(self.applied_policy),
        )

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def exec(
        self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None
    ):
        self.exec_calls.append(
            {
                "id": handle.sandbox_id,
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_s": timeout_s,
                "user": user,
            }
        )
        return SandboxExecResult(stdout="ok", stderr=None, return_code=0)

    async def upload_file(self, handle, source_path: Path, target_path: str) -> None:
        self.files[target_path] = source_path.read_bytes()

    async def download_file(self, handle, source_path: str, target_path: Path) -> None:
        target_path.write_bytes(self.files[source_path])

    async def close(self, handle: SandboxHandle) -> None:
        self.closed.append(handle.sandbox_id)

    async def aclose(self) -> None:
        self.aclosed = True


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeProvider.instances.clear()
    monkeypatch.setattr(gym_opensandbox, "OpenSandboxProvider", FakeProvider)


def make_backend(**overrides):
    from nemo_rl.environments.sandbox.backends.opensandbox import (
        OpenSandboxEpisodeBackend,
    )

    settings: dict[str, Any] = {"egress": build_egress_policy(default_action="allow")}
    settings.update(overrides)
    return OpenSandboxEpisodeBackend(**settings)


def make_spec(**overrides) -> SanitizedEpisodeSpec:
    settings: dict[str, Any] = {
        "job_id": "job-1",
        "image": "registry.example.com/swe/grader:1.0",
        "ttl_s": 300.0,
        "metadata": {"nemo-rl-job-id": "job-1"},
        "egress": build_egress_policy(default_action="allow"),
    }
    settings.update(overrides)
    return SanitizedEpisodeSpec(**settings)


# --------------------------------------------------------------------------------------------
# Egress
# --------------------------------------------------------------------------------------------


def test_egress_policy_reaches_the_provider_at_construction():
    make_backend()

    policy = FakeProvider.instances[0].create_config["network_policy"]
    assert policy["defaultAction"] == "allow"
    denied = {rule["target"] for rule in policy["egress"] if rule["action"] == "deny"}
    assert set(DEFAULT_CLUSTER_DENY_TARGETS) == denied


def test_broker_owned_policy_overrides_a_configured_one():
    # Whatever a deployment puts in the create block, the broker's policy is what reaches the SDK.
    make_backend(
        create={
            "network_policy": {"defaultAction": "allow", "egress": []},
            "retries": 5,
        }
    )

    provider = FakeProvider.instances[0]
    assert provider.create_config["retries"] == 5
    assert provider.create_config["network_policy"]["egress"] != []


@pytest.mark.asyncio
async def test_create_fails_and_reaps_when_the_policy_did_not_apply():
    backend = make_backend()
    # Server reports a policy that silently dropped the cluster denials.
    FakeProvider.instances[0].applied_policy = {"defaultAction": "allow", "egress": []}

    with pytest.raises(
        EpisodeBackendError, match="did not apply the requested egress policy"
    ):
        await backend.create(make_spec())

    # A sandbox whose isolation cannot be confirmed is not left running.
    assert FakeProvider.instances[0].closed == ["sandbox-1"]


@pytest.mark.asyncio
async def test_create_fails_when_default_action_was_flipped():
    backend = make_backend()
    provider = FakeProvider.instances[0]
    provider.applied_policy = {
        "defaultAction": "deny",
        "egress": [
            {"action": "deny", "target": target}
            for target in DEFAULT_CLUSTER_DENY_TARGETS
        ],
    }

    with pytest.raises(EpisodeBackendError):
        await backend.create(make_spec())


@pytest.mark.asyncio
async def test_create_succeeds_when_the_policy_matches():
    backend = make_backend()

    backend_id = await backend.create(make_spec())

    assert backend_id == "sandbox-1"
    assert FakeProvider.instances[0].closed == []


@pytest.mark.asyncio
async def test_verification_can_be_disabled():
    backend = make_backend(verify_egress=False)
    FakeProvider.instances[
        0
    ].applied_policy = None  # get_egress_policy would raise if called

    assert await backend.create(make_spec()) == "sandbox-1"


def test_policy_rendering_uses_the_sidecar_field_names():
    from nemo_rl.environments.sandbox.backends.opensandbox import to_opensandbox_policy

    rendered = to_opensandbox_policy(
        EpisodeEgressPolicy(
            default_action="allow",
            rules=(EpisodeEgressRule(target="10.0.0.0/8", action="deny"),),
        )
    )

    assert rendered == {
        "defaultAction": "allow",
        "egress": [{"action": "deny", "target": "10.0.0.0/8"}],
    }


# --------------------------------------------------------------------------------------------
# Spec translation
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_copies_only_known_fields():
    backend = make_backend()

    await backend.create(
        make_spec(
            workdir="/workspace",
            env={"CI": "1"},
            entrypoint=("/bin/sh", "-c", "sleep infinity"),
            resources=EpisodeResources(cpu=2, memory_mib=4096),
            ready_timeout_s=45.0,
        )
    )

    spec = FakeProvider.instances[0].created_specs[0]
    assert spec.image == "registry.example.com/swe/grader:1.0"
    assert spec.workdir == "/workspace"
    assert spec.env == {"CI": "1"}
    assert spec.entrypoint == ["/bin/sh", "-c", "sleep infinity"]
    assert spec.resources.cpu == 2
    assert spec.resources.memory_mib == 4096
    assert spec.ready_timeout_s == 45.0
    assert spec.metadata["nemo-rl-job-id"] == "job-1"
    # The field that carries volumes, platform, snapshots, and the privilege extension is never
    # populated, so none of them can reach the SDK regardless of what the caller sent.
    assert spec.provider_options == {}


@pytest.mark.asyncio
async def test_mounts_are_refused_rather_than_dropped():
    backend = make_backend()
    spec = make_spec(
        mounts=(PlatformMount(claim_name="job-scratch", mount_path="/job/work"),)
    )

    with pytest.raises(
        UnsupportedEpisodeOperationError, match="does not mount volumes"
    ):
        await backend.create(spec)


@pytest.mark.asyncio
async def test_staged_files_are_not_passed_through_the_spec():
    backend = make_backend()

    await backend.create(make_spec(files={"/work/seed.txt": b"hello"}))

    # The broker stages files after create; handing them to the provider spec too would upload
    # them twice.
    assert FakeProvider.instances[0].created_specs[0].files == {}


# --------------------------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_status_and_close_delegate_to_the_provider():
    backend = make_backend()
    backend_id = await backend.create(make_spec())
    provider = FakeProvider.instances[0]

    assert await backend.status(backend_id) is SandboxStatus.RUNNING

    result = await backend.exec(
        backend_id, "pytest -q", cwd="/workspace", timeout_s=30, user="root"
    )
    assert result.return_code == 0
    assert provider.exec_calls[0]["user"] == "root"
    assert provider.exec_calls[0]["timeout_s"] == 30

    await backend.close(backend_id)
    assert provider.closed == [backend_id]
    # Close is idempotent, and an unknown handle is not an error on teardown.
    await backend.close(backend_id)
    assert provider.closed == [backend_id]


@pytest.mark.asyncio
async def test_file_round_trip_through_the_public_provider_api():
    backend = make_backend()
    backend_id = await backend.create(make_spec())

    await backend.upload_file(backend_id, "/work/f.bin", b"\x00\x01binary")

    assert await backend.download_file(backend_id, "/work/f.bin") == b"\x00\x01binary"


@pytest.mark.asyncio
async def test_operations_on_unknown_episodes_are_rejected():
    backend = make_backend()

    with pytest.raises(EpisodeBackendError, match="unknown episode"):
        await backend.status("sandbox-does-not-exist")


@pytest.mark.asyncio
async def test_aclose_releases_the_provider():
    backend = make_backend()
    await backend.aclose()

    assert FakeProvider.instances[0].aclosed is True


def test_registry_builds_the_backend_with_the_configured_egress_profile():
    from nemo_rl.environments.sandbox.backends.registry import build_backend
    from nemo_rl.environments.sandbox.config import EpisodeBrokerConfig

    backend = build_backend(
        EpisodeBrokerConfig(
            job_id="job-1", backend="opensandbox", egress_allow_targets=("pypi.org",)
        )
    )

    assert backend.name == "opensandbox"
    policy = FakeProvider.instances[0].create_config["network_policy"]
    assert policy["defaultAction"] == "allow"
    # An explicit allow precedes the broad range denials it would otherwise fall inside.
    assert policy["egress"][0] == {"action": "allow", "target": "pypi.org"}
    assert {
        rule["target"] for rule in policy["egress"] if rule["action"] == "deny"
    } == set(DEFAULT_CLUSTER_DENY_TARGETS)
