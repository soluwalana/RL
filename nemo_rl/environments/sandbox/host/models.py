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

"""Models for provisioning the job-level Gym sandbox host.

Defines the create/spec and config shapes used by ``SandboxedGymHostProvider`` and
``SandboxedGymActor``. Episode egress policy construction lives in
:mod:`nemo_rl.environments.sandbox.egress`.
"""

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_rl.environments.sandbox.config import K8S_LABEL_VALUE_RE


DEFAULT_RUNTIME_HTTP_PORT = 8080
DEFAULT_MAX_ROLLOUT_BYTES = 268_435_456  # 256 MiB
DEFAULT_HOST_TTL_S = 14_400  # 4 hours
DEFAULT_HOST_READY_TIMEOUT_S = 15 * 60
DEFAULT_ROLLOUT_TIMEOUT_S = 30 * 60

FORBIDDEN_BOOTSTRAP_ENV_PREFIXES = ("OPENSANDBOX_",)
FORBIDDEN_BOOTSTRAP_ENV_KEYS = frozenset(
    {
        "OPENSANDBOX_API_KEY",
        "RAY_ADDRESS",
        "ray_head_node_address",
        "RAY_HEAD_NODE_ADDRESS",
    }
)

TProvider = TypeVar("TProvider")


class GymHostVolumeMount(BaseModel):
    """PVC mount into the job sandbox."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pvc_claim: str = Field(min_length=1)
    sub_path: str = ""
    mount_path: str = Field(min_length=1)
    read_only: bool = False

    @field_validator("mount_path")
    @classmethod
    def _absolute_mount_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError(f"mount_path must be absolute, got {value!r}")
        return value


class GymHostEgressRule(BaseModel):
    """Allowed egress destination (FQDN, Service DNS, or IP) plus port.

    ``host`` is the OpenSandbox egress sidecar target (hostname or IP). ``port``
    is recorded for platform NetworkPolicy emission; the sidecar match is
    address/hostname-only. Prefer Service DNS / FQDN for model endpoints; the
    episode broker may advertise a node IP when no Service DNS exists.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)

    @field_validator("host")
    @classmethod
    def _nonempty_host(cls, value: str) -> str:
        host = value.strip().rstrip(".")
        if not host:
            raise ValueError("egress host cannot be blank")
        return host


class GymHostSpec(BaseModel):
    """Create parameters for one job-level Gym host sandbox."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(min_length=1)
    runtime_image: str = Field(min_length=1)
    environment_mount: GymHostVolumeMount
    dataset_mount: GymHostVolumeMount | None = None
    workspace_mount: GymHostVolumeMount
    egress_allow: tuple[GymHostEgressRule, ...] = ()
    bootstrap_env: Mapping[str, str] = Field(default_factory=dict)
    labels: Mapping[str, str] = Field(default_factory=dict)
    max_request_bytes: int = Field(default=DEFAULT_MAX_ROLLOUT_BYTES, gt=0)
    max_response_bytes: int = Field(default=DEFAULT_MAX_ROLLOUT_BYTES, gt=0)
    ttl_s: int | None = DEFAULT_HOST_TTL_S
    # How long OpenSandbox may wait for the pod to become Running (image pull).
    ready_timeout_s: float | None = Field(
        default=float(DEFAULT_HOST_READY_TIMEOUT_S), gt=0
    )
    resources: Mapping[str, str] | None = None
    runtime_http_port: int = Field(default=DEFAULT_RUNTIME_HTTP_PORT, ge=1, le=65535)
    # Every policy is deny-by-default. This switch explicitly adds the safe
    # public IPv4 and DNS-suffix whitelist.
    allow_internet: bool = False
    public_dns_allow: tuple[str, ...] | None = None
    entrypoint: tuple[str, ...] | None = None

    @field_validator("job_id")
    @classmethod
    def _check_job_id(cls, value: str) -> str:
        if not K8S_LABEL_VALUE_RE.match(value):
            raise ValueError(
                f"job_id must be a valid Kubernetes label value: {value!r}"
            )
        return value

    @field_validator("bootstrap_env")
    @classmethod
    def _reject_forbidden_bootstrap(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        validate_bootstrap_env(value)
        return dict(value)

    @model_validator(mode="after")
    def _check_mount_roles(self) -> "GymHostSpec":
        if not self.environment_mount.read_only:
            raise ValueError("environment_mount must be read-only")
        if self.dataset_mount is not None and not self.dataset_mount.read_only:
            raise ValueError("dataset_mount must be read-only when present")
        if self.workspace_mount.read_only:
            raise ValueError("workspace_mount must be read-write")
        return self


@dataclass(frozen=True)
class GymHostHandle(Generic[TProvider]):
    """Handle returned after a job host is created.

    ``provider`` is the provider instance that owns the host. Its internal
    resource handle remains private to the adapter.
    """

    host_id: str
    health_url: str
    rollout_url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    provider: TProvider | None = None


class SandboxNetworkPolicy(BaseModel):
    """Allow-only egress settings under ``env.nemo_gym.sandbox.network_policy``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    egress_allow: list[GymHostEgressRule] = Field(default_factory=list)
    # Additional suffixes consulted only when ``allow_internet`` is true.
    public_dns_allow: tuple[str, ...] | None = None


class SandboxConfig(BaseModel):
    """Job sandbox settings under ``env.nemo_gym.sandbox``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str = Field(min_length=1)
    env_mount_path: str = "/job/environment"
    dataset_mount_path: str = "/job/dataset"
    work_mount_path: str = "/job/work"
    max_request_bytes: int = Field(default=DEFAULT_MAX_ROLLOUT_BYTES, gt=0)
    max_response_bytes: int = Field(default=DEFAULT_MAX_ROLLOUT_BYTES, gt=0)
    ttl_s: int = Field(default=DEFAULT_HOST_TTL_S, gt=0)
    network_policy: SandboxNetworkPolicy
    resources: dict[str, str] | None = None
    allow_internet: bool = False
    environment_pvc_claim: str = Field(min_length=1)
    environment_sub_path: str = ""
    dataset_pvc_claim: str | None = None
    dataset_sub_path: str = ""
    workspace_pvc_claim: str = Field(min_length=1)
    workspace_sub_path: str = ""
    runtime_http_port: int = Field(default=DEFAULT_RUNTIME_HTTP_PORT, ge=1, le=65535)
    ready_timeout_s: float = Field(default=float(DEFAULT_HOST_READY_TIMEOUT_S), gt=0)
    rollout_timeout_s: float = Field(default=float(DEFAULT_ROLLOUT_TIMEOUT_S), gt=0)
    host_provider_options: dict[str, Any] = Field(default_factory=dict)
    entrypoint: list[str] | None = None


class NemoGymSandboxedConfig(BaseModel):
    """Sandboxed-mode fields under ``env.nemo_gym``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sandboxed: bool = False
    host_provider: str = "opensandbox"
    environment_path: str | None = None
    sandbox: SandboxConfig | None = None
    job_id: str | None = None
    episode_broker: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_sandbox_when_enabled(self) -> "NemoGymSandboxedConfig":
        if self.sandboxed and self.sandbox is None:
            raise ValueError(
                "env.nemo_gym.sandboxed=true requires env.nemo_gym.sandbox to be set"
            )
        return self


def validate_bootstrap_env(env: Mapping[str, str]) -> None:
    """Raise if ``env`` contains OpenSandbox credentials or training Ray addresses."""
    for key in env:
        if key in FORBIDDEN_BOOTSTRAP_ENV_KEYS:
            raise ValueError(
                f"bootstrap_env must not contain {key!r}; OpenSandbox credentials and "
                "training Ray addresses stay in the trusted actor"
            )
        for prefix in FORBIDDEN_BOOTSTRAP_ENV_PREFIXES:
            if key.startswith(prefix):
                raise ValueError(
                    f"bootstrap_env must not contain OpenSandbox credential key {key!r}"
                )


def build_bootstrap_env(
    job_id: str,
    environment_path: str,
    work_path: str,
    broker_url: str,
    broker_token: str,
    max_request_bytes: int,
    max_response_bytes: int,
    dataset_path: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the bootstrap environment injected into the job sandbox."""
    env: dict[str, str] = {
        "NMP_JOB_ID": job_id,
        "NMP_ENVIRONMENT_PATH": environment_path,
        "NMP_WORK_PATH": work_path,
        "NMP_BROKER_URL": broker_url,
        "NMP_BROKER_TOKEN": broker_token,
        "NMP_MAX_REQUEST_BYTES": str(max_request_bytes),
        "NMP_MAX_RESPONSE_BYTES": str(max_response_bytes),
    }
    if dataset_path is not None:
        env["NMP_DATASET_PATH"] = dataset_path
    if extra:
        env.update(dict(extra))
    validate_bootstrap_env(env)
    return env
