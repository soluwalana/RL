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

"""Configuration for the trusted episode provisioning broker."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_rl.environments.sandbox.egress import DEFAULT_CLUSTER_DENY_TARGETS


# Metadata keys the broker owns. A caller may not set them, so a backend label can always be
# trusted to say which job an episode belongs to.
RESERVED_METADATA_PREFIX = "nemo-rl-"
JOB_ID_METADATA_KEY = "nemo-rl-job-id"

# The trusted broker buffers every request body, so this cap protects the training leader pod --
# not the episode. Uploads larger than this must be staged through the episode itself.
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024


class EpisodeBrokerConfig(BaseModel):
    """Policy and placement settings for one job's episode broker.

    Every field is trusted-side policy: the job sandbox can read the broker token but has no way to
    influence any value here. Defaults are deliberately closed -- with no ``approved_images`` or
    ``approved_image_prefixes`` configured, every episode create is refused, so a job that does not
    use episode sandboxes is unaffected and one that does must be granted images explicitly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str = Field(min_length=1)

    backend: Literal["opensandbox", "memory"] = "opensandbox"
    backend_options: dict[str, Any] = Field(default_factory=dict)
    # Second key required to select the in-memory backend, which provisions nothing real.
    allow_insecure_memory_backend: bool = False

    approved_images: tuple[str, ...] = ()
    approved_image_prefixes: tuple[str, ...] = ()
    # Keys a caller may pass in ``provider_options``. Anything else is rejected rather than
    # dropped, so an environment fails fast instead of running with silently different options.
    allowed_provider_option_keys: tuple[str, ...] = ()

    default_ttl_s: float = Field(default=300.0, gt=0)
    max_ttl_s: float = Field(default=3600.0, gt=0)
    # Caps on how long one broker request may block. A caller that could name its own timeout
    # could otherwise pin trusted-side connections open indefinitely.
    default_exec_timeout_s: float = Field(default=180.0, gt=0)
    max_exec_timeout_s: float = Field(default=1800.0, gt=0)
    max_ready_timeout_s: float = Field(default=300.0, gt=0)
    max_concurrent_episodes: int = Field(default=32, ge=1)
    max_request_bytes: int = Field(default=DEFAULT_MAX_REQUEST_BYTES, gt=0)

    max_cpu: float | None = Field(default=8.0, gt=0)
    max_memory_mib: int | None = Field(default=32768, gt=0)
    max_disk_gib: int | None = Field(default=128, gt=0)
    # Episode sandboxes grade model output on CPU; GPUs are opt-in per deployment.
    max_gpu: int = Field(default=0, ge=0)

    # Egress. Allow-by-default with cluster ranges denied, because the threat is reaching another
    # tenant rather than reaching the internet -- graders legitimately install packages mid-episode.
    # See nemo_rl.environments.sandbox.egress for why this fails open and what backs it up.
    egress_default_action: Literal["allow", "deny"] = "allow"
    egress_deny_targets: tuple[str, ...] = DEFAULT_CLUSTER_DENY_TARGETS
    egress_allow_targets: tuple[str, ...] = ()
    # Second key required to run episodes with no egress restriction whatsoever.
    allow_unrestricted_episode_egress: bool = False

    # Host advertised to the job sandbox. Left unset the broker uses the leader pod IP; set it to a
    # headless-Service DNS name where one exists, so the sandbox egress rule survives a reschedule.
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    port_range_low: int | None = Field(default=None, ge=1, le=65535)
    port_range_high: int | None = Field(default=None, ge=1, le=65535)

    @field_validator("approved_image_prefixes")
    @classmethod
    def _check_prefixes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        # A prefix must end at a path boundary. Without this, granting "registry.example.com/swe"
        # would also grant "registry.example.com/swe-attacker/anything".
        for prefix in value:
            if not prefix or not prefix.endswith("/"):
                raise ValueError(f"approved image prefix must end with '/': {prefix!r}")
        return value

    @model_validator(mode="after")
    def _check_bounds(self) -> "EpisodeBrokerConfig":
        if self.default_ttl_s > self.max_ttl_s:
            raise ValueError("default_ttl_s must not exceed max_ttl_s")
        if self.default_exec_timeout_s > self.max_exec_timeout_s:
            raise ValueError(
                "default_exec_timeout_s must not exceed max_exec_timeout_s"
            )
        low, high = self.port_range_low, self.port_range_high
        if (low is None) != (high is None):
            raise ValueError("port_range_low and port_range_high must be set together")
        if low is not None and high is not None and low >= high:
            raise ValueError("port_range_low must be less than port_range_high")
        if (
            self.egress_default_action == "allow"
            and not self.egress_deny_targets
            and not self.allow_unrestricted_episode_egress
        ):
            # Allow-by-default with nothing denied is an episode that can reach every pod and
            # Service in the cluster. Reachable only by saying so explicitly.
            raise ValueError(
                "egress_default_action='allow' with an empty egress_deny_targets places no "
                "restriction on episode network access. Set "
                "allow_unrestricted_episode_egress=true to accept that, or supply the cluster's "
                "private ranges in egress_deny_targets."
            )
        return self


class BrokerEndpoint(BaseModel):
    """Where the job sandbox reaches the broker, and the token it must present.

    ``token`` is excluded from ``repr`` so it does not reach driver logs. It is not a secret from
    the sandbox -- user code can read it -- but it should not leak any further than it must.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    host: str
    port: int
    token: str = Field(repr=False)
