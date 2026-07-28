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

"""Job-host provider protocol and registry."""

from typing import Protocol

from nemo_rl.environments.sandbox.egress import (
    EpisodeEgressPolicy,
    build_egress_policy,
)
from nemo_rl.environments.sandbox.host.models import GymHostHandle, GymHostSpec


class SandboxedGymHostProvider(Protocol):
    """Provisions and tears down the job-level Gym host sandbox."""

    name: str

    async def create_host(self, spec: GymHostSpec) -> GymHostHandle:
        """Create the host and return actor-reachable health/rollout URLs."""

    async def wait_ready(self, handle: GymHostHandle, timeout_s: float) -> None:
        """Block until ``GET health_url`` reports ready, or raise on timeout."""

    async def destroy_host(self, handle: GymHostHandle) -> None:
        """Terminate the host. Best-effort; must not raise after a successful destroy."""


def build_host_egress_policy(spec: GymHostSpec) -> EpisodeEgressPolicy:
    """Build the shared egress profile for a job host from ``GymHostSpec``."""
    allow_targets = tuple(rule.host for rule in spec.egress_allow)
    if spec.deny_internet:
        return build_egress_policy(
            default_action="deny",
            allow_targets=allow_targets,
            deny_targets=(),
        )
    return build_egress_policy(
        default_action="allow",
        allow_targets=allow_targets,
        deny_targets=spec.egress_deny_targets,
    )


def get_host_provider(name: str, options: dict | None = None) -> SandboxedGymHostProvider:
    """Construct a registered job-host provider by name."""
    options = options or {}
    if name == "opensandbox":
        from nemo_rl.environments.sandbox.host.opensandbox import (
            OpenSandboxGymHostProvider,
        )

        return OpenSandboxGymHostProvider(**options)
    raise ValueError(f"Unknown sandboxed gym host provider: {name!r}")
