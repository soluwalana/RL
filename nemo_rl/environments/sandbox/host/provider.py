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

from typing import Protocol, TypeVar

from nemo_rl.environments.sandbox.egress import (
    EpisodeEgressPolicy,
    build_sandbox_egress_policy,
)
from nemo_rl.environments.sandbox.host.models import GymHostHandle, GymHostSpec


TProvider = TypeVar("TProvider")


class SandboxedGymHostProvider(Protocol[TProvider]):
    """Provisions and tears down the job-level Gym host sandbox."""

    name: str
    # Concrete provider implementation stored on ``GymHostHandle.provider``.
    provider_class: type[TProvider]

    async def create_host(self, spec: GymHostSpec) -> GymHostHandle[TProvider]:
        """Create the host and return actor-reachable health/rollout URLs."""

    async def wait_ready(
        self, handle: GymHostHandle[TProvider], timeout_s: float
    ) -> None:
        """Block until ``GET health_url`` reports ready, or raise on timeout."""

    async def destroy_host(self, handle: GymHostHandle[TProvider]) -> None:
        """Terminate the host. Best-effort; must not raise after a successful destroy."""


def build_host_egress_policy(spec: GymHostSpec) -> EpisodeEgressPolicy:
    """Build an allow-only, deny-by-default job-host policy."""
    allow_targets = tuple(rule.host for rule in spec.egress_allow)
    return build_sandbox_egress_policy(
        endpoint_targets=allow_targets,
        allow_internet=spec.allow_internet,
        public_dns_allow=spec.public_dns_allow,
        resolver_addresses=spec.resolver_addresses,
    )


def get_host_provider(
    name: str, options: dict | None = None
) -> SandboxedGymHostProvider:
    """Construct a registered job-host provider by name."""
    options = options or {}
    if name == "opensandbox":
        from nemo_rl.environments.sandbox.host.opensandbox import (
            OpenSandboxGymHostProvider,
        )

        return OpenSandboxGymHostProvider(**options)
    raise ValueError(f"Unknown sandboxed gym host provider: {name!r}")
