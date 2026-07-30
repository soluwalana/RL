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

"""Trusted episode provisioning broker for sandboxed NeMo-Gym GRPO jobs.

``SandboxEpisodeBrokerActor`` is intentionally not re-exported here: importing it pulls in Ray,
and the HTTP surface, sanitizer, and backends are kept importable without a Ray cluster so they
can be tested directly. Import it from
:mod:`nemo_rl.environments.sandbox.broker_actor` instead.
"""

from nemo_rl.environments.sandbox.backends.base import (
    EpisodeSandboxBackend,
    SanitizedEpisodeSpec,
    UnsupportedEpisodeOperationError,
)
from nemo_rl.environments.sandbox.config import BrokerEndpoint, EpisodeBrokerConfig
from nemo_rl.environments.sandbox.egress import (
    DEFAULT_PUBLIC_DNS_SUFFIXES,
    EpisodeEgressPolicy,
    EpisodeEgressRule,
    build_egress_policy,
    build_sandbox_egress_policy,
    denied_cidrs,
    local_resolver_addresses,
)
from nemo_rl.environments.sandbox.errors import BrokerRequestError
from nemo_rl.environments.sandbox.host.models import (
    GymHostEgressRule,
    GymHostHandle,
    GymHostSpec,
    GymHostVolumeMount,
    NemoGymSandboxedConfig,
    SandboxConfig,
)
from nemo_rl.environments.sandbox.http_app import (
    begin_shutdown,
    build_broker_app,
    close_all_episodes,
)
from nemo_rl.environments.sandbox.sanitize import (
    sanitize_create_request,
    sanitize_exec_request,
)


__all__ = [
    "DEFAULT_PUBLIC_DNS_SUFFIXES",
    "BrokerEndpoint",
    "BrokerRequestError",
    "EpisodeBrokerConfig",
    "EpisodeEgressPolicy",
    "EpisodeEgressRule",
    "EpisodeSandboxBackend",
    "GymHostEgressRule",
    "GymHostHandle",
    "GymHostSpec",
    "GymHostVolumeMount",
    "NemoGymSandboxedConfig",
    "SandboxConfig",
    "SanitizedEpisodeSpec",
    "UnsupportedEpisodeOperationError",
    "begin_shutdown",
    "build_broker_app",
    "build_egress_policy",
    "build_sandbox_egress_policy",
    "denied_cidrs",
    "local_resolver_addresses",
    "close_all_episodes",
    "sanitize_create_request",
    "sanitize_exec_request",
]
