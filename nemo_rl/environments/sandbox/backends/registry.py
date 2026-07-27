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

"""Selection of the configured episode backend."""

from nemo_rl.environments.sandbox.backends.base import EpisodeSandboxBackend
from nemo_rl.environments.sandbox.backends.memory import InMemoryEpisodeBackend
from nemo_rl.environments.sandbox.config import EpisodeBrokerConfig


def build_backend(config: EpisodeBrokerConfig) -> EpisodeSandboxBackend:
    """Instantiate the episode backend named by ``config``.

    Args:
        config: Broker configuration.

    Returns:
        A backend satisfying :class:`EpisodeSandboxBackend`.

    Raises:
        ValueError: If the in-memory backend is selected without the explicit insecure opt-in, or
            the backend name is unknown.
        NotImplementedError: If the backend is known but not implemented yet.
    """
    if config.backend == "memory":
        if not config.allow_insecure_memory_backend:
            raise ValueError(
                "Episode backend 'memory' provisions no real sandbox and applies no isolation. "
                "Set allow_insecure_memory_backend=true to use it in development or tests."
            )
        return InMemoryEpisodeBackend()

    if config.backend == "opensandbox":
        raise NotImplementedError(
            "The OpenSandbox episode backend is not implemented yet. It wraps NeMo-Gym's "
            "OpenSandboxProvider behind EpisodeSandboxBackend and lands with the broker's backend "
            "phase; until then only the 'memory' backend is selectable."
        )

    raise ValueError(f"Unknown episode backend: {config.backend!r}")
