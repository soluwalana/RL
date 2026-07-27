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

"""Ray actor that hosts the trusted episode broker on the training leader.

Ray provides placement and lifecycle here, not the request path: the HTTP server runs on its own
event loop in a background thread, the way ``VllmAsyncGenerationWorker`` hosts its OpenAI-compatible
server. Requests therefore do not serialize through Ray actor method dispatch, which matters
because every episode ``exec`` in a SWE rollout crosses this boundary.

The job sandbox talks to this actor over HTTP only. It is never given a Ray handle, because a
sandbox that could join the training Ray cluster could schedule work anywhere in it.
"""

import asyncio
import logging
import os
import secrets
import socket
import threading
import time
from typing import Any

import ray

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GYM_PORT_RANGE_HIGH,
    DEFAULT_GYM_PORT_RANGE_LOW,
    _bind_socket_in_range,
    _get_node_ip_local,
)
from nemo_rl.environments.sandbox.backends.registry import build_backend
from nemo_rl.environments.sandbox.config import BrokerEndpoint, EpisodeBrokerConfig
from nemo_rl.environments.sandbox.http_app import build_broker_app, close_all_episodes
from nemo_rl.utils.venvs import create_local_venv_on_each_node


LOGGER = logging.getLogger(__name__)

BROKER_ACTOR_FQN = "nemo_rl.environments.sandbox.broker_actor.SandboxEpisodeBrokerActor"

STARTUP_TIMEOUT_S = 30.0
SHUTDOWN_DRAIN_TIMEOUT_S = 60.0
SHUTDOWN_JOIN_TIMEOUT_S = 30.0


# Deliberately no ``max_restarts``: a silently restarted broker would come back with an empty
# handle map while its episodes kept running, so a crash should fail the job instead. Restart
# support needs label-based reconciliation through ``EpisodeSandboxBackend.list_backend_ids``.
@ray.remote
class SandboxEpisodeBrokerActor:
    """Trusted, job-scoped provisioner of episode sandboxes.

    Holds the episode backend credential on the training leader so the job sandbox never has one.
    Its own API is reachable with a token user code can read, so it has to be escalation-free on
    its own terms: see :mod:`nemo_rl.environments.sandbox.sanitize`.
    """

    def __init__(self, config: EpisodeBrokerConfig | dict[str, Any]) -> None:
        self._config = (
            config
            if isinstance(config, EpisodeBrokerConfig)
            else EpisodeBrokerConfig(**config)
        )
        self._endpoint: BrokerEndpoint | None = None
        self._app = None
        self._server = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def _reserve_socket(self) -> tuple[socket.socket, int]:
        """Bind and listen on the broker port, holding the socket until uvicorn takes it.

        The port is handed to the job sandbox's egress allowlist before the sandbox exists, so it
        must be settled up front. Keeping the listening socket open and passing it straight to
        uvicorn leaves no window for another process to take the port.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self._config.port is not None:
            sock.bind(("", self._config.port))
            port = self._config.port
        else:
            port = _bind_socket_in_range(
                sock,
                self._config.port_range_low or DEFAULT_GYM_PORT_RANGE_LOW,
                self._config.port_range_high or DEFAULT_GYM_PORT_RANGE_HIGH,
            )
        sock.listen(128)
        sock.setblocking(False)
        return sock, port

    def start(self) -> BrokerEndpoint:
        """Start the broker HTTP server and return where the job sandbox should reach it.

        Returns:
            The broker endpoint, including the job-scoped token to inject into sandbox bootstrap.

        Raises:
            RuntimeError: If the broker has already been started.
        """
        if self._endpoint is not None:
            raise RuntimeError("Episode broker is already started")

        import uvicorn

        backend = build_backend(self._config)
        token = secrets.token_urlsafe(32)
        self._socket, port = self._reserve_socket()
        host = self._config.host or _get_node_ip_local()

        self._app = build_broker_app(backend=backend, config=self._config, token=token)
        # Bind on all interfaces: the caller is a sandbox pod, not a local process. Reachability is
        # constrained by the sandbox egress allowlist and the job's NetworkPolicy, not by binding.
        self._server = uvicorn.Server(
            config=uvicorn.Config(
                self._app, host="0.0.0.0", port=port, access_log=False
            )
        )

        reserved_socket = self._socket
        self._socket = None  # ownership transfers to uvicorn

        def _serve() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(self._server.serve(sockets=[reserved_socket]))
            finally:
                loop.close()

        self._thread = threading.Thread(
            target=_serve, name="nemo-rl-episode-broker", daemon=True
        )
        self._thread.start()

        # Wait for uvicorn to actually serve rather than for the thread to merely exist, so a
        # server that dies during startup surfaces here instead of as an endpoint the caller
        # publishes to the job sandbox and only finds broken at the first rollout.
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("Episode broker HTTP server exited during startup")
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Episode broker HTTP server did not start within {STARTUP_TIMEOUT_S:g}s"
                )
            time.sleep(0.05)

        self._endpoint = BrokerEndpoint(
            url=f"http://{host}:{port}", host=host, port=port, token=token
        )
        LOGGER.info(
            "Episode broker for job %s listening on %s:%s with the %s backend",
            self._config.job_id,
            host,
            port,
            self._config.backend,
        )
        return self._endpoint

    def get_endpoint(self) -> BrokerEndpoint:
        """Return the endpoint of a started broker.

        Raises:
            RuntimeError: If the broker has not been started.
        """
        if self._endpoint is None:
            raise RuntimeError("Episode broker has not been started")
        return self._endpoint

    def shutdown(self) -> None:
        """Terminate outstanding episodes, then stop the HTTP server.

        Episodes are reaped before the server stops so teardown runs on a live event loop. Callers
        should shut the job sandbox down first, so in-flight episode calls fail fast rather than
        racing this.
        """
        if self._loop is not None and self._app is not None:
            future = asyncio.run_coroutine_threadsafe(
                close_all_episodes(self._app), self._loop
            )
            try:
                future.result(timeout=SHUTDOWN_DRAIN_TIMEOUT_S)
            except Exception:
                LOGGER.exception(
                    "Episode broker failed to drain episodes during shutdown"
                )

        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                LOGGER.warning(
                    "Episode broker HTTP thread did not stop within the shutdown timeout"
                )

        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._endpoint = None


def start_episode_broker(
    config: EpisodeBrokerConfig | dict[str, Any],
    *,
    node_id: str | None = None,
    extra_ray_options: dict[str, Any] | None = None,
) -> tuple[Any, BrokerEndpoint]:
    """Create and start the broker actor, mirroring how the NeMo-Gym actor is spun up in GRPO.

    Intended to be called from ``SandboxedGymActor._spinup`` before the job sandbox is created, so
    the returned endpoint can be written into the sandbox's egress allowlist and bootstrap
    environment.

    Args:
        config: Broker configuration, or a mapping to validate into one.
        node_id: Ray node to pin the actor to, normally the training leader. Pinned hard, because
            the endpoint handed to the sandbox has to stay valid for the life of the job.
        extra_ray_options: Additional options for ``.options()``.

    Returns:
        A tuple of the actor handle and its endpoint.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    py_executable = get_actor_python_env(BROKER_ACTOR_FQN)
    if py_executable.startswith("uv"):
        py_executable = create_local_venv_on_each_node(py_executable, BROKER_ACTOR_FQN)

    options: dict[str, Any] = {
        "runtime_env": {
            "py_executable": py_executable,
            # Propagates the trusted process environment, including any episode backend credential.
            # That credential belongs here and must never be forwarded into the job sandbox's
            # bootstrap environment.
            "env_vars": {
                **os.environ,
                "VIRTUAL_ENV": py_executable,
                "UV_PROJECT_ENVIRONMENT": py_executable,
            },
        }
    }
    if node_id is not None:
        options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node_id=node_id, soft=False
        )
    if extra_ray_options:
        options.update(extra_ray_options)

    actor = SandboxEpisodeBrokerActor.options(**options).remote(config)
    endpoint = ray.get(actor.start.remote())
    return actor, endpoint
