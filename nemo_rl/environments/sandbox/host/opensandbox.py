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

"""OpenSandbox backend for the job-level Gym host."""

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nemo_rl.environments.sandbox.backends.opensandbox import to_opensandbox_policy
from nemo_rl.environments.sandbox.config import JOB_ID_METADATA_KEY
from nemo_rl.environments.sandbox.host.models import (
    GymHostHandle,
    GymHostSpec,
    GymHostVolumeMount,
)
from nemo_rl.environments.sandbox.host.provider import build_host_egress_policy


LOGGER = logging.getLogger(__name__)

_HEALTH_POLL_S = 2.0


class OpenSandboxGymHostProvider:
    """Provisions the job Gym host through OpenSandbox.

    Delegates create/close to NeMo-Gym's ``OpenSandboxProvider``. Owns PVC mounts,
    shared egress policy, and proxy URL resolution for the runtime HTTP port.
    """

    name = "opensandbox"

    def __init__(
        self,
        connection: Mapping[str, Any] | None = None,
        create: Mapping[str, Any] | None = None,
        probe: Mapping[str, Any] | None = None,
        operations: Mapping[str, Any] | None = None,
    ) -> None:
        from nemo_gym.sandbox.providers.opensandbox import OpenSandboxProvider

        self._connection = dict(connection) if connection else {}
        if "use_server_proxy" not in self._connection:
            self._connection["use_server_proxy"] = True

        self._egress = None
        self._create_options = dict(create) if create else {}
        self._probe = probe
        self._operations = operations
        self._provider_cls = OpenSandboxProvider
        self._provider = None

    def _provider_for_spec(self, spec: GymHostSpec):
        egress = build_host_egress_policy(spec)
        self._egress = egress
        create_config = {
            **self._create_options,
            "network_policy": to_opensandbox_policy(egress),
        }
        return self._provider_cls(
            connection=self._connection,
            create=create_config,
            probe=self._probe,
            operations=self._operations,
        )

    def _volume_dict(self, mount: GymHostVolumeMount, name: str) -> dict[str, Any]:
        volume: dict[str, Any] = {
            "name": name,
            "mountPath": mount.mount_path,
            "readOnly": mount.read_only,
            "pvc": {
                "claimName": mount.pvc_claim,
                "createIfNotExists": False,
            },
        }
        if mount.sub_path:
            volume["subPath"] = mount.sub_path
        return volume

    def _to_sandbox_spec(self, spec: GymHostSpec):
        from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec

        volumes = [
            self._volume_dict(spec.environment_mount, "environment"),
            self._volume_dict(spec.workspace_mount, "workspace"),
        ]
        if spec.dataset_mount is not None:
            volumes.append(self._volume_dict(spec.dataset_mount, "dataset"))

        metadata = {
            JOB_ID_METADATA_KEY: spec.job_id,
            **{str(k): str(v) for k, v in spec.labels.items()},
        }

        resources = SandboxResources()
        if spec.resources:
            cpu = spec.resources.get("cpu")
            memory = spec.resources.get("memory") or spec.resources.get("memory_mib")
            resource_kwargs: dict[str, Any] = {}
            if cpu is not None:
                resource_kwargs["cpu"] = float(str(cpu).rstrip("m")) / (
                    1000.0 if str(cpu).endswith("m") else 1.0
                )
            if memory is not None:
                memory_s = str(memory)
                if memory_s.endswith("Gi"):
                    resource_kwargs["memory_mib"] = int(float(memory_s[:-2]) * 1024)
                elif memory_s.endswith("Mi"):
                    resource_kwargs["memory_mib"] = int(float(memory_s[:-2]))
                else:
                    resource_kwargs["memory_mib"] = int(memory_s)
            resources = SandboxResources.from_mapping(resource_kwargs)

        return SandboxSpec(
            image=spec.runtime_image,
            ttl_s=spec.ttl_s,
            env=dict(spec.bootstrap_env),
            metadata=metadata,
            resources=resources,
            entrypoint=list(spec.entrypoint) if spec.entrypoint is not None else None,
            provider_options={"volumes": volumes},
        )

    async def _endpoint_urls(
        self, sandbox: Any, port: int
    ) -> tuple[str, str]:
        endpoint = await sandbox.get_endpoint(port)
        base = str(endpoint.endpoint).rstrip("/")
        # Proxy paths already include the port root; append API routes.
        health_url = f"{base}/health"
        rollout_url = f"{base}/rollouts/run"
        return health_url, rollout_url

    async def create_host(self, spec: GymHostSpec) -> GymHostHandle:
        """Create the job host with mounts and egress, then resolve proxy URLs."""
        provider = self._provider_for_spec(spec)
        self._provider = provider
        handle = await provider.create(self._to_sandbox_spec(spec))
        try:
            health_url, rollout_url = await self._endpoint_urls(
                handle.raw, spec.runtime_http_port
            )
        except Exception:
            await provider.close(handle)
            raise
        return GymHostHandle(
            host_id=handle.sandbox_id,
            health_url=health_url,
            rollout_url=rollout_url,
            opaque=handle,
        )

    async def wait_ready(self, handle: GymHostHandle, timeout_s: float) -> None:
        """Poll ``health_url`` until status is ready."""
        deadline = asyncio.get_running_loop().time() + timeout_s
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                body = await asyncio.to_thread(self._get_json, handle.health_url)
                if body.get("status") == "ready":
                    return
                last_error = RuntimeError(f"host not ready: {body!r}")
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(_HEALTH_POLL_S)
        raise TimeoutError(
            f"job host {handle.host_id} did not become ready within {timeout_s:g}s"
            + (f": {last_error}" if last_error is not None else "")
        )

    def _get_json(self, url: str) -> dict[str, Any]:
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=10) as response:
                payload = response.read()
        except HTTPError as exc:
            if exc.code == 503:
                return {"status": "starting"}
            raise
        except URLError:
            raise
        return json.loads(payload.decode("utf-8"))

    async def destroy_host(self, handle: GymHostHandle) -> None:
        """Terminate the job host."""
        opaque = handle.opaque
        if opaque is None:
            LOGGER.warning("destroy_host called with no opaque handle for %s", handle.host_id)
            return
        provider = self._provider
        if provider is None:
            from nemo_gym.sandbox.providers.opensandbox import OpenSandboxProvider

            provider = OpenSandboxProvider(connection=self._connection)
        try:
            await provider.close(opaque)
        except Exception:
            LOGGER.exception("Failed to destroy job host %s", handle.host_id)
