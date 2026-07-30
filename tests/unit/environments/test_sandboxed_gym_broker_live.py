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

"""Live episode-broker path: job sandbox -> broker -> OpenSandbox episode.

This is the plumbing test underneath any SWE rollout. A SWE agent reaches the broker through
NeMo-Gym's ``Sandbox`` API, which needs a real Gym tree in the job sandbox and several minutes of
image pull and dependency sync before the first episode is even attempted. Here the probe speaks
the broker's HTTP contract directly with the standard library, on the slim runtime image, so a
break in the network path, the token, the image policy, or the OpenSandbox episode backend is
identified in about a minute instead of being diagnosed from the tail of a SWE run.

What it establishes, in the order the probe checks it:

* the job sandbox can reach the broker at all -- the endpoint it was handed is the advertised
  reverse-tunnel Service DNS, and the egress allowlist is otherwise closed;
* the injected token authenticates, and the broker reports the job it belongs to;
* the approved-image policy fails closed for an image outside the configured prefixes;
* an approved image really provisions an OpenSandbox episode that can execute a command;
* the episode can be torn down by the sandbox that created it.

Gated behind ``LIVE_OPENSANDBOX=1`` and the same environment as the other live tests; see
``scratch/sandboxed-gym-live-env-setup.md``. The reverse tunnel must be up before running.
"""

import base64
import textwrap
import uuid
from collections.abc import Iterator

import pytest
import ray
from nemo_gym.sandbox.broker import (
    BROKER_AUTH_HEADER,
    BROKER_TOKEN_ENV,
    BROKER_URL_ENV,
)

from nemo_rl.environments.sandbox.broker_actor import start_episode_broker
from nemo_rl.environments.sandbox.config import EpisodeBrokerConfig
from nemo_rl.environments.sandbox.host.models import (
    GymHostEgressRule,
    GymHostSpec,
    GymHostVolumeMount,
    build_bootstrap_env,
)
from nemo_rl.environments.sandbox.host.opensandbox import OpenSandboxGymHostProvider
from sandboxed_gym_live_common import (
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_PORT,
    EPISODE_SMOKE_IMAGE,
    READY_TIMEOUT_S,
    RUNTIME_IMAGE,
    STUB_SANDBOX_RESOURCES,
    broker_service_cluster_ip,
    build_live_target,
    cluster_resolver_addresses,
    create_ephemeral_pvcs,
    episode_broker_block,
    live_opensandbox_enabled,
    port_forward_server,
    stub_entrypoint,
)


pytestmark = [
    pytest.mark.nemo_gym,
    pytest.mark.skipif(
        not live_opensandbox_enabled(),
        reason="Set LIVE_OPENSANDBOX=1 to run live episode-broker tests",
    ),
]

# Syntactically valid and certain to be outside any configured prefix, so the 403 it draws is the
# image policy talking and not a registry lookup.
UNAPPROVED_IMAGE = "registry.invalid/not-approved/nothing:latest"
PROBE_PATH = "/job/work/broker_probe.py"


@pytest.fixture
def live_target():
    return build_live_target()


@pytest.fixture
def port_forward(live_target) -> Iterator[tuple[str, str]]:
    yield from port_forward_server(live_target)


@pytest.fixture
def ephemeral_pvcs(live_target) -> Iterator[tuple[str, str]]:
    yield from create_ephemeral_pvcs(live_target)


def _probe_source(job_id: str) -> str:
    """Build the in-sandbox probe.

    Standard library only: the point of running on the slim image is that nothing about this
    depends on a Gym install. The broker environment variable names are interpolated from
    NeMo-Gym's own wire contract, so a rename on either side fails this test rather than silently
    turning brokered mode off in production.
    """
    return textwrap.dedent(
        f"""\
        import json
        import os
        import urllib.error
        import urllib.request

        BASE = os.environ.get({BROKER_URL_ENV!r}, "").rstrip("/")
        TOKEN = os.environ.get({BROKER_TOKEN_ENV!r}, "")
        print("BROKER_URL=" + (BASE or "<unset>"))
        print("BROKER_TOKEN_SET=" + ("yes" if TOKEN else "no"))
        if not BASE or not TOKEN:
            raise SystemExit("broker environment was not injected")


        def call(method, path, payload=None):
            data = None if payload is None else json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                BASE + path,
                data=data,
                method=method,
                headers={{
                    "Content-Type": "application/json",
                    {BROKER_AUTH_HEADER!r}: TOKEN,
                }},
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    body = response.read().decode("utf-8")
                    return response.status, (json.loads(body) if body else {{}})
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8")
                try:
                    return exc.code, json.loads(body)
                except ValueError:
                    return exc.code, {{"raw": body}}


        status, health = call("GET", "/health")
        print("HEALTH=" + str(status) + " " + json.dumps(health, sort_keys=True))
        assert status == 200, health
        assert health.get("job_id") == {job_id!r}, health

        status, denied = call("POST", "/episodes", {{"image": {UNAPPROVED_IMAGE!r}}})
        print("UNAPPROVED=" + str(status) + " " + json.dumps(denied, sort_keys=True))
        assert status == 403, denied
        assert denied.get("code") == "image_not_approved", denied

        status, created = call(
            "POST",
            "/episodes",
            {{
                "image": {EPISODE_SMOKE_IMAGE!r},
                "ready_timeout_s": 600,
                "resources": {{"cpu": 1, "memory_mib": 1024}},
                "metadata": {{"probe": "broker-live"}},
            }},
        )
        print("CREATE=" + str(status) + " " + json.dumps(created, sort_keys=True))
        assert status == 201, created
        episode_id = created["episode_id"]

        try:
            status, result = call(
                "POST",
                "/episodes/" + episode_id + "/exec",
                {{"command": "echo brokered-episode-ok", "timeout_s": 120}},
            )
            print("EXEC=" + str(status) + " " + json.dumps(result, sort_keys=True))
            assert status == 200, result
            assert result["return_code"] == 0, result
            assert "brokered-episode-ok" in (result.get("stdout") or ""), result
        finally:
            status, _ = call("DELETE", "/episodes/" + episode_id)
            print("DELETE=" + str(status))

        print("BROKER_PROBE_OK")
        """
    )


def _host_spec(
    job_id: str,
    env_claim: str,
    work_claim: str,
    endpoint,
    broker_cluster_ip: str | None,
    resolver_addresses: tuple[str, ...],
) -> GymHostSpec:
    """Job-host spec whose only permitted destination is the broker.

    ``allow_internet`` stays off and no policy endpoint is allowed, so anything the probe reaches
    it reached through the rule under test. The ClusterIP is allowed alongside the Service name
    because the deny ranges cover cluster-private space and only an address can be subtracted
    from them; the workbox cannot resolve Service DNS to supply it automatically.
    """
    egress_allow = [GymHostEgressRule(host=DEFAULT_BROKER_HOST, port=DEFAULT_BROKER_PORT)]
    if broker_cluster_ip:
        egress_allow.append(
            GymHostEgressRule(host=broker_cluster_ip, port=DEFAULT_BROKER_PORT)
        )
    return GymHostSpec(
        job_id=job_id,
        runtime_image=RUNTIME_IMAGE,
        environment_mount=GymHostVolumeMount(
            pvc_claim=env_claim,
            mount_path="/job/environment",
            read_only=True,
        ),
        workspace_mount=GymHostVolumeMount(
            pvc_claim=work_claim,
            mount_path="/job/work",
            read_only=False,
        ),
        egress_allow=tuple(egress_allow),
        bootstrap_env=build_bootstrap_env(
            job_id,
            "/job/environment",
            "/job/work",
            endpoint.url,
            endpoint.token,
            1_048_576,
            1_048_576,
        ),
        ttl_s=1800,
        ready_timeout_s=READY_TIMEOUT_S,
        resources=STUB_SANDBOX_RESOURCES,
        runtime_http_port=8080,
        allow_internet=False,
        resolver_addresses=resolver_addresses,
        entrypoint=tuple(stub_entrypoint()),
        labels={"purpose": "gym-broker-live", "component": "sandboxed-gym"},
    )


@pytest.mark.asyncio
async def test_live_broker_provisions_episodes_for_the_job_sandbox(
    live_target,
    port_forward,
    ephemeral_pvcs,
):
    domain, api_key = port_forward
    env_claim, work_claim = ephemeral_pvcs
    job_id = f"gym-broker-{uuid.uuid4().hex[:8]}"

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    broker_config = EpisodeBrokerConfig.model_validate(
        {
            "job_id": job_id,
            **episode_broker_block(
                domain, api_key, DEFAULT_BROKER_HOST, DEFAULT_BROKER_PORT
            ),
        }
    )
    broker_actor, endpoint = start_episode_broker(
        broker_config,
        node_id=ray.get_runtime_context().get_node_id(),
    )
    try:
        assert endpoint.host == DEFAULT_BROKER_HOST
        cluster_ip = broker_service_cluster_ip(live_target, DEFAULT_BROKER_HOST)
        spec = _host_spec(
            job_id,
            env_claim,
            work_claim,
            endpoint,
            cluster_ip,
            cluster_resolver_addresses(live_target),
        )
        # The OpenSandbox credential the broker holds must not have followed the endpoint into the
        # job sandbox's environment.
        assert not any(key.startswith("OPENSANDBOX_") for key in spec.bootstrap_env)

        provider = OpenSandboxGymHostProvider(
            connection={
                "domain": domain,
                "api_key": api_key,
                "protocol": "http",
                "use_server_proxy": True,
                "request_timeout_s": int(READY_TIMEOUT_S),
            },
            create={"timeout_s": READY_TIMEOUT_S, "skip_health_check": False},
        )
        handle = await provider.create_host(spec)
        try:
            await provider.wait_ready(handle, timeout_s=READY_TIMEOUT_S)

            encoded = base64.b64encode(
                _probe_source(job_id).encode("utf-8")
            ).decode("ascii")
            written = await provider.exec_host(
                handle,
                "python -c \"import base64; "
                f"open({PROBE_PATH!r},'wb').write(base64.b64decode('{encoded}'))\"",
            )
            assert written.return_code == 0, written.stderr or written.stdout

            probe = await provider.exec_host(
                handle, f"python {PROBE_PATH}", timeout_s=READY_TIMEOUT_S
            )
            output = f"{probe.stdout or ''}\n{probe.stderr or ''}"
            assert probe.return_code == 0, output
            assert "BROKER_PROBE_OK" in output, output
            assert "DELETE=200" in output, output
        finally:
            await provider.destroy_host(handle)
    finally:
        try:
            ray.get(broker_actor.shutdown.remote(), timeout=120)
        except Exception:
            pass
        ray.kill(broker_actor)
