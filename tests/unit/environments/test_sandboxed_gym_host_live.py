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

"""Live OpenSandboxGymHostProvider tests.

Gated behind ``LIVE_OPENSANDBOX=1``. Deployment targets are supplied only via
environment variables (no cluster-specific defaults in-tree).

Required when gated on:

* ``OPENSANDBOX_KUBE_CONTEXT`` — kubectl context
* ``OPENSANDBOX_SERVER_SVC`` — OpenSandbox server Service name
* ``OPENSANDBOX_API_SECRET`` — Secret holding the server API key (key ``api-key``)
* ``OPENSANDBOX_WORKLOAD_NS`` — namespace where sandboxes and test PVCs are created
* ``OPENSANDBOX_SYSTEM_NS`` — namespace of the server Service and API-key Secret

Optional:

* ``OPENSANDBOX_LIVE_STORAGE_CLASS`` — PVC storage class (cluster default if unset)
* ``OPENSANDBOX_LIVE_STORAGE_ACCESS_MODE`` — PVC access mode (default ``ReadWriteOnce``)
* ``OPENSANDBOX_LIVE_RUNTIME_IMAGE`` — sandbox image (default ``python:3.12-slim``)
* ``OPENSANDBOX_LIVE_READY_TIMEOUT_S`` — create/ready timeout seconds (default ``600``)
* ``OPENSANDBOX_EXPECT_RUNTIME_CLASS`` — assert pod ``runtimeClassName``; empty means unset

Exercises create → wait_ready → proxy health/rollout → mounts → public package
install → cluster-private / metadata denial → destroy.
"""

import base64
import json
import os
import socket
import subprocess
import textwrap
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from nemo_rl.environments.sandbox.host.models import (
    GymHostEgressRule,
    GymHostHandle,
    GymHostSpec,
    GymHostVolumeMount,
)
from nemo_rl.environments.sandbox.host.opensandbox import OpenSandboxGymHostProvider


pytestmark = [
    pytest.mark.nemo_gym,
    pytest.mark.skipif(
        os.environ.get("LIVE_OPENSANDBOX") != "1",
        reason="Set LIVE_OPENSANDBOX=1 to run live OpenSandbox host tests",
    ),
]

_LIVE_RUNTIME_PATH = Path(__file__).with_name("sandboxed_gym_host_live_runtime.py")
_LIVE_RUNTIME_SOURCE = _LIVE_RUNTIME_PATH.read_text(encoding="utf-8")

RUNTIME_IMAGE = os.environ.get(
    "OPENSANDBOX_LIVE_RUNTIME_IMAGE", "docker.io/library/python:3.12-slim"
)
READY_TIMEOUT_S = float(os.environ.get("OPENSANDBOX_LIVE_READY_TIMEOUT_S", "600"))
STORAGE_ACCESS_MODE = os.environ.get(
    "OPENSANDBOX_LIVE_STORAGE_ACCESS_MODE", "ReadWriteOnce"
)
# Tiny pure-Python package; proves DNS + public HTTPS work end-to-end.
_PIP_PACKAGE = "six==1.17.0"


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise pytest.UsageError(f"{name} must be set when LIVE_OPENSANDBOX=1")
    return value


@dataclass(frozen=True)
class LiveTarget:
    kube_context: str
    system_ns: str
    server_svc: str
    api_secret: str
    workload_ns: str
    expect_runtime_class: str | None


@pytest.fixture
def live_target() -> LiveTarget:
    expect = os.environ.get("OPENSANDBOX_EXPECT_RUNTIME_CLASS", "").strip()
    return LiveTarget(
        kube_context=_require_env("OPENSANDBOX_KUBE_CONTEXT"),
        system_ns=_require_env("OPENSANDBOX_SYSTEM_NS"),
        server_svc=_require_env("OPENSANDBOX_SERVER_SVC"),
        api_secret=_require_env("OPENSANDBOX_API_SECRET"),
        workload_ns=_require_env("OPENSANDBOX_WORKLOAD_NS"),
        expect_runtime_class=expect or None,
    )


def _kubectl(
    target: LiveTarget, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    cmd = ["kubectl", "--context", target.kube_context, *args]
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _api_key(target: LiveTarget) -> str:
    encoded = _kubectl(
        target,
        "get",
        "secret",
        "-n",
        target.system_ns,
        target.api_secret,
        "-o",
        "jsonpath={.data.api-key}",
    ).stdout.strip()
    if not encoded:
        raise RuntimeError(f"empty api-key in secret {target.api_secret}")
    return base64.b64decode(encoded).decode("utf-8")


def _server_cluster_ip(target: LiveTarget) -> str:
    ip = _kubectl(
        target,
        "get",
        "svc",
        target.server_svc,
        "-n",
        target.system_ns,
        "-o",
        "jsonpath={.spec.clusterIP}",
    ).stdout.strip()
    if not ip:
        raise RuntimeError(f"no clusterIP for svc/{target.server_svc}")
    return ip


def _server_fqdn(target: LiveTarget) -> str:
    return f"{target.server_svc}.{target.system_ns}.svc.cluster.local"


@pytest.fixture
def port_forward(live_target: LiveTarget) -> Iterator[tuple[str, str]]:
    """Port-forward the OpenSandbox server Service; yield (domain, api_key)."""
    api_key = _api_key(live_target)
    port = _free_port()
    log_path = f"/tmp/osb-pf-{live_target.server_svc}-{port}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [
            "kubectl",
            "--context",
            live_target.kube_context,
            "-n",
            live_target.system_ns,
            "port-forward",
            f"svc/{live_target.server_svc}",
            f"{port}:80",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    domain = f"127.0.0.1:{port}"
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"port-forward exited early; see {log_path}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"port-forward not ready; see {log_path}")
        yield domain, api_key
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()


@pytest.fixture
def ephemeral_pvcs(live_target: LiveTarget) -> Iterator[tuple[str, str]]:
    """Create RO-env and RW-work PVCs in the workload namespace; delete on exit."""
    suffix = uuid.uuid4().hex[:8]
    env_claim = f"gym-host-env-{suffix}"
    work_claim = f"gym-host-work-{suffix}"
    storage_class = os.environ.get("OPENSANDBOX_LIVE_STORAGE_CLASS", "").strip()
    storage_class_line = (
        f"              storageClassName: {storage_class}\n" if storage_class else ""
    )
    for claim, size in ((env_claim, "1Gi"), (work_claim, "1Gi")):
        manifest = textwrap.dedent(
            f"""\
            apiVersion: v1
            kind: PersistentVolumeClaim
            metadata:
              name: {claim}
              namespace: {live_target.workload_ns}
              labels:
                nemo-rl-live-test: gym-host
            spec:
              accessModes: ["{STORAGE_ACCESS_MODE}"]
{storage_class_line}              resources:
                requests:
                  storage: {size}
            """
        )
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                live_target.kube_context,
                "apply",
                "-f",
                "-",
            ],
            input=manifest,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"failed to create PVC {claim}: {result.stderr or result.stdout}"
            )
    try:
        yield env_claim, work_claim
    finally:
        for claim in (env_claim, work_claim):
            _kubectl(
                live_target,
                "delete",
                "pvc",
                claim,
                "-n",
                live_target.workload_ns,
                "--wait=false",
                check=False,
            )


def _host_spec(env_claim: str, work_claim: str) -> GymHostSpec:
    job_id = f"gym-host-{uuid.uuid4().hex[:8]}"
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
        egress_allow=(GymHostEgressRule(host="broker.svc.cluster.local", port=51234),),
        bootstrap_env={
            "NMP_JOB_ID": job_id,
            "NMP_ENVIRONMENT_PATH": "/job/environment",
            "NMP_WORK_PATH": "/job/work",
            "NMP_BROKER_URL": "http://broker.svc.cluster.local:51234",
            "NMP_BROKER_TOKEN": "live-test-token",
            "NMP_MAX_REQUEST_BYTES": "1048576",
            "NMP_MAX_RESPONSE_BYTES": "1048576",
        },
        ttl_s=1800,
        resources={"cpu": "250m", "memory": "512Mi"},
        runtime_http_port=8080,
        allow_internet=True,
        entrypoint=("python", "-c", _LIVE_RUNTIME_SOURCE),
        labels={"purpose": "gym-host-live", "component": "sandboxed-gym"},
    )


def _find_sandbox_pod(target: LiveTarget, host_id: str) -> str:
    deadline = time.time() + 180
    while time.time() < deadline:
        out = _kubectl(
            target,
            "get",
            "pods",
            "-n",
            target.workload_ns,
            "-o",
            "jsonpath={range .items[*]}{.metadata.name}{'\\t'}{.metadata.ownerReferences[0].name}{'\\t'}{.status.phase}{'\\n'}{end}",
            check=False,
        ).stdout
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            name, owner, phase = parts
            if phase != "Running":
                continue
            if owner == host_id or name.startswith(host_id):
                return name
        time.sleep(3)
    raise TimeoutError(
        f"no Running pod for sandbox {host_id} in {target.workload_ns}"
    )


async def _run_command(
    provider: OpenSandboxGymHostProvider,
    handle: GymHostHandle,
    command: str,
) -> tuple[int | None, str, str]:
    """Run a shell command in the sandbox; return (exit_code, stdout, stderr)."""
    execution = await provider.exec_host(handle, command)
    return execution.return_code, execution.stdout or "", execution.stderr or ""


def _proxy_get_json(url: str, headers: Mapping[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET", headers=dict(headers))
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _proxy_post_json(
    url: str, headers: Mapping[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


@pytest.mark.asyncio
async def test_live_host_functionality(
    live_target: LiveTarget,
    port_forward: tuple[str, str],
    ephemeral_pvcs: tuple[str, str],
):
    domain, api_key = port_forward
    env_claim, work_claim = ephemeral_pvcs
    spec = _host_spec(env_claim, work_claim)
    server_ip = _server_cluster_ip(live_target)
    server_dns = _server_fqdn(live_target)

    assert not any(k.startswith("OPENSANDBOX_") for k in spec.bootstrap_env)
    assert "OPENSANDBOX_API_KEY" not in spec.bootstrap_env

    provider = OpenSandboxGymHostProvider(
        connection={
            "domain": domain,
            "api_key": api_key,
            "protocol": "http",
            "use_server_proxy": True,
        },
        create={
            "timeout_s": READY_TIMEOUT_S,
            "skip_health_check": True,
        },
    )

    handle = await provider.create_host(spec)
    try:
        await provider.wait_ready(handle, timeout_s=READY_TIMEOUT_S)
        assert handle.health_url.startswith("http://")
        assert handle.health_url.endswith("/health")
        assert handle.rollout_url.endswith("/rollouts/run")

        # Proxy: health and rollout through the server, not the pod IP.
        health = _proxy_get_json(handle.health_url, handle.headers)
        assert health.get("status") == "ready"
        rollout = _proxy_post_json(
            handle.rollout_url,
            handle.headers,
            {"examples": [{"prompt": "ping"}, {"prompt": "pong"}]},
        )
        assert rollout["job_id"] == spec.job_id
        assert rollout["environment_path"] == "/job/environment"
        assert rollout["work_path"] == "/job/work"
        assert len(rollout["results"]) == 2

        # Bootstrap env is present; OpenSandbox credentials stay out.
        code, out, err = await _run_command(
            provider,
            handle,
            "python -c \"import os; "
            "print('JOB=' + os.environ.get('NMP_JOB_ID','')); "
            "print('CRED=' + ('LEAK' if any(k.startswith('OPENSANDBOX_') for k in os.environ) else 'CLEAN'))\"",
        )
        assert code == 0, err
        assert f"JOB={spec.job_id}" in out
        assert "CRED=CLEAN" in out
        assert "CRED=LEAK" not in out

        # Mount roles: workspace is writable, environment is read-only.
        code, out, err = await _run_command(
            provider,
            handle,
            "set -e; "
            "echo live-probe > /job/work/probe.txt; "
            "test \"$(cat /job/work/probe.txt)\" = live-probe; "
            "if touch /job/environment/should-fail 2>/dev/null; then echo ENV_WRITABLE; "
            "else echo ENV_RO; fi",
        )
        assert code == 0, err
        assert "ENV_RO" in out
        assert "ENV_WRITABLE" not in out

        # Public egress: DNS + pip install (real package index traffic).
        code, out, err = await _run_command(
            provider,
            handle,
            "python -m pip install --no-cache-dir --disable-pip-version-check "
            f"{_PIP_PACKAGE} "
            "&& python -c \"import six; print(six.__version__)\"",
        )
        assert code == 0, f"pip install failed:\nstdout={out}\nstderr={err}"
        assert "1.17.0" in out

        # East-west: OpenSandbox server ClusterIP and Service DNS must be denied.
        # Metadata link-local must also be denied. Drop the probe via base64 so
        # shell quoting cannot mangle the script.
        deny_script = textwrap.dedent(
            f"""\
            import socket
            import urllib.request

            def tcp(host, port, label):
                try:
                    with socket.create_connection((host, port), timeout=3):
                        print(label + "_OPEN")
                except OSError as exc:
                    print(label + "_BLOCKED:" + type(exc).__name__)

            tcp({server_ip!r}, 80, "SERVER_IP")
            try:
                socket.getaddrinfo({server_dns!r}, 80)
                tcp({server_dns!r}, 80, "SERVER_DNS")
            except OSError as exc:
                print("SERVER_DNS_BLOCKED:" + type(exc).__name__)

            try:
                urllib.request.urlopen("http://169.254.169.254/", timeout=3)
                print("METADATA_OPEN")
            except Exception as exc:
                print("METADATA_BLOCKED:" + type(exc).__name__)
            """
        )
        encoded = base64.b64encode(deny_script.encode("utf-8")).decode("ascii")
        write_code, write_out, write_err = await _run_command(
            provider,
            handle,
            "python -c \"import base64; "
            "open('/job/work/deny_probe.py','wb').write(base64.b64decode('"
            + encoded
            + "'))\"",
        )
        assert write_code == 0, write_err or write_out
        code, out, err = await _run_command(
            provider, handle, "python /job/work/deny_probe.py"
        )
        assert code == 0, err
        assert "SERVER_IP_BLOCKED" in out, out
        assert "SERVER_IP_OPEN" not in out
        assert "SERVER_DNS_OPEN" not in out, out
        assert "METADATA_OPEN" not in out, out
        assert "METADATA_BLOCKED" in out

        pod = _find_sandbox_pod(live_target, handle.host_id)
        runtime_class = _kubectl(
            live_target,
            "get",
            "pod",
            "-n",
            live_target.workload_ns,
            pod,
            "-o",
            "jsonpath={.spec.runtimeClassName}",
        ).stdout.strip()
        if live_target.expect_runtime_class is None:
            assert runtime_class == ""
        else:
            assert runtime_class == live_target.expect_runtime_class
    finally:
        await provider.destroy_host(handle)

        # Destroyed host must no longer answer on the proxy.
        with pytest.raises((urllib.error.URLError, urllib.error.HTTPError, TimeoutError)):
            _proxy_get_json(handle.health_url, handle.headers)
