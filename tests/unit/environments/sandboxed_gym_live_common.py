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

"""Shared helpers for live sandboxed Gym tests (host + Ray actor)."""

import base64
import os
import socket
import subprocess
import textwrap
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pytest


_LIVE_RUNTIME_PATH = Path(__file__).with_name("sandboxed_gym_host_live_runtime.py")
_LIVE_RUNTIME_SOURCE = _LIVE_RUNTIME_PATH.read_text(encoding="utf-8")
_GYM_HOST_RUNTIME_PATH = (
    Path(__file__).resolve().parents[3]
    / "nemo_rl/environments/sandbox/gym_host_runtime.py"
)
_GYM_HOST_RUNTIME_SOURCE = _GYM_HOST_RUNTIME_PATH.read_text(encoding="utf-8")
NEMO_RL_IMAGE_GIT_ROOT = "/opt/nemo-rl"
# Baked venv in nvcr.io/nvidia/nemo-rl. It carries nemo_rl but *not* the nemo_gym
# extra, and the image ships uv under /root/.local/bin (not /bin/uv or PATH).
NEMO_RL_IMAGE_VENV = "/opt/nemo_rl_venv"
REAL_GYM_SANDBOX_RESOURCES = {"cpu": "2", "memory": "8Gi"}
STUB_SANDBOX_RESOURCES = {"cpu": "250m", "memory": "512Mi"}

# Host image candidates (Gym lives under the image's NeMo-RL ``3rdparty`` and is
# installed into the baked venv by ``real_gym_entrypoint`` — same package set as
# ``PY_EXECUTABLES.NEMO_GYM`` / colocated ``NemoGym``). Override with
# ``OPENSANDBOX_LIVE_RUNTIME_IMAGE``. Default stays slim+stub for host-provider
# plumbing tests that do not need a real Gym tree.
NEMO_RL_BASE_IMAGE = "nvcr.io/nvidia/nemo-rl:v0.7.0"
NMP_RL_TRAINING_IMAGE = (
    "nvcr.io/0921617854601259/nemo-platform-dev/nmp-rl-training:latest"
)
RUNTIME_IMAGE = os.environ.get(
    "OPENSANDBOX_LIVE_RUNTIME_IMAGE", "docker.io/library/python:3.12-slim"
)
READY_TIMEOUT_S = float(os.environ.get("OPENSANDBOX_LIVE_READY_TIMEOUT_S", "1200"))

# Real Gym bootstrap creates per-server venvs and therefore needs public package
# registries. All policies remain default-deny; this switch adds only safe public
# CIDRs and DNS suffixes. Set 0 once those venvs are prebaked.
ALLOW_INTERNET = os.environ.get("SANDBOXED_GYM_LIVE_ALLOW_INTERNET", "1") == "1"
STORAGE_ACCESS_MODE = os.environ.get(
    "OPENSANDBOX_LIVE_STORAGE_ACCESS_MODE", "ReadWriteOnce"
)

# External vLLM / IGW used as Gym ``policy_base_url`` for live sandboxed tests.
#
# Internal requests made against external ingresses are short circuted but the port is maintained, so we use http:// here and port 443
NMP_TEMP1_BASE_URL = os.environ.get(
    "NMP_TEMP1_BASE_URL", "http://nmp-temp1.dev.aire.nvidia.com:443"
).rstrip("/")
DEFAULT_POLICY_MODEL_NAME = os.environ.get(
    "SANDBOXED_GYM_POLICY_MODEL_NAME", "default/qwen3-5-2b"
)
DEFAULT_POLICY_BASE_URL = os.environ.get(
    "SANDBOXED_GYM_POLICY_BASE_URL",
    f"{NMP_TEMP1_BASE_URL}/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
)
_POLICY_URL = urlparse(DEFAULT_POLICY_BASE_URL)
DEFAULT_POLICY_HOST = os.environ.get(
    "SANDBOXED_GYM_POLICY_HOST", _POLICY_URL.hostname or ""
)
DEFAULT_POLICY_PORT = int(
    os.environ.get("SANDBOXED_GYM_POLICY_PORT", "")
    or _POLICY_URL.port
    or (443 if _POLICY_URL.scheme == "https" else 80)
)

# Episode broker reachability from OpenSandbox pods: workbox IP is not
# cluster-routable. Live tests advertise a reverse-tunnel Service in
# ``soluwalana-dev`` (set up from the workbox) and allow that FQDN.
DEFAULT_BROKER_HOST = os.environ.get(
    "SANDBOXED_GYM_BROKER_HOST",
    "broker-reverse-tunnel.soluwalana-dev.svc.cluster.local",
)
DEFAULT_BROKER_PORT = int(os.environ.get("SANDBOXED_GYM_BROKER_PORT", "51234"))

# Episode images the broker will provision. Deliberately narrow: the broker fails closed, so
# anything an environment asks for outside these prefixes comes back as a 403 the test can read.
# ``docker.io/library/`` covers the smoke image; the SWE-bench prefix covers ``mini_swe_agent_2``.
EPISODE_IMAGE_PREFIXES = tuple(
    prefix
    for prefix in os.environ.get(
        "SANDBOXED_GYM_EPISODE_IMAGE_PREFIXES",
        "docker.io/library/,docker.io/swebench/",
    ).split(",")
    if prefix.strip()
)
EPISODE_SMOKE_IMAGE = os.environ.get(
    "SANDBOXED_GYM_EPISODE_IMAGE", "docker.io/library/python:3.12-slim"
)


def live_opensandbox_enabled() -> bool:
    return os.environ.get("LIVE_OPENSANDBOX") == "1"


def require_env(name: str) -> str:
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


def build_live_target() -> LiveTarget:
    expect = os.environ.get("OPENSANDBOX_EXPECT_RUNTIME_CLASS", "").strip()
    return LiveTarget(
        kube_context=require_env("OPENSANDBOX_KUBE_CONTEXT"),
        system_ns=require_env("OPENSANDBOX_SYSTEM_NS"),
        server_svc=require_env("OPENSANDBOX_SERVER_SVC"),
        api_secret=require_env("OPENSANDBOX_API_SECRET"),
        workload_ns=require_env("OPENSANDBOX_WORKLOAD_NS"),
        expect_runtime_class=expect or None,
    )


def kubectl(
    target: LiveTarget, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    cmd = ["kubectl", "--context", target.kube_context, *args]
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def api_key(target: LiveTarget) -> str:
    encoded = kubectl(
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


def port_forward_server(target: LiveTarget) -> Iterator[tuple[str, str]]:
    """Port-forward the OpenSandbox server Service; yield (domain, api_key)."""
    key = api_key(target)
    port = free_port()
    log_path = f"/tmp/osb-pf-{target.server_svc}-{port}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [
            "kubectl",
            "--context",
            target.kube_context,
            "-n",
            target.system_ns,
            "port-forward",
            f"svc/{target.server_svc}",
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
        yield domain, key
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()


def create_ephemeral_pvcs(target: LiveTarget) -> Iterator[tuple[str, str]]:
    """Create RO-env and RW-work PVCs; delete on exit."""
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
              namespace: {target.workload_ns}
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
                target.kube_context,
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
            kubectl(
                target,
                "delete",
                "pvc",
                claim,
                "-n",
                target.workload_ns,
                "--wait=false",
                check=False,
            )


def stub_entrypoint() -> list[str]:
    return ["python", "-c", _LIVE_RUNTIME_SOURCE]


def real_gym_entrypoint() -> list[str]:
    """Start ``gym_host_runtime`` inside the NeMo-RL image.

    The image's venv already resolves every heavy dependency, so we add only the
    ``nemo_gym`` extra into it (``--inexact`` keeps the existing packages, and the
    baked uv cache makes it an offline ~5s install of ~20 small wheels). Building a
    fresh ``uv run`` environment instead would re-materialize torch and friends.

    ``uv`` is looked up across known locations because the published image puts it
    in ``/root/.local/bin`` and OpenSandbox's bootstrap shell has no login PATH.
    """
    encoded = base64.b64encode(_GYM_HOST_RUNTIME_SOURCE.encode("utf-8")).decode("ascii")
    script = textwrap.dedent(f"""
        set -eu
        venv={NEMO_RL_IMAGE_VENV}
        root={NEMO_RL_IMAGE_GIT_ROOT}
        uv=""
        for c in "$(command -v uv || true)" /root/.local/bin/uv /usr/local/bin/uv /bin/uv; do
            if [ -n "$c" ] && [ -x "$c" ]; then uv="$c"; break; fi
        done
        if [ -z "$uv" ]; then
            echo "gym-host: no uv binary found in image" >&2
            exit 127
        fi
        echo "gym-host: syncing nemo_gym extra into $venv using $uv" >&2
        VIRTUAL_ENV="$venv" UV_PROJECT_ENVIRONMENT="$venv" UV_OFFLINE=1 \\
            "$uv" sync --locked --inexact --active --no-group build \\
            --extra nemo_gym --directory "$root" >&2
        echo "{encoded}" | base64 -d > /tmp/gym_host_runtime.py
        cd "$root"
        exec "$venv/bin/python" /tmp/gym_host_runtime.py
    """).strip()
    return ["/bin/sh", "-c", script]


def colocated_parity_global_config_dict() -> dict:
    """Inline Gym global config matching ``test_nemo_gym.py::nemo_gym`` (example_multi_step + vLLM)."""
    from yaml import safe_load

    yaml_str = r"""example_multi_step_resources_server:
  resources_servers:
    example_multi_step:
      entrypoint: app.py
      domain: instruction_following
example_multi_step_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_multi_step_resources_server
      model_server:
        type: responses_api_models
        name: openai_model
openai_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: true
      uses_reasoning_parser: true
global_aiohttp_client_request_debug: true
"""
    loaded = safe_load(yaml_str)
    assert isinstance(loaded, dict)
    return loaded


def live_actor_py_executable() -> str:
    """Ray ``py_executable`` for sandboxed Gym actor live tests.

    Prefer ``SANDBOXED_GYM_ACTOR_PY_EXECUTABLE``. Otherwise use the repo
    ``.venv/bin/python`` when present: Ray's worker script lives under the
    ``uv``-managed env cache, and ``uv run`` rejects executing from inside
    ``~/.cache/uv``. Production training still uses ``PY_EXECUTABLES.NEMO_GYM``.
    """
    override = os.environ.get("SANDBOXED_GYM_ACTOR_PY_EXECUTABLE", "").strip()
    if override:
        return override
    repo_root = _LIVE_RUNTIME_PATH.resolve().parents[3]
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )
    from nemo_rl.environments.sandbox.nemo_gym_actor import (
        SANDBOXED_GYM_ACTOR_FQN,
    )

    return get_actor_python_env(SANDBOXED_GYM_ACTOR_FQN)


CLUSTER_DNS_SERVICES: tuple[tuple[str, str], ...] = (
    ("kube-system", "kube-dns"),
    ("kube-system", "coredns"),
)


def service_addresses(target: LiveTarget, namespace: str, name: str) -> tuple[str, ...]:
    """Return the ClusterIP a Service answers on.

    The ClusterIP is the whole answer: the sandbox's rules match the destination it dials, before
    kube-proxy rewrites it to an endpoint, so the pod IPs behind the Service never need allowing.
    A deployed actor gets this from the cluster resolver; the workbox cannot resolve cluster names
    at all, so it asks the API server instead.
    """
    cluster_ip = kubectl(
        target,
        "get",
        "svc",
        name,
        "-n",
        namespace,
        "-o",
        "jsonpath={.spec.clusterIP}",
        check=False,
    ).stdout.strip()
    return (cluster_ip,) if cluster_ip and cluster_ip != "None" else ()


def cluster_resolver_addresses(target: LiveTarget) -> tuple[str, ...]:
    """Look up the addresses a sandbox's DNS queries actually reach.

    The policy defaults to the trusted process's own nameservers, which is right for a deployed
    actor and wrong from a workbox: the workbox resolves through a systemd stub on loopback while
    the sandbox asks kube-dns. Left denied, every lookup is dropped and nothing resolves --
    including the allowed names.
    """
    for namespace, name in CLUSTER_DNS_SERVICES:
        addresses = service_addresses(target, namespace, name)
        if addresses:
            return addresses
    return ()


def broker_service_addresses(target: LiveTarget, broker_host: str) -> tuple[str, ...]:
    """Address behind a ``<name>.<namespace>.svc.cluster.local`` broker host."""
    parts = broker_host.split(".")
    if len(parts) < 3 or parts[2:5] != ["svc", "cluster", "local"]:
        return ()
    return service_addresses(target, parts[1], parts[0])


def episode_broker_block(
    domain: str,
    api_key_value: str,
    broker_host: str,
    broker_port: int,
    resolver_addresses: tuple[str, ...] | None = None,
) -> dict:
    """Build the ``env.nemo_gym.episode_broker`` block for live tests.

    The broker runs in the trusted actor's process and is the only side holding the OpenSandbox
    credential, so its connection settings are configured here rather than reaching the job
    sandbox. ``host``/``port`` advertise the reverse-tunnel Service DNS instead of the workbox IP,
    which OpenSandbox pods cannot route to.
    """
    block: dict = {
        "host": broker_host,
        "port": broker_port,
        "backend": "opensandbox",
        "backend_options": {
            "connection": {
                "domain": domain,
                "api_key": api_key_value,
                "protocol": "http",
                "use_server_proxy": True,
                "request_timeout_s": int(READY_TIMEOUT_S),
            },
            "create": {
                "timeout_s": READY_TIMEOUT_S,
                "request_timeout_s": int(READY_TIMEOUT_S),
                "skip_health_check": False,
            },
        },
        "approved_image_prefixes": EPISODE_IMAGE_PREFIXES,
        # SWE-bench images pull for minutes and the harness holds one episode for a whole
        # trajectory, so the defaults (300s ready, 3600s TTL) would clamp a real run.
        "max_ready_timeout_s": READY_TIMEOUT_S,
        "default_ttl_s": 1800.0,
        "max_ttl_s": 18000.0,
        "allow_internet": ALLOW_INTERNET,
    }
    if resolver_addresses is not None:
        block["resolver_addresses"] = resolver_addresses
    return block


def sandboxed_env_block(
    domain: str,
    api_key_value: str,
    env_claim: str,
    work_claim: str,
    job_id: str | None = None,
    with_stub_entrypoint: bool = True,
    policy_host: str | None = None,
    policy_port: int | None = None,
    broker_host: str | None = None,
    broker_port: int | None = None,
    broker_addresses: tuple[str, ...] = (),
    resolver_addresses: tuple[str, ...] | None = None,
) -> dict:
    """Build the ``env.nemo_gym`` sandboxed block used by actor / training jobs."""
    job = job_id or f"gym-host-{uuid.uuid4().hex[:8]}"
    vllm_host = policy_host or DEFAULT_POLICY_HOST
    vllm_port = DEFAULT_POLICY_PORT if policy_port is None else policy_port
    tunnel_host = broker_host or DEFAULT_BROKER_HOST
    tunnel_port = DEFAULT_BROKER_PORT if broker_port is None else broker_port
    runtime_image = (
        RUNTIME_IMAGE
        if with_stub_entrypoint
        else os.environ.get("OPENSANDBOX_LIVE_RUNTIME_IMAGE", NEMO_RL_BASE_IMAGE)
    )
    egress_allow = [
        {"host": vllm_host, "port": vllm_port},
        {"host": tunnel_host, "port": tunnel_port},
    ]
    egress_allow.extend(
        {"host": address, "port": tunnel_port} for address in broker_addresses
    )
    network_policy: dict = {"egress_allow": egress_allow}
    if resolver_addresses is not None:
        network_policy["resolver_addresses"] = resolver_addresses
    sandbox: dict = {
        "image": runtime_image,
        "environment_pvc_claim": env_claim,
        "workspace_pvc_claim": work_claim,
        "network_policy": network_policy,
        "resources": (
            STUB_SANDBOX_RESOURCES
            if with_stub_entrypoint
            else REAL_GYM_SANDBOX_RESOURCES
        ),
        "ttl_s": 1800,
        "ready_timeout_s": READY_TIMEOUT_S,
        "rollout_timeout_s": READY_TIMEOUT_S,
        "allow_internet": ALLOW_INTERNET,
        "host_provider_options": {
            "connection": {
                "domain": domain,
                "api_key": api_key_value,
                "protocol": "http",
                "use_server_proxy": True,
                "request_timeout_s": int(READY_TIMEOUT_S),
            },
            "create": {
                "timeout_s": READY_TIMEOUT_S,
                "request_timeout_s": int(READY_TIMEOUT_S),
                "skip_health_check": False,
            },
        },
    }
    if with_stub_entrypoint:
        sandbox["entrypoint"] = stub_entrypoint()
    else:
        sandbox["entrypoint"] = real_gym_entrypoint()
    block: dict = {
        "sandboxed": True,
        "host_provider": "opensandbox",
        "job_id": job,
        "environment_path": "/job/environment",
        "sandbox": sandbox,
        "episode_broker": episode_broker_block(
            domain, api_key_value, tunnel_host, tunnel_port, resolver_addresses
        ),
    }
    if with_stub_entrypoint:
        block["config_paths"] = [
            "resources_servers/example_multi_step/configs/example_multi_step.yaml"
        ]
    else:
        block["initial_global_config_dict"] = colocated_parity_global_config_dict()
    return block
