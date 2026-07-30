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

import ipaddress

import pytest

from nemo_rl.environments.sandbox.egress import safe_public_ipv4_cidrs
from nemo_rl.environments.sandbox.host.models import (
    GymHostEgressRule,
    GymHostSpec,
    GymHostVolumeMount,
    NemoGymSandboxedConfig,
    SandboxConfig,
    SandboxNetworkPolicy,
    build_bootstrap_env,
    validate_bootstrap_env,
)
from nemo_rl.environments.sandbox.host.provider import build_host_egress_policy


def _mount(claim: str, path: str, read_only: bool) -> GymHostVolumeMount:
    return GymHostVolumeMount(
        pvc_claim=claim,
        mount_path=path,
        read_only=read_only,
    )


def test_bootstrap_env_rejects_opensandbox_credentials():
    with pytest.raises(ValueError, match="OPENSANDBOX_API_KEY"):
        validate_bootstrap_env({"OPENSANDBOX_API_KEY": "secret"})


def test_build_bootstrap_env_sets_required_keys():
    env = build_bootstrap_env(
        "job-1",
        "/job/environment",
        "/job/work",
        "http://broker:51234",
        "token",
        1024,
        2048,
        dataset_path="/job/dataset",
    )
    assert env["NMP_JOB_ID"] == "job-1"
    assert env["NMP_BROKER_URL"] == "http://broker:51234"
    assert env["NMP_DATASET_PATH"] == "/job/dataset"
    assert "OPENSANDBOX_API_KEY" not in env


def test_gym_host_spec_enforces_mount_roles():
    with pytest.raises(ValueError, match="environment_mount must be read-only"):
        GymHostSpec(
            job_id="job-1",
            runtime_image="runtime:dev",
            environment_mount=_mount("claim", "/job/environment", False),
            workspace_mount=_mount("claim", "/job/work", False),
        )


def test_egress_rule_accepts_ip_or_fqdn():
    assert GymHostEgressRule(host="10.0.0.5", port=8000).host == "10.0.0.5"
    assert GymHostEgressRule(host="vllm.svc.cluster.local", port=8000).host == (
        "vllm.svc.cluster.local"
    )
    with pytest.raises(ValueError, match="blank"):
        GymHostEgressRule(host="  ", port=8000)


def test_host_egress_policy_allow_internet_is_still_default_deny():
    spec = GymHostSpec(
        job_id="job-1",
        runtime_image="runtime:dev",
        environment_mount=_mount("claim", "/job/environment", True),
        workspace_mount=_mount("claim", "/job/work", False),
        egress_allow=(GymHostEgressRule(host="vllm.svc.cluster.local", port=8000),),
        allow_internet=True,
    )
    policy = build_host_egress_policy(spec)
    assert policy.default_action == "deny"
    assert any(
        r.target == "vllm.svc.cluster.local" and r.action == "allow"
        for r in policy.rules
    )
    assert any(r.target == "*.com" for r in policy.rules)
    assert set(safe_public_ipv4_cidrs()).issubset({r.target for r in policy.rules})
    assert all(r.action == "allow" for r in policy.rules)


def test_host_egress_policy_without_internet_has_endpoints_only():
    spec = GymHostSpec(
        job_id="job-1",
        runtime_image="runtime:dev",
        environment_mount=_mount("claim", "/job/environment", True),
        workspace_mount=_mount("claim", "/job/work", False),
        egress_allow=(GymHostEgressRule(host="broker.svc.cluster.local", port=51234),),
        allow_internet=False,
        public_dns_allow=("*.io",),
    )
    policy = build_host_egress_policy(spec)
    assert policy.default_action == "deny"
    assert len(policy.rules) == 1
    assert policy.rules[0].target == "broker.svc.cluster.local"


def test_host_egress_policy_adds_custom_public_dns_suffixes():
    spec = GymHostSpec(
        job_id="job-1",
        runtime_image="runtime:dev",
        environment_mount=_mount("claim", "/job/environment", True),
        workspace_mount=_mount("claim", "/job/work", False),
        allow_internet=True,
        public_dns_allow=("*.io",),
    )
    targets = {rule.target for rule in build_host_egress_policy(spec).rules}
    assert {"*.com", "*.org", "*.io"} <= targets


@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.1",
        "100.64.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "172.16.0.1",
        "192.168.0.1",
        "198.18.0.1",
        "224.0.0.1",
        "240.0.0.1",
    ],
)
def test_safe_public_ipv4_cidrs_exclude_non_public_space(address):
    addr = ipaddress.ip_address(address)
    networks = [ipaddress.ip_network(cidr) for cidr in safe_public_ipv4_cidrs()]
    assert not any(addr in network for network in networks)


def test_safe_public_ipv4_cidrs_include_public_space_without_allow_all():
    networks = [ipaddress.ip_network(cidr) for cidr in safe_public_ipv4_cidrs()]
    assert any(ipaddress.ip_address("8.8.8.8") in network for network in networks)
    assert ipaddress.ip_network("0.0.0.0/0") not in networks


def test_sandboxed_config_requires_sandbox_block():
    with pytest.raises(ValueError, match="sandbox"):
        NemoGymSandboxedConfig(sandboxed=True)


def _sandboxed_actor_cfg(global_config):
    return {
        "model_name": "meta-llama/Llama-3.1-8B",
        "base_urls": ["http://vllm-0.svc.cluster.local:8000/v1"],
        "initial_global_config_dict": global_config,
        "sandboxed": {
            "sandboxed": True,
            "sandbox": {
                "image": "runtime:dev",
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env-pvc",
                "workspace_pvc_claim": "work-pvc",
            },
            "job_id": "job-1",
        },
    }


def test_sandbox_global_config_preserves_gym_config():
    from nemo_rl.environments.sandbox.nemo_gym_actor import build_sandbox_global_config

    global_config = build_sandbox_global_config(
        _sandboxed_actor_cfg(
            {
                "config_paths": ["/job/environment/nemo-environment.yaml"],
                "effort_levels": {"low": 1},
                "ray_head_node_address": "10.0.0.5:6379",
            }
        )
    )

    assert global_config["config_paths"] == ["/job/environment/nemo-environment.yaml"]
    assert "effort_levels" not in global_config
    assert "ray_head_node_address" not in global_config
    assert global_config["policy_model_name"] == "meta-llama/Llama-3.1-8B"
    assert global_config["policy_base_url"] == [
        "http://vllm-0.svc.cluster.local:8000/v1"
    ]
    assert global_config["default_host"] == "127.0.0.1"
    assert global_config["port_range_low"] < global_config["port_range_high"]


def test_gym_host_spec_carries_global_config_in_bootstrap():
    import json

    from nemo_rl.environments.sandbox.nemo_gym_actor import (
        GYM_GLOBAL_CONFIG_ENV_KEY,
        _gym_host_spec_from_config,
    )

    cfg = _sandboxed_actor_cfg({"config_paths": ["/job/environment/env.yaml"]})
    sandboxed = NemoGymSandboxedConfig.model_validate(cfg["sandboxed"])
    spec = _gym_host_spec_from_config(
        cfg,
        sandboxed,
        "http://broker.svc.cluster.local:51234",
        "token",
        "broker.svc.cluster.local",
        51234,
    )

    embedded = json.loads(spec.bootstrap_env[GYM_GLOBAL_CONFIG_ENV_KEY])
    assert embedded["config_paths"] == ["/job/environment/env.yaml"]
    assert any(rule.host == "broker.svc.cluster.local" for rule in spec.egress_allow)


def test_sandbox_config_round_trip():
    cfg = SandboxConfig(
        image="runtime:dev",
        network_policy=SandboxNetworkPolicy(
            egress_allow=[GymHostEgressRule(host="vllm.svc", port=8000)]
        ),
        environment_pvc_claim="env-pvc",
        workspace_pvc_claim="work-pvc",
    )
    assert cfg.env_mount_path == "/job/environment"
    assert cfg.network_policy.egress_allow[0].host == "vllm.svc"
    assert cfg.network_policy.public_dns_allow is None


def test_collect_gym_host_egress_allows_discovers_and_dedupes():
    from nemo_rl.environments.sandbox.nemo_gym_actor import (
        collect_gym_host_egress_allows,
    )

    rules = collect_gym_host_egress_allows(
        configured=[
            GymHostEgressRule(host="vllm.svc", port=8000),
            GymHostEgressRule(host="broker.svc", port=51234),
        ],
        broker_host="broker.svc",
        broker_port=51234,
        base_urls=[
            "http://vllm.svc:8000/v1",
            "https://64.181.219.176/v1",
        ],
    )

    assert [(rule.host, rule.port) for rule in rules] == [
        ("vllm.svc", 8000),
        ("broker.svc", 51234),
        ("64.181.219.176", 443),
    ]
