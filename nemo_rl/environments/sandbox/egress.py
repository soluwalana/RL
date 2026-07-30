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

"""Allow-only egress policies for untrusted sandboxes.

Every policy is deny-by-default. Callers may add whitelist targets, but cannot
submit deny rules. When ``allow_internet`` is true, the whitelist also receives
safe public IPv4 CIDRs and public DNS suffixes; private, metadata, loopback,
CGNAT, multicast, and reserved addresses remain blocked by omission.
"""

from __future__ import annotations

import ipaddress
from functools import lru_cache
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict


class EpisodeEgressRule(BaseModel):
    """One whitelist rule. ``target`` is an FQDN, IP, or CIDR."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str
    action: Literal["allow"] = "allow"


class EpisodeEgressPolicy(BaseModel):
    """Egress policy applied to a sandbox.

    Always sent explicitly, even when it has no rules. A backend that treats an absent policy as
    allow-all would otherwise grant unrestricted network access by default, and on OpenSandbox
    create is the only moment the default action can be established at all -- the post-create
    egress calls merge rules but preserve ``defaultAction``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_action: Literal["deny"] = "deny"
    rules: tuple[EpisodeEgressRule, ...] = ()


DEFAULT_PUBLIC_DNS_SUFFIXES: tuple[str, ...] = ("*.com", "*.org")

_NON_PUBLIC_IPV4_CIDRS: tuple[str, ...] = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
)


@lru_cache(maxsize=1)
def safe_public_ipv4_cidrs() -> tuple[str, ...]:
    """Return public IPv4 as a minimal CIDR whitelist, never ``0.0.0.0/0``."""
    remaining = [ipaddress.ip_network("0.0.0.0/0")]
    for excluded_raw in _NON_PUBLIC_IPV4_CIDRS:
        excluded = ipaddress.ip_network(excluded_raw)
        updated: list[ipaddress.IPv4Network] = []
        for network in remaining:
            if network.subnet_of(excluded):
                continue
            if excluded.subnet_of(network):
                updated.extend(network.address_exclude(excluded))
            else:
                updated.append(network)
        remaining = updated
    return tuple(str(network) for network in remaining)


def _unique_targets(targets: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            target.strip().rstrip(".")
            for target in targets
            if target and target.strip().rstrip(".")
        )
    )


def build_sandbox_allow_targets(
    *,
    endpoint_targets: Iterable[str] = (),
    allow_internet: bool = False,
    public_dns_allow: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Build a sandbox whitelist.

    Public CIDRs and DNS suffixes are added *only* when ``allow_internet=True``.
    """
    targets = list(endpoint_targets)
    if allow_internet:
        targets.extend(DEFAULT_PUBLIC_DNS_SUFFIXES)
        targets.extend(public_dns_allow or ())
        targets.extend(safe_public_ipv4_cidrs())
    return _unique_targets(targets)


def build_sandbox_egress_policy(
    *,
    endpoint_targets: Iterable[str] = (),
    allow_internet: bool = False,
    public_dns_allow: tuple[str, ...] | None = None,
) -> EpisodeEgressPolicy:
    """Build a deny-by-default policy containing allow rules only."""
    allow_targets = build_sandbox_allow_targets(
        endpoint_targets=endpoint_targets,
        allow_internet=allow_internet,
        public_dns_allow=public_dns_allow,
    )
    return EpisodeEgressPolicy(
        default_action="deny",
        rules=tuple(
            EpisodeEgressRule(target=target, action="allow") for target in allow_targets
        ),
    )


def build_egress_policy(*, allow_targets: tuple[str, ...] = ()) -> EpisodeEgressPolicy:
    """Compatibility wrapper for an explicit, strict whitelist."""
    return build_sandbox_egress_policy(endpoint_targets=allow_targets)
