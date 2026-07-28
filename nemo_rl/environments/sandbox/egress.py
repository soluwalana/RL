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

"""Egress profile for sandboxes that run untrusted environment code.

The threat this addresses is **east-west**: user code reaching another tenant's pods or Services,
or the cloud metadata endpoint. Public egress is deliberately permitted -- graders legitimately
install packages mid-episode -- so the shape is allow-by-default with an explicit deny list of
cluster-internal ranges, which is what RFC section 5 describes ("allow public tool egress ... but
deny Pod/Service CIDRs, node-local, and metadata").

Two properties of that shape are worth stating plainly, because they drive how it is deployed:

* **It fails open.** Anything missing from the deny list is reachable. The defaults below cover
  what a standard cluster uses, but the authoritative ranges are cluster fact and the platform
  should supply them.
* **It is not the primary control.** A Kubernetes NetworkPolicy over the sandbox pods -- selected
  by the ``nemo-rl-job-id`` metadata the broker stamps, which OpenSandbox propagates to pod labels
  -- is enforced at the CNI, outside the pod, where a compromised in-pod sidecar cannot bypass it.
  It is also the only layer that can express ports; the OpenSandbox egress sidecar matches on
  domain/IP/CIDR with no port. This profile is the in-band second layer.

The profile is shared rather than episode-specific: the job sandbox has identical exposure and
should be given the same ranges.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class EpisodeEgressRule(BaseModel):
    """One egress rule. ``target`` is an FQDN, IP, or CIDR."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str
    action: Literal["allow", "deny"] = "allow"


class EpisodeEgressPolicy(BaseModel):
    """Egress policy applied to a sandbox.

    Always sent explicitly, even when it has no rules. A backend that treats an absent policy as
    allow-all would otherwise grant unrestricted network access by default, and on OpenSandbox
    create is the only moment the default action can be established at all -- the post-create
    egress calls merge rules but preserve ``defaultAction``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_action: Literal["allow", "deny"] = "deny"
    rules: tuple[EpisodeEgressRule, ...] = ()


# Cluster-internal ranges denied by default. Deny rules become nftables element sets in the
# OpenSandbox egress sidecar, so they match on the resolved destination address -- a hostile
# domain whose A record points into the cluster is still dropped.
DEFAULT_CLUSTER_DENY_TARGETS: tuple[str, ...] = (
    # RFC1918. Pod and Service CIDRs on most clusters.
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    # Carrier-grade NAT. A common Pod CIDR on GKE and on EKS with custom networking. Omitting it
    # is the likeliest way for this deny list to silently cover nothing on a real cluster.
    "100.64.0.0/10",
    # Link-local, which includes the cloud metadata endpoint at 169.254.169.254 that serves node
    # IAM credentials.
    "169.254.0.0/16",
    "127.0.0.0/8",
    # IPv6 loopback, unique-local, and link-local mirror the ranges above.
    "::1/128",
    "fc00::/7",
    "fe80::/10",
    # IPv4-mapped IPv6. Without this, ::ffff:10.0.5.7 reaches 10.0.5.7 straight past an
    # IPv4-only deny list -- a standard SSRF filter bypass.
    "::ffff:0:0/96",
)


def build_egress_policy(
    *,
    default_action: Literal["allow", "deny"],
    allow_targets: tuple[str, ...] = (),
    deny_targets: tuple[str, ...] = DEFAULT_CLUSTER_DENY_TARGETS,
) -> EpisodeEgressPolicy:
    """Build an egress policy from a default action plus explicit allow and deny targets.

    Allow rules are emitted before deny rules so a narrow carve-out -- an in-cluster package
    mirror, say -- precedes the broad range denial that would otherwise cover it. The sidecar
    evaluates domain rules first-match.

    Args:
        default_action: What happens to traffic no rule matches.
        allow_targets: Domains, IPs, or CIDRs to allow explicitly.
        deny_targets: Domains, IPs, or CIDRs to deny explicitly.

    Returns:
        The assembled :class:`EpisodeEgressPolicy`.
    """
    rules = tuple(
        EpisodeEgressRule(target=target, action="allow") for target in allow_targets
    ) + tuple(
        EpisodeEgressRule(target=target, action="deny") for target in deny_targets
    )
    return EpisodeEgressPolicy(default_action=default_action, rules=rules)
