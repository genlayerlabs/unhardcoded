"""Host-side route identity.

MEASURING per-route reliability/latency is the host's job (llm-router #14/#15: the
algebra reads them as per-candidate fields, like price). The measurement used to
be an in-process EMA folded here; since #4a it is DERIVED on the fly from the raw
per-attempt `route_observations` ledger (`host_store.route_stats`) — fleet-
consistent and surviving restarts, instead of a per-pod dict.

What stays here is the one thing that is pure identity, not measurement: the
`route_key` that names a route (a specific seller peer serving a family, or the
provider itself for a peerless gateway route). It is reused by route_cache,
route_tool_capability and the offer-stamp sites, and matches the key
`route_stats` aggregates on. It stays entirely host-internal; the algebra never
sees it.
"""
from __future__ import annotations


def route_key(provider_id: str, model_family: str, peer_id: str) -> str:
    """Identity of a route = provider|family|served_by (the peer for a marketplace
    route, else the provider). Host-internal; never enters the signature."""
    return f"{provider_id}|{model_family}|{peer_id}"


def reset() -> None:
    """Test hook. Reliability/latency state now lives in route_observations (reset
    by truncating the host store); kept as a no-op so callers need no change."""
