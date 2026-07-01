"""
Provider sources: read-only feeds of live pricing and balances per provider.

Strictly off the request path — a source being down never affects routing;
the router coasts on last-known prices (or the metrics seed before the
first refresh). See docs/superpowers/specs/2026-06-10-provider-sources-design.md.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Literal, Protocol, TypedDict


class Price(TypedDict):
    provider_id: str
    served_model_id: str          # the provider's wire id, e.g. "openai/gpt-5.5"
    model_family: str | None      # mapped curated family; None = unmapped
    price_in_usd_per_mtok: float
    price_out_usd_per_mtok: float


class Balance(TypedDict):
    kind: Literal["credits_usd", "deposits_usdc", "quota_window"]
    value: float | None
    detail: dict
    fetched_at: int


class ProviderSource(Protocol):
    name: str
    provider_ids: list[str]
    poll_interval_s: int | None   # None = passive-only (no refresh task)

    async def pricing(self) -> list[Price]: ...
    async def balances(self) -> dict[str, Balance]: ...


# source name -> {last_ok, error, prices_pushed, balances}
# Serialized (without secrets — there are none here) by the shim's /x/runtime.
SOURCE_STATE: dict[str, dict[str, Any]] = {}


def _served_pairs(catalog: dict) -> set[tuple[str, str]]:
    """Every (provider_id, family) pair the catalog routes. Marketplace
    providers can serve ANY curated family (their candidates come from
    offers), so they pair with every model."""
    pairs = set()
    families = list((catalog.get("models") or {}).keys())
    for family, model in (catalog.get("models") or {}).items():
        for served in model.get("served_by") or []:
            if served.get("provider"):
                pairs.add((served["provider"], family))
    for pid, p in (catalog.get("providers") or {}).items():
        if isinstance(p, dict) and p.get("discovery") == "marketplace":
            for family in families:
                pairs.add((pid, family))
    return pairs


def push_prices(host: Any, catalog: dict, prices: list[Price]) -> int:
    """Write mapped prices into the core's metrics store (the one ranking
    and price-ceiling filters read). Unmapped or un-cataloged prices are
    skipped — sources never widen the catalog.

    Prices are stored raw. The host/core boundary applies the current
    `<provider>.price_multiplier` knob at selection time so changing the
    multiplier does not require waiting for the next source refresh."""
    pairs = _served_pairs(catalog)
    now = int(time.time())
    pushed = 0
    for p in prices:
        family = p.get("model_family")
        provider = p.get("provider_id")
        if not family or (provider, family) not in pairs:
            continue
        host.update_metrics(provider, family, {
            "price_in": p["price_in_usd_per_mtok"],
            "price_out": p["price_out_usd_per_mtok"],
            "price_refreshed_at": now,
        })
        pushed += 1
    return pushed


async def refresh_once(host: Any, catalog: dict, source: ProviderSource) -> None:
    """One refresh tick. Never raises: failures land in SOURCE_STATE and the
    last-known data stays in place."""
    state = SOURCE_STATE.setdefault(source.name, {
        "last_ok": None, "error": None, "prices_pushed": 0, "balances": {},
    })
    try:
        prices = await source.pricing()
        state["prices_pushed"] = push_prices(host, catalog, prices)
        state["balances"] = await source.balances()
        if hasattr(source, "market_book"):
            state["book"] = source.market_book()
        state["last_ok"] = int(time.time())
        state["error"] = None
    except Exception as exc:  # noqa: BLE001 — isolation is the contract
        state["error"] = f"{type(exc).__name__}: {exc}"


async def _run_source(host: Any, catalog: dict, source: ProviderSource) -> None:
    while True:
        await refresh_once(host, catalog, source)
        # jitter ±10% so multiple sources don't sync up
        await asyncio.sleep(source.poll_interval_s * (0.9 + 0.2 * random.random()))


def start_refresh_tasks(host: Any, catalog: dict,
                        registry: list[ProviderSource]) -> list[asyncio.Task]:
    """Call from an async context (FastAPI startup). First refresh runs
    immediately; cadence is per-source."""
    return [asyncio.create_task(_run_source(host, catalog, s))
            for s in registry if s.poll_interval_s]
