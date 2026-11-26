"""Configuration and cost handling for shared upstream spend admission."""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation, ROUND_CEILING

SCALE = Decimal("0.000000000001")
MAX_USD = Decimal("1000000000")
SpendConfig = tuple[dict[str, tuple[Decimal, Decimal]], dict[str, str]]


def usd(value, *, reported=False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("Expected a USD amount")
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or not 0 <= amount <= MAX_USD:
            raise ValueError("USD amount out of range")
        rounded = amount.quantize(SCALE, rounding=ROUND_CEILING)
        if not reported and rounded != amount:
            raise ValueError("USD amount has more than 12 decimal places")
    except InvalidOperation as exc:
        raise ValueError("Invalid USD amount") from exc
    return rounded


def load(raw: str) -> SpendConfig:
    """Return limits by upstream account and the mapping from catalog IDs."""
    try:
        data = json.loads(raw, parse_float=Decimal)
        if not isinstance(data, dict):
            raise ValueError
        limits = {}
        identities = {}
        for identity, entry in data.items():
            if (not identity or len(identity) > 128 or not isinstance(entry, dict)
                    or set(entry) - {"daily_usd", "max_attempt_usd", "provider_ids"}):
                raise ValueError
            limit = usd(entry.get("daily_usd"))
            reserve = usd(entry.get("max_attempt_usd"))
            providers = entry.get("provider_ids", [identity])
            if not reserve or not isinstance(providers, list) or not providers:
                raise ValueError
            for provider_id in providers:
                if (not isinstance(provider_id, str) or not provider_id or len(provider_id) > 128
                        or provider_id in identities):
                    raise ValueError
                identities[provider_id] = identity
            limits[identity] = (limit, reserve)
        return limits, identities
    except ValueError as exc:
        raise ValueError("Invalid ROUTER_PROVIDER_DAILY_SPEND configuration") from exc


def actual_cost(result: dict) -> Decimal | None:
    if result.get("ok"):
        try:
            return usd((result.get("response") or {}).get("cost_reported"), reported=True)
        except ValueError:
            pass
    return None
