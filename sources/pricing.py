"""Shared pricing helpers for provider sources.

Sources report USD-per-million-token prices. The provider catalog may then
define an effective-price multiplier so routing can compare cash-equivalent
costs without mutating the raw upstream catalog data.
"""
from __future__ import annotations

from typing import Any


def effective_multiplier(provider_cfg: dict[str, Any] | None) -> float:
    raw = (provider_cfg or {}).get("effective_price_multiplier", 1.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 1.0
    return value if value > 0 else 1.0


def effective_price(provider_cfg: dict[str, Any] | None,
                    price_in: float,
                    price_out: float) -> tuple[float, float]:
    m = effective_multiplier(provider_cfg)
    return price_in * m, price_out * m
