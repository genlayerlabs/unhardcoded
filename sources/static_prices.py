"""Static stopgap prices for direct providers.

This is intentionally a source, not catalog data: the catalog says which
providers can serve which families; sources decide what those routes cost until
provider-native price feeds exist.
"""
from __future__ import annotations

from sources import Balance, Price


# USD per million tokens, keyed by provider id and provider-local model id.
# Keep this small: only provider/model pairs already declared in config.live.lua.
STATIC_PRICES_USD_PER_MTOK: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {
        "gpt-5.5": (5.0, 30.0),
        "gpt-5.4": (2.5, 15.0),
    },
    "anthropic": {
        "claude-opus-4-8": (5.0, 25.0),
        "claude-sonnet-4-6": (3.0, 15.0),
    },
    "gemini": {
        "gemini-3.1-pro-preview": (2.0, 12.0),
    },
    "bedrock_mantle": {
        "qwen.qwen3-235b-a22b-2507": (0.09, 0.10),
    },
}


class StaticPriceSource:
    name = "static_prices"
    poll_interval_s = 3600

    def __init__(self, catalog: dict):
        providers = catalog.get("providers") or {}
        self._served: list[tuple[str, str, str]] = []
        for family, model in (catalog.get("models") or {}).items():
            for served in model.get("served_by") or []:
                provider = served.get("provider")
                provider_model = served.get("provider_model_id")
                prices = STATIC_PRICES_USD_PER_MTOK.get(provider or "")
                if prices and provider in providers and provider_model in prices:
                    self._served.append((provider, provider_model, family))
        self.provider_ids = sorted({provider for provider, _model, _family in self._served})

    async def pricing(self) -> list[Price]:
        prices: list[Price] = []
        for provider, provider_model, family in self._served:
            table = STATIC_PRICES_USD_PER_MTOK.get(provider) or {}
            price = table.get(provider_model)
            if price is None:
                continue
            pin, pout = price
            prices.append({
                "provider_id": provider,
                "served_model_id": provider_model,
                "model_family": family,
                "price_in_usd_per_mtok": pin,
                "price_out_usd_per_mtok": pout,
            })
        return prices

    async def balances(self) -> dict[str, Balance]:
        return {}
