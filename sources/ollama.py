"""
Ollama source: discovers models from local Ollama instance or Ollama Cloud.

Local:  GET http://localhost:11434/api/tags (no auth)
Cloud: GET https://ollama.com/api/tags (requires auth)

Authentication methods (in order of preference):
1. Ed25519 OAuth - Uses ~/.ollama/id_ed25519 key (automatic, no config needed)
2. API Key - Uses OLLAMA_API_KEY environment variable
"""
from __future__ import annotations

import os
from typing import Any

from sources import Balance, Price
from sources.ollama_auth import (
    get_ollama_auth_header,
    has_ollama_key,
    can_use_ed25519_auth,
    OllamaAuthError,
)


BASE_URL_LOCAL = "http://localhost:11434"
BASE_URL_CLOUD = "https://ollama.com"
HTTP_TIMEOUT_S = 15.0


class OllamaSource:
    """Provider source for Ollama (local and cloud)."""

    name = "ollama"
    provider_ids = ["ollama"]
    poll_interval_s = 300  # 5 minutes

    def __init__(self, catalog: dict, env_get=os.environ.get, client: Any = None):
        self._env_get = env_get
        self._client = client
        self._use_cloud = env_get("OLLAMA_CLOUD") == "1"
        self._api_key = env_get("OLLAMA_API_KEY")
        self._local_base = env_get("OLLAMA_BASE_URL") or BASE_URL_LOCAL

        # Ed25519 auth support
        self._use_ed25519 = can_use_ed25519_auth()

        # Cached offers from last refresh
        self._offers: list[dict] = []
        self._local_models: list[dict] = []
        self._cloud_models: list[dict] = []

    def _get_auth_headers(self, url: str) -> dict:
        """
        Get authentication headers for Ollama Cloud.

        Priority:
        1. Ed25519 OAuth (if key exists)
        2. API Key (if OLLAMA_API_KEY set)
        """
        if self._use_ed25519:
            try:
                auth_token = get_ollama_auth_header(
                    method="GET",
                    url=url,
                    body=b""
                )
                return {"Authorization": auth_token}
            except OllamaAuthError:
                # Fall back to API key
                pass

        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}

        return {}

    async def _get(self, base_url: str, path: str, headers: dict | None = None) -> dict:
        """HTTP GET with optional auth headers."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_S)
        resp = await self._client.get(f"{base_url}{path}", headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"ollama GET {path} -> {resp.status_code}")
        return resp.json()

    async def _fetch_local_models(self) -> list[dict]:
        """Fetch models from local Ollama instance (no auth)."""
        try:
            body = await self._get(self._local_base, "/api/tags")
            return body.get("models") or []
        except Exception:
            return []  # Local not running, skip silently

    async def _fetch_cloud_models(self) -> list[dict]:
        """Fetch models from Ollama Cloud.

        Uses Ed25519 OAuth if key exists, otherwise API key.
        Requires either OLLAMA_API_KEY or ~/.ollama/id_ed25519.
        """
        # Prefer Ed25519 auth when available (no config needed)
        if not self._use_ed25519 and not self._api_key:
            return []

        try:
            headers = self._get_auth_headers(f"{BASE_URL_CLOUD}/api/tags")
            if not headers:
                return []

            body = await self._get(BASE_URL_CLOUD, "/api/tags", headers=headers)
            return body.get("models") or []
        except Exception:
            return []  # Cloud unavailable, skip silently

    def _extract_capabilities(self, model: dict) -> dict:
        """Extract capabilities from Ollama model info."""
        caps = {}
        details = model.get("details") or {}

        # Context length
        ctx = model.get("context_length") or details.get("context_length")
        if ctx:
            caps["context"] = int(ctx)

        # Capabilities from model details
        if details.get("supports_tools"):
            caps["supports_tools"] = True
        if details.get("supports_vision"):
            caps["supports_vision"] = True

        return caps

    async def pricing(self) -> list[Price]:
        """Discover models and return zero-cost prices (subscription model)."""
        prices: list[Price] = []

        # Always try local
        self._local_models = await self._fetch_local_models()

        # Try cloud if configured or Ed25519 key available
        if self._use_cloud and (self._use_ed25519 or self._api_key):
            self._cloud_models = await self._fetch_cloud_models()

        # Build offers (no duplicates, prefer local)
        self._offers = []
        seen = set()

        # Local models (free)
        for m in self._local_models:
            model_id = m.get("name")
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            offer = {
                "model_family": model_id,
                "wire_model_id": model_id,
                "seller_endpoint": self._local_base,
                "price_in_usd_per_mtok": 0.0,
                "price_out_usd_per_mtok": 0.0,
                "capabilities": self._extract_capabilities(m),
                "source": "local",
            }
            self._offers.append(offer)
            prices.append({
                "provider_id": "ollama",
                "served_model_id": model_id,
                "model_family": model_id,
                "price_in_usd_per_mtok": 0.0,
                "price_out_usd_per_mtok": 0.0,
            })

        # Cloud models (subscription, marginal cost = 0)
        for m in self._cloud_models:
            model_id = m.get("name")
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            offer = {
                "model_family": model_id,
                "wire_model_id": model_id,
                "seller_endpoint": BASE_URL_CLOUD,
                "price_in_usd_per_mtok": 0.0,
                "price_out_usd_per_mtok": 0.0,
                "capabilities": self._extract_capabilities(m),
                "source": "cloud",
            }
            self._offers.append(offer)
            prices.append({
                "provider_id": "ollama",
                "served_model_id": model_id,
                "model_family": model_id,
                "price_in_usd_per_mtok": 0.0,
                "price_out_usd_per_mtok": 0.0,
            })

        return prices

    def offers_sync(self, provider_id: str) -> list[dict]:
        """Return cached offers (called during ranking)."""
        return self._offers

    async def balances(self) -> dict[str, Balance]:
        """Ollama has no balance tracking (local = free, cloud = subscription)."""
        return {}