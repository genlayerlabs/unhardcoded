"""
Unit tests for Ollama source: model discovery from local and cloud.
No network: HTTP is mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sources.ollama import OllamaSource  # noqa: E402


class MockClient:
    """Mock httpx.AsyncClient for testing."""

    def __init__(self, responses: list):
        self.responses = responses
        self.call_index = 0

    async def get(self, url: str, headers: dict | None = None):
        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data

            def json(self):
                return self._json

        resp = self.responses[self.call_index]
        self.call_index += 1
        return MockResponse(200, resp)


@pytest.mark.asyncio
async def test_local_discovery_no_auth():
    """Local models discovered without auth."""
    source = OllamaSource({}, env_get=lambda k: None)
    source._client = MockClient([
        {"models": [{"name": "llama3.2:latest", "details": {"context_length": 128000}}]}
    ])

    prices = await source.pricing()

    assert len(prices) == 1
    assert prices[0]["model_family"] == "llama3.2:latest"
    assert prices[0]["price_in_usd_per_mtok"] == 0.0
    assert prices[0]["price_out_usd_per_mtok"] == 0.0

    # Check offers were cached
    offers = source.offers_sync("ollama")
    assert len(offers) == 1
    assert offers[0]["source"] == "local"
    assert offers[0]["seller_endpoint"] == "http://localhost:11434/v1"


@pytest.mark.asyncio
async def test_cloud_discovery_with_api_key():
    """Cloud models discovered with API key."""
    source = OllamaSource(
        {},
        env_get=lambda k: "test-key" if k == "OLLAMA_API_KEY"
                         else "1" if k == "OLLAMA_CLOUD" else None
    )
    source._use_cloud = True
    source._client = MockClient([
        {"models": []},  # local (empty)
        {"models": [{"name": "gpt-oss:120b", "details": {"supports_tools": True}}]},  # cloud
    ])

    prices = await source.pricing()

    assert len(prices) == 1
    assert prices[0]["model_family"] == "gpt-oss:120b"

    offers = source.offers_sync("ollama")
    assert offers[0]["source"] == "cloud"
    assert offers[0]["seller_endpoint"] == "https://ollama.com/api/v1"
    assert offers[0]["capabilities"].get("supports_tools") is True


@pytest.mark.asyncio
async def test_local_unavailable_graceful_skip():
    """Local Ollama not running -> skip, try cloud if configured."""
    source = OllamaSource(
        {},
        env_get=lambda k: "key" if k == "OLLAMA_API_KEY"
                         else "1" if k == "OLLAMA_CLOUD" else None
    )
    source._use_cloud = True

    # Mock client that raises on local, succeeds on cloud
    class FailingOnLocalClient:
        def __init__(self):
            self.call_count = 0

        async def get(self, url: str, headers: dict | None = None):
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("Connection refused")  # local fails
            class MockResponse:
                status_code = 200
                def json(self):
                    return {"models": [{"name": "cloud-model"}]}
            return MockResponse()

    source._client = FailingOnLocalClient()

    prices = await source.pricing()

    # Should have recovered and found cloud model
    assert len(prices) == 1
    assert prices[0]["model_family"] == "cloud-model"


@pytest.mark.asyncio
async def test_no_duplicates_local_and_cloud():
    """Same model in local and cloud -> only one offer (prefer local)."""
    source = OllamaSource(
        {},
        env_get=lambda k: "key" if k == "OLLAMA_API_KEY"
                         else "1" if k == "OLLAMA_CLOUD" else None
    )
    source._use_cloud = True
    source._client = MockClient([
        {"models": [{"name": "llama3.2:latest"}]},  # local
        {"models": [{"name": "llama3.2:latest"}]},  # cloud (same)
    ])

    prices = await source.pricing()

    assert len(prices) == 1  # No duplicate
    assert prices[0]["model_family"] == "llama3.2:latest"

    offers = source.offers_sync("ollama")
    assert offers[0]["source"] == "local"  # Prefer local


@pytest.mark.asyncio
async def test_capability_extraction():
    """Capabilities extracted from model details."""
    source = OllamaSource({}, env_get=lambda k: None)
    source._client = MockClient([
        {"models": [{
            "name": "llama3.2:latest",
            "details": {
                "context_length": 128000,
                "supports_tools": True,
                "supports_vision": True,
            }
        }]}
    ])

    await source.pricing()

    offers = source.offers_sync("ollama")
    caps = offers[0]["capabilities"]
    assert caps["context"] == 128000
    assert caps["supports_tools"] is True
    assert caps["supports_vision"] is True


@pytest.mark.asyncio
async def test_balances_returns_empty_dict():
    """Ollama has no balance tracking (local = free, cloud = subscription)."""
    source = OllamaSource({}, env_get=lambda k: None)

    balances = await source.balances()

    assert balances == {}


@pytest.mark.asyncio
async def test_multiple_local_models_discovered():
    """Multiple local models all discovered and priced correctly."""
    source = OllamaSource({}, env_get=lambda k: None)
    source._client = MockClient([
        {"models": [
            {"name": "llama3.2:latest", "details": {"context_length": 128000}},
            {"name": "mistral:7b", "details": {"supports_tools": True}},
            {"name": "gemma:2b", "details": {}},
        ]}
    ])

    prices = await source.pricing()

    assert len(prices) == 3
    for price in prices:
        assert price["price_in_usd_per_mtok"] == 0.0
        assert price["price_out_usd_per_mtok"] == 0.0
        assert price["provider_id"] == "ollama"

    offers = source.offers_sync("ollama")
    assert len(offers) == 3
    model_names = {o["model_family"] for o in offers}
    assert model_names == {"llama3.2:latest", "mistral:7b", "gemma:2b"}


@pytest.mark.asyncio
async def test_cloud_disabled_without_env_var():
    """Cloud fetch is skipped when OLLAMA_CLOUD env var is not set."""
    source = OllamaSource(
        {},
        env_get=lambda k: "key" if k == "OLLAMA_API_KEY" else None  # No OLLAMA_CLOUD
    )

    source._client = MockClient([
        {"models": [{"name": "local-model"}]},
    ])

    prices = await source.pricing()

    # Only local should be fetched
    assert len(prices) == 1
    assert prices[0]["model_family"] == "local-model"
    assert source._client.call_index == 1  # Only called once (local)


@pytest.mark.asyncio
async def test_cloud_disabled_without_api_key():
    """Cloud fetch is skipped when OLLAMA_API_KEY env var is not set."""
    source = OllamaSource(
        {},
        env_get=lambda k: "1" if k == "OLLAMA_CLOUD" else None  # No OLLAMA_API_KEY
    )

    source._client = MockClient([
        {"models": [{"name": "local-model"}]},
    ])

    prices = await source.pricing()

    # Only local should be fetched
    assert len(prices) == 1
    assert prices[0]["model_family"] == "local-model"
    assert source._client.call_index == 1  # Only called once (local)


@pytest.mark.asyncio
async def test_custom_local_base_url():
    """Custom local base URL is used when OLLAMA_BASE_URL is set."""
    source = OllamaSource(
        {},
        env_get=lambda k: "http://custom:8080" if k == "OLLAMA_BASE_URL" else None
    )

    source._client = MockClient([
        {"models": [{"name": "local-model"}]},
    ])

    await source.pricing()

    offers = source.offers_sync("ollama")
    assert offers[0]["seller_endpoint"] == "http://custom:8080/v1"


@pytest.mark.asyncio
async def test_capability_context_from_model_level():
    """Context length can be at model level, not just details level."""
    source = OllamaSource({}, env_get=lambda k: None)
    source._client = MockClient([
        {"models": [{
            "name": "llama3.2:latest",
            "context_length": 8192,  # At model level
            "details": {},
        }]}
    ])

    await source.pricing()

    offers = source.offers_sync("ollama")
    caps = offers[0]["capabilities"]
    assert caps["context"] == 8192


@pytest.mark.asyncio
async def test_capability_details_context_overrides():
    """Model-level context_length is used when both levels are present."""
    source = OllamaSource({}, env_get=lambda k: None)
    source._client = MockClient([
        {"models": [{
            "name": "llama3.2:latest",
            "context_length": 8192,  # At model level (preferred)
            "details": {
                "context_length": 128000,  # At details level
            },
        }]}
    ])

    await source.pricing()

    offers = source.offers_sync("ollama")
    caps = offers[0]["capabilities"]
    # model.context_length takes precedence over details.context_length
    assert caps["context"] == 8192


@pytest.mark.asyncio
async def test_empty_models_list_handled():
    """Empty models list is handled gracefully."""
    source = OllamaSource({}, env_get=lambda k: None)
    source._client = MockClient([
        {"models": []},
    ])

    prices = await source.pricing()

    assert len(prices) == 0
    assert source.offers_sync("ollama") == []


@pytest.mark.asyncio
async def test_models_with_missing_name_skipped():
    """Models without names are skipped gracefully."""
    source = OllamaSource({}, env_get=lambda k: None)
    source._client = MockClient([
        {"models": [
            {"name": "valid-model"},
            {},  # Missing name
            {"name": "another-valid"},
        ]}
    ])

    prices = await source.pricing()

    assert len(prices) == 2
    model_names = {p["model_family"] for p in prices}
    assert model_names == {"valid-model", "another-valid"}


@pytest.mark.asyncio
async def test_both_local_and_cloud_fail_gracefully():
    """When both local and cloud fail, returns empty prices."""
    source = OllamaSource(
        {},
        env_get=lambda k: "key" if k == "OLLAMA_API_KEY"
                         else "1" if k == "OLLAMA_CLOUD" else None
    )
    source._use_cloud = True

    class AlwaysFailingClient:
        async def get(self, url: str, headers: dict | None = None):
            raise RuntimeError("Always fails")

    source._client = AlwaysFailingClient()

    prices = await source.pricing()

    # Should return empty list, not raise
    assert prices == []
    assert source.offers_sync("ollama") == []