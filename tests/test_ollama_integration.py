"""
Integration tests for Ollama with live instance.
Requires running Ollama (`ollama serve`).
Skips if Ollama not available.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sources.ollama import OllamaSource  # noqa: E402

# Skip all tests in this module if Ollama not installed
pytestmark = pytest.mark.skipif(
    not os.path.exists("/usr/local/bin/ollama") and
    not os.path.exists("/opt/homebrew/bin/ollama") and
    not os.environ.get("OLLAMA_INTEGRATION_TESTS"),
    reason="Ollama not installed or OLLAMA_INTEGRATION_TESTS not set"
)


@pytest.mark.asyncio
async def test_live_local_discovery():
    """Discover models from running local Ollama."""
    source = OllamaSource({})

    try:
        prices = await source.pricing()
    except Exception as e:
        pytest.skip(f"Ollama not reachable: {e}")

    # May be 0 if no models pulled, but should not raise
    assert isinstance(prices, list)

    # If models exist, check structure
    for p in prices:
        assert "model_family" in p
        assert "served_model_id" in p
        assert p["price_in_usd_per_mtok"] == 0.0
        assert p["price_out_usd_per_mtok"] == 0.0


@pytest.mark.asyncio
async def test_live_offers_cached():
    """Offers are cached after discovery."""
    source = OllamaSource({})

    try:
        await source.pricing()
    except Exception as e:
        pytest.skip(f"Ollama not reachable: {e}")

    # offers_sync should return cached results without network
    offers = source.offers_sync("ollama")
    assert isinstance(offers, list)