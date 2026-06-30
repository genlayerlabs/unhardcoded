"""Unit tests for the shim's max_tokens defaulting.

Pure-function tests for `_request_to_contract` — no Lua host required. They
pin the behaviour that the OpenAI-compat shim supplies a default `max_tokens`
when the client omits it, so spec-compliant clients (which may leave the
optional field out) don't get rejected by upstreams/policies that require it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shim import ChatRequest, _request_to_contract, DEFAULT_MAX_TOKENS_FALLBACK  # noqa: E402


def _req(**kw) -> ChatRequest:
    base = {"model": "", "messages": [{"role": "user", "content": "hi"}]}
    base.update(kw)
    return ChatRequest(**base)


def test_explicit_max_tokens_is_preserved():
    c = _request_to_contract(_req(max_tokens=128), "default", default_max_tokens=4096)
    assert c["max_tokens"] == 128


def test_missing_max_tokens_gets_the_default():
    c = _request_to_contract(_req(), "default", default_max_tokens=4096)
    assert c["max_tokens"] == 4096


def test_default_is_applied_via_create_app_fallback():
    # When no per-call value is given and no explicit default is threaded,
    # the shim still supplies its module-level fallback rather than omitting it.
    c = _request_to_contract(_req(), "default")
    assert c["max_tokens"] == DEFAULT_MAX_TOKENS_FALLBACK


def test_default_none_means_omit():
    # Opting out (default_max_tokens=None) keeps the old behaviour: omit when absent.
    c = _request_to_contract(_req(), "default", default_max_tokens=None)
    assert "max_tokens" not in c


def test_first_token_timeout_is_forwarded_to_contract():
    c = _request_to_contract(
        _req(first_token_timeout_ms=2500),
        "default",
        default_max_tokens=4096,
    )
    assert c["first_token_timeout_ms"] == 2500
