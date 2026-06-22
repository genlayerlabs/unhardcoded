"""
Host-side credential layer: provider `auth` descriptor → request headers.
The router carries the `auth` blob opaquely; all auth semantics live here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router_host import _resolve_auth_headers, _prepare_openai_call  # noqa: E402


def _env(mapping):
    return lambda k: mapping.get(k)


# ---- auth kinds --------------------------------------------------------

def test_bare_auth_env_is_treated_as_bearer():
    headers, err = _resolve_auth_headers(
        {"auth_env": "MY_KEY"}, _env({"MY_KEY": "sk-123"}))
    assert err is None
    assert headers == {"Authorization": "Bearer sk-123"}


def test_bearer_missing_env_is_auth_error():
    headers, err = _resolve_auth_headers(
        {"auth_env": "MY_KEY"}, _env({}))
    assert headers is None
    assert err["error_kind"] == "auth_error"


def test_kind_none_sends_no_authorization_header():
    headers, err = _resolve_auth_headers(
        {"auth": {"kind": "none"}, "base_url": "http://localhost:8377/v1"}, _env({}))
    assert err is None
    assert headers == {}, "no Authorization header for kind=none"


def test_kind_bearer_explicit_env():
    headers, err = _resolve_auth_headers(
        {"auth": {"kind": "bearer", "env": "HEURIST"}}, _env({"HEURIST": "tok"}))
    assert err is None
    assert headers["Authorization"] == "Bearer tok"


def test_kind_oauth_uses_token_provider():
    headers, err = _resolve_auth_headers(
        {"auth": {"kind": "oauth", "provider": "codex"}},
        _env({}),
        token_providers={"codex": lambda: "oauth-token"},
    )
    assert err is None
    assert headers["Authorization"] == "Bearer oauth-token"


def test_kind_oauth_without_provider_is_error():
    headers, err = _resolve_auth_headers(
        {"auth": {"kind": "oauth", "provider": "codex"}}, _env({}))
    assert headers is None
    assert err["error_kind"] == "auth_error"


def test_unknown_kind_is_error():
    headers, err = _resolve_auth_headers({"auth": {"kind": "magic"}}, _env({}))
    assert headers is None
    assert err["error_kind"] == "auth_error"


# ---- request prep ------------------------------------------------------

def test_prepare_builds_url_body_and_headers():
    prep, err = _prepare_openai_call(
        {
            "served_model_id": "minimax/minimax-m2.7",
            "base_url": "https://openrouter.ai/api/v1/",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
            "temperature": 0.5,
            "auth": {"kind": "none"},
        },
        _env({}), extra={}, timeout_s=30.0,
    )
    assert err is None
    url, body, headers, timeout = prep
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert body["model"] == "minimax/minimax-m2.7"
    assert body["tools"] == [{"type": "function"}]
    assert body["temperature"] == 0.5
    assert headers["Content-Type"] == "application/json"
    assert "Authorization" not in headers
    assert timeout == 30.0


def test_prepare_propagates_auth_error():
    prep, err = _prepare_openai_call(
        {"served_model_id": "m", "base_url": "http://x", "auth_env": "UNSET"},
        _env({}), extra={}, timeout_s=10.0,
    )
    assert prep is None
    assert err["error_kind"] == "auth_error"


def test_prepare_openai_call_prefers_offer_wire_model_id():
    request = {
        "served_model_id": "qwen3-235b-a22b",       # curated family name
        "offer": {"wire_model_id": "qwen3-235b-instruct"},  # what the peer serves
        "base_url": "http://localhost:8377/v1",
        "auth": {"kind": "none"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    prep, err = _prepare_openai_call(request, lambda k: None, {}, 30.0, None)
    assert err is None
    _url, body, _headers, _timeout = prep
    assert body["model"] == "qwen3-235b-instruct"


def test_prepare_openai_call_pins_marketplace_peer_per_request():
    # The browse-mode AntSeed buyer disables auto-selection; the host pins the
    # offer's peer per request so the policy-selected peer is the one served.
    request = {
        "served_model_id": "claude-opus-4-8",
        "offer": {"wire_model_id": "claude-opus-4-8",
                  "peer_id": "0329c5d3920e301740f78d6e17b8d1a11cca9b2c"},
        "base_url": "http://localhost:8377/v1",
        "auth": {"kind": "none"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    prep, err = _prepare_openai_call(request, lambda k: None, {}, 30.0, None)
    assert err is None
    _url, _body, headers, _timeout = prep
    assert headers["x-antseed-pin-peer"] == "0329c5d3920e301740f78d6e17b8d1a11cca9b2c"

    # no offer peer -> no pin header (non-marketplace providers are untouched)
    request["offer"] = {"wire_model_id": "claude-opus-4-8"}
    prep, err = _prepare_openai_call(request, lambda k: None, {}, 30.0, None)
    assert err is None
    assert "x-antseed-pin-peer" not in prep[2]


# ---- Ollama auth --------------------------------------------------------

def test_ollama_local_no_auth():
    """Local Ollama requires no authorization header (even with API key set)."""
    request = {
        "served_model_id": "llama3",
        "base_url": "http://localhost:11434/v1",
        "offer": {"seller_endpoint": "http://localhost:11434"},
        "auth_env": "OLLAMA_API_KEY",  # Explicitly set auth to verify suppression
        "messages": [{"role": "user", "content": "hi"}],
    }
    prep, err = _prepare_openai_call(
        request, _env({"OLLAMA_API_KEY": "test-key"}), {}, 30.0, None)
    assert err is None
    _url, _body, headers, _timeout = prep
    # Local Ollama should NOT use auth headers even if OLLAMA_API_KEY is set
    assert "Authorization" not in headers


def test_ollama_127_no_auth():
    """Ollama at 127.0.0.1 requires no authorization header (even with API key set)."""
    request = {
        "served_model_id": "llama3",
        "base_url": "http://127.0.0.1:11434/v1",
        "offer": {"seller_endpoint": "http://127.0.0.1:11434"},
        "auth_env": "OLLAMA_API_KEY",  # Explicitly set auth to verify suppression
        "messages": [{"role": "user", "content": "hi"}],
    }
    prep, err = _prepare_openai_call(
        request, _env({"OLLAMA_API_KEY": "test-key"}), {}, 30.0, None)
    assert err is None
    _url, _body, headers, _timeout = prep
    # Local Ollama should NOT use auth headers
    assert "Authorization" not in headers


def test_ollama_cloud_requires_api_key():
    """Cloud Ollama requires API key as Bearer token."""
    request = {
        "served_model_id": "llama3",
        "base_url": "https://ollama.com/v1",
        "offer": {"seller_endpoint": "https://ollama.com"},
        "messages": [{"role": "user", "content": "hi"}],
    }

    prep, err = _prepare_openai_call(
        request, _env({"OLLAMA_API_KEY": "sk-test-123"}), {}, 30.0, None)
    assert err is None
    _url, _body, headers, _timeout = prep
    assert headers["Authorization"] == "Bearer sk-test-123"


def test_ollama_cloud_missing_api_key_is_error():
    """Cloud Ollama with missing API key returns auth error."""
    request = {
        "served_model_id": "llama3",
        "base_url": "https://ollama.com/v1",
        "offer": {"seller_endpoint": "https://ollama.com"},
        "messages": [{"role": "user", "content": "hi"}],
    }

    prep, err = _prepare_openai_call(
        request, _env({}), {}, 30.0, None)
    assert prep is None
    assert err["error_kind"] == "auth_error"
    assert "OLLAMA_API_KEY" in err["error_message"]


def test_ollama_detected_by_provider_id():
    """Ollama can be detected by provider_id, not just base_url."""
    request = {
        "provider_id": "ollama",
        "served_model_id": "llama3",
        "base_url": "http://some-host:11434/v1",
        "offer": {"seller_endpoint": "http://some-host:11434"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    prep, err = _prepare_openai_call(
        request, _env({"OLLAMA_API_KEY": "test-key"}), {}, 30.0, None)
    assert err is None
    _url, _body, headers, _timeout = prep
    # Local endpoint (not ollama.com) should NOT use auth
    assert "Authorization" not in headers


def test_non_ollama_provider_unaffected():
    """Non-Ollama providers use existing auth logic."""
    request = {
        "provider_id": "openai",
        "served_model_id": "gpt-4",
        "base_url": "https://api.openai.com/v1",
        "auth_env": "OPENAI_API_KEY",
        "messages": [{"role": "user", "content": "hi"}],
    }
    prep, err = _prepare_openai_call(
        request, _env({"OPENAI_API_KEY": "sk-abc"}), {}, 30.0, None)
    assert err is None
    _url, _body, headers, _timeout = prep
    assert headers["Authorization"] == "Bearer sk-abc"
