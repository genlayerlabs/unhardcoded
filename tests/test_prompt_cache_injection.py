"""Essence for #74: the router injects anthropic prompt-cache breakpoints.

Anthropic caching is OPT-IN per request (`cache_control`); policy-driven
clients are model-agnostic by design and never know the chosen provider, so
the router is the only place that can mark the request. Without the markers
an agentic session re-buys its whole growing prefix at full input price on
every call (measured live: 3 identical-prefix opus calls, tokens_cached=0,
full price each — 2026-07-06).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from provider_adapters.anthropic import _anthropic_request           # noqa: E402
from provider_adapters.openai_compatible import _prepare_openai_call  # noqa: E402

EPH = {"type": "ephemeral"}


def _openai_request(provider_id: str, family: str, messages: list[dict]) -> dict:
    return {
        "provider_id": provider_id,
        "model_family": family,
        "served_model_id": family,
        "base_url": "https://openrouter.ai/api/v1",
        "auth_env": "OPENROUTER_API_KEY",
        "messages": messages,
    }


def test_native_anthropic_marks_system_and_last_message():
    url, body, headers = _anthropic_request({
        "served_model_id": "claude-opus-4-8",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "turn 2"},
        ],
    }, "tok", {})

    assert body["system"] == [{"type": "text", "text": "Be terse.",
                               "cache_control": EPH}]
    # ONLY the last message carries the rolling breakpoint
    assert body["messages"][-1]["content"] == [
        {"type": "text", "text": "turn 2", "cache_control": EPH}]
    assert body["messages"][0]["content"] == "turn 1"
    assert body["messages"][1]["content"] == "ok"


def test_openrouter_claude_gets_breakpoints_without_mutating_the_request():
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "turn 1"},
    ]
    request = _openai_request("openrouter", "claude-opus-4-8", messages)

    prepared, err = _prepare_openai_call(request, {"OPENROUTER_API_KEY": "k"}.get, {}, 30.0)
    assert err is None
    _url, body, _headers, _timeout = prepared

    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "Be terse.", "cache_control": EPH}]
    assert body["messages"][-1]["content"] == [
        {"type": "text", "text": "turn 1", "cache_control": EPH}]
    # the caller's request is untouched — it is shared with retries and
    # other rank candidates
    assert messages[0]["content"] == "Be terse."
    assert messages[1]["content"] == "turn 1"


def test_non_anthropic_and_unknown_surfaces_are_untouched():
    messages = [{"role": "user", "content": "hi"}]

    for provider, family in (
        ("openrouter", "gpt-5.5"),          # relaying surface, wrong family
        ("heurist", "claude-opus-4-8"),     # anthropic family, unknown surface
    ):
        request = _openai_request(provider, family, messages)
        prepared, err = _prepare_openai_call(
            request, {"OPENROUTER_API_KEY": "k", "HEURIST_API_KEY": "k"}.get, {}, 30.0)
        assert err is None
        _url, body, _headers, _timeout = prepared
        assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_breakpoints_cap_at_two_marks():
    # ≤4 breakpoints allowed by Anthropic; we spend exactly two (system +
    # rolling last), whatever the conversation length
    long = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": f"t{i}"} for i in range(10)]
    request = _openai_request("openrouter_market", "claude-fable-5", long)

    prepared, err = _prepare_openai_call(request, {"OPENROUTER_API_KEY": "k"}.get, {}, 30.0)
    assert err is None
    _url, body, _headers, _timeout = prepared

    marked = sum(
        1 for m in body["messages"]
        if isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and p.get("cache_control") for p in m["content"]))
    assert marked == 2


# ── read-back: every adapter must surface its provider's cache counters ──────
# (bedrock and openai-compat/codex already did; anthropic-native and google
# were BLIND — the meter and the cost discount never saw the hits)

def test_every_usage_shape_reaches_tokens_cached():
    from provider_adapters.anthropic import _parse_anthropic_response
    from provider_adapters.google import _parse_gemini_response
    from provider_adapters.common import cached_tokens

    anth = _parse_anthropic_response({
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 100, "output_tokens": 5,
                  "cache_read_input_tokens": 90},
    }, 200, 10)
    assert anth["response"]["tokens_cached"] == 90

    gem = _parse_gemini_response({
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5,
                          "totalTokenCount": 105, "cachedContentTokenCount": 40},
    }, 200, 10)
    assert gem["response"]["tokens_cached"] == 40

    # the shared helper covers openai-compat, codex-responses and raw anthropic
    assert cached_tokens({"prompt_tokens_details": {"cached_tokens": 7}}) == 7
    assert cached_tokens({"input_tokens_details": {"cached_tokens": 8}}) == 8
    assert cached_tokens({"cache_read_input_tokens": 9}) == 9
    assert cached_tokens({"prompt_tokens": 5}) is None
