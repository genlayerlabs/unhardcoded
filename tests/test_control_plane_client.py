"""Unit tests for control_plane_client (no Postgres, no network).

The HTTP boundary is a fake async client monkeypatched onto the module-level
`_client` (suite convention — no respx). Feature flags are monkeypatched module
attributes; every test starts from reset_for_tests().
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import control_plane_client as cpc  # noqa: E402


class _FakeResp:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


class _FakeClient:
    """Scripted responses per URL prefix; records every call."""

    def __init__(self, responses):
        # responses: list of _FakeResp | Exception, consumed in order
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def get(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    cpc.reset_for_tests()
    monkeypatch.setattr(cpc, "CONTROL_PLANE_URL", "http://cp.test")
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "s3cret")
    yield
    cpc.reset_for_tests()


def _install(monkeypatch, responses) -> _FakeClient:
    fake = _FakeClient(responses)
    monkeypatch.setattr(cpc, "_client", fake)
    return fake


def _active(consumer="acme", tenant_id=7, rate=None, burst=None):
    return {"active": True, "consumer": consumer, "tenant_id": tenant_id,
            "rate_per_min": rate, "burst": burst}


DIGEST = "ab" * 32


# ---- resolve_key -------------------------------------------------------------

def test_feature_off_returns_none_without_http(monkeypatch):
    monkeypatch.setattr(cpc, "CONTROL_PLANE_URL", "")
    fake = _install(monkeypatch, [])
    out = asyncio.run(cpc.resolve_key(DIGEST))
    assert out is None
    assert fake.calls == []


def test_resolve_fetches_then_serves_from_cache(monkeypatch):
    fake = _install(monkeypatch, [_FakeResp(200, _active(rate=60, burst=20))])
    first = asyncio.run(cpc.resolve_key(DIGEST))
    assert first.active and first.consumer == "acme" and first.tenant_id == 7
    assert first.rate_per_min == 60 and first.burst == 20
    second = asyncio.run(cpc.resolve_key(DIGEST))
    assert second is first
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["url"].endswith("/internal/keys/resolve")
    assert call["params"] == {"sha256": DIGEST}
    assert call["headers"] == {"x-internal-secret": "s3cret"}


def test_negative_answer_is_cached(monkeypatch):
    fake = _install(monkeypatch, [_FakeResp(200, {"active": False})])
    first = asyncio.run(cpc.resolve_key(DIGEST))
    assert first is not None and first.active is False
    second = asyncio.run(cpc.resolve_key(DIGEST))
    assert second.active is False
    assert len(fake.calls) == 1


def test_ttl_expiry_revalidates_and_definitive_no_replaces_positive(monkeypatch):
    fake = _install(monkeypatch, [_FakeResp(200, _active()),
                                  _FakeResp(200, {"active": False})])
    assert asyncio.run(cpc.resolve_key(DIGEST)).active is True
    monkeypatch.setattr(cpc, "RESOLVE_TTL_S", 0.0)   # everything positive expired
    out = asyncio.run(cpc.resolve_key(DIGEST))
    assert out.active is False                        # revoked upstream wins
    assert len(fake.calls) == 2


def test_stale_grace_serves_positive_on_transport_error_only(monkeypatch):
    fake = _install(monkeypatch, [_FakeResp(200, _active()),
                                  httpx.ConnectError("down"),
                                  httpx.ConnectError("down")])
    assert asyncio.run(cpc.resolve_key(DIGEST)).active is True
    monkeypatch.setattr(cpc, "RESOLVE_TTL_S", 0.0)
    # CP unreachable inside the grace window -> stale positive keeps working
    assert asyncio.run(cpc.resolve_key(DIGEST)).consumer == "acme"
    # grace exhausted -> None (caller 401s)
    monkeypatch.setattr(cpc, "RESOLVE_STALE_GRACE_S", 0.0)
    assert asyncio.run(cpc.resolve_key(DIGEST)) is None
    assert len(fake.calls) == 3


def test_negative_entries_get_no_grace(monkeypatch):
    _install(monkeypatch, [_FakeResp(200, {"active": False}),
                           httpx.ConnectError("down")])
    assert asyncio.run(cpc.resolve_key(DIGEST)).active is False
    monkeypatch.setattr(cpc, "NEGATIVE_TTL_S", 0.0)
    assert asyncio.run(cpc.resolve_key(DIGEST)) is None


def test_5xx_counts_as_unreachable(monkeypatch):
    _install(monkeypatch, [_FakeResp(200, _active()), _FakeResp(500, {})])
    assert asyncio.run(cpc.resolve_key(DIGEST)).active is True
    monkeypatch.setattr(cpc, "RESOLVE_TTL_S", 0.0)
    assert asyncio.run(cpc.resolve_key(DIGEST)).consumer == "acme"   # stale grace


def test_single_flight_coalesces_concurrent_resolves(monkeypatch):
    fake = _FakeClient([])
    started = asyncio.Event()

    async def slow_get(url, params=None, headers=None):
        fake.calls.append({"url": url})
        started.set()
        await asyncio.sleep(0.02)
        return _FakeResp(200, _active())

    fake.get = slow_get
    monkeypatch.setattr(cpc, "_client", fake)

    async def run():
        return await asyncio.gather(cpc.resolve_key(DIGEST), cpc.resolve_key(DIGEST))

    a, b = asyncio.run(run())
    assert a.consumer == b.consumer == "acme"
    assert len(fake.calls) == 1


def test_malformed_body_is_a_definitive_negative(monkeypatch):
    _install(monkeypatch, [_FakeResp(200, ValueError("not json"))])
    out = asyncio.run(cpc.resolve_key(DIGEST))
    assert out is not None and out.active is False


# ---- tenant_env --------------------------------------------------------------

def test_tenant_env_filters_through_allowlist_and_caches(monkeypatch):
    fake = _install(monkeypatch, [_FakeResp(200, {"env": {
        "OPENAI_API_KEY": "sk-tenant", "EVIL_PATH_OVERRIDE": "x",
        "ANTHROPIC_API_KEY": ""}})])
    env = asyncio.run(cpc.tenant_env(7))
    assert env == {"OPENAI_API_KEY": "sk-tenant"}   # allowlist + empty dropped
    assert asyncio.run(cpc.tenant_env(7)) == env
    assert len(fake.calls) == 1
    assert fake.calls[0]["url"].endswith("/internal/tenants/7/provider-env")


def test_tenant_env_fail_soft_to_platform_keys(monkeypatch):
    _install(monkeypatch, [httpx.ConnectError("down")])
    assert asyncio.run(cpc.tenant_env(9)) == {}


def test_tenant_env_stale_grace_then_empty(monkeypatch):
    _install(monkeypatch, [_FakeResp(200, {"env": {"OPENAI_API_KEY": "sk-t"}}),
                           httpx.ConnectError("down"),
                           httpx.ConnectError("down")])
    assert asyncio.run(cpc.tenant_env(7)) == {"OPENAI_API_KEY": "sk-t"}
    monkeypatch.setattr(cpc, "TENANT_ENV_TTL_S", 0.0)
    assert asyncio.run(cpc.tenant_env(7)) == {"OPENAI_API_KEY": "sk-t"}   # grace
    monkeypatch.setattr(cpc, "TENANT_ENV_STALE_GRACE_S", 0.0)
    assert asyncio.run(cpc.tenant_env(7)) == {}


# ---- env_get / context isolation ---------------------------------------------

def test_env_get_prefers_active_tenant_map_then_process_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    assert cpc.env_get("OPENAI_API_KEY") == "sk-platform"
    token = cpc.activate_tenant_env({"OPENAI_API_KEY": "sk-tenant"})
    try:
        assert cpc.env_get("OPENAI_API_KEY") == "sk-tenant"
        assert cpc.env_get("OPENROUTER_API_KEY") == os.environ.get("OPENROUTER_API_KEY")
    finally:
        cpc.reset_tenant_env(token)
    assert cpc.env_get("OPENAI_API_KEY") == "sk-platform"


def test_env_get_isolation_across_tasks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    seen: dict[str, str | None] = {}

    async def tenant_task(name: str, key: str | None):
        if key is not None:
            cpc.activate_tenant_env({"OPENAI_API_KEY": key})
        await asyncio.sleep(0.01)

        async def child():
            seen[name] = cpc.env_get("OPENAI_API_KEY")

        # tasks created AFTER activation copy the context (streaming/flow nodes)
        await asyncio.create_task(child())

    async def run():
        await asyncio.gather(tenant_task("a", "sk-a"), tenant_task("b", "sk-b"),
                             tenant_task("none", None))

    asyncio.run(run())
    assert seen == {"a": "sk-a", "b": "sk-b", "none": "sk-platform"}


# ---- internal_secret_ok --------------------------------------------------------

def test_internal_secret_ok(monkeypatch):
    assert cpc.internal_secret_ok({"x-internal-secret": "s3cret"}) is True
    assert cpc.internal_secret_ok({"x-internal-secret": "wrong"}) is False
    assert cpc.internal_secret_ok({}) is False
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "")
    assert cpc.internal_secret_ok({"x-internal-secret": ""}) is False
