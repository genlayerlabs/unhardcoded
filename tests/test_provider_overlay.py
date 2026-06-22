"""
The add-provider flow: overlay validation/merge (provider_overlay), the
shim's hot-add endpoint (POST /x/providers), and the dashboard endpoint
that persists + applies (POST /dashboard/api/provider-keys/add).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import provider_overlay as po  # noqa: E402
from llm_router_host import LLMRouterHost  # noqa: E402
from shim import create_app  # noqa: E402


@pytest.fixture
def host():
    h = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "core" / "config.example.lua",
        metrics_path=ROOT / "core" / "metrics.example.lua",
        now_ms=lambda: 1_000_000,
    )
    h.init()
    return h


def _entry(host, **over):
    fam = next(iter(host.catalog()["models"]))
    base = {"base_url": "https://api.example.com/v1",
            "api_kind": "openai_compatible", "tier": "partner",
            "auth_env": "EXAMPLE_API_KEY",
            "served_models": [{"family": fam, "provider_model_id": "wire-id"}]}
    base.update(over)
    return base


# ---- validation -------------------------------------------------------------

def test_validate_entry_catches_each_problem(host):
    catalog = host.catalog()
    ok = po.validate_entry("newprov", _entry(host), catalog)
    assert ok == []
    assert po.validate_entry("Bad Id!", _entry(host), catalog)
    existing = next(iter(catalog["providers"]))
    assert any("already exists" in e
               for e in po.validate_entry(existing, _entry(host), catalog))
    assert any("base_url" in e for e in po.validate_entry(
        "newprov", _entry(host, base_url="ftp://x"), catalog))
    assert any("api_kind" in e for e in po.validate_entry(
        "newprov", _entry(host, api_kind="openai_codex"), catalog))
    assert any("auth_env" in e for e in po.validate_entry(
        "newprov", _entry(host, auth_env="lower"), catalog))
    assert any("served_models" in e for e in po.validate_entry(
        "newprov", _entry(host, served_models=[]), catalog))
    assert any("unknown model family" in e for e in po.validate_entry(
        "newprov", _entry(host, served_models=[{"family": "nope"}]), catalog))


# ---- overlay merge into the live Lua config -----------------------------------

def test_apply_to_host_adds_provider_and_served_by(host):
    fam = next(iter(host.catalog()["models"]))
    applied = po.apply_to_host(host, {"providers": {"newprov": _entry(host)}})
    assert applied == ["newprov"]
    host.init()
    catalog = host.catalog()
    assert catalog["providers"]["newprov"]["base_url"] == "https://api.example.com/v1"
    served = catalog["models"][fam]["served_by"]
    mine = [s for s in served if s.get("provider") == "newprov"]
    assert len(mine) == 1 and mine[0]["provider_model_id"] == "wire-id"
    # the new provider actually enters ranking for that family
    ranked, rejected = host.rank({"profile": "default",
                                  "requirements": {"context": 1000}})
    pairs = {(r["candidate"]["provider_id"], r["candidate"]["model_family"])
             for r in ranked} | {(r.get("provider_id"), r.get("model_family"))
                                 for r in rejected}
    assert any(p == "newprov" for p, _ in pairs)


def test_apply_to_host_never_overwrites_and_never_duplicates(host):
    existing = next(iter(host.catalog()["providers"]))
    before = host.catalog()["providers"][existing]
    assert po.apply_to_host(host, {"providers": {existing: _entry(host)}}) == []
    assert host.catalog()["providers"][existing] == before
    # applying the same new provider twice -> one served_by row
    fam = next(iter(host.catalog()["models"]))
    po.apply_to_host(host, {"providers": {"newprov": _entry(host)}})
    po.apply_to_host(host, {"providers": {"newprov": _entry(host)}})
    served = host.catalog()["models"][fam]["served_by"]
    assert len([s for s in served if s.get("provider") == "newprov"]) == 1


def test_overlay_load_save_roundtrip(tmp_path):
    path = tmp_path / "providers.local.json"
    assert po.load_overlay(path) == {"providers": {}}
    overlay = {"providers": {"groq": {"base_url": "https://x", "auth_env": "G_KEY",
                                      "served_models": [{"family": "f"}]}}}
    po.save_overlay(overlay, path)
    assert po.load_overlay(path) == overlay
    path.write_text("not json")
    assert po.load_overlay(path) == {"providers": {}}


# ---- shim hot-add endpoint -----------------------------------------------------

def test_shim_add_provider_hot_applies_with_state_preserved(host, monkeypatch):
    client = TestClient(create_app(host, default_profile="default"))
    fam = next(iter(host.catalog()["models"]))
    # live state that must survive the re-init
    host.update_metrics("openrouter" if "openrouter" in host.catalog()["providers"]
                        else next(iter(host.catalog()["providers"])), fam,
                        {"price_in": 9.75, "price_out": 19.5})
    ema_before = {k: v for k, v in host.dump_state()["ema_metrics"].items()
                  if v.get("price_in") == 9.75}
    assert ema_before

    r = client.post("/x/providers", json={
        "id": "hotprov", "base_url": "https://hot.example.com/v1",
        "auth_env": "HOTPROV_API_KEY", "key": "sk-hot-123",
        "served_models": [{"family": fam}]})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["key_installed"] is True

    import os
    assert os.environ.get("HOTPROV_API_KEY") == "sk-hot-123"
    assert host._env.get("HOTPROV_API_KEY") == "sk-hot-123"
    assert "hotprov" in host.catalog()["providers"]
    ema_after = host.dump_state()["ema_metrics"]
    for k, v in ema_before.items():
        assert ema_after[k]["price_in"] == 9.75, "EMA state lost across re-init"
    os.environ.pop("HOTPROV_API_KEY", None)

    # duplicate -> 409; invalid -> 400
    assert client.post("/x/providers", json={
        "id": "hotprov", "base_url": "https://hot.example.com/v1",
        "auth_env": "HOTPROV_API_KEY",
        "served_models": [{"family": fam}]}).status_code == 409
    assert client.post("/x/providers", json={
        "id": "badprov", "base_url": "https://x.example.com",
        "auth_env": "X_KEY", "served_models": [{"family": "nope"}]
    }).status_code == 400


# ---- dashboard add endpoint -----------------------------------------------------

def test_dashboard_add_provider_persists_and_hot_applies(monkeypatch, tmp_path):
    import auth_proxy
    env_path = tmp_path / ".env.secrets"
    overlay_path = tmp_path / "providers.local.json"
    monkeypatch.setattr(auth_proxy, "DASHBOARD_KEY_ENV_PATH", str(env_path))
    monkeypatch.setenv("PROVIDERS_OVERLAY_PATH", str(overlay_path))
    monkeypatch.setattr(auth_proxy, "_load_policy_config", lambda: {
        "providers": {"openrouter": {}},
        "models": {"llama-3.3-70b": {"served_by": []}},
    })
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET", "test-secret")

    calls = []

    class FakeResp:
        status_code = 200

    class FakeClient:
        async def post(self, url, json=None, timeout=None):
            calls.append((url, json))
            return FakeResp()

    monkeypatch.setattr(auth_proxy, "_client", FakeClient())

    payload = {"id": "groq", "base_url": "https://api.groq.com/openai/v1",
               "tier": "partner", "auth_env": "GROQ_API_KEY", "key": "gsk-raw",
               "served_models": [{"family": "llama-3.3-70b",
                                  "provider_model_id": "llama-3.3-70b-versatile"}]}

    # gated on admin role now (not a per-name 'admin' gate): anonymous 401,
    # any admin allowed (the admin session below is just one such admin).
    assert TestClient(auth_proxy.app).post(
        "/dashboard/api/provider-keys/add", json=payload).status_code == 401

    admin = TestClient(auth_proxy.app)
    admin.cookies.set(auth_proxy.DASHBOARD_COOKIE_NAME,
                       auth_proxy._make_dashboard_session("admin"))
    r = admin.post("/dashboard/api/provider-keys/add", json=payload)
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["applied_live"] is True

    assert "GROQ_API_KEY=gsk-raw" in env_path.read_text()         # key persisted
    saved = json.loads(overlay_path.read_text())
    assert saved["providers"]["groq"]["auth_env"] == "GROQ_API_KEY"
    assert "key" not in saved["providers"]["groq"]                 # never in overlay
    url, body = calls[0]
    assert url.endswith("/x/providers") and body["key"] == "gsk-raw"

    # second add of the same id is rejected (overlay merged into catalog view)
    monkeypatch.setattr(auth_proxy, "_load_policy_config",
                        lambda: auth_proxy._merge_provider_overlay({
                            "providers": {"openrouter": {}},
                            "models": {"llama-3.3-70b": {"served_by": []}}}))
    assert admin.post("/dashboard/api/provider-keys/add",
                       json=payload).status_code == 400


def test_merge_provider_overlay_folds_into_catalog(monkeypatch, tmp_path):
    import auth_proxy
    overlay_path = tmp_path / "providers.local.json"
    overlay_path.write_text(json.dumps({"providers": {"groq": {
        "base_url": "https://api.groq.com/openai/v1", "auth_env": "GROQ_API_KEY",
        "served_models": [{"family": "llama-3.3-70b",
                           "provider_model_id": "llama-3.3-70b-versatile"}],
        "added_at": 1}}}))
    monkeypatch.setenv("PROVIDERS_OVERLAY_PATH", str(overlay_path))
    cfg = auth_proxy._merge_provider_overlay({
        "providers": {"openrouter": {}},
        "models": {"llama-3.3-70b": {"served_by": [{"provider": "openrouter"}]}}})
    assert cfg["providers"]["groq"]["api_kind"] == "openai_compatible"
    served = cfg["models"]["llama-3.3-70b"]["served_by"]
    assert {"provider": "groq", "provider_model_id": "llama-3.3-70b-versatile"} in served


def test_dashboard_html_has_add_provider_form(monkeypatch):
    import auth_proxy
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET", "test-secret")
    client = TestClient(auth_proxy.app)
    client.cookies.set(auth_proxy.DASHBOARD_COOKIE_NAME,
                       auth_proxy._make_dashboard_session("admin"))
    html = client.get("/dashboard").text
    for needle in ("toggleAddProvider", "addProvSubmit", "addProvModels",
                   "/dashboard/api/provider-keys/add", "parseServedModels"):
        assert needle in html, needle
