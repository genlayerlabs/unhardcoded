"""
live_smoke.py — drive the router against real LLM providers.

Run from repo root:
    source .env
    nix-shell -p 'python3.withPackages(ps: [ps.lupa ps.httpx])' \
        --run 'python hosts/python_shim/live_smoke.py'

Each scenario prints the chosen provider, every attempted candidate, and the
trace decision path. Total cost should be a fraction of a cent (Heurist gives
free credits to the `genlayer` referral, OpenRouter free tier covers fallback).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # hosts/python_shim
ROOT = HERE.parents[1]                           # repo root
sys.path.insert(0, str(ROOT / "hosts" / "python"))

from llm_router_host import LLMRouterHost, make_http_call_provider  # noqa: E402


def banner(label: str) -> None:
    print()
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)


def show(result: dict) -> None:
    trace = result.get("trace") or {}
    ranked = trace.get("ranked") or []
    if ranked:
        print("  ranked:")
        for i, r in enumerate(ranked, 1):
            print(f"    {i}. {r['provider_id']:11s} {r['model_family']:18s} "
                  f"score={r['score']:.3f}  tier={r['tier']}")
    print("  attempts:")
    for evt in trace.get("decision_path") or []:
        event = evt.get("event")
        if event == "attempted":
            err = evt.get("error_kind") or "ok"
            print(f"    - attempted   {evt['provider_id']:11s} "
                  f"attempt={evt['attempt']}  "
                  f"{evt.get('latency_ms', 0)}ms  result={err}"
                  + (f"  http={evt['http_status']}" if evt.get("http_status") else ""))
        elif event == "retry_scheduled":
            print(f"    - retry       {evt['provider_id']:11s} "
                  f"attempt={evt['attempt']}  backoff={evt.get('backoff_ms')}ms")
        elif event == "provider_disabled":
            print(f"    - DISABLED    {evt['provider_id']:11s}  reason={evt['reason']}")
        elif event == "skipped":
            print(f"    - skipped     {evt['provider_id']:11s}  reason={evt['reason']}")

    if result["ok"]:
        chosen = result["chosen"]
        text   = (result["response"].get("text") or "").strip()
        tokens = result["response"].get("tokens_total")
        print(f"  chosen: {chosen['provider_id']} -> {chosen['served_model_id']}")
        print(f"  tokens: {tokens}")
        print(f"  text:   {text[:200]!r}")
    else:
        print(f"  ERROR: {result.get('error')}")

    print(f"  total: {trace.get('total_latency_ms')}ms")


def make_host() -> LLMRouterHost:
    h = LLMRouterHost(
        router_path = ROOT / "router.lua",
        config_path = HERE / "config.live.lua",
        call_provider = make_http_call_provider(),
    )
    h.init()
    return h


def disable(host: LLMRouterHost, provider_id: str, reason: str = "smoke_disabled") -> None:
    """Inject a disabled marker into RUNTIME via the test backdoor."""
    host.router._test.runtime()["disabled_providers"][provider_id] = reason


def main() -> int:
    missing = [k for k in ("HEURIST_API_KEY", "IONET_API_KEY", "OPENROUTER_API_KEY")
               if not os.environ.get(k)]
    if missing:
        print(f"Missing env vars: {missing}.  `source .env` first.", file=sys.stderr)
        return 2

    contract = {
        "prompt":     "Reply with exactly one word: pong",
        "profile":    "default",
        "max_tokens": 8,
        "temperature": 0.0,
        # Hard abort per-call: 8s. Distinct from scoring's max_latency_ms.
        "timeout_ms": 8000,
    }

    # ---- Scenario 1: happy path
    banner("1. Happy path — router picks top candidate")
    h = make_host()
    info = h.info()
    print(f"  loaded: providers={info['providers_loaded']}, candidates={info['candidates']}")
    show(h.execute(contract))

    # ---- Scenario 2: top partner disabled -> next partner
    banner("2. Top partner disabled -> cascade to second partner")
    h = make_host()
    # Pick whichever partner ended up at the top, knock it out, re-run
    ranked0, _ = h.rank(contract)
    if ranked0:
        top_provider = ranked0[0]["candidate"]["provider_id"]
        print(f"  (disabling top: {top_provider})")
        disable(h, top_provider)
    show(h.execute(contract))

    # ---- Scenario 3: both partners out -> fallback
    banner("3. Both partners out -> cascade to fallback")
    h = make_host()
    print("  (disabling: heurist + io_net)")
    disable(h, "heurist")
    disable(h, "io_net")
    show(h.execute(contract))

    # ---- Scenario 4: per-call policy_ir -> cheapest (cost-only)
    # (sigma-pol/v2) weighted overrides are gone; rank with a raw field scorer.
    banner("4. per-call policy_ir = cheapest (neg-normalized price)")
    h = make_host()
    show(h.execute({
        **contract,
        "policy_ir": ["policy",
                      ["and", ["meets_req"], ["not", ["is", "disabled"]]],
                      ["neg", ["normalize", ["field", "price_in"]]],
                      ["argmax"], ["id"], ["always", {"action": "next_candidate"}]],
    }))

    return 0


if __name__ == "__main__":
    sys.exit(main())
