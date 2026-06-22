"""
Behave environment for the unhardcoded user-flow BDD suite.

Drives the LIVE local stack (ingress :8080 + router) — the same endpoints the
dashboard frontend consumes — so the assertions prove the data the UI renders is
present AND correct, not just that endpoints return 200.

Assumptions (local-dev):
  * stack up at BASE_URL (default http://127.0.0.1:8080)
  * DASHBOARD_NO_AUTH=1 so /dashboard/api/* is reachable as admin
  * a working $0 route exists for family gpt-5.5 (codex) so chat tests are FREE

Run: nix-shell -p "python3.withPackages(ps: with ps; [behave requests])" \
        --run 'behave features'
"""
import os
import json
import requests

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8080")

# A $0, price-first policy pinned to gpt-5.5 -> resolves to codex (subscription,
# cost 0). Keeps every end-to-end chat test free.
FREE_POLICY_IR = [
    "policy",
    ["and", ["meets_req"], ["not", ["is", "disabled"]], ["family_eq", "gpt-5.5"]],
    ["neg", ["normalize", ["field", "price_in"]]],
    ["argmax"], ["id"], ["always", {"action": "next_candidate"}],
]


def _mint_caller_token(consumer="bdd-test"):
    # NO_AUTH dashboard -> we can mint a consumer key with no session.
    r = requests.post(f"{BASE_URL}/dashboard/api/keys",
                      json={"consumer": consumer}, timeout=30)
    r.raise_for_status()
    return r.json()["api_key"]


def before_all(context):
    context.base_url = BASE_URL
    context.session = requests.Session()

    # Sanity: stack is up.
    h = requests.get(f"{BASE_URL}/healthz", timeout=10)
    assert h.status_code == 200, f"stack not healthy: {h.status_code}"

    # Sanity: dashboard is reachable without a login (NO_AUTH expected locally).
    d = requests.get(f"{BASE_URL}/dashboard/api/full", timeout=10)
    assert d.status_code == 200, (
        "dashboard /api/full needs DASHBOARD_NO_AUTH=1 for the BDD suite "
        f"(got {d.status_code}). Set it in .env.secrets and restart ingress."
    )

    context.caller_token = _mint_caller_token()

    # Seed REAL activity so Activity / usage / stats / catalog have data to show:
    # one chat + one 2-node flow, both $0 via codex.
    _seed_activity(context)


def before_scenario(context, scenario):
    tags = scenario.effective_tags
    if "antseed" in tags:
        # Only run AntSeed scenarios when the funded sidecar is actually up;
        # otherwise skip (keeps the default suite green without a wallet).
        try:
            w = (requests.get(f"{BASE_URL}/dashboard/api/market", timeout=10)
                 .json().get("wallet") or {})
        except Exception:
            w = {}
        if w.get("connection") != "connected":
            scenario.skip("antseed sidecar not up/funded (wallet not connected)")
            return
        # Real-money spend is gated behind an explicit opt-in env var.
        if "spend" in tags and os.environ.get("RUN_ANTSEED_SPEND") != "1":
            scenario.skip("real-money antseed spend — set RUN_ANTSEED_SPEND=1 to run")
            return
    if "browser" in scenario.effective_tags:
        import shutil
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1400,2000")
        opts.binary_location = shutil.which("chromium") or shutil.which("chromium-browser")
        service = Service(executable_path=shutil.which("chromedriver"))
        context.driver = webdriver.Chrome(service=service, options=opts)
        context.driver.set_page_load_timeout(60)


def after_scenario(context, scenario):
    d = getattr(context, "driver", None)
    if d is not None:
        try:
            d.quit()
        except Exception:
            pass
        context.driver = None


def _seed_activity(context):
    hdr = {"Authorization": f"Bearer {context.caller_token}",
           "Content-Type": "application/json"}
    # one routed chat (codex, $0)
    requests.post(f"{context.base_url}/v1/chat/completions", headers=hdr, json={
        "model": "", "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with one short sentence."}],
        "policy_ir": FREE_POLICY_IR,
    }, timeout=120)
    # one flow (two codex nodes -> merge), $0
    flow = ["flow", {
        "q": {"kind": "input"},
        "a": {"kind": "llm", "system": "Be concise.",
              "policy": FREE_POLICY_IR, "inputs": ["q"]},
        "b": {"kind": "llm", "system": "Synthesize.",
              "policy": FREE_POLICY_IR, "inputs": ["a"],
              "template": "Refine: $1"},
        "out": {"kind": "output", "inputs": ["b"]},
    }]
    requests.post(f"{context.base_url}/v1/chat/completions", headers=hdr, json={
        "model": "", "max_tokens": 120,
        "messages": [{"role": "user", "content": "Say hello."}],
        "flow_ir": flow,
    }, timeout=180)
    context.seeded = True


# ---- helpers used by steps ------------------------------------------------

def auth_headers(context):
    return {"Authorization": f"Bearer {context.caller_token}",
            "Content-Type": "application/json"}


def jpath(obj, path):
    """Tiny dotted/indexed JSON getter: 'a.b[0].c'. Returns SENTINEL if missing."""
    cur = obj
    for part in _tokenize(path):
        try:
            if isinstance(part, int):
                cur = cur[part]
            else:
                cur = cur[part]
        except (KeyError, IndexError, TypeError):
            return _MISSING
    return cur


class _Missing:
    def __repr__(self):
        return "<MISSING>"


_MISSING = _Missing()
SENTINEL = _MISSING


def _tokenize(path):
    out = []
    for seg in path.split("."):
        while "[" in seg:
            name, rest = seg.split("[", 1)
            if name:
                out.append(name)
            idx, seg = rest.split("]", 1)
            out.append(int(idx))
        if seg:
            out.append(seg)
    return out
