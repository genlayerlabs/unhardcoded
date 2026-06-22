# Set up unhardcoded — agent runbook

A step-by-step setup you can follow yourself, or **point your coding agent at
this file** and say *"set up unhardcoded by following SETUP.md"*. It takes you
from nothing to a running router + dashboard and a first routed call.

Every step is a copy-pasteable command with the expected result. Stop and fix if
a result doesn't match.

---

## Prerequisites

- **git** and **Docker** (with `docker compose`).
- At least one provider key to route over. The easiest is an **OpenRouter** key
  (`OPENROUTER_API_KEY`) — get one at https://openrouter.ai/keys. Codex (ChatGPT
  subscription) and AntSeed (on-chain marketplace) are optional, covered at the end.

> The core Σ_pol engine is a **git submodule**. If you skip `--recursive` the
> Docker build fails fast at `test -f core/router.lua` — that's the missing
> submodule, fix with step 1b.

---

## 1. Clone with the submodule

```bash
git clone --recursive https://github.com/genlayerlabs/unhardcoded.git
cd unhardcoded
```

**1b.** If you already cloned without `--recursive`:

```bash
git submodule update --init
```

Expected: `core/router.lua` and `core/llm_policy.lua` exist.

---

## 2. Configure secrets

```bash
cp .env.example .env.secrets
chmod 600 .env.secrets
```

Edit `.env.secrets` and set, at minimum:

```ini
OPENROUTER_API_KEY=sk-or-...        # your provider key
DASHBOARD_NO_AUTH=1                 # LOCAL DEV ONLY: open the dashboard without a login
PUBLIC_BASE_URL=http://127.0.0.1:8080/v1
```

> `DASHBOARD_NO_AUTH=1` makes the dashboard reachable as admin with no password —
> **only on a machine no one else can reach.** For a real deployment, instead set
> `DASHBOARD_PASSWORD_SHA256` + `DASHBOARD_SESSION_SECRET` (see `.env.example`).

---

## 3. Start the stack

```bash
docker compose up -d --build
```

This builds `unhardcoded:local` and starts two services: `router` (the engine +
data plane) and `ingress` (auth proxy + dashboard, bound to `127.0.0.1:8080`).

Expected: `docker compose ps` shows both **healthy**.

---

## 4. Verify it's up

```bash
curl -fsS http://127.0.0.1:8080/healthz
# -> {"ok":true,"initialized":true}
```

Open the dashboard: **http://127.0.0.1:8080/dashboard** (loads straight in with
`DASHBOARD_NO_AUTH=1`). Tabs: Analytics · Builder · Activity · Catalog · Config ·
Consumers · Provider keys.

---

## 5. Mint a consumer (caller) key

The `/v1` endpoint requires a per-service bearer token. Generate one from the
dashboard API (works under `DASHBOARD_NO_AUTH`):

```bash
curl -s -X POST http://127.0.0.1:8080/dashboard/api/keys \
  -H 'content-type: application/json' -d '{"consumer":"my-app"}'
# -> {"api_key":"llmr_...", "sha256_prefix":"...", ...}   (shown ONCE — save it)
```

(Or in the dashboard: **Consumers → Generate key**, then copy the setup blurb.)

---

## 6. Make your first routed call

Point any OpenAI-compatible client at the ingress with that token. Empty `model`
runs the `default` policy (the router picks/falls-back over your keys):

```bash
TOKEN=llmr_...   # from step 5
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"","messages":[{"role":"user","content":"Reply: pong"}],"max_tokens":8}'
```

Expected: a `chat.completion`. The non-standard `x_router` field shows which
provider/model was chosen and the full decision trace. The run also appears in
the dashboard **Activity** tab.

OpenAI SDK form:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="llmr_...")
r = client.chat.completions.create(model="", messages=[{"role":"user","content":"hi"}])
print(r.choices[0].message.content)
```

**You now have a working router.** Send `model="family:<name>"`, `pin:<provider>/<family>`,
`profile:<name>`, or a per-call `policy_ir` / `flow_ir` term in the body — see the
[README](./README.md) Policies table and `SKILL.md` for authoring.

---

## Optional providers

### Codex (ChatGPT subscription) — unofficial / ToS-risky
`codex login` to produce `~/.codex/auth.json`, set `CODEX_AUTH_PATH` to it (the
compose mounts it into the router), and restart. See
[`docs/OPENAI-CODEX.md`](./docs/OPENAI-CODEX.md). The token auto-refreshes; an
expired/dead token disables the provider.

### AntSeed (decentralized marketplace) — REAL on-chain money
Opt-in (`--profile antseed`) and spends real USDC on Base mainnet. Use a
**dedicated dev wallet**, never a production key:

```bash
./scripts/gen-dev-wallet.sh        # prints ANTSEED_IDENTITY_HEX + ANTSEED_CONTROL_TOKEN
# paste both into .env, then:
docker compose --profile antseed up -d --build
docker compose exec antseed antseed buyer balance --json   # -> the address to fund
# fund that address with a little USDC + ETH (gas) on Base, then Deposit it into
# escrow from the dashboard Catalog (wallet cell).
```

See [`docs/PROVIDERS.md`](./docs/PROVIDERS.md). Note: the deposits contract locks
funds — an immediate `withdraw` after `deposit` reverts.

---

## Run the test suites

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Host unit tests (mocked providers, no spend):

```bash
python -m pytest tests -q
```

End-to-end BDD user-flow suite (drives the live stack; chats route to a $0 path
so it's free; the browser pass needs Chrome/Chromium installed):

```bash
behave
```

*(Nix users: a `nix-shell` with the same packages works too.)*

The full catalogue of user flows is in [`user_flows.json`](./user_flows.json);
the `.feature` files under `features/` are lowered from it.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Docker build fails at `test -f core/router.lua` | Submodule not cloned → `git submodule update --init` |
| Dashboard returns 401 / shows a login | `DASHBOARD_NO_AUTH=1` not set (or set a password) and restart `ingress` |
| `/v1` → 401 `caller_auth` | Missing/unknown bearer token → mint one (step 5) |
| `/v1` → 403 `caller_route_not_allowed` | The key's `allowed_routes` blocks that route |
| Chat → 502/`payment_required` from a provider | That provider has no credit / no funded wallet; the router falls back to others |
| `genlayer-web` network error on `compose up` | The compose references an external network — create it (`docker network create genlayer-web`) or remove that network from `compose.yml` for a standalone run |
