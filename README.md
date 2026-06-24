# unhardcoded

An OpenAI-compatible **LLM router**. Instead of hardcoding a model, a caller
sends a **policy** with each request: the host filters the candidate models to
those that qualify, ranks the survivors and picks one (e.g. the cheapest that
passes) — over your own provider keys — and falls back automatically when a
provider errors. Every run records which models passed, which were rejected and
why.

Concretely it's an async FastAPI shim that runs the
[`unhardcoded-engine`](https://github.com/genlayerlabs/unhardcoded-engine) core
and inherits its provider selection, fallback, retry and per-provider auth. The core
is vendored as a git submodule under `core/`; this repo is the *host* (the I/O,
auth, providers, the HTTP service) plus an operator **dashboard** to compose,
test and analyse policies. The policy *algebra* (Σ_pol) lives in the core.

## The model: callers send a policy

A request carries its own **Σ_pol policy** as data — a `policy_ir` term in the
body. The host admits it (sorts, arity, depth/size bounds), ∧-composes the
host's `policy_envelope` so a caller can only *narrow* the host's invariants,
then runs it: filter the candidate models, score the survivors, pick and fall
back. Compose, preview, **test live** and download policies in the dashboard
Builder.

There are **no intelligence tiers**. When a caller sends no policy, a single
declarative `default` policy (balanced quality/cost) is the fallback — itself a
Σ_pol term with an identity, editable like any other.

## Layout

```
core/                  -- git submodule -> genlayerlabs/unhardcoded-engine (the pure Σ_pol core)
llm_router_host.py     -- embeds the core via lupa; sync/async backends; auth resolver
shim.py                -- OpenAI-compatible app: /v1/chat/completions (+ per-call policy_ir) and /x/* operator endpoints
auth_proxy.py          -- ingress + operator dashboard (Analytics · Builder · Activity · Market · Settings)
serve.py               -- entry point: wires the api_kind dispatcher + runs uvicorn
config.live.lua        -- catalog (providers + models), the `default` policy, and the observation fields (incl. OpenRouter benchmark/modality/capability fields)
scripts/refresh_model_meta.py -- the job: writes model_meta.lua (model-level traits pulled from OpenRouter)
codex_auth.py / codex_backend.py -- the ChatGPT-subscription (Codex) provider
metrics.live.lua       -- EMA seed (PLACEHOLDER/fake — see docs/METRICS.md)
docs/METRICS.md        -- the metrics seed: format, codex≈0, regeneration (it's fake)
docs/PROVIDERS.md      -- per-provider auth + the AntSeed node
docs/OPENAI-CODEX.md   -- the ChatGPT-subscription provider (unofficial / ToS-risky)
docs/USAGE_ENDPOINTS.md -- per-key usage stats API endpoints and auth behavior
live_smoke.py          -- drive real providers end-to-end
tests/                 -- the full host unit-test suite
features/              -- BDD user-flow suite (behave); user_flows.json is the spec
SETUP.md               -- agent/human setup runbook (clone -> running -> first call)
scripts/gen-dev-wallet.sh -- generate a local AntSeed dev wallet (testing)
```

## Quickstart

Point a coding agent at **[`SETUP.md`](./SETUP.md)** ("set up unhardcoded by
following SETUP.md") for a guided, copy-pasteable runbook — or do it yourself:

```bash
git clone --recursive https://github.com/genlayerlabs/unhardcoded.git && cd unhardcoded
cp .env.example .env.secrets && chmod 600 .env.secrets
# in .env.secrets: set OPENROUTER_API_KEY, and DASHBOARD_NO_AUTH=1 for local dev
docker compose up -d --build
curl -fsS http://127.0.0.1:8080/healthz            # -> {"ok":true,"initialized":true}
```

Open the dashboard at **http://127.0.0.1:8080/dashboard**, mint a caller key
(`POST /dashboard/api/keys {"consumer":"my-app"}`, or Consumers → Generate key),
then call:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
  -d '{"model":"","messages":[{"role":"user","content":"Reply: pong"}]}'
```

Empty `model` runs the `default` policy; or send `family:`/`pin:`/`profile:` or a
per-call `policy_ir`/`flow_ir` (below). Optional providers (Codex, AntSeed) and
troubleshooting are in [`SETUP.md`](./SETUP.md).

### Develop without Docker (raw shim — no dashboard/auth)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python serve.py --config config.live.lua --default-profile default --host 127.0.0.1 --port 8080
```

`serve.py` is the data plane only (no ingress auth, no dashboard); provider keys
come from the process env. The bearer-token contract and the dashboard live in
the `ingress` service (`docker compose`). *(Nix users: a `nix-shell` with the
same packages works too.)*

## Policies (the `model` field + `policy_ir`)

Routing is **server-side**. The primary path is a per-call policy; the `model`
prefix is sugar for the common cases:

| client sends                    | host does                                               |
|---------------------------------|---------------------------------------------------------|
| `policy_ir` term in the body    | run that Σ_pol policy (the primary path)                |
| `""` / unprefixed `model`       | the `default` policy                                    |
| `model = "profile:NAME"`        | a named profile from the catalog (only `default` ships) |
| `model = "family:FAMILY"`       | default, pinned to a model family                       |
| `model = "pin:PROVIDER/FAMILY"` | default, pinned to one (provider, family)               |

A **policy is a Σ_pol term**: `filter` (which models qualify) → `score` (rank
the survivors) → `pick` (the selector) + request transform + failover. Write it
as raw IR, or compose it in the Builder (structured rows ↔ raw term). The
vocabulary it filters/scores over includes live fields (`price_in`/`price_out`,
`latency_ms`, `success_rate`, …) and host-declared fields — the OpenRouter
benchmarks/modalities/capabilities (`bench_intelligence`, `in_image`,
`cap_tools`, …). See the core's `core/docs/SIGMA-POL.md` for the algebra;
`config.live.lua` declares the host fields and the `default` policy.

## Dashboard

`auth_proxy` serves an operator console at `/dashboard`:

- **Analytics** — spend, traffic and errors over time, filtered by timeframe,
  consumer, provider and model.
- **Builder** — compose a policy over raw + benchmark fields, preview the live
  ranking, **Test call** it with a prompt, and download the term to run per call.
- **Activity** — per-request trace: the policy that ran (copyable), the ordered
  fallback chain with the error at each step, and the cost paid.
- **Market** — live price book per model family.
- **Settings** — consumer keys and provider keys.

## Tests

Install the test deps once (a virtualenv keeps them off your system Python):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

**Unit** — boots a real host with mocked provider responses (only the outbound
HTTP to upstream providers is mocked):

```bash
python -m pytest tests -q
```

**BDD user-flow suite** (`features/`, behave) — drives the live stack end to end
the way the dashboard does and asserts the rendered data is correct, including a
real headless-browser pass (needs Chrome/Chromium installed; Selenium fetches the
matching driver). Free & repeatable: end-to-end chats route to a $0 path. The
catalogue of flows it covers is [`user_flows.json`](./user_flows.json):

```bash
behave
```

*(Nix users: `nix-shell -p ...` with the same packages — plus `chromium
chromedriver` for the browser pass — works as before.)*
