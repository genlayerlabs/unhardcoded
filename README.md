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
tests/                 -- the full host test suite
```

## Clone (with the core submodule)

```bash
git clone --recursive https://github.com/genlayerlabs/unhardcoded.git
# or, after a plain clone:
git submodule update --init
```

## Run

```bash
nix-shell -p 'python3.withPackages(ps: with ps; [lupa httpx fastapi uvicorn pydantic])' \
    --run 'python serve.py --config config.live.lua --default-profile default --host 127.0.0.1 --port 8080'
```

Put provider keys in the environment (`OPENROUTER_API_KEY`, `HEURIST_API_KEY`,
…). For the ChatGPT-subscription (Codex) provider, `codex login` first and pass
`--codex-auth ~/.codex/auth.json` — see [`docs/OPENAI-CODEX.md`](./docs/OPENAI-CODEX.md)
(unofficial / ToS-risky).

A client points at the shim and either sends its own policy or lets the
`default` decide:

```ini
endpoint = "http://127.0.0.1:8080/v1/chat/completions"
api_key  = "dummy"   # required non-empty; the shim ignores client auth
model    = ""        # empty -> the default policy; or family:/pin:/profile: (below)
# for a custom policy, POST a `policy_ir` term in the request body instead.
```

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

```bash
nix-shell -p 'python3.withPackages(ps: with ps; [lupa httpx fastapi uvicorn pydantic pytest pytest-asyncio])' \
    --run 'python -m pytest tests -q'
```

The tests boot a real host with mocked provider responses; only the outbound
HTTP to upstream providers is mocked. The Codex live streaming call is not
covered (no subscription in CI); everything around it is.
