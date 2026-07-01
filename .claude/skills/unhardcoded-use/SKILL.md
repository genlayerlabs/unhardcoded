---
name: unhardcoded-use
description: >-
  Use a running unhardcoded router — as a CALLER (send an OpenAI-compatible
  request carrying a Σ_pol policy_ir / Σ_flow flow_ir, or a profile/family/pin
  model string; session pinning; streaming; dry-run before spending) and as an
  OPERATOR (dashboard, /x/* endpoints, mint keys, hot-add providers, AntSeed
  wallet). Load this when integrating an app or agent against the router, wiring
  a client, minting keys, debugging a route, or operating the host. To AUTHOR a
  policy term against this host's live catalog use the root SKILL.md
  (sigma-policy-author); to develop the host itself use unhardcoded-contribute.
---

# Using a running unhardcoded router

The router is an **OpenAI-compatible endpoint**. You do not pick a model name —
you send a **policy** (data) and the host filters/ranks/picks a `(provider,
model)` over the *operator's* keys, falling back on provider errors. When a
policy is present the request's `model` field is ignored for selection.

There are two roles: **caller** (hits `/v1/*` with a bearer key) and **operator**
(the dashboard + the internal `/x/*` endpoints). This skill orients both; the
detail lives in `docs/` (linked inline).

## As a caller

**Auth.** Bearer key, prefix `llmr_…`. Mint one from the dashboard (Consumers)
or `POST /dashboard/api/keys`. Send it as `Authorization: Bearer llmr_…`.

**Send a call.** `POST /v1/chat/completions` (also `POST /v1/responses`), body =
standard OpenAI + one of:
- `policy_ir`: a Σ_pol term — decides which model serves this one call.
- `flow_ir`: a Σ_flow term (`["flow", nodes]`) — a DAG of calls, each node
  carrying its own policy.
- no policy → the host's `default` policy runs. Or steer via the `model` string:
  `profile:<name>`, `family:<f>`, `pin:<provider>/<family>`. You can also
  path-address a profile: `POST /{profile}/v1/chat/completions`.

**Session pinning.** Set `X-Unhardcoded-Session: <conversation-id>` (or a
`session` body field — body wins). The router keeps the conversation on its
cache-hot peer (warm prompt-cache across turns) and meters spend per session.

**Streaming.** `stream: true`. Errors *before the first token* return a normal
JSON error (no partial SSE commit), so client-side fallback still works.

**Other caller endpoints.** `GET /v1/models` (families served), `POST /v1/compact`
(stateless context sealing — hand it your turns, get a sealed summary back).

**Dry-run before you spend** (these cost nothing):
- `POST /x/policy/normalize` — admit a `policy_ir` + get its `fingerprint`/version.
- `POST /x/rank` — the ordered candidates this host *would* try for a term.
- `POST /x/policy/build` — elaborate a declarative spec → a `policy_ir`.
- `POST /x/flow/normalize` — admit + identify a `flow_ir`.
- `GET /x/fields` — the live field schema a policy may reference.

**Authoring the policy itself is a separate skill.** The root `SKILL.md`
(`sigma-policy-author`) is the canonical authoring guide: it embeds this host's
**live** model/provider catalog and field vocabulary and is test-gated so it can
only teach ops the core actually admits. Load it to write terms; the normative
grammar is `core/docs/SIGMA-POL.md`.

## As an operator

The `/x/*` endpoints are **operator-only** — the ingress proxy hides them from
consumers. The **dashboard** (Analytics · Debugger · Activity · Catalog · SKILL.md · Config · Settings)
wraps most of them:
- **Keys:** mint/list consumer keys.
- **Providers:** hot-add or re-key a provider at runtime — `POST /x/providers`,
  `POST /x/provider-key` (persisted as overlays; no redeploy).
- **Observe:** `GET /x/runtime` (breakers, EMA metrics, source freshness),
  `GET /x/market` (full price book per family), `GET /x/calls` (recent ledger),
  `GET /x/session/{sid}` and `GET /x/sessions` (per-session meters).
- **Reload:** `POST /x/config/reload`, `POST /x/codex/reload`.
- **AntSeed wallet:** `POST /x/wallet/deposit|withdraw|refresh` (funds the
  marketplace buyer escrow — see the `antseed-prod-deposit` runbook for prod).

## Gotchas

- **`model` is ignored when a policy is present** — the policy drives selection.
- **The host envelope only narrows.** `config.policy_envelope` is `∧`-composed
  onto every caller policy: a caller can tighten the host's invariants, never
  widen them. A term that passes your local check can still be narrowed by the host.
- **Seed prices are a placeholder.** `metrics.live.lua` is intentionally fake
  (see `docs/METRICS.md`); real EMA prices/latency accrue from live calls. Don't
  trust the seed numbers for cost reasoning on a fresh host.
- **Codex ranks near-free by design** (subscription = sunk cost, seeded ≈ $0), so
  a cheapest-first policy prefers it when available — deliberate, not a bug.

## Where to read more

- Root `SKILL.md` — author Σ_pol/Σ_flow against this host's live catalog (the authoring canon).
- `docs/DEPLOY.md` — the client contract (OpenAI SDK base_url), deploy + provider modes.
- `docs/USAGE_ENDPOINTS.md` — per-key usage stats API + the consumer self-service model.
- `docs/PROVIDERS.md` — per-provider auth (bearer/oauth/none), the AntSeed node + wallet.
- `docs/OPENAI-CODEX.md` — the ChatGPT-subscription (Codex) provider, its wire shape and setup.
- `docs/METRICS.md` — the metrics seed format and why it's fake.
