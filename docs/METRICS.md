# Metrics: the EMA seed (`--metrics`)

The router ranks candidates partly on **price, latency, quality and success
rate**. Those live in `ctx.state` as exponential moving averages (EMAs). The host
**seeds** them at init from a metrics file (`--metrics <file>`), and the **live
EMA overwrites the seed as real calls happen**. So the seed only matters for the
cold start (before a candidate has been called).

```bash
python serve.py --config config.live.lua --metrics metrics.live.lua --default-profile agent
```

## ⚠️ The committed `metrics.live.lua` is a PLACEHOLDER, not real data

The `metrics.live.lua` checked into this repo is a **hand-written seed with
fake/representative numbers** — enough to demonstrate policy routing (and to make
`codex` rank first; see below). It is **not** scraped live pricing and **not**
authoritative. Treat it as an example. Two things make it safe:

1. The live EMA overwrites every seeded value once traffic flows.
2. AntSeed price caps are enforced by the **buyer proxy** (`maxPricing`), not by
   these numbers — so a stale/optimistic seed can't make you overpay on AntSeed.

For a real deployment, regenerate it (below) or just let the EMA learn from a warm-up.

## Format

Keys are `"<model_family>@<provider_id>"`. Provider ids must match
`config.live.lua` (`openai`, `antseed`, `openrouter`, …).

```lua
return {
  generated_at_iso = "2026-06-02T00:00:00Z",
  providers = { openrouter = { last_seen_ok = "…", free_credits_remaining_usd = 50.0 } },
  models = {
    ["gpt-5.5@antseed"] = {
      price_in_usd_per_mtok  = 0.17,
      price_out_usd_per_mtok = 0.57,
      success_rate_24h       = 0.97,
      last_quality_eval      = 0.95,
      tok_s_p50 = 40, ttft_ms_p50 = 400,   -- optional
    },
  },
}
```

## The one load-bearing value: `codex ≈ 0`

`gpt-5.5-codex@openai` is seeded at **price 0** on purpose. The ChatGPT/Codex
subscription is a sunk cost, so its marginal price is ≈ 0 until the weekly cap.
That zero is what makes a cost-led policy rank codex first (then cascade on
its rate-limit). Without it, at a tie-on-cost cold start the rank falls back to
quality and codex won't lead.

## Regenerating realistic AntSeed prices

The live AntSeed prices come from the network, queryable without spending:

```bash
antseed network browse --services --json --top 50 > browse.json
```

Each peer's `metadata.providers[].servicePricing[<model>]` gives
`inputUsdPerMillion` / `outputUsdPerMillion`. Take, per model, the **cheapest
peer** as the `@antseed` seed (the buyer proxy will pick a peer at or below the
policy's per-call ceiling anyway). OpenRouter prices come from its model list /
pricing page. Codex stays at 0.

This is only a seed — once the proxies are live, the EMA is the source of truth.
