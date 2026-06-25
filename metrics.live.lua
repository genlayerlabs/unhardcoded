-- metrics.live.lua — EMA seed for the live catalog (config.live.lua).
--
-- ⚠️ PLACEHOLDER / FAKE DATA. These numbers are hand-written, representative
-- seeds — NOT scraped live pricing and NOT authoritative. They exist only to
-- demonstrate tier routing (and to make codex rank first). The live EMA
-- overwrites every value once real traffic flows, and AntSeed price caps are
-- enforced by the buyer proxy (maxPricing), not by these numbers. Regenerate for
-- a real deployment — see docs/METRICS.md.
--
-- The router reads this at init to seed price/latency/quality EMAs; live calls
-- overwrite them as traffic flows. Keys are "<model_family>@<provider_id>".
--
-- The one economically-load-bearing value is codex at price 0: the ChatGPT
-- subscription is a sunk cost (≈0 marginal until its weekly cap), so with cost
-- as the dominant rank weight, codex wins the `edge` rank and is used first;
-- on its rate-limit the cascade falls to the cheapest candidate under the
-- $5/$25 ceiling. AntSeed seeds are the cheapest observed peers from
-- `antseed network browse`; OpenRouter seeds are representative frontier rates.
-- All are starting points — the live EMA corrects them.
return {
    generated_at_iso = "2026-06-04T09:35:54Z",

    providers = {
        openai_codex  = { last_seen_ok = "2026-06-04T09:35:54Z" },  -- codex (subscription)
        heurist       = { last_seen_ok = "2026-06-04T09:35:54Z" },
        io_net        = { last_seen_ok = "2026-06-04T09:35:54Z" },
        openai        = { last_seen_ok = "2026-06-04T09:35:54Z" },
        anthropic     = { last_seen_ok = "2026-06-04T09:35:54Z" },
        gemini        = { last_seen_ok = "2026-06-04T09:35:54Z" },
        bedrock_mantle = { last_seen_ok = "2026-06-04T09:35:54Z" },
        -- antseed prices/availability come from live marketplace offers
        -- (sources/antseed.py), not this seed.
        openrouter    = { last_seen_ok = "2026-06-04T09:35:54Z" },
    },

    models = {
        -- codex: sunk cost → 0 marginal (the pivot that puts it first).
        ["gpt-5.5@openai_codex"]            = { price_in_usd_per_mtok = 0.0,  price_out_usd_per_mtok = 0.0,  success_rate_24h = 0.99, last_quality_eval = 0.95 },
        ["gpt-5.3-codex-spark@openai_codex"] = { price_in_usd_per_mtok = 0.0,  price_out_usd_per_mtok = 0.0,  success_rate_24h = 0.99, last_quality_eval = 0.90 },

        -- gpt-5.5
        ["gpt-5.5@openai"]                  = { price_in_usd_per_mtok = 1.25, price_out_usd_per_mtok = 10.0, success_rate_24h = 0.99, last_quality_eval = 0.95 },
        ["gpt-5.5@openrouter"]              = { price_in_usd_per_mtok = 1.25, price_out_usd_per_mtok = 10.0, success_rate_24h = 0.99, last_quality_eval = 0.95 },

        -- gpt-5.4
        ["gpt-5.4@openai"]                  = { price_in_usd_per_mtok = 0.80, price_out_usd_per_mtok = 6.0,  success_rate_24h = 0.99, last_quality_eval = 0.90 },
        ["gpt-5.4@openrouter"]              = { price_in_usd_per_mtok = 0.80, price_out_usd_per_mtok = 6.0,  success_rate_24h = 0.99, last_quality_eval = 0.90 },

        -- claude-opus-4-8 (no codex path; antseed/openrouter only)
        ["claude-opus-4-8@anthropic"]       = { price_in_usd_per_mtok = 5.0,  price_out_usd_per_mtok = 25.0, success_rate_24h = 0.99, last_quality_eval = 0.93 },
        ["claude-opus-4-8@openrouter"]      = { price_in_usd_per_mtok = 5.0,  price_out_usd_per_mtok = 25.0, success_rate_24h = 0.99, last_quality_eval = 0.93 },

        -- gemini-3.1-pro-preview (no codex path)
        ["gemini-3.1-pro-preview@gemini"]   = { price_in_usd_per_mtok = 1.25, price_out_usd_per_mtok = 5.0,  success_rate_24h = 0.99, last_quality_eval = 0.92 },
        ["gemini-3.1-pro-preview@openrouter"]   = { price_in_usd_per_mtok = 1.25, price_out_usd_per_mtok = 5.0,  success_rate_24h = 0.99, last_quality_eval = 0.92 },

        -- affordable edge fallback verified against the current OpenRouter credit ceiling
        ["qwen3-235b-a22b@bedrock_mantle"]  = { price_in_usd_per_mtok = 0.20, price_out_usd_per_mtok = 0.60, success_rate_24h = 0.99, last_quality_eval = 0.90 },
        ["qwen3-235b-a22b@openrouter"]      = { price_in_usd_per_mtok = 0.20, price_out_usd_per_mtok = 0.60, success_rate_24h = 0.99, last_quality_eval = 0.90 },

        -- medium tier (free AntSeed only where the network browse showed $0 peers)
        ["claude-sonnet-4-6@anthropic"]     = { price_in_usd_per_mtok = 3.0,  price_out_usd_per_mtok = 15.0, success_rate_24h = 0.99, last_quality_eval = 0.88 },
        ["claude-sonnet-4-6@openrouter"]    = { price_in_usd_per_mtok = 3.0,  price_out_usd_per_mtok = 15.0, success_rate_24h = 0.99, last_quality_eval = 0.88 },
        ["deepseek-v4-pro@openrouter"]      = { price_in_usd_per_mtok = 0.50, price_out_usd_per_mtok = 1.5,  success_rate_24h = 0.99, last_quality_eval = 0.85 },
        ["glm-5.1@openrouter"]              = { price_in_usd_per_mtok = 0.60, price_out_usd_per_mtok = 2.0,  success_rate_24h = 0.99, last_quality_eval = 0.84 },
        ["kimi-k2.6@openrouter"]            = { price_in_usd_per_mtok = 0.55, price_out_usd_per_mtok = 2.5,  success_rate_24h = 0.99, last_quality_eval = 0.83 },
        ["minimax-m2.7@openrouter"]         = { price_in_usd_per_mtok = 0.10, price_out_usd_per_mtok = 0.50, success_rate_24h = 0.98, last_quality_eval = 0.80 },

        -- dummy tier
        ["deepseek-v4-flash@openrouter"]    = { price_in_usd_per_mtok = 0.10, price_out_usd_per_mtok = 0.40, success_rate_24h = 0.99, last_quality_eval = 0.76 },
        ["llama-3.3-70b@heurist"]           = { price_in_usd_per_mtok = 0.0,  price_out_usd_per_mtok = 0.0,  success_rate_24h = 0.94, last_quality_eval = 0.72 },
        ["llama-3.3-70b@io_net"]            = { price_in_usd_per_mtok = 0.12, price_out_usd_per_mtok = 0.30, success_rate_24h = 0.94, last_quality_eval = 0.72 },
        ["llama-3.3-70b@openrouter"]        = { price_in_usd_per_mtok = 0.12, price_out_usd_per_mtok = 0.30, success_rate_24h = 0.99, last_quality_eval = 0.72 },
        ["gpt-oss-120b@openrouter"]         = { price_in_usd_per_mtok = 0.08, price_out_usd_per_mtok = 0.30, success_rate_24h = 0.99, last_quality_eval = 0.70 },
        ["gemma-3-27b@openrouter"]          = { price_in_usd_per_mtok = 0.07, price_out_usd_per_mtok = 0.20, success_rate_24h = 0.99, last_quality_eval = 0.65 },
    },
}
