-- config.live.lua — provider catalog used by `hosts/python_shim/live_smoke.py`.
-- Primary model is minimax-m2.7 served by OpenRouter — fast, current,
-- and inexpensive. Llama-3.3-70b is kept as a secondary candidate so
-- the cascade behaviour can still be demonstrated when needed.

-- Tier policies live in their own files under policies/ for clarity; this dir is
-- resolved relative to the process cwd (run from the repo root) or $LLM_POLICY_DIR.
local HERE = os.getenv("LLM_POLICY_DIR") or "."

-- Registered model-level traits (benchmarks/modalities/capabilities), generated
-- from OpenRouter by scripts/refresh_model_meta.py. They are properties of the
-- model family — identical whoever serves it — so a per-family lookup feeds the
-- field schema below. Provider-level pricing/caching flow live via the EMA.
local MM_OK, MM = pcall(dofile, HERE .. "/model_meta.lua")
if not MM_OK then MM = {} end
-- Per-family model trait getter. Curated families resolve from the static,
-- deterministic model_meta.lua (MM) — the on-chain path. Discovered marketplace
-- families (e.g. live OpenRouter models) aren't in MM; they carry their full
-- live traits inline on the offer, so fall back to c.offer.traits. Either way a
-- discovered family ranks on its real benchmark, not a placeholder.
local function mfield(name, sort, default)
    return { sort = sort, default = default, group = "model",
             get = function(c)
                 local m = MM[c.model_family]
                 if m ~= nil and m[name] ~= nil then return m[name] end
                 local o = c.offer
                 if o ~= nil and o.traits ~= nil and o.traits[name] ~= nil then return o.traits[name] end
                 return nil
             end }
end

return {

    providers = {
        heurist = {
            discovery = "static",
            base_url  = "https://llm-gateway.heurist.xyz/v1",
            api_kind  = "openai_compatible",
            auth_env  = "HEURIST_API_KEY",
            tier      = "partner",
            notes     = "Free credits via referral code 'genlayer'",
        },
        io_net = {
            discovery = "static",
            base_url  = "https://api.intelligence.io.solutions/api/v1",
            api_kind  = "openai_compatible",
            auth_env  = "IONET_API_KEY",
            tier      = "partner",
        },
        openrouter = {
            discovery = "static",
            base_url  = "https://openrouter.ai/api/v1",
            api_kind  = "openai_compatible",
            auth_env  = "OPENROUTER_API_KEY",
            tier      = "fallback",
            notes     = "Last-resort gateway",
        },
        openai = {
            discovery = "static",
            base_url  = "https://api.openai.com/v1",
            api_kind  = "openai_compatible",
            auth_env  = "OPENAI_API_KEY",
            tier      = "partner",
            notes     = "Native OpenAI API.",
        },
        anthropic = {
            discovery = "static",
            base_url  = "https://api.anthropic.com/v1",
            api_kind  = "anthropic",
            auth_env  = "ANTHROPIC_API_KEY",
            tier      = "partner",
            notes     = "Native Anthropic Messages API.",
        },
        gemini = {
            discovery = "static",
            base_url  = "https://generativelanguage.googleapis.com/v1beta",
            api_kind  = "google",
            auth_env  = "GEMINI_API_KEY",
            tier      = "partner",
            notes     = "Native Gemini generateContent API.",
        },
        bedrock_mantle = {
            discovery = "static",
            base_url  = os.getenv("BEDROCK_MANTLE_BASE_URL")
                     or "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
            api_kind  = "openai_compatible",
            auth_env  = "AWS_BEARER_TOKEN_BEDROCK",
            tier      = "partner",
            notes     = "Amazon Bedrock Mantle OpenAI-compatible endpoint. "
                     .. "Use BEDROCK_MANTLE_BASE_URL to select region/path.",
        },
        -- Live discovery of the WHOLE OpenRouter catalog (every model it serves,
        -- straight from /models — no hand curation). Candidates/prices come from
        -- the discover hook (sources/openrouter.py offers_sync); the curated
        -- families above stay served by the static `openrouter` provider and
        -- keep their benchmark ranking, so this covers the long tail. Each
        -- discovered family carries its full live traits inline on the offer
        -- (c.offer.traits), so it ranks on its real benchmark when OpenRouter
        -- reports one, and falls back to price + learned latency otherwise.
        -- Same key/base_url;
        -- first-party gateway → no peer pinning. market_price_cap is just a wide
        -- outer ceiling (the per-call Σ_pol policy sets the real price gate).
        openrouter_market = {
            discovery        = "marketplace",
            discovery_id     = "openrouter_market",
            base_url         = "https://openrouter.ai/api/v1",
            api_kind         = "openai_compatible",
            auth_env         = "OPENROUTER_API_KEY",
            tier             = "marketplace",
            market_price_cap = { input = 1000, output = 1000 },
            -- OpenRouter marketplace rows default `vendor/model` to the
            -- provider-neutral family `model`, while wire_model_id keeps the
            -- exact OpenRouter slug. Keep this map only for the canonicalization
            -- exceptions where stripping the vendor isn't the right family
            -- (dated or suffixed slugs).
            service_aliases  = {
                ["anthropic/claude-opus-4.8"]      = "claude-opus-4-8",
                ["google/gemma-3-27b-it"]          = "gemma-3-27b",
                ["qwen/qwen3-235b-a22b-2507"]      = "qwen3-235b-a22b",
            },
        },
        -- AntSeed buyer proxies: candidates and prices come from the live
        -- peer market (sources/antseed.py reads the /market dump and feeds
        -- the discover hook) — no hardcoded antseed model rows anywhere.
        -- market_price_cap is the single source of truth for each proxy's
        -- price band; the compose buyer --max-*-usd-per-million flags must
        -- match it. error_map turns AntSeed's error bodies into canonical
        -- kinds (insufficient deposits = out of credits; a peer that
        -- doesn't sell a service = that family is unavailable, not a
        -- provider failure). See docs/superpowers/specs/2026-06-10-provider-sources-design.md.
        -- One AntSeed buyer (no tiers). The price band is no longer baked per
        -- proxy: the cap here is just a wide outer ceiling; each call's policy
        -- sets the real price ceiling with cmp(price_out, le, X). One wallet to
        -- fund (antseed buyer deposit), one /market dump, one compose service.
        antseed = {
            discovery    = "marketplace",
            discovery_id = "antseed",
            -- Where the router reaches the buyer proxy. k8s runs antseed as a
            -- container in the SAME pod -> localhost:8377 (default). docker
            -- compose runs it on its own network (the daemon binds 127.0.0.1,
            -- socat re-exposes it) -> set ANTSEED_BASE_URL=http://antseed:8378/v1.
            base_url     = os.getenv("ANTSEED_BASE_URL") or "http://localhost:8377/v1",
            api_kind     = "openai_compatible",
            auth         = { kind = "none" },
            -- marketplace, NOT fallback: AntSeed serves the same model families
            -- as everyone else, so it inherits the same per-family OpenRouter
            -- benchmark and competes head-to-head (cheaper -> often wins). The
            -- old "fallback / not quality-rankable" framing was backwards.
            tier         = "marketplace",
            -- wide outer ceiling; the real per-call price gate is the caller's
            -- Σ_pol policy. Must stay <= the buyer's ANTSEED_MAX_* spend rails.
            market_price_cap = { input = 1000, output = 1000 },
            service_aliases  = { ["qwen3-235b-instruct"] = "qwen3-235b-a22b" },
            error_map = {
                ["insufficient_deposits"]             = "payment_required",
                ["not served by this peer"]           = "model_unavailable",
                ["outside your buyer routing policy"] = "model_unavailable",
            },
        },
        openai_codex = {
            discovery = "static",
            base_url  = "https://chatgpt.com/backend-api/codex",
            api_kind  = "openai_codex",
            auth      = { kind = "oauth", provider = "codex" },
            tier      = "partner",
            notes     = "ChatGPT subscription via Codex proxy. UNOFFICIAL / ToS-risky — "
                     .. "the backend mimics the Codex CLI. See docs/OPENAI-CODEX.md.",
        },
        ollama = {
            discovery = "marketplace",
            discovery_id = "ollama",
            base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1",
            cloud_url = "https://ollama.com/api/v1",
            api_kind = "openai_compatible",
            auth_env = "OLLAMA_API_KEY",
            tier = "partner",
            notes = "Local Ollama (no auth) or Ollama Cloud (subscription). "
                  .. "Set OLLAMA_CLOUD=1 to prefer cloud endpoint.",
        },
    },

    models = {
        ["minimax-m2.7"] = {
            served_by = {
                { provider = "openrouter", provider_model_id = "minimax/minimax-m2.7" },
            },
            capabilities = {
                context            = 200000,
                supports_tools     = true,
                supports_json_mode = true,
            },
            static_quality_hint = 0.80,
        },
        ["llama-3.3-70b"] = {
            served_by = {
                { provider = "heurist",    provider_model_id = "meta-llama/llama-3.3-70b-instruct" },
                { provider = "io_net",     provider_model_id = "meta-llama/Llama-3.3-70B-Instruct" },
                { provider = "openrouter", provider_model_id = "meta-llama/llama-3.3-70b-instruct" },
            },
            capabilities = {
                context            = 128000,
                supports_tools     = true,
                supports_json_mode = true,
            },
            static_quality_hint = 0.72,
        },
        -- (deepseek-v3.1 removed: it was served only by the legacy `antseed`
        --  provider entry, which marketplace discovery replaces.)

        -- ── `edge` tier: frontier models (quality ≥ 0.90) ──────────────────────
        -- Codex (sunk-cost subscription) is a ROUTE of gpt-5.5, not a separate
        -- family — its slug on the ChatGPT backend is plain "gpt-5.5". Its $0
        -- cost is held in check by the host scarcity price ramp (sources/codex.py),
        -- which lifts codex's ranking price as the subscription quota fills so
        -- paid routes take over before the 429 wall. Claude/Gemini have no codex
        -- path, so they cascade antseed_edge → openrouter.
        ["gpt-5.5"] = {
            served_by = {
                { provider = "openai_codex", provider_model_id = "gpt-5.5" },
                { provider = "openai",       provider_model_id = "gpt-5.5" },
                { provider = "openrouter",   provider_model_id = "openai/gpt-5.5" },
            },
            capabilities = { context = 400000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.95,
        },
        ["gpt-5.4"] = {
            served_by = {
                { provider = "openai",       provider_model_id = "gpt-5.4" },
                { provider = "openrouter",   provider_model_id = "openai/gpt-5.4" },
            },
            capabilities = { context = 400000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.90,
        },
        ["claude-opus-4-8"] = {
            served_by = {
                { provider = "anthropic",    provider_model_id = "claude-opus-4-8" },
                { provider = "openrouter",   provider_model_id = "anthropic/claude-opus-4-8" },
            },
            capabilities = { context = 200000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.93,
        },
        ["gemini-3.1-pro-preview"] = {
            served_by = {
                { provider = "gemini",       provider_model_id = "gemini-3.1-pro-preview" },
                { provider = "openrouter",   provider_model_id = "google/gemini-3.1-pro-preview" },
            },
            capabilities = { context = 1000000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.92,
        },
        -- Emergency affordable edge fallback for low OpenRouter credit states.
        -- Verified 2026-06-04 with a ~20k-token Hermes/t4pebot prompt + tools:
        -- expensive frontier models were rejected by OpenRouter credit ceilings,
        -- while this Qwen route returned valid content/tool-call responses.
        ["qwen3-235b-a22b"] = {
            served_by = {
                { provider = "bedrock_mantle", provider_model_id = "qwen.qwen3-235b-a22b-2507" },
                { provider = "openrouter", provider_model_id = "qwen/qwen3-235b-a22b-2507" },
            },
            capabilities = { context = 262000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.90,
        },

        -- Free codex safety net for `edge`: spark (subscription, ~0 marginal)
        -- ranks just below gpt-5.5-codex and above every paid candidate.
        ["gpt-5.3-codex-spark"] = {
            served_by = {
                { provider = "openai_codex", provider_model_id = "gpt-5.3-codex-spark" },
            },
            capabilities = { context = 400000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.90,
        },

        -- ── `medium` tier: quality [0.78, 0.90). ───────────────────────────────
        -- Normal cascade: free AntSeed → cheap paid AntSeed → openrouter. (The
        -- duplicated gpt-5.3-codex-spark-{medium,dummy} banding families were
        -- removed: spark is one family; tier banding is the caller policy's job,
        -- not a forked model.)
        ["claude-sonnet-4-6"] = {
            served_by = {
                { provider = "anthropic",     provider_model_id = "claude-sonnet-4-6" },
                { provider = "openrouter",    provider_model_id = "anthropic/claude-sonnet-4-6" },
            },
            capabilities = { context = 200000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.88,
        },
        ["deepseek-v4-pro"] = {
            served_by = {
                { provider = "openrouter",    provider_model_id = "deepseek/deepseek-v4-pro" },
            },
            capabilities = { context = 128000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.85,
        },
        ["glm-5.1"] = {
            served_by = {
                { provider = "openrouter",    provider_model_id = "z-ai/glm-5.1" },
            },
            capabilities = { context = 200000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.84,
        },
        ["kimi-k2.6"] = {
            served_by = {
                { provider = "openrouter",    provider_model_id = "moonshotai/kimi-k2.6" },
            },
            capabilities = { context = 256000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.83,
        },

        -- ── `dummy` tier: quality < 0.78. Free AntSeed → cheap → OR ────────────
        ["deepseek-v4-flash"] = {
            served_by = {
                { provider = "openrouter",    provider_model_id = "deepseek/deepseek-v4-flash" },
            },
            capabilities = { context = 128000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.76,
        },
        ["gpt-oss-120b"] = {
            served_by = {
                { provider = "openrouter",    provider_model_id = "openai/gpt-oss-120b" },
            },
            capabilities = { context = 128000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.70,
        },
        ["gemma-3-27b"] = {
            served_by = {
                { provider = "openrouter",    provider_model_id = "google/gemma-3-27b-it" },
            },
            capabilities = { context = 96000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.65,
        },
    },

    profiles = {
        -- No tiers. Each caller sends its own policy as a Σ_pol term (policy_ir,
        -- e.g. from the dashboard builder). `default` is only the fallback when a
        -- caller sends no policy at all: a balanced, DECLARATIVE policy — so it
        -- lowers to a Σ_pol term with an identity (copyable, testable in the
        -- builder), unlike the old closure-based tier profiles.
        default = {
            -- (sigma-pol/v2) the composite scorer atoms AND quality/quality_hint
            -- were removed; score on real fields. Balanced = benchmark (per
            -- model, identical whoever serves it) vs price (cheaper wins — this
            -- is where AntSeed, serving the same families, competes and often
            -- wins) vs learned reliability. success_rate is the live EMA of
            -- observed success per (provider, family) — the only signal that is
            -- genuinely per-provider for the same model (OpenRouter's benchmark
            -- can't tell you a peer's reliability). Its cold-start default is 1,
            -- so it's neutral until real traffic differentiates: a peer that
            -- starts failing has its EMA fall and is PROGRESSIVELY demoted as the
            -- failures accumulate. Weighted 0.30 (was 0.10): at 0.10 the demotion
            -- was too weak to overcome a failing peer's price edge, so dead peers
            -- (e.g. a marketplace seller timing out every request) kept being
            -- re-chosen. At 0.30 a peer whose EMA collapses falls below a reliable
            -- alternative — self-healing, and it climbs back if it recovers.
            -- (Raw latency_ms is deliberately NOT scored here: its cold-start
            -- default is +inf, which would freeze out every never-tried peer.)
            scorer       = { "add",
                { "scale", 0.35, { "field", "bench_intelligence" } },
                { "scale", 0.35, { "neg", { "normalize", { "field", "price_in" } } } },
                { "scale", 0.30, { "field", "success_rate" } },
            },
            filter       = { "requirements", "not_disabled" },
            selector     = "argmax",
            retry_policy = "balanced",
        },
    },

    retry_policies = {
        balanced = {
            rate_limit        = { action = "next_candidate", open_breaker_ms = 30000 },
            timeout           = { action = "next_candidate" },
            server_error      = { action = "retry_same", attempts = 1, backoff_ms = 500,
                                  then_action = "next_candidate" },
            auth_error        = { action = "disable_provider" },
            bad_request       = { action = "next_candidate" },
            content_filter    = { action = "next_candidate" },
            bad_response      = { action = "next_candidate" },
            model_unavailable = { action = "next_provider_same_model", mark_unavailable_ms = 300000 },
            network_error     = { action = "retry_same", attempts = 2, backoff_ms = { 200, 600 },
                                  then_action = "next_candidate" },
            -- A context overflow on ONE route says nothing about the others:
            -- a provider-neutral family (e.g. family:gpt-5.4) spans candidates
            -- with heterogeneous context windows, so the next one may well fit.
            -- retry_same would be futile (same model, same window) but
            -- next_candidate is not — fall through. If every candidate overflows
            -- the request still ends cleanly in `exhausted: context_overflow`.
            context_overflow  = { action = "next_candidate" },
            -- A stream that died AFTER content reached the client cannot
            -- fall through (the next candidate would append a second answer
            -- to a half-delivered one): abort, the shim reports in-stream.
            stream_interrupted = { action = "abort" },
            -- Out of credits (OpenRouter 402, AntSeed insufficient_deposits).
            -- Won't heal on retry: fall through, and keep the breaker open
            -- long (5 min) so dead-broke providers stop eating latency.
            payment_required  = { action = "next_candidate", open_breaker_ms = 300000 },
            unknown           = { action = "next_candidate" },
        },
    },

    -- Model-level observation fields (registered traits from OpenRouter, read
    -- per family). Gateable with `cmp`/`is`, scorable with `field` — e.g.
    -- cmp(bench_intelligence, ge, 0.6), is(in_image), field(bench_coding).
    -- Missing family/trait falls back to the conservative default.
    fields = {
        bench_intelligence     = mfield("bench_intelligence",     "Num",  0),
        bench_coding           = mfield("bench_coding",           "Num",  0),
        bench_agentic          = mfield("bench_agentic",          "Num",  0),
        bench_arena            = mfield("bench_arena",            "Num",  0),
        -- catalog ranks (1 = best): "in the top k by X" = cmp(<X>_rank, le, k);
        -- intersection of shortlists = the `and` of those. Static, deterministic.
        -- Default huge so a model without the benchmark is outside every top-k.
        bench_intelligence_rank = mfield("bench_intelligence_rank", "Num", 1e9),
        bench_coding_rank       = mfield("bench_coding_rank",       "Num", 1e9),
        bench_agentic_rank      = mfield("bench_agentic_rank",      "Num", 1e9),
        bench_arena_rank        = mfield("bench_arena_rank",        "Num", 1e9),
        in_image               = mfield("in_image",               "Bool", false),
        in_audio               = mfield("in_audio",               "Bool", false),
        in_file                = mfield("in_file",                "Bool", false),
        in_video               = mfield("in_video",               "Bool", false),
        out_image              = mfield("out_image",              "Bool", false),
        cap_tools              = mfield("cap_tools",              "Bool", false),
        cap_tool_choice        = mfield("cap_tool_choice",        "Bool", false),
        cap_parallel_tools     = mfield("cap_parallel_tools",     "Bool", false),
        cap_structured_outputs = mfield("cap_structured_outputs", "Bool", false),
        cap_response_format    = mfield("cap_response_format",    "Bool", false),
        cap_reasoning          = mfield("cap_reasoning",          "Bool", false),
        cap_seed               = mfield("cap_seed",               "Bool", false),
        cap_logprobs           = mfield("cap_logprobs",           "Bool", false),
        -- Per-route on-chain reputation (0-100), stamped on the offer by
        -- sources/antseed.py (the buyer's own admission signal). Gate with
        -- cmp(reputation_score, ge, N) or weight with field(reputation_score).
        -- Default 0: a route with no reported reputation scores neutral-low,
        -- never NaN; the default `default` profile does not use it.
        reputation_score = { sort = "Num", default = 0, group = "route",
            get = function(c)
                local o = c.offer
                if o ~= nil and o.reputation_score ~= nil then return o.reputation_score end
                return nil
            end },
    },

    -- Σ_pol host envelope: ∧-ed by the core onto every per-call `policy_ir`,
    -- so callers can only NARROW what this host allows, never widen it.
    -- Floor: the contract's requirements must hold, and auth-disabled
    -- providers stay out no matter what the caller's term says.
    policy_envelope = { "and", { "meets_req" }, { "not", { "is", "disabled" } } },
}
