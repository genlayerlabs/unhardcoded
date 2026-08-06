-- config.live.lua — production provider/model catalog.  Plain
-- OpenAI-compatible requests use the identified default policy below; callers
-- can override it with a per-call Σ_pol policy_ir.

-- Tier policies live in their own files under policies/ for clarity; this dir is
-- resolved relative to the process cwd (run from the repo root) or $LLM_POLICY_DIR.
local HERE = os.getenv("LLM_POLICY_DIR") or "."

-- Registered model-level traits (benchmarks/modalities/capabilities), generated
-- from OpenRouter by scripts/refresh_model_meta.py. They are properties of the
-- model family — identical whoever serves it — so a per-family lookup feeds the
-- field schema below. Provider-level pricing/caching flow live via the EMA.
local MM_OK, MM = pcall(dofile, HERE .. "/model_meta.lua")
if not MM_OK then MM = {} end
local BEDROCK_REGION = os.getenv("BEDROCK_REGION") or "us-east-1"
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

-- The same bounded product defaults exposed by policy_templates.py.  A parity
-- test compares the normalized terms so the Python template and this
-- no-policy profile cannot drift silently.
local DEFAULT_PROVIDER_ORDER = {
    { "openai_codex" },
    { "antseed" },
    { "bedrock", "bedrock_market" },
    { "openrouter", "openrouter_market" },
}
local DEFAULT_EXPECTED_INPUT_SHARE = 0.8
local DEFAULT_RELIABILITY_FLOOR = 0.8
local DEFAULT_MAX_PRICE_IN = 5.0
local DEFAULT_MAX_PRICE_OUT = 25.0
local DEFAULT_QUALITY_TOP_N = 5
local DEFAULT_COST_WEIGHT = 0.75
local DEFAULT_INTELLIGENCE_WEIGHT = 0.25

local BALANCED_RETRY = {
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
    -- A context overflow on one route says nothing about the others.
    context_overflow  = { action = "next_candidate" },
    -- A stream that died after content reached the client cannot fall through.
    stream_interrupted = { action = "abort" },
    -- Out of credits will not heal on retry; keep the breaker open for 5 min.
    payment_required  = { action = "next_candidate", open_breaker_ms = 300000 },
    unknown           = { action = "next_candidate" },
}

local function fail_plan(actions)
    local keys = {}
    for reason, _ in pairs(actions) do
        if reason ~= "unknown" then keys[#keys + 1] = reason end
    end
    table.sort(keys)
    local out = { "always", actions.unknown }
    for _, reason in ipairs(keys) do
        out = { "override", out, reason, actions[reason] }
    end
    return out
end

local function provider_pred(group)
    if #group == 1 then return { "provider_eq", group[1] } end
    local out = { "or" }
    for _, provider in ipairs(group) do
        out[#out + 1] = { "provider_eq", provider }
    end
    return out
end

local function default_policy_ir()
    local provider_preds = {}
    local allowed = { "or" }
    for _, group in ipairs(DEFAULT_PROVIDER_ORDER) do
        local pred = provider_pred(group)
        provider_preds[#provider_preds + 1] = pred
        allowed[#allowed + 1] = pred
    end

    -- Cost dominates the value score inside a provider, with intelligence as
    -- the quality/tie-break component.  The top-five preference is soft: a
    -- lower-ranked family remains available when requirements (or an explicit
    -- `family:` model) leave no top-five candidate.
    local selector = { "argmax" }
    for i = #provider_preds, 1, -1 do
        selector = { "prefer", provider_preds[i], selector }
    end
    selector = {
        "prefer",
        { "cmp", "bench_intelligence_rank", "le", DEFAULT_QUALITY_TOP_N },
        selector,
    }
    selector = {
        "prefer",
        { "not", { "is", "breaker_open" } },
        selector,
    }

    return {
        "policy",
        { "and",
            { "meets_req" },
            { "not", { "is", "disabled" } },
            { "cmp", "success_rate", "ge", DEFAULT_RELIABILITY_FLOOR },
            { "cmp", "price_in", "le", DEFAULT_MAX_PRICE_IN },
            { "cmp", "price_out", "le", DEFAULT_MAX_PRICE_OUT },
            allowed,
        },
        { "add",
            { "scale", DEFAULT_COST_WEIGHT,
                { "neg", { "normalize",
                    { "add",
                        { "scale", DEFAULT_EXPECTED_INPUT_SHARE, { "field", "price_in" } },
                        { "scale", 1.0 - DEFAULT_EXPECTED_INPUT_SHARE, { "field", "price_out" } },
                    },
                } },
            },
            { "scale", DEFAULT_INTELLIGENCE_WEIGHT,
                { "normalize", { "field", "bench_intelligence" } },
            },
        },
        selector,
        { "id" },
        fail_plan(BALANCED_RETRY),
    }
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
        bedrock = {
            discovery = "static",
            base_url  = "bedrock://" .. BEDROCK_REGION,
            api_kind  = "bedrock",
            aws_region = BEDROCK_REGION,
            tier      = "partner",
            source    = "bedrock",
            notes     = "Amazon Bedrock Runtime via native AWS auth "
                     .. "(IRSA/EKS Pod Identity in AWS, AWS_PROFILE locally).",
        },
        bedrock_market = {
            discovery        = "marketplace",
            discovery_id     = "bedrock_market",
            base_url         = "bedrock://" .. BEDROCK_REGION,
            api_kind         = "bedrock",
            aws_region       = BEDROCK_REGION,
            tier             = "partner",
            source           = "bedrock",
            market_price_cap = { input = 1000, output = 1000 },
            notes            = "Dynamic Bedrock model/profile offers, priced "
                            .. "from AWS public Bedrock price lists.",
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
            source           = "openrouter",
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
            -- Exact wire-name -> curated family, and the ONLY way a peer's name
            -- reaches a family the canonicalizer refuses to guess at. It folds
            -- vendor prefixes and separators (`opus-4.8`, `anthropic/claude-opus-4.8`)
            -- and gives serving-mode variants their own `<family>@<variant>`, but it
            -- never bridges a letter/digit boundary (`gemma4-31b` vs `gemma-4-31b`)
            -- and never crosses vendors — that judgement is the operator's, here.
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

    -- `vendor` (optional): who actually makes the weights. Only the AntSeed
    -- marketplace canonicalizer reads it (sources/antseed.py `_family_vendor`).
    --
    -- A peer's wire name may claim a vendor (`z-ai-glm-5.1`), and that claim is
    -- refused unless this family's vendor is KNOWN and agrees — so `x-ai-glm-5.1`
    -- and `deepseek-llama-3.3-70b` (the naming shape of a real distill with
    -- DIFFERENT weights) never reach these families. Unknown vendor + a vendor
    -- claim = refused, and the offer stays routable under its raw wire name.
    --
    -- So annotate ONLY a family whose own name does not already carry its vendor
    -- token: `claude-*`, `gemini-*`, `deepseek-*` and `minimax-*` are read off
    -- the name and need no line. Annotating is what re-opens the legitimate
    -- `<vendor>-<family>` spellings peers really advertise (they mirror the
    -- OpenRouter slugs in `served_by` below) — and only those. Write the vendor
    -- as the wire token (`mistralai`) or its canonical id (`mistral`); anything
    -- else is dead config and warns.
    models = {
        ["minimax-m2.7"] = {  -- vendor read off the name: `minimax-`
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
            vendor = "meta",
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
        -- Codex (sunk-cost subscription) is a ROUTE of each compatible OpenAI
        -- family, not a separate family. Its $0 cost is held in check by the
        -- host scarcity price ramp (sources/codex.py), which lifts codex's
        -- ranking price as the subscription quota fills so paid routes take
        -- over before the 429 wall. Claude/Gemini have no codex path, so they
        -- cascade through their other configured providers.
        ["gpt-5.5"] = {
            vendor = "openai",
            served_by = {
                { provider = "openai_codex", provider_model_id = "gpt-5.5" },
                { provider = "openai",       provider_model_id = "gpt-5.5" },
                { provider = "openrouter",   provider_model_id = "openai/gpt-5.5" },
            },
            capabilities = { context = 400000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.95,
        },
        ["gpt-5.4"] = {
            vendor = "openai",
            served_by = {
                { provider = "openai_codex", provider_model_id = "gpt-5.4" },
                { provider = "openai",       provider_model_id = "gpt-5.4" },
                { provider = "openrouter",   provider_model_id = "openai/gpt-5.4" },
            },
            capabilities = { context = 400000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.90,
        },
        ["gpt-5.4-mini"] = {
            vendor = "openai",
            served_by = {
                { provider = "openai_codex", provider_model_id = "gpt-5.4-mini" },
                { provider = "openai",       provider_model_id = "gpt-5.4-mini" },
                { provider = "openrouter",   provider_model_id = "openai/gpt-5.4-mini" },
            },
            capabilities = { context = 400000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.86,
        },
        ["claude-opus-4-8"] = {
            served_by = {
                { provider = "anthropic",    provider_model_id = "claude-opus-4-8" },
                { provider = "openrouter",   provider_model_id = "anthropic/claude-opus-4-8" },
            },
            capabilities = { context = 200000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.93,
        },
        -- The two Opus releases before 4.8. Both are sold live by several AntSeed
        -- peers, and until they were curated every one of those peers' spellings
        -- was its own unreachable family (`family_eq` is an exact compare). Same id
        -- scheme as 4.8 at both providers, so the direct routes are a rename of a
        -- route we already call — not a guess about a model that might not exist.
        ["claude-opus-4-7"] = {
            served_by = {
                { provider = "anthropic",    provider_model_id = "claude-opus-4-7" },
                { provider = "openrouter",   provider_model_id = "anthropic/claude-opus-4-7" },
            },
            capabilities = { context = 200000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.92,
        },
        ["claude-opus-4-6"] = {
            served_by = {
                { provider = "anthropic",    provider_model_id = "claude-opus-4-6" },
                { provider = "openrouter",   provider_model_id = "anthropic/claude-opus-4-6" },
            },
            capabilities = { context = 200000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.91,
        },
        -- Curated so it has a DIRECT (non-marketplace) fallback: until now
        -- claude-fable-5 lived only as raw marketplace offers (2 thin/failing
        -- sellers), so a fable-5 request had no route home when they failed.
        -- Direct anthropic/openrouter routes give it a fallback, and being
        -- curated lets the marketplace canonicalizer fold fable-5 wire variants.
        ["claude-fable-5"] = {
            served_by = {
                { provider = "anthropic",    provider_model_id = "claude-fable-5" },
                { provider = "openrouter",   provider_model_id = "anthropic/claude-fable-5" },
            },
            capabilities = { context = 200000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.90,
        },
        ["gemini-3.1-pro-preview"] = {
            served_by = {
                { provider = "gemini",       provider_model_id = "gemini-3.1-pro-preview" },
                { provider = "openrouter",   provider_model_id = "google/gemini-3.1-pro-preview" },
            },
            capabilities = { context = 1000000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.92,
        },
        -- The Flash tier of the same Gemini 3 line as the Pro preview above: same
        -- id at the gemini API, same `google/<id>` slug at OpenRouter, and the
        -- official-pricing scraper anchors on the family verbatim
        -- (sources/official_pricing's gemini_html parser), so it prices itself.
        ["gemini-3-flash-preview"] = {
            served_by = {
                { provider = "gemini",       provider_model_id = "gemini-3-flash-preview" },
                { provider = "openrouter",   provider_model_id = "google/gemini-3-flash-preview" },
            },
            capabilities = { context = 1000000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.85,
        },
        -- Emergency affordable edge fallback for low OpenRouter credit states.
        -- Verified 2026-06-04 with a ~20k-token Hermes/t4pebot prompt + tools:
        -- expensive frontier models were rejected by OpenRouter credit ceilings,
        -- while this Qwen route returned valid content/tool-call responses.
        ["qwen3-235b-a22b"] = {
            vendor = "qwen",
            served_by = {
                { provider = "bedrock", provider_model_id = "qwen.qwen3-vl-235b-a22b" },
                { provider = "openrouter", provider_model_id = "qwen/qwen3-235b-a22b-2507" },
            },
            capabilities = { context = 262000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.90,
        },

        -- Free codex safety net for `edge`: spark (subscription, ~0 marginal)
        -- ranks just below gpt-5.5-codex and above every paid candidate.
        ["gpt-5.3-codex-spark"] = {
            vendor = "openai",
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
                { provider = "bedrock",       provider_model_id = "us.anthropic.claude-sonnet-4-6" },
                { provider = "openrouter",    provider_model_id = "anthropic/claude-sonnet-4-6" },
            },
            capabilities = { context = 200000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.88,
        },
        -- The Sonnet/Haiku releases before 4.6, curated for the same reason as the
        -- older Opus pair: live on AntSeed under several spellings, each of which
        -- was its own unreachable family until the curated name existed to fold
        -- them onto. Bedrock is left off deliberately — sources/bedrock only maps
        -- the families in its `_FAMILY_PATTERNS`, and claiming a bedrock route it
        -- cannot price would be a route that 404s on first call.
        ["claude-sonnet-4-5"] = {
            served_by = {
                { provider = "anthropic",     provider_model_id = "claude-sonnet-4-5" },
                { provider = "openrouter",    provider_model_id = "anthropic/claude-sonnet-4-5" },
            },
            capabilities = { context = 200000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.86,
        },
        ["claude-haiku-4-5"] = {
            served_by = {
                { provider = "anthropic",     provider_model_id = "claude-haiku-4-5" },
                { provider = "openrouter",    provider_model_id = "anthropic/claude-haiku-4-5" },
            },
            capabilities = { context = 200000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.80,
        },
        ["deepseek-v4-pro"] = {
            served_by = {
                { provider = "openrouter",    provider_model_id = "deepseek/deepseek-v4-pro" },
            },
            capabilities = { context = 128000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.85,
        },
        -- Already reachable on TWO routes the repo names itself — sources/bedrock's
        -- `_FAMILY_PATTERNS` maps AWS's ids onto this exact family, and the
        -- OpenRouter slug is `deepseek/deepseek-v3.2`. Curating it is what lets
        -- bedrock stamp a context on its offers (_capabilities_for reads it here).
        ["deepseek-v3.2"] = {
            served_by = {
                { provider = "openrouter",    provider_model_id = "deepseek/deepseek-v3.2" },
            },
            capabilities = { context = 128000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.82,
        },
        ["glm-5.1"] = {
            vendor = "z-ai",
            served_by = {
                { provider = "openrouter",    provider_model_id = "z-ai/glm-5.1" },
            },
            capabilities = { context = 200000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.84,
        },
        ["kimi-k2.6"] = {
            vendor = "moonshot",
            served_by = {
                { provider = "openrouter",    provider_model_id = "moonshotai/kimi-k2.6" },
            },
            capabilities = { context = 256000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.83,
        },
        ["qwen3-coder"] = {
            vendor = "qwen",
            served_by = {
                { provider = "openrouter",    provider_model_id = "qwen/qwen3-coder" },
            },
            capabilities = { context = 262000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.82,
        },
        ["mistral-large"] = {
            vendor = "mistral",
            served_by = {
                { provider = "openrouter",    provider_model_id = "mistralai/mistral-large" },
            },
            capabilities = { context = 128000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.80,
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
            vendor = "openai",
            served_by = {
                { provider = "bedrock",       provider_model_id = "openai.gpt-oss-120b-1:0" },
                { provider = "openrouter",    provider_model_id = "openai/gpt-oss-120b" },
            },
            capabilities = { context = 128000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.70,
        },
        ["gemma-3-27b"] = {
            vendor = "google",
            served_by = {
                { provider = "bedrock",       provider_model_id = "google.gemma-3-27b-it" },
                { provider = "openrouter",    provider_model_id = "google/gemma-3-27b-it" },
            },
            capabilities = { context = 96000, supports_tools = true, supports_json_mode = true },
            static_quality_hint = 0.65,
        },
        -- NOT curated, on purpose: `grok-4.3`, `minimax-m3`, `kimi-k2.7-code`,
        -- `qwen3.6-27b`, `gemma-4-31b-it` and `gemma-4-26b-a4b-it` are named by
        -- live policies, but no provider here has a route we can name for them —
        -- each would need a `served_by` extrapolated FORWARD from a different
        -- model's id, and `validate_model` (core/llm_policy/candidate.lua) would
        -- happily accept the invention. They stay reachable through the AntSeed
        -- market under their raw wire names; the AntSeed source now ranks them by
        -- distinct-seller count in `_stats.unbound_top` (with a `near_miss` when
        -- one alias would bind them, e.g. `gemma4-31b-it` -> `gemma-4-31b-it`), so
        -- curating each is a one-entry job the moment a real route is confirmed.
    },

    profiles = {
        -- Callers may send their own policy_ir.  Plain OpenAI-compatible
        -- requests use this identified policy: healthy Codex -> AntSeed ->
        -- Bedrock -> OpenRouter, with safe price/reliability rails and the best
        -- intelligence-ranked families preferred inside each provider.
        default = {
            policy_ir    = default_policy_ir(),
            selector     = "argmax",
            retry_policy = "balanced",
        },
    },

    retry_policies = {
        balanced = BALANCED_RETRY,
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
    --
    -- Third clause — ANTSEED FUNDING ADMISSION. AntSeed pays every call out of an
    -- on-chain USDC escrow, and opening a payment channel reserves ~1 USDC; below
    -- that, every routed call 402s `insufficient_deposits`. `credits` carries the
    -- buyer's live `deposits_available` (pushed by sources.push_credits on each
    -- balances refresh), so this keeps an unfundable buyer out of ranking.
    --
    -- Deliberately SCOPED to antseed by the `or`: every other provider bills
    -- against its own quota/credit mechanics with no on-chain escrow, and the
    -- engine's `credits` field defaults to 0 — an unscoped clause would reject
    -- the entire catalog. The scoped form needs no core change precisely because
    -- the `not provider_eq` branch is true for them.
    --
    -- This fails CLOSED (default 0 < 1.0 => rejected) where the offer-side
    -- tourniquet in sources/antseed.py fails OPEN. That asymmetry is intended —
    -- envelope = belt, offers_sync = braces — and the cold-start hole it opens is
    -- closed by sources.seed_credits, which publishes the last known escrow from
    -- the durable buyer_status row before the app serves its first request.
    policy_envelope = { "and", { "meets_req" }, { "not", { "is", "disabled" } },
        { "or", { "not", { "provider_eq", "antseed" } },
                { "cmp", "credits", "ge", 1.0 } } },
}
