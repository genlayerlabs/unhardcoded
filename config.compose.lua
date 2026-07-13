-- config.compose.lua — local Compose/BDD overlay.
--
-- Production keeps using config.live.lua and its `compact` profile. The local
-- Compose stack has a hermetic Ollama sidecar with one explicitly documented
-- test model, so give that stack its own server-owned compaction profile. The
-- public request may narrow this profile but still cannot choose its policy.

local HERE = os.getenv("LLM_POLICY_DIR") or "."
local cfg = dofile(HERE .. "/config.live.lua")

cfg.profiles.compact_bdd_ollama = {
    policy_ir = { "policy",
        { "and",
            { "meets_req" },
            { "not", { "is", "disabled" } },
            { "provider_eq", "ollama" },
            { "family_eq", "qwen2.5:0.5b" },
            { "cmp", "billing_price_in", "ge", 0.0 },
            { "cmp", "billing_price_in", "le", 0.0 },
            { "cmp", "billing_price_out", "ge", 0.0 },
            { "cmp", "billing_price_out", "le", 0.0 },
        },
        { "field", "success_rate" },
        { "argmax" },
        { "id" },
        { "always", { action = "abort" } },
    },
}

return cfg
