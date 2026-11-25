return {
    providers = {
        fixture = {
            discovery = "static",
            base_url = "http://upstream:8000/v1",
            api_kind = "openai_compatible",
            auth = { kind = "none" },
            tier = "partner",
        },
    },
    models = {
        ["load-model"] = {
            served_by = { { provider = "fixture", provider_model_id = "load-model" } },
            capabilities = { context = 8192 },
        },
    },
    profiles = { default = { scorer = { "zero" } } },
}
