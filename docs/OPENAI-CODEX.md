# OpenAI via ChatGPT subscription (Codex proxy)

> **Unofficial and ToS-risky.** This routes inference to your **ChatGPT
> subscription** through the Codex backend, the way `codex login` does. It is
> **not** a supported way for a third-party app to bill a user's subscription.
> Anthropic and Google closed their equivalents in 2026; OpenAI may follow, and
> accounts that abuse it can be limited or banned. Decide with eyes open. The
> supported alternative is a normal OpenAI API key (`api_kind =
> "openai_compatible"`, `auth_env = "OPENAI_API_KEY"`).

## Why the Apps SDK does not work for this

The [Apps SDK OAuth](https://developers.openai.com/apps-sdk/build/auth)
authenticates users into *your* MCP server and issues tokens for *your* tools.
It does **not** grant a token to call OpenAI inference on the user's plan. The
only path that does is the Codex one below.

## How it works here

```
router (api_kind="openai_codex")
  → codex_backend.make_codex_async_call_provider
      → POST {base_url}/responses           (https://chatgpt.com/backend-api/codex/responses)
        Authorization: Bearer <access_token from ~/.codex/auth.json, auto-refreshed>
        chatgpt-account-id: <account_id>
        body: Responses API schema, stream=true (SSE)
      ← SSE aggregated back into the router's response shape
```

- **Token source:** `codex_auth.CodexAuth` reads `~/.codex/auth.json`
  (`access_token`, `refresh_token`, `id_token`, `account_id`; both top-level and
  nested `tokens` layouts are accepted).
- **Refresh:** when the access token's JWT `exp` is near, it refreshes against
  `https://auth.openai.com/oauth/token` with the public Codex client id
  `app_EMoamEEZ73f0CkXaXp7hrann` and writes the new tokens back to `auth.json`.
- **Wire shape:** the Codex endpoint speaks the **Responses API** (not
  chat/completions) and streams SSE; `codex_backend` translates the router's
  chat-style request to `input` items and folds the
  `response.output_text.delta` / `response.completed` events back into text +
  usage.

## Setup

```bash
# 1. Authenticate (opens a browser; "Sign in with ChatGPT")
codex login                       # writes ~/.codex/auth.json

# 2. Start the shim; it picks up auth.json lazily on the first codex call
python -m hosts.python_shim --config hosts/python_shim/config.live.lua \
    --codex-auth ~/.codex/auth.json
```

`--codex-auth` defaults to `~/.codex/auth.json`. Treat that file like a
password — it holds bearer tokens.

## Caveats

- The backend endpoint is undocumented and can change without notice; if the
  Responses event names or the `/responses` path change, `codex_backend` needs
  updating.
- The live streaming call is **not** covered by tests (no subscription in CI).
  The token read/refresh, request translation, header building and SSE
  aggregation are unit-tested in `hosts/python_shim/tests/test_codex.py`.
- Tool calls are not yet translated to the Responses API tool schema; v1 sends
  text in / text out. Add tool translation in `build_codex_body` when needed.
