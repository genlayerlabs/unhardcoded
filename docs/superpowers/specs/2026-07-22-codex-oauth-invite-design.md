# Codex OAuth invite flow — design

Date: 2026-07-22
Status: approved (design), pending implementation plan

## Goal

Replace the manual "run `codex login` locally, extract JSON with jq, paste
auth.json into the dashboard" onboarding with a server-driven OAuth flow:

- The operator clicks **Generate invite link** in the dashboard, names the
  account (e.g. `team-1`), and sends the resulting single-use link to a
  teammate.
- The teammate (who has **no dashboard access**) opens the link, signs in with
  their ChatGPT account via OpenAI's device-code flow, and the server captures
  the tokens, stores the account, and hot-reloads the router.

Multi-account model (decided): **one `openai_codex` provider, account pool
behind it.** Each completed invite adds one named account to
`CODEX_ACCOUNTS_DIR` (`/codex/accounts/<name>.json`), discovered by
`CodexAuthStore` and selectable from the dashboard (specific account or
"balanced" rotation). No catalog/ranking changes.

## Verified protocol (probed 2026-07-22 against auth.openai.com)

Mirrors `codex login --device-auth` (source: `openai/codex`,
`codex-rs/login/src/device_code_auth.rs`). Issuer `https://auth.openai.com`,
public client id `app_EMoamEEZ73f0CkXaXp7hrann` (already in `codex_auth.py`).

1. `POST {issuer}/api/accounts/deviceauth/usercode` with JSON
   `{"client_id": ...}` → 200
   `{"device_auth_id", "user_code", "interval" (string, seconds),
   "expires_at"}`. Confirmed working with plain HTTP client (no Cloudflare
   block on this path; the bare `/deviceauth/*` paths ARE blocked — the
   `/api/accounts` prefix is required).
2. User signs in at `{issuer}/codex/device` and enters `user_code`. Codes
   expire in 15 minutes.
3. Poll `POST {issuer}/api/accounts/deviceauth/token` with
   `{"device_auth_id", "user_code"}`. 403/404 = pending (keep polling at
   `interval`); 200 → `{"authorization_code", "code_challenge",
   "code_verifier"}` (OpenAI generates the PKCE pair server-side).
4. Exchange at `{issuer}/oauth/token` (standard authorization-code grant):
   `grant_type=authorization_code`, `code=authorization_code`,
   `redirect_uri={issuer}/deviceauth/callback`, `client_id`, `code_verifier`
   → `{access_token, refresh_token, id_token}`.
5. `account_id` = `chatgpt_account_id` from the id_token's
   `https://api.openai.com/auth` claim (same claim the existing refresh path
   in `codex_auth.py` handles).

## Components

### Device-flow client — `codex_auth.py`

New module-level functions (it already owns `OAUTH_TOKEN_URL` and
`CODEX_CLIENT_ID`):

- `device_usercode_request()` → step 1
- `device_token_poll(device_auth_id, user_code)` → one poll attempt (step 3);
  returns pending / success / fatal
- `device_code_exchange(authorization_code, code_verifier)` → step 4, returns
  the auth.json-shaped dict (`tokens.{access_token, refresh_token, id_token,
  account_id}`, `last_refresh`)

### Invite store + endpoints — `auth_proxy.py`

Invites persisted to `{CODEX_ACCOUNTS_DIR}/invites.json` (same PVC as
accounts). Invite record: `{token, name, created_at, expires_at, status,
device_auth_id?, user_code?, interval?, device_started_at?, last_poll_at?}`.
Token: `secrets.token_urlsafe(32)`, single use, 24 h expiry.

Privileged (existing dashboard auth, same guard as other
`/dashboard/api/codex/*` routes):

- `POST /dashboard/api/codex/invites {name}` → create; returns
  `{url, name, expires_at}`. A new invite with the same name replaces any
  pending invite for that name (one pending link per name). The URL host
  comes from the request's own origin (scheme + Host header, honoring
  `X-Forwarded-Proto`/`X-Forwarded-Host` from the ingress), so no new config
  is required.
- `GET /dashboard/api/codex/invites` → list with status
  (`pending` / `awaiting sign-in` / `used` / `expired`) and copyable URLs.
- `DELETE /dashboard/api/codex/invites/{token}` → revoke.

Public, token-gated (no dashboard auth; unknown/expired/used token → friendly
"link expired" page / 404 JSON):

- `GET /codex/onboard/{token}` → onboarding HTML page.
- `POST /codex/onboard/{token}/start` → runs step 1, stores device state on
  the invite, returns `{verification_url: "{issuer}/codex/device",
  user_code}`. Re-invocable (restarts the device flow) while the invite is
  unexpired and unused.
- `GET /codex/onboard/{token}/status` → drives the flow **without background
  tasks**: each call performs at most one upstream poll (step 3), guarded by
  OpenAI's `interval` since `last_poll_at`. On success it runs step 4/5,
  writes the account through the same code path as the existing paste flow
  (`dashboard_add_codex_account` internals: write
  `/codex/accounts/<name>.json`, hot-apply via router `/x/codex/reload`),
  marks the invite `used`, and returns `{status: "connected", name}`.

### Onboarding page (HTML in `auth_proxy.py`, existing dashboard style)

One button: **Sign in with ChatGPT**. After start: shows the one-time code
(large, copy button) + link to OpenAI's sign-in page; polls `/status` every
~5 s; success state: "Connected as `<name>` — you can close this page."

Transparency copy (required): the page states it connects the visitor's
ChatGPT account to this router, and warns that OpenAI's page shows a
"if a website gave you this code, cancel" notice — continue only because a
trusted operator sent the link. (Consistent with the ToS caveats in
`docs/OPENAI-CODEX.md` — this is an internal/trusted-team tool.)

### Dashboard panel additions (JS/HTML in the existing codex panel)

"Generate invite link" (name input + button), invite list with status +
copy-link + revoke. The existing paste-auth.json form stays as a fallback.

## Error handling

- usercode request fails → page shows the error with a retry button.
- Poll timeout (15 min device-code expiry) → invite drops back to `pending`;
  teammate can hit Sign in again.
- Proxy restart mid-flow → device state is persisted with the invite, so
  `/status` polling resumes; if the device code meanwhile expired, teammate
  restarts sign-in.
- Name collision on completion → replace the existing account file (identical
  to today's paste-flow add/replace semantics).
- Tokens are never logged; invites.json holds no tokens (only device_auth_id
  + user_code, which expire in 15 min and are useless without approval).

## Testing

Unit/endpoint tests with mocked OpenAI endpoints, following
`tests/test_codex_auth_store.py` / `tests/test_auth_proxy_dashboard_full.py`
patterns:

- device client: usercode parse (string `interval`), poll pending vs success
  vs fatal, exchange → auth.json shape, account_id extraction from id_token.
- invite lifecycle: create (replaces same-name pending), list statuses,
  revoke, expiry, single-use enforcement, unknown-token 404.
- onboard flow: start → status pending (interval guard: no upstream call
  before `interval` elapses) → status success writes the account file,
  triggers reload, marks used; restart-mid-flow resume.

The live end-to-end flow is not covered in CI (no ChatGPT subscription),
matching the existing caveat in `docs/OPENAI-CODEX.md`.

## Docs

Update `docs/OPENAI-CODEX.md` setup section: invite flow is the primary path;
`codex login` + paste remains the fallback.

## Out of scope

- Per-account catalog providers (rejected: one provider + account pool).
- Any change to ranking, pricing, or the balanced-rotation selection logic.
- Refresh-token handling changes (existing `CodexAuth` refresh is reused).
