# Deployment

This stack exposes one OpenAI-compatible endpoint for your services while keeping provider API keys and OAuth files on the router host.

## Client contract

Set three env vars and keep using the normal OpenAI SDK:

```bash
LLM_BASE_URL=http://127.0.0.1:8080/v1        # or http://llm-router.internal:8080/v1 from a joined Docker network
LLM_API_KEY=<per-service-token>              # token from CALLER_KEYS_JSON
LLM_MODEL=                      # empty -> the default policy; or family:/pin:, or send a policy_ir in the body
```

Python:

```python
import os
from openai import OpenAI
client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
resp = client.chat.completions.create(model=os.getenv("LLM_MODEL", ""), messages=[{"role":"user","content":"ping"}])
print(resp.choices[0].message.content)
```

Node:

```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: process.env.LLM_BASE_URL, apiKey: process.env.LLM_API_KEY });
const r = await client.chat.completions.create({ model: process.env.LLM_MODEL ?? "", messages: [{ role: "user", content: "ping" }] });
console.log(r.choices[0].message.content);
```

## Deploy

1. Clone with submodules:

```bash
git clone --recursive https://github.com/genlayerlabs/unhardcoded.git
```

2. Create `.env.secrets` from `.env.example` or inject the same env vars from the platform secret store:

```bash
cp .env.example .env.secrets
chmod 600 .env.secrets
# Fill provider keys and dashboard auth values.
# Generate DASHBOARD_PASSWORD_SHA256 with the command in .env.example.
# Generate DASHBOARD_SESSION_SECRET with: python -c 'import secrets; print(secrets.token_urlsafe(32))'
# Optional Codex OAuth: set CODEX_AUTH_PATH to a host-side codex auth.json.
```

Register one caller token per consuming service. The registration helper stores only `sha256(token) -> caller` by default and prints the raw token once:

```bash
python scripts/register_consumer_key.py service-a --env .env.secrets
python scripts/register_consumer_key.py service-b --env .env.secrets
```

3. Start the default stack:

```bash
docker compose -f compose.yml up -d --build
```

4. Start with AntSeed buyer sidecars enabled:

```bash
docker compose -f compose.yml --profile antseed up -d --build
```

## Provider modes in this local stack

- `openai_codex`: ChatGPT/Codex OAuth via `CODEX_AUTH_PATH`, mounted into the router as `/codex/auth.json`. This is unofficial and should stay single-replica because the auth file can be refreshed/written by the backend.
- `openai`: normal OpenAI API using `OPENAI_API_KEY`; currently optional because the local key may be absent/stale.
- `openrouter`: normal OpenAI-compatible gateway using `OPENROUTER_API_KEY`.
- `antseed`: a single AntSeed buyer proxy in browse mode (no pinned peer — it discovers sellers on the network). The sidecar shares the router network namespace so its localhost-only proxy port is reachable by the router but not published to the host. The per-call price ceiling comes from the policy (`cmp(price_out, le, X)`); the proxy's `--max-*-usd-per-million` is just a wide outer bound.

Routing model:

- There are no named intelligence tiers. Each caller sends its own Σ_pol policy
  (`policy_ir` in the request body, e.g. composed in the dashboard Builder).
- When a caller sends no policy, the single declarative `default` profile
  (balanced quality/cost) is the fallback. Codex (api_kind `openai_codex`), the
  AntSeed marketplace proxy and OpenRouter are all in the candidate pool; the
  policy decides the order and the cascade.

## Security shape

- `router` is not published; only the `ingress` proxy can reach it on the Compose network.
- `ingress` binds `127.0.0.1:${LLM_ROUTER_HOST_PORT:-8080}` only. Put a private LB, Tailscale, service mesh, or Caddy internal-only route in front if other hosts need access.
- The ingress checks the OpenAI-SDK bearer token against `CALLER_KEYS_JSON`; each token maps to a caller name for audit logs.
- Provider secrets and OAuth files are read only by the router/sidecars from env/secrets or host mounts. Clients never receive provider keys.
- Logs are JSON lines with caller, route, status, latency, and router-chosen provider/model when available.

## Smoke tests

```bash
curl -fsS http://127.0.0.1:8080/healthz | jq .

curl -fsS \
  -H "Authorization: Bearer <per-service-token>" \
  http://127.0.0.1:8080/v1/models | jq '.data[:5]'

curl -fsS \
  -H "Authorization: Bearer <per-service-token>" \
  -H 'Content-Type: application/json' \
  -d '{"model":"","messages":[{"role":"user","content":"Reply with exactly pong"}],"max_tokens":8}' \
  http://127.0.0.1:8080/v1/chat/completions | jq '{model, content:.choices[0].message.content, router:.x_router}'
```

Unauthorized callers should fail:

```bash
curl -i http://127.0.0.1:8080/v1/models
# expected: HTTP/1.1 401 Unauthorized
```

AntSeed status checks:

```bash
docker compose -f compose.yml --profile antseed ps
docker compose -f compose.yml exec antseed antseed buyer status
docker compose -f compose.yml exec antseed antseed network browse --services --top 5
```

## Profile/catalog edits

`config.live.lua` is the server-side catalog. To add/change behavior, edit one provider/model/profile sentence, then recreate the router container. Clients keep sending the same `model=profile:<name>` string.
