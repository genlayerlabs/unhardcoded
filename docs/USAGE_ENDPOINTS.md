# Per-key usage endpoints

The auth proxy exposes sanitized usage statistics for a specific consumer API key.

These endpoints are for exact-key observability: they report usage for one API key, not broad provider credentials or unrestricted dashboard data.

## Security model

- Raw API keys are accepted only as authentication credentials or as the dashboard lookup input.
- Raw API keys are never returned in responses.
- Full SHA-256 key digests are never returned in responses.
- Responses expose only `key_sha256_prefix`, the first 12 hex characters of the key digest.
- Provider credentials and environment-derived secrets are never returned.
- Dashboard/admin APIs require dashboard authentication; consumer bearer tokens do not satisfy dashboard auth.
- Do not pass API keys in query strings. Dashboard lookup uses a JSON POST body.

## Persistent history

Per-key usage events are appended to `ROUTER_USAGE_HISTORY_PATH` as JSONL. If unset, the default path is `/run/llm-router/usage-history.jsonl`. The provided Compose config sets it to `/run/llm-router/secrets/usage-history.jsonl`, backed by the host `./secrets` mount, so history survives ingress container recreation as well as ordinary restarts.

The persistent event log stores sanitized event metadata and the key SHA-256 digest for lookup. It does not store raw API keys, bearer tokens, provider credentials, request bodies, or response bodies. API responses still expose only `key_sha256_prefix`, never the full digest.

Persistent history lets `/v1/usage`, `/api/usage`, and `/dashboard/api/key-usage` rebuild totals after an ingress restart. In-memory stats are still kept for live dashboard overview pages.

## Consumer self-service

```http
GET /v1/usage?since=2026-06-01T00:00:00Z&limit=50&offset=0
Authorization: Bearer <consumer-api-key>
```

Alias:

```http
GET /api/usage?window=24h&limit=50&offset=0
Authorization: Bearer <consumer-api-key>
```

Returns full sanitized usage detail for the exact bearer key used in the request. “Full” means all detail the proxy records for that key: totals, latency, token counts, provider/model/route/status breakdowns, consumer settings, key metadata, route-health summary, health summary, daily/monthly totals, cost estimates when pricing is available, and sanitized recent request rows. It does not include request/response bodies because the proxy does not persist them in usage stats.

## Time windows and pagination

Supported query/body controls:

- `since`: Unix timestamp or ISO-8601 datetime; include events at or after this timestamp.
- `until`: Unix timestamp or ISO-8601 datetime; include events at or before this timestamp.
- `window`: relative window such as `15m`, `24h`, `7d`, or `4w`. Example: `?window=24h`. When `window` is set it overrides `since`.
- `limit`: recent-event page size, clamped to 1–500.
- `offset`: recent-event offset for pagination.

Totals and breakdowns are computed over the selected time window. `recent` is then paginated using `limit`/`offset`.

The response `window` object includes the normalized values plus `recent_total` and `recent_returned`.

## Cost estimates

Responses include `cost_estimate` when the router metrics file has pricing entries for the observed `model_family@provider` pair.

The proxy reads `price_in_usd_per_mtok` and `price_out_usd_per_mtok` from `DASHBOARD_POLICY_METRICS_PATH` / `metrics.live.lua` and estimates:

```text
(tokens_in / 1_000_000 * input_price) + (tokens_out / 1_000_000 * output_price)
```

Cost values are estimates, rounded to six decimals, and include `priced_events`, `unpriced_events`, and `price_sources` metadata.

## Consumer success response

```json
{
  "schema_version": 3,
  "kind": "router_key_usage",
  "detail_level": "full",
  "viewer": "consumer:crm",
  "generated_at": 1760000000,
  "consumer": "crm",
  "key_sha256_prefix": "abcdef123456",
  "source": {
    "persistent_history": true,
    "history_path_configured": true
  },
  "window": {
    "since": 1760000000,
    "until": null,
    "window": "24h",
    "limit": 50,
    "offset": 0,
    "recent_total": 1,
    "recent_returned": 1
  },
  "key": {
    "sha256_prefix": "abcdef123456",
    "status": "active"
  },
  "consumer_settings": {
    "status": "active",
    "allowed_routes": [],
    "rate_per_min": 600,
    "burst": 200,
    "effective_per_min": 600
  },
  "totals": {
    "requests": 1,
    "errors": 0,
    "tokens_in": 10,
    "tokens_out": 3,
    "tokens_total": 13,
    "latency_ms_avg": 123.4,
    "latency_ms_max": 123.4,
    "error_rate": 0.0,
    "last_seen": 1760000000
  },
  "cost_estimate": {
    "estimated": true,
    "usd": 0.000123,
    "priced_events": 1,
    "unpriced_events": 0,
    "source": "/path/to/metrics.live.lua",
    "price_sources": {}
  },
  "daily_totals": [],
  "monthly_totals": [],
  "by_provider": {},
  "by_model_family": {},
  "by_route": {},
  "by_served_model": {},
  "by_status": {},
  "route_health": [],
  "health_summary": {},
  "recent": [],
  "security": {
    "sanitized": true,
    "raw_api_key_exposed": false,
    "full_sha256_exposed": false,
    "provider_credentials_exposed": false
  }
}
```

`recent` rows include sanitized recorded fields such as timestamp, method, path, status, latency, requested model, provider, model family, served model, token counts, cost estimate, error code/type/message, and decision trace when recorded. Raw bearer tokens, full SHA-256 digests, and obvious provider key patterns are redacted from nested detail.

## Dashboard lookup

```http
POST /dashboard/api/key-usage
Cookie: <dashboard session>
Content-Type: application/json

{
  "api_key": "<consumer-api-key>",
  "window": "24h",
  "limit": 50,
  "offset": 0
}
```

The dashboard endpoint hashes the submitted key server-side and returns sanitized usage for that key if it is configured or has recorded usage. The dashboard UI has a **Key usage** tab for this lookup and renders totals, cost estimates, recent rows, and JSON detail for daily/monthly totals.

### Dashboard auth behavior

- Missing dashboard auth returns `401` with code `dashboard_auth`.
- A consumer bearer token on this dashboard endpoint also returns `401` with code `dashboard_auth`.
- Unknown/unseen keys return `404` with code `key_usage_not_found`.

## OpenAPI docs/spec

FastAPI exposes these endpoints in `/openapi.json`:

- `GET /v1/usage`
- `GET /api/usage`
- `POST /dashboard/api/key-usage`

## Error response shape

Endpoints use the existing router error envelope:

```json
{
  "error": {
    "message": "unauthorized caller",
    "type": "auth_error",
    "code": "caller_auth"
  }
}
```

## Implementation notes

The proxy records per-key counters in memory alongside existing per-consumer stats and appends sanitized events to persistent JSONL history:

- `by_key_sha256`
- `by_key_provider`
- `by_key_model_family`
- `by_key_route`
- `by_key_served_model`
- `by_key_status`
- `key_owner`

The usage endpoints rebuild windowed totals from recent in-memory events plus persistent history, de-duplicating by `usage_event_id`.
