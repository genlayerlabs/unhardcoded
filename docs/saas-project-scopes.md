# SaaS project/environment bridge (version 2)

This extends the existing control-plane bridge; it does not introduce a router,
policy algebra or evaluator. Legacy operator authentication and non-tenant calls
are unchanged. The engine submodule is unchanged.

## Authentication and scope

The ingress advertises `scope_version=2` when resolving a consumer key. A scoped
response must contain positive integer `project_id` and `environment_id`, with
`scope_version: 2` and the existing `tenant_id`. Partial or unknown versions are
inactive, never downgraded to tenant-wide access. Old control planes that return
the original tenant-only response remain supported.

The cloud must reject scoped keys when this capability is absent. Deploy this
dataplane before activating scoped keys in cloud; an old ingress cannot safely
serve a scoped key.

On every inference request the ingress resolves:

```text
GET /internal/tenants/{tenant}/projects/{project}/environments/{environment}/routes/{name}
    ?key_sha256={authenticated_key_digest}
```

The cloud checks that the key still exists, is unrevoked/unexpired and belongs to
that exact scope, then resolves the active unpaused revision. This check is not
cached: stale ingress authentication cannot bypass key revocation. Requests that
already passed authorization may finish; revocation is not cancellation.

The ingress strips client-supplied scope headers and stamps only resolved values:
`x-unhardcoded-scope-version`, `x-unhardcoded-project`,
`x-unhardcoded-environment`, plus the existing trusted tenant/internal headers.
Chat Completions and Responses follow the same contract resolver and Lua engine.

The shim validates the scope and fetches fresh credentials from the corresponding
`.../provider-env` endpoint. Both route and credential responses must echo the
matching version, tenant, project and environment. Missing/mismatched scopes or
control-plane failures reject scoped requests; no legacy credential fallback is
allowed. Sessions and provider-discovery caches include the environment identity.
The bounded routing summary records project/environment IDs, not secrets.

## Rollout boundary

This protocol is only one delivery block. The cloud must also enforce project
permissions in the dashboard, assign all issued keys/routes to environments and
restrict provider assignments by development/production grants before claiming
complete environment isolation. Do not expose a UI-only project boundary.

Cloud migration and deployment need a coordinated SaaS-only write pause so an old
control-plane instance cannot keep editing the original credential tables after
their encrypted blobs are migrated. Preserve originals for recovery; do not roll
back across subsequent project configuration changes without a validated restore
plan. No production deployment is authorized by these changes.

## Scoped usage and activity

The internal `/internal/usage` and `/internal/usage/recent` endpoints accept
`project_id` and `environment_id` together with the required caller slug. Partial
or nonpositive scopes are rejected. PostgreSQL applies all three filters before
aggregation or the recent-call limit; historical rows without a scope do not
appear in environment reports. Responses echo `scope_version: 2` and both IDs so
Cloud can reject an old ingress that ignores the new filters. Database errors
return 503; reads run off the ingress event loop and have a statement deadline.
The ledger is best-effort telemetry, not billing-grade accounting.

The companion Cloud #3 now wires scope through the entire dashboard. It includes
an opt-in two-process test using the actual Django bridge and this dataplane,
with concurrent Chat Completions/Responses, streaming/fallback, scope forgery,
revoked cached keys and removed connection assignments. Provider calls are fake;
this verifies isolation/recovery rather than a production throughput target.

## Bridge transport

HTTPS is required by default for all control-plane HTTP requests and the ingress
hop that carries trusted tenant headers to the shim. TLS certificate verification
remains enabled. Configure trusted internal certificates (including a CA bundle
where needed); do not expose the shim or internal control-plane endpoints publicly.

`CP_ALLOW_INSECURE_HTTP=1` is an explicit exception for isolated local development
or an operator-managed transport boundary. It does not add encryption. The local
Compose/test fixtures set it deliberately; SaaS production must leave it disabled.
Cloud's production settings also reject plaintext router/ingress URLs. Existing
operator-only deployments are unaffected because their control-plane integration
is disabled. Existing HTTP control-plane deployments must configure HTTPS or
explicitly opt into their existing insecure transport before adopting this release.

## Key limits, budgets and reconciliation

The route response may carry the calling key's identity and limits:

```json
"key": {"id": 42, "subject": "g:7", "labels": {"team": "search"}},
"limits": {"monthly_budget_usd": "40.00", "rate_per_min": 120, "burst": 40}
```

`subject` (`g:<group id>` or `k:<key id>`) is the unit that limits apply to; any
limit may be `null`. Both objects absent means no key limits (older control
planes). A present but malformed object makes the route unavailable (`503
route_unavailable`): a limit is never silently dropped.

After the plan rate check the ingress enforces, in order:

1. the subject token bucket `"{caller}|{subject}"` (`429 key_rate_limit` with
   `Retry-After`);
2. an atomic PostgreSQL reservation of `CLOUD_BUDGET_RESERVATION_USD` (default
   0.05, capped at the budget itself) against `(tenant, subject, UTC month)`,
   admitted only if `spent + reserved + reservation <= budget`
   (`402 key_budget_exhausted` with `budget_usd` and `spent_usd`);
3. the per-pod capacity permit.

An unavailable store fails closed (`503 key_rate_limit_unavailable` /
`key_budget_unavailable`). The reservation is settled synchronously when the
call finishes on every path (success, upstream error, exception, capacity
rejection, stream end or client disconnect): it is released and the actual cost
(unknown cost counts as 0) is added. A reservation whose holder died expires
after 15 minutes. Overshoot is bounded by concurrent in-flight calls times
(actual cost - reservation). The recorded `routing_summary` includes `key_id`
and `subject`.

Because the scoped route lookup is uncached and carries the key digest, a key
revoked in the control plane is refused on its very next request even while the
key resolution itself is still cached.

Reconciliation endpoints (same shared secret; `caller` and the complete
project/environment pair behave as for `/internal/usage`, filtering before
aggregation or paging):

* `GET /internal/usage?...&until_ts=&group_by=key|model|route|day|hour` adds a
  `groups` list (`key` groups by the full key digest). Windows are half-open
  (`since_ts <= ts < until_ts`). Every response carries `watermark_ts`
  (`now - LEDGER_WATERMARK_LAG_S`, capped by the oldest call still queued in
  this replica's ledger writer); a window with `until_ts <= watermark_ts` is
  closed.
* `GET /internal/usage/export?caller=&after_id=&limit=<=5000` returns ledger
  rows in id order with `next_after_id` and `watermark_ts`. Rows are delivered
  once they were inserted at least `LEDGER_WATERMARK_LAG_S` (default 120) ago, so
  the id cursor cannot pass a lower id still committing on another replica.
  `watermark_ts` is `now - LEDGER_WATERMARK_LAG_S`, capped by the oldest call
  still queued in this replica's ledger writer and by the oldest matching row
  not yet delivered: every call with `ts < watermark_ts` is in a page at or
  before `next_after_id`. Other replicas' writer backlogs are covered only by
  the lag.
* `GET /internal/budgets?tenant_id=&period=YYYY-MM&subject=...` (repeatable, at
  most 500) returns `spent_usd` and live `reserved_usd` per subject.
