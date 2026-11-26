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

The version 2 credential response may include a `managed_providers` array of
up to 128 distinct, exact provider IDs. Each ID must exist in the router catalog
and be configured in `SAAS_SHARED_PROVIDERS`. A nonempty array requires an empty
`env` object and empty or omitted `connections`. Invalid grants reject the entire
scoped request with 503. Missing or empty grants preserve tenant credential
routing. Legacy tenant-wide responses cannot grant operator access. The router
checks grants on each credential lookup, so removal applies to the next request.

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
`key_budget_unavailable`). Settlement runs on a bounded background queue.
For the key-only reservation, a successful call with unknown cost books the
reservation. A failed call with unknown cost books zero. A lost reservation
releases after 15 minutes and reported actual cost can exceed the estimate.
The recorded `routing_summary` includes `key_id` and `subject`.

A published route can also include `limits.shared_budget`:

```json
{
  "scope": "search",
  "monthly_budget_usd": "100.00",
  "max_call_cost_usd": "0.25",
  "guidance": "Contact your administrator for more capacity.",
  "help_url": "https://example.org/limits"
}
```

Keys in the same tenant and UTC month share spend when their scope IDs match.
Use a stable ID even when changing the limit. Each route resolution supplies the
current limit and ceiling. The lookup is uncached, so revocations and limit
changes apply to the next resolution. Enable shared scopes after every ingress
replica supports the field. Use decimal strings to transmit exact
amounts with up to 12 fractional digits. A zero limit rejects all calls.

The publisher must supply a positive ceiling covering the entire request,
including fallbacks and any auxiliary calls. The ingress charges that ceiling
against the scope and any key/group limit in one transaction. On exhaustion,
the ingress returns `402 shared_budget_exhausted` or `402 key_budget_exhausted`.
The shared error includes the optional `guidance` and `help_url`.

A known actual cost replaces the precharge on settlement. A request rejected
before forwarding settles at zero. Unknown cost and lost settlement leave the
full precharge in place, even after expiration. Reconciliation records become
eligible for pruning one day after expiry. An actual cost above the published
ceiling is logged and charged in full, so a publisher that supplies a ceiling
below actual upstream spend can exceed its configured budget.

### Shared allowances across months

`GET /internal/budgets/shared/{tenant_id}/{scope}` returns durable totals for
the published scope ID across UTC months. Both GET and PUT require
`x-internal-secret`. The account tracks guarded admissions made by ingress
replicas running this version. Upgrade every ingress replica before configuring
an allowance. To carry a starting balance from earlier activity, provide a
signed opening adjustment in `correction_usd`.

`PUT` replaces the allowance and cumulative correction at the supplied
settings revision. For example, the first write for a scope can use:

```json
{"revision": 0, "allowance_usd": "100.00", "correction_usd": "0"}
```

The response has the same shape as GET and advances `revision` by one.
A stale revision returns HTTP 409. Read the current state before retrying a
conflicting change. Admissions and settlements update the totals independently
of the settings revision. Use decimal strings with up to 12 fractional digits.
A null allowance disables the cross-month cap while preserving accounting.
Zero pauses admissions. A positive cap applies in addition to the published
monthly limits. An admission that would exceed the cap receives
`402 shared_allowance_exhausted`. When the cap is active, new-month admissions
can proceed with an earlier reservation still open because both charges count
toward the cross-month cap.

Monetary totals in the response are decimal strings. `precharged_usd` is the
sum of admitted ceilings. `replaced_usd` is the portion of those ceilings with
a known settlement, and `settled_usd` is the actual cost that replaced them.
Their difference `precharged_usd - replaced_usd` includes open reservations,
unknown costs and lost settlements. The total used for admission is
`committed_usd = precharged_usd - replaced_usd + settled_usd + correction_usd`.
The cumulative correction can be positive or negative and applies only to the
cross-month total. A known settlement applies once, to the scope and original
UTC month, while its reservation is retained. A correction can account for
information that arrives after the reservation is gone. These totals survive
reservation pruning and are separate from the ordinary usage export.

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

`GET /internal/usage/refusals?tenant_id=<id>&since_ts=<epoch>&until_ts=<epoch>`
uses the same `x-internal-secret` as usage. Both timestamps are required and
must fall on UTC hour boundaries. The range includes the start hour and
excludes the end hour. The response includes `tenant_id`, the requested
`window`, and `groups` with `subject`, `reason`, `count`, `first_ts`,
and `last_ts`. The timestamps in a group are the first and last refusal
times in epoch seconds.

For authenticated SaaS tenants, the reason is the response error code
(`key_budget_exhausted`, `key_rate_limit`, `consumer_budget_exhausted`,
or `caller_rate_limit`). A tenant-level limit without a resolved key
subject appears with `subject: null`. Unauthorized requests lack a
verified tenant and appear only in the existing reason-level
`llm_router_rejections_total` metric.

PostgreSQL keeps one counter per tenant, subject, reason, and UTC hour.
The read covers retained hours (`ROUTER_DB_RETENTION_DAYS`, default 30),
and an unavailable store returns 503 `refusal_store_unavailable`.
Recording is best effort. Counts can arrive late or be dropped during
an outage or when the bounded background queue fills.
