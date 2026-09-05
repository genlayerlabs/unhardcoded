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

```
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
