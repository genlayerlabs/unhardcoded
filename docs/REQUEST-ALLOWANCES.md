# Tenant request allowances

Set `REQUEST_ALLOWANCES_ENABLED=1` on each ingress replica that shares the
admission database. The setting defaults to off. When enabled, the operator's
internal route resolver can include this optional top-level field in a tenant's
route resolution response:

```json
{
  "request_allowance": {
    "period_start": 1796083200,
    "period_end": 1798761600,
    "request_count": 1000
  }
}
```

The timestamps are integer Unix seconds in UTC. The example permits 1,000
requests during December 2026. The start is inclusive and the end is exclusive.
Use the same period boundaries and current count for every route and key of the
tenant. The count can change during the period without resetting usage. Give
the next period a new start. Omit the field for unlimited requests, or set the
count to zero to refuse all requests until the period ends. An invalid or stale
period returns 503.

All replicas reserve against the same PostgreSQL counter before forwarding a
request. When the allowance is full, ingress responds with HTTP 429 and a
`Retry-After` delay in seconds, rounded up to the end of the period. Admission
store failures return HTTP 503.

Upstream failures release their slot when the background writer processes the
settlement. Successful
responses consume a slot, including a stream cut short by bounded draining
after the client disconnects. An unsettled slot can be reclaimed after an hour,
including when a replica exits or the background writer drops a settlement.
Each period has one counter and only outstanding requests have reservation
rows. The store's retention sweep removes ended periods and their reservations.
