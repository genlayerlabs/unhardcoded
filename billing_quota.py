"""Atomic monthly admission shared by every Cloud ingress replica.

Quota comes only from a freshly resolved control-plane route. Reserve before
forwarding; settle once after the full response. Unknown crash outcomes remain
reserved until they expire without counting as successful usage.
"""
import os
import time
import uuid

import host_store


def _pending_max_age() -> int:
    try:
        return max(1, int(os.getenv("CLOUD_QUOTA_PENDING_MAX_AGE_S", "3600")))
    except ValueError:
        return 3600


def _expire_period(conn, tenant_id, start, cutoff):
    conn.execute('''WITH expired AS (
        UPDATE cloud_quota_reservations SET state = 'expired'
        WHERE tenant_id = %s AND period_start = %s AND state = 'pending'
          AND created_at < %s RETURNING id
    )
    UPDATE cloud_quota_periods SET used = used - (SELECT count(*) FROM expired)
    WHERE tenant_id = %s AND period_start = %s AND EXISTS (SELECT 1 FROM expired)''',
    [tenant_id, start, cutoff, tenant_id, start])


def expire_pending(now: int) -> None:
    """Release abandoned slots, including periods that get no further requests."""
    cutoff = now - _pending_max_age()
    with host_store._get_pool().connection() as conn:
        periods = conn.execute('''SELECT DISTINCT tenant_id, period_start
            FROM cloud_quota_reservations WHERE state = 'pending' AND created_at < %s''',
            [cutoff]).fetchall()
    # Hold locks for a single tenant and period per transaction.
    for tenant_id, start in periods:
        with host_store._get_pool().connection() as conn:
            _expire_period(conn, tenant_id, start, cutoff)


def reserve(tenant_id: int, quota: object) -> str | None:
    if not isinstance(quota, dict):
        raise ValueError('Monthly quota is required')
    start, end, limit = [quota.get(k) for k in ('period_start', 'period_end', 'limit')]
    if any(type(v) is not int for v in (tenant_id, start, end, limit)):
        raise ValueError('Invalid quota contract')
    now = int(time.time())
    if tenant_id <= 0 or limit < 0 or not start <= now < end:
        raise ValueError('Expired or invalid quota contract')
    overage = quota.get('overage_enabled') is True
    customer, meter = quota.get('stripe_customer_id'), quota.get('meter_event_name')
    if overage and (not isinstance(customer, str) or not customer.startswith('cus_')
                    or meter != 'unhardcoded_requests_202609_v2'):
        raise ValueError('Metered usage requires a configured billing destination')
    if limit == 0 and not overage:
        return None
    ticket = uuid.uuid4().hex
    with host_store._get_pool().connection() as conn:
        # Release abandoned slots before checking the current allowance.
        _expire_period(conn, tenant_id, start, now - _pending_max_age())
        row = conn.execute('''INSERT INTO cloud_quota_periods (tenant_id, period_start, period_end, used)
            VALUES (%s, %s, %s, 1) ON CONFLICT (tenant_id, period_start) DO UPDATE
            SET used = cloud_quota_periods.used + 1, period_end = EXCLUDED.period_end
            WHERE (%s OR cloud_quota_periods.used < %s) RETURNING used''',
            [tenant_id, start, end, overage, limit]).fetchone()
        if row is None:
            return None
        conn.execute('''INSERT INTO cloud_quota_reservations
            (id, tenant_id, period_start, created_at, stripe_customer_id, meter_event_name)
            VALUES (%s, %s, %s, %s, %s, %s)''',
            [ticket, tenant_id, start, now, customer if overage else None, meter if overage else None])
    return ticket


def settle(ticket: str, success: bool) -> None:
    with host_store._get_pool().connection() as conn:
        row = conn.execute('''UPDATE cloud_quota_reservations SET state = %s
            WHERE id = %s AND state = 'pending' RETURNING tenant_id, period_start''',
            ['success' if success else 'failed', ticket]).fetchone()
        if row and not success:
            conn.execute('''UPDATE cloud_quota_periods SET used = used - 1
                WHERE tenant_id = %s AND period_start = %s''', row)
