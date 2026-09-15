"""Atomic monthly admission shared by every Cloud ingress replica.

Quota comes only from a freshly resolved control-plane route. Reserve before
forwarding; settle once after the full response. Unknown crash outcomes remain
reserved until reconciled, never released speculatively to permit overruns.
"""
import time
import uuid
import host_store

SCHEMA = [
    '''CREATE TABLE IF NOT EXISTS cloud_quota_periods (
       tenant_id BIGINT NOT NULL, period_start BIGINT NOT NULL, period_end BIGINT NOT NULL,
       used BIGINT NOT NULL DEFAULT 0 CHECK (used >= 0),
       PRIMARY KEY (tenant_id, period_start))''',
    '''CREATE TABLE IF NOT EXISTS cloud_quota_reservations (
       id TEXT PRIMARY KEY, tenant_id BIGINT NOT NULL, period_start BIGINT NOT NULL,
       state TEXT NOT NULL DEFAULT 'pending', created_at BIGINT NOT NULL,
       FOREIGN KEY (tenant_id, period_start) REFERENCES cloud_quota_periods(tenant_id, period_start))''',
]
SCHEMA += [
    "ALTER TABLE cloud_quota_reservations ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT",
    "ALTER TABLE cloud_quota_reservations ADD COLUMN IF NOT EXISTS meter_event_name TEXT",
    "ALTER TABLE cloud_quota_reservations ADD COLUMN IF NOT EXISTS usage_batch_id TEXT",
    """CREATE TABLE IF NOT EXISTS cloud_usage_batches (
        id TEXT PRIMARY KEY, stripe_customer_id TEXT NOT NULL, meter_event_name TEXT NOT NULL,
        value BIGINT NOT NULL CHECK (value > 0), event_timestamp BIGINT NOT NULL,
        first_attempt_at BIGINT, reported_at BIGINT)""",
    """CREATE INDEX IF NOT EXISTS cloud_usage_unbatched ON cloud_quota_reservations(created_at)
        WHERE state='success' AND stripe_customer_id IS NOT NULL AND usage_batch_id IS NULL""",
]



def reserve(tenant_id, quota):
    if not isinstance(quota, dict):
        raise ValueError('Monthly quota is required')
    start, end, limit = [quota.get(k) for k in ('period_start', 'period_end', 'limit')]
    if any(type(v) is not int for v in (tenant_id, start, end, limit)):
        raise ValueError('Invalid quota contract')
    if tenant_id <= 0 or limit < 0 or not start <= int(time.time()) < end:
        raise ValueError('Expired or invalid quota contract')
    overage = quota.get('overage_enabled') is True
    customer, meter = quota.get('stripe_customer_id'), quota.get('meter_event_name')
    if overage and (not isinstance(customer, str) or not customer.startswith('cus_')
                    or meter != 'unhardcoded_requests_202609_v2'):
        raise ValueError('Metered usage requires a configured billing destination')
    if limit == 0:
        return None
    ticket = uuid.uuid4().hex
    with host_store._get_pool().connection() as conn:
        row = conn.execute('''INSERT INTO cloud_quota_periods (tenant_id,period_start,period_end,used)
            VALUES (%s,%s,%s,1) ON CONFLICT (tenant_id,period_start) DO UPDATE
            SET used=cloud_quota_periods.used+1,period_end=EXCLUDED.period_end
            WHERE (%s OR cloud_quota_periods.used < %s) RETURNING used''',
            [tenant_id,start,end,overage,limit]).fetchone()
        if row is None:
            return None
        conn.execute('''INSERT INTO cloud_quota_reservations (id,tenant_id,period_start,created_at,stripe_customer_id,meter_event_name)
            VALUES (%s,%s,%s,%s,%s,%s)''', [ticket,tenant_id,start,int(time.time()),customer if overage else None,meter if overage else None])
    return ticket


def settle(ticket, success):
    with host_store._get_pool().connection() as conn:
        row = conn.execute('''UPDATE cloud_quota_reservations SET state=%s
            WHERE id=%s AND state='pending' RETURNING tenant_id,period_start''',
            ['success' if success else 'failed',ticket]).fetchone()
        if row and not success:
            conn.execute('''UPDATE cloud_quota_periods SET used=used-1
                WHERE tenant_id=%s AND period_start=%s''', list(row))
