"""Run every minute in a dedicated Cloud worker with DATABASE_URL and
CLOUD_STRIPE_USAGE_SECRET_KEY. Only confirmed successes are sent; provider keys
and request content never leave the ledger. Exit nonzero on reporting failure.
"""
import json
import os
import time
import uuid
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import host_store


def prepare_batch():
    with host_store._get_pool().connection() as conn:
        group = conn.execute("""SELECT tenant_id,period_start,stripe_customer_id,meter_event_name
            FROM cloud_quota_reservations WHERE state='success' AND stripe_customer_id IS NOT NULL
              AND usage_batch_id IS NULL AND created_at < %s
            ORDER BY created_at LIMIT 1""", [int(time.time()) - 60]).fetchone()
        if not group:
            return None
        rows = conn.execute("""SELECT id,created_at FROM cloud_quota_reservations
            WHERE tenant_id=%s AND period_start=%s AND stripe_customer_id=%s AND meter_event_name=%s
              AND state='success' AND usage_batch_id IS NULL AND created_at < %s
            ORDER BY created_at LIMIT 10000 FOR UPDATE SKIP LOCKED""",
            [*group, int(time.time()) - 60]).fetchall()
        if not rows:
            return None
        batch = uuid.uuid4().hex
        conn.execute("""INSERT INTO cloud_usage_batches
            (id,stripe_customer_id,meter_event_name,value,event_timestamp) VALUES (%s,%s,%s,%s,%s)""",
            [batch, group[2], group[3], len(rows), min(r[1] for r in rows)])
        conn.execute('UPDATE cloud_quota_reservations SET usage_batch_id=%s WHERE id=ANY(%s)',
                     [batch, [r[0] for r in rows]])
        return batch


def send_to_stripe(batch):
    key = os.environ.get('CLOUD_STRIPE_USAGE_SECRET_KEY', '')
    mode = os.environ.get('CLOUD_STRIPE_USAGE_MODE', 'test')
    if mode not in {'test', 'live'} or not key.startswith((f'sk_{mode}_', f'rk_{mode}_')):
        raise RuntimeError('Configure the matching Stripe usage key and mode')
    data = urlencode({'event_name': batch[2], 'identifier': batch[0],
                      'timestamp': batch[4], 'payload[stripe_customer_id]': batch[1],
                      'payload[value]': str(batch[3])}).encode()
    request = Request('https://api.stripe.com/v1/billing/meter_events', data=data,
                      headers={'Authorization': 'Bearer ' + key, 'Idempotency-Key': batch[0],
                               'Content-Type': 'application/x-www-form-urlencoded'})
    with urlopen(request, timeout=20) as response:
        obj = json.load(response)
    if obj.get('identifier') != batch[0]:
        raise RuntimeError('Stripe usage acknowledgement mismatch')


def report_batch(batch_id, send=send_to_stripe):
    # Persist the first attempt before network I/O, including on connection loss.
    with host_store._get_pool().connection() as conn:
        conn.execute('''UPDATE cloud_usage_batches SET first_attempt_at=%s
            WHERE id=%s AND first_attempt_at IS NULL''', [int(time.time()), batch_id])
    with host_store._get_pool().connection() as conn:
        batch = conn.execute('''SELECT id,stripe_customer_id,meter_event_name,value,event_timestamp,
            first_attempt_at,reported_at FROM cloud_usage_batches WHERE id=%s FOR UPDATE''', [batch_id]).fetchone()
        if not batch or batch[6] is not None:
            return False
        if int(time.time()) - batch[5] >= 23 * 3600:
            # Stripe only guarantees identifier deduplication for 24 hours.
            # Do not charge twice after an ambiguous outcome; reconcile manually.
            raise RuntimeError('Usage batch needs reconciliation before retry: ' + batch_id)
        send(batch)
        conn.execute('UPDATE cloud_usage_batches SET reported_at=%s WHERE id=%s', [int(time.time()), batch_id])
        return True


def run(max_batches=100):
    reported = 0
    for _ in range(max_batches):
        with host_store._get_pool().connection() as conn:
            row = conn.execute('SELECT id FROM cloud_usage_batches WHERE reported_at IS NULL ORDER BY event_timestamp LIMIT 1').fetchone()
        batch_id = row[0] if row else prepare_batch()
        if batch_id is None:
            break
        reported += report_batch(batch_id)
    return reported


if __name__ == '__main__':
    try:
        print('Usage batches reported:', run())
    finally:
        if host_store._pool is not None:
            host_store._pool.close()
