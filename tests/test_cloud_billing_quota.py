"""Billing ledger integration tests against disposable Postgres."""
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
import billing_quota
import host_store
import report_cloud_usage as reporter


@pytest.fixture(autouse=True)
def ledger():
    with host_store._get_pool().connection() as conn:
        conn.execute('TRUNCATE cloud_quota_reservations, cloud_quota_periods, cloud_usage_batches')


def contract(limit=10, paid=False, customer='cus_test'):
    now=int(time.time())
    return dict(limit=limit, period_start=now-1000, period_end=now+1000,
                overage_enabled=paid, stripe_customer_id=customer if paid else None,
                meter_event_name='unhardcoded_requests_202609_v2' if paid else None)


def age_requests():
    with host_store._get_pool().connection() as conn:
        conn.execute('UPDATE cloud_quota_reservations SET created_at=created_at-120')


def test_free_concurrent_requests_stop_at_cap():
    quota=contract()
    with ThreadPoolExecutor(max_workers=12) as pool:
        tickets=list(pool.map(lambda _: billing_quota.reserve(1,quota),range(40)))
    assert len([x for x in tickets if x])==10


def test_paid_goes_beyond_included_requests():
    quota=contract(limit=2,paid=True)
    tickets=[billing_quota.reserve(1,quota) for _ in range(5)]
    assert all(tickets)
    for ticket in tickets: billing_quota.settle(ticket,True)
    age_requests()
    batch=reporter.prepare_batch()
    sender=Mock()
    assert reporter.report_batch(batch,sender)
    assert sender.call_args.args[0][3]==5  # Stripe's tier excludes the first two.
    assert not reporter.report_batch(batch,sender)
    assert sender.call_count==1


def test_failed_and_pending_requests_never_billed():
    quota=contract(paid=True)
    failed=billing_quota.reserve(1,quota)
    billing_quota.settle(failed,False)
    billing_quota.settle(failed,False)
    billing_quota.reserve(1,quota)
    age_requests()
    assert reporter.prepare_batch() is None
    with host_store._get_pool().connection() as conn:
        assert conn.execute('SELECT used FROM cloud_quota_periods').fetchone()[0]==1


def test_free_success_has_no_stripe_event():
    ticket=billing_quota.reserve(1,contract())
    billing_quota.settle(ticket,True)
    age_requests()
    assert reporter.prepare_batch() is None


def test_failure_releases_exactly_one_slot():
    quota=contract(limit=1)
    ticket=billing_quota.reserve(1,quota)
    assert billing_quota.reserve(1,quota) is None
    billing_quota.settle(ticket,False)
    billing_quota.settle(ticket,False)
    assert billing_quota.reserve(1,quota)
    assert billing_quota.reserve(1,quota) is None


def test_unknown_or_expired_meter_contract_fails_closed():
    q=contract(paid=True);q['stripe_customer_id']=None
    with pytest.raises(ValueError): billing_quota.reserve(1,q)
    q=contract();q['period_end']=int(time.time())-1
    with pytest.raises(ValueError): billing_quota.reserve(1,q)


def test_organization_and_periods_are_isolated():
    first=contract(limit=1)
    assert billing_quota.reserve(1,first)
    assert billing_quota.reserve(2,first)
    second={**first,'period_start':first['period_start']+1}
    assert billing_quota.reserve(1,second)
    assert billing_quota.reserve(1,first) is None


def test_retry_uses_same_immutable_batch_after_network_failure():
    ticket=billing_quota.reserve(1,contract(paid=True))
    billing_quota.settle(ticket,True);age_requests()
    batch=reporter.prepare_batch()
    sender=Mock(side_effect=[OSError('connection lost'),None])
    with pytest.raises(OSError): reporter.report_batch(batch,sender)
    assert reporter.report_batch(batch,sender)
    assert sender.call_args_list[0].args==sender.call_args_list[1].args


def test_ambiguous_old_batch_requires_reconciliation():
    ticket=billing_quota.reserve(1,contract(paid=True))
    billing_quota.settle(ticket,True);age_requests()
    batch=reporter.prepare_batch()
    with host_store._get_pool().connection() as conn:
        conn.execute('UPDATE cloud_usage_batches SET first_attempt_at=%s WHERE id=%s',
                     [int(time.time())-24*3600,batch])
    sender=Mock()
    with pytest.raises(RuntimeError,match='reconciliation'): reporter.report_batch(batch,sender)
    sender.assert_not_called()


def test_batch_preparation_concurrency_cannot_duplicate_usage():
    for _ in range(20):
        billing_quota.settle(billing_quota.reserve(1,contract(paid=True)),True)
    age_requests()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _:reporter.prepare_batch(),range(4)))
    with host_store._get_pool().connection() as conn:
        assert conn.execute('SELECT SUM(value) FROM cloud_usage_batches').fetchone()[0]==20
    assert reporter.prepare_batch() is None


@pytest.fixture(scope='session', autouse=True)
def close_pool_at_exit():
    yield
    if host_store._pool is not None:
        host_store._pool.close()
        host_store._pool = None
