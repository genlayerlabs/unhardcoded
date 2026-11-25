"""Monthly quota integration tests against disposable PostgreSQL."""
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import billing_quota
import host_store


@pytest.fixture(autouse=True)
def ledger(host_store_clean):
    with host_store._get_pool().connection() as conn:
        conn.execute("TRUNCATE cloud_quota_reservations, cloud_quota_periods")


def contract(limit=1, paid=False):
    now = int(time.time())
    return dict(limit=limit, period_start=now - 1000, period_end=now + 1000,
                overage_enabled=paid, stripe_customer_id="cus_test" if paid else None,
                meter_event_name="unhardcoded_requests_202609_v2" if paid else None)


def test_concurrent_requests_contend_for_last_slot():
    quota = contract(limit=2)
    assert billing_quota.reserve(1, quota)
    with ThreadPoolExecutor(max_workers=12) as pool:
        tickets = list(pool.map(lambda _: billing_quota.reserve(1, quota), range(40)))
    assert sum(ticket is not None for ticket in tickets) == 1


@pytest.mark.parametrize("limit", [0, 2])
def test_paid_goes_beyond_included_requests(limit):
    quota = contract(limit=limit, paid=True)
    tickets = [billing_quota.reserve(1, quota) for _ in range(5)]
    assert all(tickets)
    for ticket in tickets:
        billing_quota.settle(ticket, True)
    with host_store._get_pool().connection() as conn:
        assert conn.execute(
            "SELECT state, stripe_customer_id FROM cloud_quota_reservations"
        ).fetchall() == [("success", "cus_test")] * 5


def test_failure_releases_exactly_one_slot():
    quota = contract()
    ticket = billing_quota.reserve(1, quota)
    assert billing_quota.reserve(1, quota) is None
    billing_quota.settle(ticket, False)
    billing_quota.settle(ticket, False)
    assert billing_quota.reserve(1, quota)
    assert billing_quota.reserve(1, quota) is None


def test_expiry_releases_slot_and_late_settlement_cannot_bill(monkeypatch):
    monkeypatch.setenv("CLOUD_QUOTA_PENDING_MAX_AGE_S", "60")
    quota = contract(paid=True)
    ticket = billing_quota.reserve(1, quota)
    quota["overage_enabled"] = False
    assert billing_quota.reserve(1, quota) is None
    with host_store._get_pool().connection() as conn:
        conn.execute("UPDATE cloud_quota_reservations SET created_at = created_at - 120")
    assert billing_quota.reserve(1, quota)
    billing_quota.settle(ticket, True)
    assert billing_quota.reserve(1, quota) is None
    with host_store._get_pool().connection() as conn:
        assert conn.execute(
            "SELECT state FROM cloud_quota_reservations WHERE id = %s", [ticket]
        ).fetchone() == ("expired",)


def test_invalid_meter_and_expired_period_fail_closed():
    with pytest.raises(ValueError):
        billing_quota.reserve(1, {**contract(paid=True), "stripe_customer_id": None})
    with pytest.raises(ValueError):
        billing_quota.reserve(1, {**contract(), "period_end": int(time.time()) - 1})


def test_tenants_and_periods_are_isolated():
    first = contract()
    assert billing_quota.reserve(1, first)
    assert billing_quota.reserve(2, first)
    second = {**first, "period_start": first["period_start"] + 1}
    assert billing_quota.reserve(1, second)
    assert billing_quota.reserve(1, first) is None


def test_sweep_expires_pending_in_inactive_periods(monkeypatch):
    monkeypatch.setenv("CLOUD_QUOTA_PENDING_MAX_AGE_S", "60")
    quota = contract(limit=2)
    pending = billing_quota.reserve(1, quota)
    successful = billing_quota.reserve(1, quota)
    billing_quota.settle(successful, True)
    billing_quota.expire_pending(quota["period_end"] + 1)
    with host_store._get_pool().connection() as conn:
        assert dict(conn.execute(
            "SELECT id, state FROM cloud_quota_reservations"
        ).fetchall()) == {pending: "expired", successful: "success"}
        assert conn.execute("SELECT used FROM cloud_quota_periods").fetchone() == (1,)
