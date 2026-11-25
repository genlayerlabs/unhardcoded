"""Process-local provider facts shared by the router and its tenant runtimes."""
from __future__ import annotations

import asyncio
import logging
import random
import threading
from time import monotonic

import host_store
from env_coerce import env_int

_MAX_ROWS = 10_000
_log = logging.getLogger(__name__)


class ProviderFacts:
    def __init__(self):
        self._snapshot = (None, {})
        self._count = 0
        # Serialize the read-and-publish and commit-and-publish operations.
        # Ranking never takes this lock.
        self._updates = threading.Lock()
        self._refresh_failed = False

    def is_true(self, name: str, provider_id: str, now_ms: int, max_age_ms: int) -> bool:
        loaded_at, facts = self._snapshot
        if loaded_at is None or (monotonic() - loaded_at) * 1000 > max_age_ms:
            return False
        providers = facts.get(name)
        observed_at = providers.get(provider_id) if providers is not None else None
        return observed_at is not None and 0 <= now_ms - observed_at <= max_age_ms

    def refresh(self) -> None:
        with self._updates:
            loaded_at = monotonic()
            try:
                rows = host_store.load_provider_facts(_MAX_ROWS)
            except Exception:
                if not self._refresh_failed:
                    _log.warning("Provider fact refresh failed; retaining the previous snapshot")
                self._refresh_failed = True
                return
            self._refresh_failed = False
            if len(rows) == _MAX_ROWS:
                _log.warning("Provider fact refresh reached the %d-row limit", _MAX_ROWS)
            facts = {}
            for name, provider_id, observed_at in rows:
                facts.setdefault(name, {})[provider_id] = observed_at
            self._snapshot = (loaded_at, facts)
            self._count = len(rows)

    def write(self, provider_id: str, name: str, value: bool,
              observed_at_ms: int, detail) -> bool:
        with self._updates:
            if not host_store.upsert_provider_fact(provider_id, name, value, observed_at_ms, detail):
                return False
            _, facts = self._snapshot
            providers = facts.get(name)
            if value:
                if providers is None or provider_id not in providers:
                    if self._count == _MAX_ROWS:
                        # Make room for this write while keeping the cache bounded.
                        first_name = next(iter(facts))
                        self._remove(facts, first_name, next(iter(facts[first_name])))
                    facts.setdefault(name, {})[provider_id] = observed_at_ms
                    self._count += 1
                else:
                    providers[provider_id] = observed_at_ms
            else:
                self._remove(facts, name, provider_id)
            return True

    def _remove(self, facts, name, provider_id):
        providers = facts.get(name)
        if providers is not None and providers.pop(provider_id, None) is not None:
            self._count -= 1
            if not providers:
                del facts[name]

    async def run(self) -> None:
        interval = max(1, env_int("ROUTER_PROVIDER_FACT_REFRESH_S", 5))
        while True:
            started = monotonic()
            await asyncio.to_thread(self.refresh)
            await asyncio.sleep(max(0, started + interval * random.uniform(0.8, 1) - monotonic()))
