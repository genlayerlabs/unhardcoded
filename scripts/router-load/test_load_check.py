"""Tests for benchmark result validation and the comparison exit criteria."""

import importlib.util
from pathlib import Path
from threading import Event
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "load_check", Path(__file__).resolve().parents[1] / "router-load-check.py",
)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)

from driver import percentiles, valid_workload


class LoadCheckTests(unittest.TestCase):
    def test_nearest_rank_percentiles(self):
        self.assertEqual(percentiles(list(range(100, 0, -1))), {"p50": 50, "p95": 95, "p99": 99})
        self.assertIsNone(percentiles([]))

    def test_refusals_and_stream_tokens_are_required(self):
        result = {
            "statuses_by_key": {
                "valid": {"200": 8}, "invalid": {"401": 1}, "limited": {"429": 1, "200": 1},
            },
            "overhead_ms": {"p99": 1},
            "first_streamed_token_ms": {"p99": 41},
        }
        self.assertTrue(valid_workload(result))
        result["statuses_by_key"]["invalid"]["200"] = 1
        self.assertFalse(valid_workload(result))
        del result["statuses_by_key"]["invalid"]["200"]
        result["first_streamed_token_ms"] = None
        self.assertFalse(valid_workload(result))

    def test_comparison_checks_each_replica_and_requires_an_increase_above_margin(self):
        baseline = {
            "overhead_ms": {"p99": 100}, "queries_per_request": 10,
            "peak_memory_bytes": {"replica-a": 100, "replica-b": 100},
        }
        candidate = {
            "overhead_ms": {"p99": 131}, "queries_per_request": 14,
            "peak_memory_bytes": {"replica-a": 130, "replica-b": 131},
        }
        self.assertEqual(check.compare(baseline, baseline, 30), [])
        self.assertEqual([entry["metric"] for entry in check.compare(baseline, candidate, 30)],
                         ["p99 overhead", "queries per request", "replica-b peak memory"])

    def test_resource_samples_match_container_ids_and_track_peaks(self):
        stop = Event()
        readings = iter(["4|1", "3|2"])

        def sql(_statement):
            result = next(readings)
            if result == "3|2":
                stop.set()
            return result

        with patch.object(check, "run", side_effect=[
            "replica-b 98MiB / 4GiB\nreplica-a 1.5GiB / 4GiB",
            "replica-a 1GiB / 4GiB\nreplica-b 102MiB / 4GiB",
        ]):
            result = check.sample(lambda *_args: _args[-1], sql, stop)

        self.assertEqual(result["peak_memory_bytes"],
                         {"replica-a": 1536 * 1024**2, "replica-b": 102 * 1024**2})
        self.assertEqual(result["postgres_peak_connections"], 4)
        self.assertEqual(result["postgres_peak_active_connections"], 2)


if __name__ == "__main__":
    unittest.main()
