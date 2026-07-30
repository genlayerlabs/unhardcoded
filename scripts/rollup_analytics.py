#!/usr/bin/env python3
"""Build bounded hourly dashboard analytics outside the request-serving pod."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import host_store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=2,
                        help="rolling hours to recompute (default: 2)")
    parser.add_argument("--backfill-days", type=int, default=0,
                        help="one-off historical backfill window")
    parser.add_argument("--retain-days", type=int, default=30,
                        help="history target for incremental backfill")
    parser.add_argument("--backfill-hours-per-run", type=int, default=24,
                        help="older history added per scheduled run")
    args = parser.parse_args()
    if args.hours < 1 or args.hours > 48:
        parser.error("--hours must be between 1 and 48")
    if args.backfill_days < 0 or args.backfill_days > 90:
        parser.error("--backfill-days must be between 0 and 90")
    if not 1 <= args.retain_days <= 90:
        parser.error("--retain-days must be between 1 and 90")
    if not 0 <= args.backfill_hours_per_run <= 48:
        parser.error("--backfill-hours-per-run must be between 0 and 48")

    end = int(time.time())
    if args.backfill_days:
        results = [host_store.rollup_analytics(
            end - args.backfill_days * 86400, end)]
    else:
        results = [host_store.rollup_analytics(end - args.hours * 3600, end)]
        state = host_store.analytics_rollup_state()
        covered_from = int(state.get("covered_from") or end)
        target = end - args.retain_days * 86400
        if args.backfill_hours_per_run and covered_from > target:
            older_start = max(target, covered_from - args.backfill_hours_per_run * 3600)
            results.append(host_store.rollup_analytics(older_start, covered_from))
    print(json.dumps({"ok": True, "rollups": results,
                      "state": host_store.analytics_rollup_state()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
