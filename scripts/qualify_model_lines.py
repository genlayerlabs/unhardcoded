#!/usr/bin/env python3
"""Qualify successors for model lines (see model_line_qualifier.py).

Runs outside the request-serving pods, like scripts/rollup_analytics.py. Probes
go through the router ingress with a consumer key so their spend is ledgered;
heads and audit events are read and written in the host store (DATABASE_URL).

    LINE_QUALIFIER_ROUTER_URL=http://llm-router \\
    LINE_QUALIFIER_API_KEY=... \\
    python scripts/qualify_model_lines.py --suites /etc/line-suites --mode propose

Modes: `report` probes and prints only; `propose` (default) also records audit
events and standing proposals; `auto` moves heads. Exit status is 0 when the run
completed (whatever it decided), 1 when it could not run at all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

import model_line_qualifier as qualifier  # noqa: E402


def make_transport(base_url: str, api_key: str, timeout_s: float):
    """`post(body)` / `get(path)` against the router's OpenAI-compatible surface,
    each returning (http_status, parsed_json_or_None)."""
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers,
                          timeout=httpx.Timeout(timeout_s, connect=10.0))

    def parsed(response: httpx.Response):
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, None

    def post(body: dict):
        return parsed(client.post("/v1/chat/completions", json=body))

    def get(path: str):
        return parsed(client.get(path))

    return post, get, client


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--router-url",
                        default=os.getenv("LINE_QUALIFIER_ROUTER_URL", ""),
                        help="router base URL (env LINE_QUALIFIER_ROUTER_URL)")
    parser.add_argument("--suites", action="append", default=[],
                        help="suite file or directory of *.json (repeatable; "
                             "env LINE_QUALIFIER_SUITES, comma-separated)")
    parser.add_argument("--mode", choices=qualifier.MODES,
                        default=os.getenv("LINE_QUALIFIER_MODE", "propose"))
    parser.add_argument("--lines", default="",
                        help="comma-separated subset of lines (default: all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="families probed in parallel (prompts within a "
                             "family are always sequential)")
    parser.add_argument("--timeout-s", type=float, default=60.0,
                        help="HTTP timeout per probe; keep it above the suite's "
                             "own timeout_ms so the router's verdict wins")
    args = parser.parse_args()

    if args.mode not in qualifier.MODES:
        parser.error(f"LINE_QUALIFIER_MODE must be one of {qualifier.MODES}")
    if not args.router_url:
        parser.error("--router-url / LINE_QUALIFIER_ROUTER_URL is required")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    suite_paths = args.suites or [
        p.strip() for p in os.getenv("LINE_QUALIFIER_SUITES", "").split(",") if p.strip()]
    if not suite_paths:
        parser.error("--suites / LINE_QUALIFIER_SUITES is required")

    try:
        suites = qualifier.load_suites(suite_paths)
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": f"suites: {exc}"}))
        return 1

    # The key is read from the environment only: never accepted on the command
    # line, where it would land in the process list and the job spec.
    post, get, client = make_transport(
        args.router_url, os.getenv("LINE_QUALIFIER_API_KEY", ""), args.timeout_s)
    only = [n.strip() for n in args.lines.split(",") if n.strip()] or None
    try:
        report = qualifier.run(post=post, get=get, suites=suites, mode=args.mode,
                               only_lines=only, workers=args.workers)
    except Exception as exc:  # noqa: BLE001 — one JSON line either way
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    finally:
        client.close()
    print(json.dumps({"ok": True, **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
