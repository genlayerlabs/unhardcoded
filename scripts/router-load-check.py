#!/usr/bin/env python3
"""Run an offline two-replica router benchmark against locally available images."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
import time
from uuid import uuid4

FIXTURES = Path(__file__).resolve().parent / "router-load"
sys.path.insert(0, str(FIXTURES))
from driver import valid_workload

REPLICAS = ("replica-a", "replica-b")
DB_FILTER = "datname = current_database() AND usename = 'router'"
QUERY_COUNT = """
SELECT COALESCE(sum(calls), 0) FROM pg_stat_statements
WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
  AND userid = 'router'::regrole
"""
MEMORY_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}


def run(*command, env=None, timeout=30):
    return subprocess.run(
        command, env=env, check=True, stdout=subprocess.PIPE, text=True, timeout=timeout,
    ).stdout.strip()


def sample(compose, sql, stop):
    containers = {compose("ps", "-q", replica): replica for replica in REPLICAS}
    peak_memory = dict.fromkeys(REPLICAS, 0)
    peak_connections = peak_active = samples = 0
    while True:
        stats = run("docker", "stats", "--no-stream", "--no-trunc", "--format",
                    "{{.ID}} {{.MemUsage}}", *containers)
        for line in stats.splitlines():
            container, usage = line.split(maxsplit=1)
            value = usage.split("/")[0].strip()
            unit = next(unit for unit in MEMORY_UNITS if value.endswith(unit)
                        and value.removesuffix(unit).replace(".", "", 1).isdigit())
            used = round(float(value.removesuffix(unit)) * MEMORY_UNITS[unit])
            replica = containers[container]
            peak_memory[replica] = max(peak_memory[replica], used)
        connections, active = map(int, sql(
            "SELECT count(*), count(*) FILTER (WHERE state = 'active') "
            f"FROM pg_stat_activity WHERE {DB_FILTER}"
        ).split("|"))
        peak_connections = max(peak_connections, connections)
        peak_active = max(peak_active, active)
        samples += 1
        if stop.wait(0.25):
            break
    if not all(peak_memory.values()) or not peak_connections:
        raise RuntimeError("could not sample both replicas and their database connections")
    return {
        "peak_memory_bytes": peak_memory,
        "postgres_peak_connections": peak_connections,
        "postgres_peak_active_connections": peak_active,
        "resource_samples": samples,
    }


def benchmark(image, harness, args):
    env = {
        **os.environ, "LOAD_ROUTER_IMAGE": image, "LOAD_HARNESS_IMAGE": harness,
        "LOAD_LATENCY_MS": str(args.latency_ms),
    }
    prefix = (
        "docker", "compose", "--env-file", os.devnull, "-f", str(FIXTURES / "compose.yml"),
        "-p", f"router-load-{uuid4().hex[:12]}",
    )

    def compose(*command, timeout=30):
        return run(*prefix, *command, env=env, timeout=timeout)

    def sql(statement):
        return compose("exec", "-T", "postgres", "psql", "-XAt", "-v", "ON_ERROR_STOP=1",
                       "-U", "postgres", "-d", "router_load", "-c", statement)

    def drive(seconds):
        return json.loads(compose(
            "run", "--rm", "-T", "--no-deps", "driver",
            str(seconds), str(args.concurrency), str(args.latency_ms),
            timeout=seconds + 30,
        ))

    try:
        print(f"Starting {image}", file=sys.stderr)
        compose("up", "-d", "--wait", "--wait-timeout", "75", timeout=90)
        compose("exec", "-T", "replica-a", "python", "/load/driver.py", "seed")
        warmup = drive(5)
        if not valid_workload(warmup):
            raise RuntimeError(f"unexpected warmup responses: {warmup['statuses_by_key']}")
        time.sleep(1)
        before = int(sql(QUERY_COUNT))
        stop = Event()
        with ThreadPoolExecutor(max_workers=1) as executor:
            monitor = executor.submit(sample, compose, sql, stop)
            try:
                result = drive(args.duration)
            finally:
                stop.set()
            resources = monitor.result()
        # Allow queued usage writes from the last requests to reach the store.
        time.sleep(1)
        queries = int(sql(QUERY_COUNT)) - before
        if queries <= 0:
            raise RuntimeError("PostgreSQL recorded no benchmark queries")
        return {
            "image": image, **result, **resources, "postgres_queries": queries,
            "queries_per_request": queries / result["requests"],
        }
    except (subprocess.SubprocessError, RuntimeError):
        print(compose("logs", "--tail", "20"), file=sys.stderr)
        raise
    finally:
        compose("down", "--volumes", "--remove-orphans", timeout=60)


def compare(baseline, candidate, margin_pct):
    metrics = [
        ("p99 overhead", baseline["overhead_ms"]["p99"], candidate["overhead_ms"]["p99"]),
        ("queries per request", baseline["queries_per_request"], candidate["queries_per_request"]),
    ]
    for replica in REPLICAS:
        metrics.append((f"{replica} peak memory", baseline["peak_memory_bytes"][replica],
                        candidate["peak_memory_bytes"][replica]))
    return [
        {"metric": name, "baseline": before, "candidate": after,
         "limit": before * (1 + margin_pct / 100)}
        for name, before, after in metrics if after > before * (1 + margin_pct / 100)
    ]


def print_table(results):
    rows = [("Metric", *(result["image"] for result in results))]
    rows.append(("Requests/s", *(f"{r['requests_per_second']:.1f}" for r in results)))
    rows.append(("Errors by status", *(json.dumps(r["errors_by_status"], sort_keys=True) for r in results)))
    for title, key in (("Overhead", "overhead_ms"), ("First streamed token", "first_streamed_token_ms")):
        rows.append((f"{title} p50/p95/p99 (ms)", *(
            "/".join(f"{value:.1f}" for value in r[key].values()) if r[key] else "n/a" for r in results
        )))
    for replica in REPLICAS:
        rows.append((f"{replica} peak memory (MiB)", *(
            f"{r['peak_memory_bytes'][replica] / 1024**2:.1f}" for r in results
        )))
    rows.extend([
        ("Peak PostgreSQL connections", *(str(r["postgres_peak_connections"]) for r in results)),
        ("Peak active PostgreSQL connections", *(str(r["postgres_peak_active_connections"]) for r in results)),
        ("Queries/request", *(f"{r['queries_per_request']:.2f}" for r in results)),
    ])
    widths = [max(map(len, column)) for column in zip(*rows)]
    for row in rows:
        print("  ".join(item.ljust(width) for item, width in zip(row, widths)).rstrip(), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="local router image tag")
    parser.add_argument("--compare", metavar="IMAGE", help="local candidate image tag")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=30, help="measurement seconds per image")
    parser.add_argument("--latency-ms", type=float, default=40)
    parser.add_argument("--margin-pct", type=float, default=30)
    args = parser.parse_args()
    if (args.concurrency < 1 or args.duration < 1 or args.latency_ms < 0 or args.margin_pct < 0
            or not all(map(math.isfinite, (args.duration, args.latency_ms, args.margin_pct)))):
        parser.error("concurrency and duration must be at least 1, with finite nonnegative latency and margin")

    images = [args.baseline]
    if args.compare:
        images.append(args.compare)
    try:
        for image in dict.fromkeys([*images, "postgres:16", "nginx:alpine"]):
            run("docker", "image", "inspect", image)
        results = [benchmark(image, args.baseline, args) for image in images]
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Load check failed: {exc}\n")

    invalid = [result["image"] for result in results if not valid_workload(result)]
    regressions = compare(*results, args.margin_pct) if args.compare and not invalid else []
    print_table(results)
    print(json.dumps({
        "settings": {"duration_seconds": args.duration, "concurrency": args.concurrency,
                     "upstream_latency_ms": args.latency_ms, "margin_pct": args.margin_pct},
        "runs": results, "invalid_workloads": invalid, "regressions": regressions,
    }, indent=2))
    if invalid:
        print(f"Unexpected benchmark responses for {', '.join(invalid)}", file=sys.stderr)
    for regression in regressions:
        print(f"Regression in {regression['metric']}: {regression['baseline']:.3g} -> "
              f"{regression['candidate']:.3g} (limit {regression['limit']:.3g})", file=sys.stderr)
    return int(bool(invalid or regressions))


if __name__ == "__main__":
    sys.exit(main())
