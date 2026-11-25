"""Seed disposable keys or drive a closed-loop workload inside the load network."""

import argparse
import asyncio
from collections import Counter
from hashlib import sha256
from itertools import cycle
import json
import math
import sys
import time

# Keys are fixtures for this disposable network.
KEYS = {name: f"router-load-test-{name}" for name in ("valid", "limited", "invalid")}
MIX = [
    ("valid", False), ("valid", True), ("valid", True), ("limited", False),
    ("valid", False), ("invalid", True), ("valid", True), ("valid", False),
    ("valid", False), ("valid", True),
]


def seed():
    sys.path.insert(0, "/app")
    from host_store import set_consumer_keys

    records = {}
    digests = {}
    for name, rate, burst in [("valid", 1_000_000, 1_000_000), ("limited", 1, 1)]:
        digest = sha256(KEYS[name].encode()).hexdigest()
        records[name] = {
            "status": "active", "rate_per_min": rate, "burst": burst,
            "keys": [{"sha256_prefix": digest[:12], "status": "active"}],
        }
        digests[digest] = name
    if not set_consumer_keys(records, key_digests=digests):
        raise SystemExit("could not seed benchmark keys")


def percentiles(values):
    ordered = sorted(values)
    return {
        f"p{percentile}": ordered[math.ceil(len(ordered) * percentile / 100) - 1]
        for percentile in (50, 95, 99)
    } if ordered else None


async def load(base_url, duration, concurrency, latency_ms):
    import httpx

    mix = cycle(MIX)
    statuses = {name: Counter() for name in KEYS}
    overhead = []
    first_tokens = []
    async with httpx.AsyncClient(
        base_url=base_url, trust_env=False, timeout=10,
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
    ) as client:
        start = time.perf_counter()

        async def worker():
            while time.perf_counter() < start + duration:
                name, stream = next(mix)
                headers = {"Authorization": f"Bearer {KEYS[name]}"}
                body = {
                    "model": "profile:default", "stream": stream, "max_tokens": 8,
                    "messages": [{"role": "user", "content": "Reply OK."}],
                }
                sent = time.perf_counter()
                first_token = None
                finished = False
                try:
                    async with client.stream("POST", "/v1/chat/completions",
                                             headers=headers, json=body) as response:
                        status = str(response.status_code)
                        if status == "200" and stream:
                            async for line in response.aiter_lines():
                                if line == "data: [DONE]":
                                    finished = True
                                elif first_token is None and line.startswith("data:"):
                                    arrived = time.perf_counter()
                                    choices = json.loads(line[5:]).get("choices", [])
                                    if choices and choices[0].get("delta", {}).get("content"):
                                        first_token = (arrived - sent) * 1000
                            if first_token is None or not finished:
                                status = "invalid_stream"
                        else:
                            await response.aread()
                            if status == "200" and not response.json()["choices"][0]["message"]["content"]:
                                status = "invalid_response"
                except httpx.RequestError:
                    status = "transport_error"
                except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                    status = "invalid_response"
                statuses[name][status] += 1
                if name == "valid" and status == "200":
                    overhead.append(max(0, (time.perf_counter() - sent) * 1000 - latency_ms))
                    if stream:
                        first_tokens.append(first_token)

        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - start
    total = sum(statuses.values(), Counter())
    requests = total.total()
    return {
        "requests": requests,
        "requests_per_second": requests / elapsed,
        "elapsed_seconds": elapsed,
        "errors_by_status": {status: count for status, count in total.items() if status != "200"},
        "statuses_by_key": statuses,
        "overhead_ms": percentiles(overhead),
        "first_streamed_token_ms": percentiles(first_tokens),
    }


def valid_workload(result):
    statuses = result["statuses_by_key"]
    return (
        bool(result["overhead_ms"] and result["first_streamed_token_ms"])
        and set(statuses["valid"]) == {"200"}
        and set(statuses["invalid"]) == {"401"}
        and "429" in statuses["limited"]
        and set(statuses["limited"]) <= {"200", "429"}
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["seed"]:
        seed()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--base-url", default="http://proxy")
        parser.add_argument("duration", type=float)
        parser.add_argument("concurrency", type=int)
        parser.add_argument("latency_ms", type=float)
        args = parser.parse_args()
        print(json.dumps(asyncio.run(load(args.base_url, args.duration, args.concurrency, args.latency_ms))))
