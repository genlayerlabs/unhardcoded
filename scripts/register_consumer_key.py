#!/usr/bin/env python3
"""Register per-consumer caller tokens for llm-router ingress.

Stores SHA-256(token) -> caller mappings in CALLER_KEYS_SHA256_JSON inside
.env.secrets by default. The raw token is printed once and is never stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
from pathlib import Path


LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _read_env(path: Path) -> tuple[list[str], dict[str, str]]:
    lines = path.read_text().splitlines() if path.exists() else []
    vals: dict[str, str] = {}
    for line in lines:
        m = LINE_RE.match(line)
        if m and not line.lstrip().startswith("#"):
            vals[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return lines, vals


def _set_env(path: Path, key: str, value: str) -> None:
    lines, _ = _read_env(path)
    out: list[str] = []
    replaced = False
    for line in lines:
        if LINE_RE.match(line or "") and line.split("=", 1)[0] == key:
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        if out and out[-1] != "":
            out.append("")
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def _load_json_obj(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise SystemExit("caller map must be a JSON object of string -> string")
    return data


def _token() -> str:
    return "llmr_" + secrets.token_urlsafe(32)


def main() -> None:
    p = argparse.ArgumentParser(description="Register a per-consumer llm-router API token")
    p.add_argument("consumer", help="stable consumer name, e.g. crm, subastas, wingston")
    p.add_argument("--env", default=".env.secrets", help="env file to update")
    p.add_argument("--token", default=None, help="optional pre-generated token; otherwise one is generated")
    p.add_argument("--plaintext", action="store_true", help="store token in legacy CALLER_KEYS_JSON instead of SHA-256 map")
    args = p.parse_args()

    env_path = Path(args.env)
    lines, vals = _read_env(env_path)
    token = args.token or _token()
    if len(token) < 16:
        raise SystemExit("token is too short")

    if args.plaintext:
        mapping = _load_json_obj(vals.get("CALLER_KEYS_JSON"))
        mapping[token] = args.consumer
        _set_env(env_path, "CALLER_KEYS_JSON", json.dumps(mapping, sort_keys=True, separators=(",", ":")))
        storage = "CALLER_KEYS_JSON"
    else:
        mapping = _load_json_obj(vals.get("CALLER_KEYS_SHA256_JSON"))
        digest = hashlib.sha256(token.encode()).hexdigest()
        mapping[digest] = args.consumer
        _set_env(env_path, "CALLER_KEYS_SHA256_JSON", json.dumps(mapping, sort_keys=True, separators=(",", ":")))
        storage = "CALLER_KEYS_SHA256_JSON"

    print(json.dumps({"consumer": args.consumer, "storage": storage, "api_key": token}, indent=2))
    print("Save the api_key now; only its hash is stored when using the default mode.")


if __name__ == "__main__":
    main()
