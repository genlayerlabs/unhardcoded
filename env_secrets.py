"""
Load operator-managed secrets from the PVC env file into os.environ at startup.

The dashboard persists provider API keys (and dashboard-issued consumer key
hashes) to DASHBOARD_KEY_ENV_PATH (default /run/llm-router/.env.secrets) via
auth_proxy._upsert_env_line / _upsert_env_json. Under docker-compose this file
is wired in with `env_file:`, but on Kubernetes the container env comes only
from the Secret (ESO <- SSM) and nothing reloads the file — so a key edited
from the dashboard is silently lost on the next pod restart.

This loader closes that gap: it reads the file at process start and writes every
KEY=value into os.environ, OVERRIDING whatever the container env provided. The
PVC file is the source of truth; the Secret/SSM value is only a bootstrap (and
may legitimately be a CHANGE_ME placeholder the dashboard then overwrites).

Call it as early as possible — before anything snapshots os.environ
(LLMRouterHost.__init__) or reads keys at import time (auth_proxy module body).
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_SECRETS_PATH = "/run/llm-router/.env.secrets"


def load_env_secrets(path: str | os.PathLike | None = None) -> list[str]:
    """Merge KEY=value lines from the env-secrets file into os.environ.

    The file wins over the existing environment (dashboard/PVC is the source of
    truth). Returns the list of keys loaded. Never raises — a missing or
    unreadable file is a no-op so startup is unaffected.
    """
    target = Path(path or os.getenv("DASHBOARD_KEY_ENV_PATH", DEFAULT_ENV_SECRETS_PATH))
    loaded: list[str] = []
    try:
        if not target.exists():
            return loaded
        for raw in target.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            os.environ[key] = value
            loaded.append(key)
    except OSError:
        return loaded
    return loaded
