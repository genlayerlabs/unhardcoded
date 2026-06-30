"""Env-var coercion helpers — a leaf (depends on nothing but `os`) so both
`settings.py` and `providers.py` read knob defaults the same way without importing
each other (which would reintroduce the settings ↔ providers cycle)."""
from __future__ import annotations

import os


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
