from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env_secrets import load_env_secrets  # noqa: E402


def test_load_env_secrets_overrides_existing_environment(monkeypatch, tmp_path):
    # The PVC file is the source of truth: a CHANGE_ME placeholder in the
    # container env must be overridden by the dashboard-persisted value.
    monkeypatch.setenv("HEURIST_API_KEY", "CHANGE_ME")
    monkeypatch.delenv("IONET_API_KEY", raising=False)
    env_file = tmp_path / ".env.secrets"
    env_file.write_text(
        "# operator keys\n"
        "HEURIST_API_KEY=sk-heurist-real\n"
        "\n"
        "IONET_API_KEY=sk-ionet-real\n"
        'CALLER_KEYS_SHA256_JSON={"abc":"crm"}\n'
        "MALFORMED_LINE_NO_EQUALS\n"
    )

    loaded = load_env_secrets(env_file)

    assert os.environ["HEURIST_API_KEY"] == "sk-heurist-real"   # overrode CHANGE_ME
    assert os.environ["IONET_API_KEY"] == "sk-ionet-real"       # newly set
    assert os.environ["CALLER_KEYS_SHA256_JSON"] == '{"abc":"crm"}'
    assert set(loaded) == {"HEURIST_API_KEY", "IONET_API_KEY", "CALLER_KEYS_SHA256_JSON"}
    assert "MALFORMED_LINE_NO_EQUALS" not in os.environ


def test_load_env_secrets_missing_file_is_noop(tmp_path):
    assert load_env_secrets(tmp_path / "does-not-exist") == []
