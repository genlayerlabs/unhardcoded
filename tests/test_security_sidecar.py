"""Image/compose hardening that no runtime test would notice regressing."""
import fnmatch
import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _dockerignored(name: str) -> bool:
    ignored = False
    for line in (_ROOT / ".dockerignore").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        neg = line.startswith("!")
        if fnmatch.fnmatch(name, line.lstrip("!")):
            ignored = not neg
    return ignored


def test_build_context_never_ships_codex_oauth_tokens():
    # compose's default CODEX_AUTH_PATH is ./codex-auth.json in the repo root,
    # and the router image does `COPY . .`.
    for name in ("codex-auth.json", "codex-accounts", ".env.secrets", ".env", "secrets"):
        assert _dockerignored(name), name
    assert not _dockerignored(".env.example")


def test_router_image_defaults_to_the_authenticated_ingress():
    cmd = next(ln for ln in (_ROOT / "Dockerfile").read_text().splitlines() if ln.startswith("CMD "))
    argv = json.loads(cmd[4:])
    assert argv[:2] == ["uvicorn", "auth_proxy:app"], argv
    assert "serve.py" not in argv


def test_compose_has_no_default_database_password():
    src = (_ROOT / "compose.yml").read_text()
    assert "hoststore:hoststore@" not in src
    assert re.search(r"POSTGRES_PASSWORD: \$\{POSTGRES_PASSWORD:\?", src)
    for dsn in re.findall(r"DATABASE_URL: (.+)", src):
        assert "${POSTGRES_PASSWORD:?" in dsn, dsn
