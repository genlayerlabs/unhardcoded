#!/usr/bin/env bash
# Run the router and SDK's local demo topology against a disposable test database.
set -euo pipefail

if [[ -z ${ESLA_BIN:-} || ! -x ${ESLA_BIN:-} ]]; then
  echo "Set ESLA_BIN to the absolute path of a built ESLA SDK executable." >&2
  exit 2
fi
if [[ -z ${DATABASE_URL:-} ]]; then
  echo "Set DATABASE_URL to a dedicated PostgreSQL test database. The demo truncates its test tables." >&2
  exit 2
fi

cd "$(dirname "$0")/.."
exec python -m pytest -q -s -rs tests/test_esla.py::test_esla_window_matches_router_attempts
