"""Cross-language schema contract guard for the host store's Node-written tables.

`peer_offers` and `buyer_status` are CREATED by the Python host store
(`host_store.py`) but WRITTEN by the Node antseed sidecar
(`antseed/write-market.js`, `antseed/store.js`) and seeded in tests by Python
mimics (`tests/conftest.py`). Three independent places must agree on the column
set, and nothing at runtime makes them: the readers (`host_store.peer_offers` /
`buyer_status`) are fail-soft, so a column renamed/added/dropped on one side
degrades antseed to "no candidates" in production *silently* — and invisibly to
the unit suite, which seeds via the Python mimic, not the real Node writer.

These tests parse the column list out of all three sources and assert it matches,
per table, so any drift turns red here. Pure text parsing — no DB, no node
runtime — so it runs in the ordinary unit suite. The live behave e2e remains the
only thing that exercises the real Node writer end to end; this guards the part
that actually drifts: the column contract.
"""
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_HOST_STORE = (_ROOT / "host_store.py").read_text()
_WRITE_MARKET = (_ROOT / "antseed" / "write-market.js").read_text()
_STORE_JS = (_ROOT / "antseed" / "store.js").read_text()
_CONFTEST = (_ROOT / "tests" / "conftest.py").read_text()

_CONSTRAINT_KEYWORDS = {"PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT", "CHECK"}


def _create_columns(sql: str, table: str) -> set:
    """Column names declared in the host store's CREATE TABLE literal for
    `table`. The composite-PK clause is dropped first so its inner `(col, col)`
    is not mistaken for two columns."""
    m = re.search(
        r'CREATE TABLE IF NOT EXISTS\s+' + re.escape(table) + r'\s*\((.*?)\n\s*\)"""',
        sql, re.DOTALL)
    assert m, f"no CREATE TABLE for {table} found in host_store.py"
    body = re.sub(r"PRIMARY KEY\s*\([^)]*\)", "", m.group(1))
    cols = set()
    for part in body.split(","):
        tok = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", part)
        if tok and tok.group(1).upper() not in _CONSTRAINT_KEYWORDS:
            cols.add(tok.group(1))
    return cols


def _insert_columns(src: str, table: str) -> set:
    """Column names in the first `INSERT INTO <table> ( ... )` of `src` (a Node
    writer or a Python test mimic). The capture stops at the column list's close
    paren, before VALUES; quotes/newlines from literal concatenation are harmless
    because we extract identifiers rather than split on punctuation."""
    m = re.search(r'INSERT INTO\s+' + re.escape(table) + r'\s*\((.*?)\)', src, re.DOTALL)
    assert m, f"no INSERT INTO {table} found"
    return set(re.findall(r"[a-z_][a-z0-9_]*", m.group(1)))


def test_peer_offers_column_contract():
    schema = _create_columns(_HOST_STORE, "peer_offers")
    node = _insert_columns(_WRITE_MARKET, "peer_offers")
    mimic = _insert_columns(_CONFTEST, "peer_offers")
    assert schema, "parsed no columns — the CREATE-TABLE parser drifted"
    assert schema == node, (
        "peer_offers: Python schema vs Node writer (antseed/write-market.js) disagree."
        f"\n  schema-only: {sorted(schema - node)}\n  writer-only: {sorted(node - schema)}")
    assert schema == mimic, (
        "peer_offers: Python schema vs test mimic (conftest.seed_peer_offers) disagree."
        f"\n  schema-only: {sorted(schema - mimic)}\n  mimic-only: {sorted(mimic - schema)}")


def test_buyer_status_column_contract():
    schema = _create_columns(_HOST_STORE, "buyer_status")
    node = _insert_columns(_STORE_JS, "buyer_status")
    mimic = _insert_columns(_CONFTEST, "buyer_status")
    assert schema, "parsed no columns — the CREATE-TABLE parser drifted"
    assert schema == node, (
        "buyer_status: Python schema vs Node writer (antseed/store.js) disagree."
        f"\n  schema-only: {sorted(schema - node)}\n  writer-only: {sorted(node - schema)}")
    assert schema == mimic, (
        "buyer_status: Python schema vs test mimic (conftest.seed_buyer_status) disagree."
        f"\n  schema-only: {sorted(schema - mimic)}\n  mimic-only: {sorted(mimic - schema)}")
