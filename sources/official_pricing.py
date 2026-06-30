"""Direct-provider list prices from each provider's OFFICIAL pricing page.

OpenAI / Anthropic / Google publish prices on documentation pages, not via an API:
  - Anthropic & OpenAI serve a stable Markdown twin (`<page>.md`) — parsed as
    Markdown tables.
  - Google's devsite page is HTML whose model sections are anchored by stable
    heading ids (`<h2 id="gemini-3.1-pro-preview">`, the catalog family verbatim)
    with `<td>Input price</td><td>$…</td>` rows — parsed from the HTML.

The source is fail-safe and strictly OFF the request path. It scrapes on a slow
cadence, upserts `host_store.provider_prices` (the DURABLE source of truth), and
returns in/out prices for the core. If a scrape fails or a page is restyled past
the parser, pricing() COASTS on the last-known rows from the table — a broken page
degrades coverage, it never zeroes a price back to +inf. Cache-read prices are
stored for effective-cost work (C/D); the core ranks on in/out only.
"""
from __future__ import annotations

import logging
import os
import re
from html.parser import HTMLParser
from typing import Any

import host_store
from sources import Balance, Price

_log = logging.getLogger("sources.official_pricing")

# provider_id -> (pricing page url, parser format). Anthropic/OpenAI expose a
# Markdown twin; Google's devsite page is HTML.
DIRECT_PAGES: dict[str, tuple[str, str]] = {
    "openai": ("https://developers.openai.com/api/docs/pricing.md", "openai_jsx"),
    "anthropic": ("https://platform.claude.com/docs/en/about-claude/pricing.md", "markdown"),
    "google": ("https://ai.google.dev/gemini-api/docs/pricing", "gemini_html"),
}


def _slug(s: str) -> str:
    """Canonical match key tolerant of dot/dash/space/case differences, so a
    page's model label lines up with a catalog family: "Claude Opus 4.8",
    "claude-opus-4-8" and "gemini-3.1-pro-preview" reduce to a bare token."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _money(cell: str) -> "float | None":
    """First USD amount in a cell: "$5.00" / "$1.25 / MTok" / "$1.25, per 1M" ->
    5.0 / 1.25. None when there is no number ("Free of charge", "—")."""
    m = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)", cell)
    return float(m.group(1)) if m else None


# ---- Markdown parser (OpenAI & Anthropic .md twins) ----------------------------

def _md_cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c)


def _strip_md(cell: str) -> str:
    """A model-name cell minus markdown links and trailing parentheticals:
    "Claude Opus 4.1 ([deprecated](/x))" -> "Claude Opus 4.1"."""
    cell = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cell)   # [text](url) -> text
    cell = re.sub(r"\([^)]*\)", "", cell)                   # drop (notes)
    return cell.replace("*", "").strip()


def _price_columns(header: list[str]) -> dict[str, "int | None"]:
    """Locate the input / output / cache-read columns in a table header, by
    keyword. "Cached Input" / "Cache Hits & Refreshes" -> cached; a plain
    "Input" / "Base Input Tokens" -> input; "Output Tokens" -> output."""
    cols: dict[str, "int | None"] = {"in": None, "out": None, "cached": None}
    for idx, h in enumerate(header):
        hl = h.lower()
        if "output" in hl and cols["out"] is None:
            cols["out"] = idx
        elif "cache" in hl and any(k in hl for k in ("hit", "read", "cached", "refresh")) \
                and cols["cached"] is None:
            cols["cached"] = idx
        elif "input" in hl and "cache" not in hl and cols["in"] is None:
            cols["in"] = idx
    return cols


def parse_markdown_pricing(text: str) -> list[dict]:
    """Every Markdown pricing table in `text` -> records
    {family_hint, price_in, price_out, price_cached_in}. A row contributes only
    when its table has both an input and an output column and the row carries
    both numbers. Earlier tables (standard pricing) are emitted before later ones
    (batch/flex), so a first-wins resolve keeps standard rates."""
    out: list[dict] = []
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        block = []
        while i < n and lines[i].lstrip().startswith("|"):
            block.append(lines[i])
            i += 1
        if len(block) < 2:
            continue
        header = _md_cells(block[0])
        if not _is_separator(_md_cells(block[1])):
            continue
        cols = _price_columns(header)
        if cols["in"] is None or cols["out"] is None:
            continue
        need = max(c for c in cols.values() if c is not None)
        for raw in block[2:]:
            cells = _md_cells(raw)
            if len(cells) <= need:
                continue
            name = _strip_md(cells[0])
            pin, pout = _money(cells[cols["in"]]), _money(cells[cols["out"]])
            if not name or pin is None or pout is None:
                continue
            cached = _money(cells[cols["cached"]]) if cols["cached"] is not None else None
            out.append({"family_hint": name, "price_in": pin,
                        "price_out": pout, "price_cached_in": cached})
    return out


# ---- OpenAI JSX parser (developers.openai.com .md embeds a component) ----------

def parse_openai_jsx(text: str) -> list[dict]:
    """OpenAI's `.md` is MDX: the pricing lives in a `<TextTokenPricingTables>`
    component as JS array rows `["gpt-5.5 (<272K context length)", 5, 0.5, 30]` =
    [name, input, cached, output]. Standard-tier rows appear before batch/flex/
    priority, so a first-wins resolve keeps standard rates."""
    out: list[dict] = []
    for m in re.finditer(r'\[\s*"([^"]+)"\s*,([^\]\[]*)\]', text):
        name = _strip_md(m.group(1))
        parts = [p.strip().strip('"') for p in m.group(2).split(",")]

        def _num(s: str) -> "float | None":
            try:
                return float(s)
            except (TypeError, ValueError):
                return None

        if len(parts) == 2:                       # [name, input, output]
            pin, cached, pout = _num(parts[0]), None, _num(parts[1])
        elif len(parts) >= 3:                     # [name, input, cached, output]
            pin, cached, pout = _num(parts[0]), _num(parts[1]), _num(parts[-1])
        else:
            continue
        if not name or pin is None or pout is None:
            continue
        out.append({"family_hint": name, "price_in": pin,
                    "price_out": pout, "price_cached_in": cached})
    return out


# ---- Gemini HTML parser (Google devsite) ---------------------------------------

class _GeminiPriceParser(HTMLParser):
    """Pulls the STANDARD-tier Input / Output / Context-caching prices out of the
    Gemini pricing page. Model sections are `<h2 id="gemini-…">` (the id IS the
    family); processing tiers are `<h3 id="standard|batch|…">`; rows are
    `<td>Input price</td><td>…</td>`. Only the standard tier is taken."""

    def __init__(self) -> None:
        super().__init__()
        self.records: dict[str, dict] = {}
        self._model: "str | None" = None
        self._tier: "str | None" = None
        self._cap = False            # accumulating a heading/cell's text
        self._buf = ""
        self._row: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, "str | None"]]) -> None:
        a = dict(attrs)
        if tag in ("h2", "h3"):
            self._cap, self._buf, self._pending_id = True, "", (a.get("id") or "")
            self._pending_tag = tag
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._in_cell, self._buf = True, ""
        elif tag == "br" and self._in_cell:
            # Gemini packs the cache-READ and the per-hour STORAGE price into one
            # cell split by <br> ("$0.15<br>$1.00 … per hour"); keep them apart.
            self._buf += "\n"

    def handle_data(self, data: str) -> None:
        if self._cap or self._in_cell:
            self._buf += data

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h2", "h3") and self._cap:
            self._cap = False
            hid = (self._pending_id or "").lower()
            if self._pending_tag == "h2":
                # a model section starts; only gemini-* headings carry prices
                self._model = hid if hid.startswith("gemini-") else None
                self._tier = None
            else:  # h3: processing tier (strip the devsite "_1" disambiguators)
                self._tier = re.sub(r"_\d+$", "", hid)
        elif tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            self._row.append(self._buf.strip())
        elif tag == "tr":
            self._row_done()

    def _row_done(self) -> None:
        if not self._model or self._tier != "standard" or not self._row:
            return
        label = self._row[0].lower()
        values = self._row[1:]
        rec = self.records.setdefault(self._model, {
            "family_hint": self._model, "price_in": None,
            "price_out": None, "price_cached_in": None})
        if "input price" in label:
            rec["price_in"] = _first_money(values)
        elif "output price" in label:
            rec["price_out"] = _first_money(values)
        elif "context caching price" in label:
            # the cache-READ price is the FIRST $ in the cell; the per-hour storage
            # price follows it (after the <br>, now a newline), so first-money wins.
            rec["price_cached_in"] = _first_money(values)


def _first_money(cells: list[str]) -> "float | None":
    for c in cells:
        v = _money(c)
        if v is not None:
            return v
    return None


def parse_gemini_html(html: str) -> list[dict]:
    p = _GeminiPriceParser()
    p.feed(html)
    return [r for r in p.records.values()
            if r["price_in"] is not None and r["price_out"] is not None]


_PARSERS = {
    "markdown": parse_markdown_pricing,
    "openai_jsx": parse_openai_jsx,
    "gemini_html": parse_gemini_html,
}


# ---- the source ----------------------------------------------------------------

class OfficialPriceSource:
    poll_interval_s = 3600          # provider pages move ~hourly at most

    def __init__(self, catalog: dict, provider_id: str, env_get=os.environ.get,
                 client: Any = None, url: "str | None" = None, fmt: "str | None" = None):
        self.provider_id = provider_id
        self.name = f"{provider_id}_pricing"
        self.provider_ids = [provider_id]
        default_url, default_fmt = DIRECT_PAGES.get(provider_id, (None, "markdown"))
        self._url = url or default_url
        self._fmt = fmt or default_fmt
        self._env_get = env_get
        self._client = client          # injected in tests; lazy httpx otherwise
        # slug -> catalog family, scoped to the families THIS provider serves (and
        # their wire ids). push_prices also filters, but resolving here keeps junk
        # page rows out of the durable table.
        self._family_by_slug: dict[str, str] = {}
        for family, model in (catalog.get("models") or {}).items():
            for served in model.get("served_by") or []:
                if served.get("provider") != provider_id:
                    continue
                self._family_by_slug[_slug(family)] = family
                pmid = served.get("provider_model_id")
                if pmid:
                    self._family_by_slug[_slug(pmid)] = family

    async def _fetch_text(self, url: str) -> str:
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=20.0, follow_redirects=True)
        resp = await self._client.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"{self.name} GET {url} -> {resp.status_code}")
        return resp.text

    def _resolve(self, records: list[dict]) -> dict[str, dict]:
        """Page records -> {catalog_family: record}, first-wins, catalog-served
        families only (standard rates beat later batch/flex rows)."""
        rows: dict[str, dict] = {}
        for r in records:
            fam = self._family_by_slug.get(_slug(r["family_hint"]))
            if fam and fam not in rows:
                rows[fam] = r
        return rows

    def _prices(self, items) -> list[Price]:
        prices: list[Price] = []
        for fam, r in items:
            if r.get("price_in") is None or r.get("price_out") is None:
                continue
            prices.append({
                "provider_id": self.provider_id,
                "served_model_id": fam,
                "model_family": fam,
                "price_in_usd_per_mtok": float(r["price_in"]),
                "price_out_usd_per_mtok": float(r["price_out"]),
            })
        return prices

    async def pricing(self) -> list[Price]:
        try:
            text = await self._fetch_text(self._url)
            records = _PARSERS[self._fmt](text)
            rows = self._resolve(records)
            if rows:
                host_store.set_provider_prices([
                    {"provider_id": self.provider_id, "model_family": fam,
                     "price_in": r["price_in"], "price_out": r["price_out"],
                     "price_cached_in": r.get("price_cached_in")}
                    for fam, r in rows.items()])
                return self._prices(rows.items())
            _log.warning("%s parsed no catalog-served prices; coasting", self.name)
        except Exception as exc:  # noqa: BLE001 — scrape is best-effort, never raises
            _log.warning("%s scrape failed, coasting on host_store: %s", self.name, exc)
        # coast on the durable table (also warm-starts a fresh process)
        stored = host_store.get_provider_prices(self.provider_id)
        return self._prices((row["model_family"], row) for row in stored)

    async def balances(self) -> dict[str, Balance]:
        return {}
