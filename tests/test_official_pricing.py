"""Direct-provider official-pricing source: the Markdown/HTML parsers, the
slug→family resolve, and the fail-safe coast on host_store. No network — pages
are fed as fixtures mirroring the real OpenAI/Anthropic/Google layouts."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import host_store  # noqa: E402
from sources import official_pricing as op  # noqa: E402

# --- fixtures: the real table shapes, trimmed -----------------------------------

ANTHROPIC_MD = """# Pricing

## Model pricing

| Model | Base Input Tokens | 5m Cache Writes | 1h Cache Writes | Cache Hits & Refreshes | Output Tokens |
| ----- | ----------------- | --------------- | --------------- | ---------------------- | ------------- |
| Claude Opus 4.8 | $5 / MTok | $6.25 / MTok | $10 / MTok | $0.50 / MTok | $25 / MTok |
| Claude Sonnet 4.6 | $3 / MTok | $3.75 / MTok | $6 / MTok | $0.30 / MTok | $15 / MTok |
| Claude Haiku 4.5 ([retired](/x)) | $1 / MTok | $1.25 / MTok | $2 / MTok | $0.10 / MTok | $5 / MTok |

### Batch processing

| Model | Batch input | Batch output |
| ----- | ----------- | ------------ |
| Claude Opus 4.8 | $2.50 / MTok | $12.50 / MTok |
"""

# OpenAI's .md is MDX: a JSX component with [name, input, cached, output] rows,
# standard pane before the hidden batch/flex/priority panes.
OPENAI_MD = """# Pricing

<div data-content-switcher-pane data-value="standard">
  <TextTokenPricingTables rows={[
    ["gpt-5.5 (<272K context length)", 5, 0.5, 30],
    ["gpt-5.4 (<272K context length)", 2.5, 0.25, 15],
    ["gpt-5.5-pro (<272K context length)", 30, "", 180],
  ]} />
</div>
<div data-content-switcher-pane data-value="batch" hidden>
  <TextTokenPricingTables rows={[
    ["gpt-5.5 (<272K context length)", 2.5, 0.25, 15],
  ]} />
</div>
"""

GEMINI_HTML = """
<h2 id="gemini-3.1-pro-preview" data-text="Gemini 3.1 Pro">Gemini 3.1 Pro</h2>
<h3 id="standard">Standard</h3>
<table>
<tr><td>Input price</td><td>Free of charge</td><td>$2.00, prompts &lt;= 200k tokens$4.00, prompts &gt; 200k</td></tr>
<tr><td>Output price (including thinking tokens)</td><td>Free of charge</td><td>$12.00</td></tr>
<tr><td>Context caching price</td><td>Free of charge</td><td>$0.20<br>$4.50 / 1,000,000 tokens per hour (storage price)</td></tr>
</table>
<h3 id="batch">Batch</h3>
<table>
<tr><td>Input price</td><td>Not available</td><td>$1.00</td></tr>
<tr><td>Output price (including thinking tokens)</td><td>Not available</td><td>$6.00</td></tr>
</table>
"""

CATALOG = {"models": {
    "claude-opus-4-8": {"served_by": [{"provider": "anthropic", "provider_model_id": "claude-opus-4-8"}]},
    "claude-sonnet-4-6": {"served_by": [{"provider": "anthropic", "provider_model_id": "claude-sonnet-4-6"}]},
    "gpt-5.5": {"served_by": [{"provider": "openai", "provider_model_id": "gpt-5.5"}]},
    "gemini-3.1-pro-preview": {"served_by": [{"provider": "google", "provider_model_id": "gemini-3.1-pro-preview"}]},
}}


# --- pure parsers ---------------------------------------------------------------

def test_markdown_parser_reads_anthropic_input_output_and_cache_read():
    recs: dict = {}    # first-wins, as the source's _resolve keeps standard over batch
    for r in op.parse_markdown_pricing(ANTHROPIC_MD):
        recs.setdefault(r["family_hint"], r)
    opus = recs["Claude Opus 4.8"]
    assert (opus["price_in"], opus["price_out"], opus["price_cached_in"]) == (5.0, 25.0, 0.50)
    # the cache-WRITE columns (1.25x/2x) are not mistaken for the cache-read price
    assert recs["Claude Sonnet 4.6"]["price_cached_in"] == 0.30
    # the markdown-link parenthetical is stripped from the model name
    assert "Claude Haiku 4.5" in recs


def test_openai_jsx_parser_reads_name_input_cached_output():
    recs: dict = {}     # first-wins keeps the standard pane over batch
    for r in op.parse_openai_jsx(OPENAI_MD):
        recs.setdefault(r["family_hint"], r)
    # the "(<272K context length)" suffix is stripped to the bare family
    assert (recs["gpt-5.5"]["price_in"], recs["gpt-5.5"]["price_cached_in"],
            recs["gpt-5.5"]["price_out"]) == (5.0, 0.50, 30.0)
    assert recs["gpt-5.5"]["price_in"] != 2.5            # not the batch pane
    # an empty cached field ("") parses as no cache price, not a crash
    assert recs["gpt-5.5-pro"]["price_cached_in"] is None


def test_gemini_html_parser_takes_standard_tier_and_cache_read():
    recs = {r["family_hint"]: r for r in op.parse_gemini_html(GEMINI_HTML)}
    g = recs["gemini-3.1-pro-preview"]
    assert g["price_in"] == 2.0           # base (<=200k) tier, NOT the >200k 4.00
    assert g["price_out"] == 12.0
    assert g["price_cached_in"] == 0.20   # cache read, NOT the per-hour storage
    # the batch tier ($1.00 input) is ignored — standard only
    assert g["price_in"] != 1.0


def test_slug_matches_dotted_and_dashed_family_names():
    assert op._slug("Claude Opus 4.8") == op._slug("claude-opus-4-8")
    assert op._slug("Gemini 3.1 Pro Preview") == op._slug("gemini-3.1-pro-preview")


# --- the source: resolve + push + fail-safe coast -------------------------------

class _Resp:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status


class _Client:
    def __init__(self, text=None, boom=False):
        self._text, self._boom = text, boom

    async def get(self, url):
        if self._boom:
            raise RuntimeError("network down")
        return _Resp(self._text)


def _src(provider_id, catalog=CATALOG, **kw):
    return op.OfficialPriceSource(catalog, provider_id, **kw)


def test_pricing_resolves_to_families_and_upserts_with_cache(monkeypatch):
    captured = {}
    monkeypatch.setattr(host_store, "set_provider_prices",
                        lambda rows: captured.setdefault("rows", rows) or True)
    src = _src("anthropic", client=_Client(ANTHROPIC_MD))
    prices = asyncio.run(src.pricing())
    by_fam = {p["model_family"]: p for p in prices}
    assert by_fam["claude-opus-4-8"]["price_in_usd_per_mtok"] == 5.0
    assert by_fam["claude-opus-4-8"]["price_out_usd_per_mtok"] == 25.0
    # cache-read price reaches the durable table even though the core Price omits it
    opus_row = next(r for r in captured["rows"] if r["model_family"] == "claude-opus-4-8")
    assert opus_row["price_cached_in"] == 0.50


def test_pricing_coasts_on_host_store_when_scrape_fails(monkeypatch):
    monkeypatch.setattr(host_store, "get_provider_prices", lambda pid=None: [
        {"provider_id": "openai", "model_family": "gpt-5.5",
         "price_in": 4.2, "price_out": 28.0, "price_cached_in": 0.4}])
    src = _src("openai", client=_Client(boom=True))   # network down
    prices = asyncio.run(src.pricing())               # must NOT raise
    assert [(p["model_family"], p["price_in_usd_per_mtok"]) for p in prices] \
        == [("gpt-5.5", 4.2)]


def test_pricing_ignores_uncataloged_page_rows(monkeypatch):
    monkeypatch.setattr(host_store, "set_provider_prices", lambda rows: True)
    # google catalog serves only gemini-3.1-pro-preview; the page row resolves to it
    src = _src("google", client=_Client(GEMINI_HTML))
    prices = asyncio.run(src.pricing())
    assert {p["model_family"] for p in prices} == {"gemini-3.1-pro-preview"}
