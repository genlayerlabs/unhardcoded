"""Curated OpenRouter slugs must be the ids OpenRouter serves (Anthropic uses dotted versions)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_curated_anthropic_openrouter_slugs_use_dotted_versions():
    lua = (ROOT / "config.live.lua").read_text()
    slugs = re.findall(r"provider\s*=\s*\"openrouter\",\s*provider_model_id\s*=\s*\"(anthropic/[^\"]+)\"", lua)
    assert slugs, "expected curated Anthropic families served through OpenRouter"
    # OpenRouter lists e.g. anthropic/claude-opus-4.7; anthropic/claude-opus-4-7 does not exist.
    assert [s for s in slugs if re.search(r"-\d+-\d+$", s)] == []
