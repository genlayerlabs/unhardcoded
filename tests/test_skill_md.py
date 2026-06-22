"""SKILL.md conformance: every Σ_pol op the authoring guide teaches must exist
in the core signature (`core/llm_policy/sig.lua` S.ops).

The guide mirrors the sigma-pol/v1 grammar for authoring convenience; the field
vocabulary is injected live (so it can't drift), but the *operators* are prose.
This gate closes that gap: if an op referenced anywhere a reader copies from —
the complete ```json examples AND the inline `[...]` snippets of the reference
table and the scorer prose — was renamed, removed, or typo'd, the test fails.
The guide can never hand an assistant a term this host would reject at
admission. Template holes (`<field>`, `<scorer>`, the `base` metavariable) are
coerced to a literal so the op at position 0 is still gated; the op is never a
hole, so coercing the holes neither hides nor invents one."""
from __future__ import annotations

import json
import re
from pathlib import Path

from lupa import LuaRuntime

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "SKILL.md"
SIG = ROOT / "core" / "llm_policy" / "sig.lua"


def _sig_ops() -> set[str]:
    lua = LuaRuntime()
    lua.globals()["__p"] = str(SIG)
    S = lua.eval("dofile(__p)")
    return {k for k in S.ops}


def _coerce_template(snippet: str) -> str:
    """Make a template snippet JSON-parseable: angle placeholders (`<field>`,
    `<scorer>`, …) and the bare `base` metavariable → 0. The op sits at array
    position 0 and is never a hole, so this exposes its op without altering it.
    Snippets that still don't parse (e.g. a bareword count `N`) are skipped by
    the caller — their ops are covered by the complete examples anyway."""
    snippet = re.sub(r"<[^>]*>", "0", snippet)
    return re.sub(r"\bbase\b", "0", snippet)


def _json_blocks(md: str) -> list:
    """Every gateable term form in the guide:
    - complete ```json examples — the copy-pasteable canon. Template holes are
      coerced, then the block MUST parse: a malformed real example FAILS the
      gate rather than silently disappearing (the whole point of the gate is to
      catch a bad op in an example, so a typo must never skip the check).
    - inline `[...]` term snippets from the reference table and scorer prose,
      coerced too. These may carry barewords that aren't angle-holes (`N`,
      `...`) which can't be coerced cleanly, so an unparseable inline snippet is
      skipped — its ops are covered by the complete examples and guarded by
      `test_skill_md_gate_covers_reference_table_and_prose_ops`."""
    blocks = []
    for raw in re.findall(r"```json\n(.*?)```", md, flags=re.S):
        blocks.append(json.loads(_coerce_template(raw)))   # strict
    for span in re.findall(r"`(\[[^`]*\])`", md):
        try:
            blocks.append(json.loads(_coerce_template(span)))
        except json.JSONDecodeError:
            continue
    return blocks


def _collect_ops(term, seen: set[str]) -> None:
    """Walk a Σ_pol term: position 0 is the op; only array-valued elements are
    subterms (scalars/strings/records are parameters), so flow `inputs` lists
    and string params are never mistaken for ops."""
    if not isinstance(term, list) or not term or not isinstance(term[0], str):
        return
    seen.add(term[0])
    for child in term[1:]:
        if isinstance(child, list):
            _collect_ops(child, seen)


def _ops_in_block(block, seen: set[str]) -> None:
    if isinstance(block, dict):
        if isinstance(block.get("policy_ir"), list):
            _collect_ops(block["policy_ir"], seen)
        return
    if isinstance(block, list) and block and block[0] == "flow":
        for node in (block[1] or {}).values():
            if isinstance(node, dict) and isinstance(node.get("policy"), list):
                _collect_ops(node["policy"], seen)
        return
    _collect_ops(block, seen)


def test_skill_md_examples_use_only_real_sigma_pol_ops():
    blocks = _json_blocks(SKILL.read_text())
    assert blocks, "SKILL.md should contain ```json policy examples"
    seen: set[str] = set()
    for b in blocks:
        _ops_in_block(b, seen)
    assert seen, "no Σ_pol ops were extracted — parser/guide drift"

    ops = _sig_ops()
    unknown = seen - ops
    assert not unknown, (
        f"SKILL.md uses ops absent from sigma-pol/v1 (sig.lua): {sorted(unknown)}")


def test_skill_md_gate_covers_reference_table_and_prose_ops():
    """Regression guard for the inline pass: ops taught only in the reference
    table / scorer prose (never in a complete ```json example) must still be
    extracted and gated. If the inline extraction regresses, these silently
    stop being checked — the exact gap this follow-up closed."""
    seen: set[str] = set()
    for b in _json_blocks(SKILL.read_text()):
        _ops_in_block(b, seen)
    table_and_prose_only = {
        "family_eq", "min_tier", "or", "gate", "neg", "zero", "sample",
    }
    missing = table_and_prose_only - seen
    assert not missing, f"inline gate regressed — not extracted: {sorted(missing)}"
    # and they are real ops (the whole point: the guide teaches no phantom op)
    assert table_and_prose_only <= _sig_ops()
