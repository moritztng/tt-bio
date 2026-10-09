"""Every lever a ladder model's fold resolves must already be in that model's baseline.

The size-ladder leg reports "new lever not in the baseline" at every rung for a census lever
the baseline has never seen, and a release gate only finds that out after folding the whole
ladder: 40-100 min per model and arch. SDPA_FUSED_PADDED reached main that way and failed
every ladder leg of the 1dd0e40b6 gate on Wormhole. Whether the fold will resolve the lever is
readable from the baseline itself: a lever's module is imported in a model's fold exactly when
another lever from that module was recorded there as resolved. So this catches it card-free,
when the lever lands, and names the one-fold-per-rung fix.

Host-only: reads docs/size_ladder_baseline* and scripts/lever_census.py.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import lever_census  # noqa: E402

MODULE = {flag: mod for flag, mod, *_ in lever_census.LEVERS}


def _entries(docs: pathlib.Path) -> dict:
    """(card, model) -> recorded entry, the per-model fragments overriding the monolith."""
    out = {}
    files = [docs / "size_ladder_baseline.json", *sorted((docs / "size_ladder_baseline.d").glob("*.json"))]
    for f in files:
        if f.exists():
            for card, c in json.loads(f.read_text()).get("cards", {}).items():
                out.update({(card, m): e for m, e in c.get("models", {}).items()})
    return out


def missing_levers(docs: pathlib.Path, models=None) -> list[str]:
    owed = []
    for (card, model), e in sorted(_entries(docs).items()):
        if models is not None and model not in models:
            continue                        # exempt from the ladder: its entry is never checked
        for rung, rows in sorted((e.get("levers") or {}).items()):
            imported = {MODULE.get(f) for f, r in rows.items() if r.get("resolved") != "not-imported"}
            new = sorted(f for f, mod in MODULE.items() if f not in rows and mod in imported)
            if new:
                owed.append(f"{card} {model}: {','.join(new)}")
                break
    return owed


def test_every_lever_an_imported_module_registers_is_in_the_baseline():
    pytest.importorskip("torch")                # release_gate imports it at module level
    spec = importlib.util.spec_from_file_location("release_gate", REPO / "scripts" / "release_gate.py")
    rg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rg)
    owed = missing_levers(REPO / "docs", set(rg.SIZE_LADDER_MODELS))
    assert not owed, (
        "the size ladder will fail 'new lever not in the baseline' for:\n  " + "\n  ".join(owed)
        + "\nrecord them on each card: scripts/release_gate.py --model size-ladder "
          "--size-ladder-models <model> --size-ladder-record-lever <FLAG,...>")


def test_a_lever_of_a_module_the_model_never_imports_is_not_owed(tmp_path):
    (tmp_path / "size_ladder_baseline.d").mkdir()
    flag, mod = next((f, m) for f, m in MODULE.items() if m == "tt_bio.tenstorrent")
    other = next(f for f, m in MODULE.items() if m != mod)
    rows = {f: {"resolved": "True"} for f, m in MODULE.items() if m == mod and f != flag}
    rows[other] = {"resolved": "not-imported"}
    (tmp_path / "size_ladder_baseline.json").write_text(json.dumps(
        {"cards": {"c": {"models": {"m": {"levers": {"256": rows}}}}}}))
    assert missing_levers(tmp_path) == [f"c m: {flag}"]
    rows = {f: {"resolved": "not-imported"} for f in MODULE if f != flag}
    (tmp_path / "size_ladder_baseline.json").write_text(json.dumps(
        {"cards": {"c": {"models": {"m": {"levers": {"256": rows}}}}}}))
    assert missing_levers(tmp_path) == []
