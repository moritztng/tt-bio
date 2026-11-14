"""--size-ladder-record-lever skips a rung that already holds the lever, and folds the rest.

Device-free and torch-free. On 2026-10-10 the release gate ran `--record-lever SDPA_FUSED_PADDED`
for every ladder model, and seven record legs failed with "already in the baseline" at every rung
of models a sibling row had spliced earlier: a finished splice read as a refused one.

Run: python3 tests/test_size_ladder_add_lever_idempotent.py, or via pytest.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import release_gate as rg

ROW = {"resolved": True, "served": 4, "declined": 0, "frac": 1.0, "how": "counter"}


def _setup(tmp_path, monkeypatch, have_at):
    base = tmp_path / "size_ladder_baseline.json"
    levers = {str(r): ({"OLD": dict(ROW), "NEW": dict(ROW)} if r in have_at else {"OLD": dict(ROW)})
              for r in (256, 512)}
    base.write_text(json.dumps({"cards": {"p300c": {"models": {"boltz2": {
        "levers": levers, "runtime_s": {"256": 4.2, "512": 10.9}}}}}}))
    folds = []

    def fake(model, rung, workdir, tag, need_runtime=True):
        folds.append(rung)
        return {"levers": {"OLD": dict(ROW), "NEW": dict(ROW)}, "runtime_s": 1.0}

    monkeypatch.setattr(rg, "_run_census_fold", fake)
    monkeypatch.setattr(rg, "_size_ladder_card_type", lambda: "p300c")
    monkeypatch.setattr(rg, "_size_ladder_model_rungs", lambda m: (256, 512))
    monkeypatch.setattr(rg, "_size_ladder_precondition", lambda m: None)
    monkeypatch.setattr(rg, "lever_census_flags", lambda: ("OLD", "NEW"))
    monkeypatch.setattr(rg, "SIZE_LADDER_WORKDIR", tmp_path / "work")
    return base, folds


def test_a_lever_already_at_every_rung_passes_without_a_fold(tmp_path, monkeypatch):
    base, folds = _setup(tmp_path, monkeypatch, have_at=(256, 512))
    out = rg.run_size_ladder_add_lever("NEW", True, base, models=["boltz2"])
    assert out["legs"][0]["gate"], out
    assert folds == [], folds


def test_only_the_missing_rungs_are_folded_and_spliced(tmp_path, monkeypatch):
    base, folds = _setup(tmp_path, monkeypatch, have_at=(256,))
    out = rg.run_size_ladder_add_lever("NEW", True, base, models=["boltz2"])
    assert out["legs"][0]["gate"], out
    assert folds == [512], folds
    levers = json.loads(base.read_text())["cards"]["p300c"]["models"]["boltz2"]["levers"]
    assert "NEW" in levers["512"], levers


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
