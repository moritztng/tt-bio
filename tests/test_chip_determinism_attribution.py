"""A cross-chip mismatch is not automatically a chip difference.

The harness varies one axis (which chip) and used to report DIVERGENT on any mismatch, which
attributes the difference to the only axis it happened to vary. A host-side step that is not
reproducible run to run -- a ligand conformer embedding, an unseeded shuffle -- produces the
same mismatch with no chip involved. So the run now also folds the same chip twice at the
same seed, and the verdict says which axis the evidence actually supports.

Device-free: `fold` and `compare` are stubbed, so every branch is reachable without a card.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

MATCH = {"comparable": True, "all_sha_match": True, "max_abs_delta": 0.0,
         "files": [], "missing_on_one_side": []}


def _differ(delta):
    return {"comparable": True, "all_sha_match": False, "max_abs_delta": delta,
            "files": [], "missing_on_one_side": []}


def _load():
    spec = importlib.util.spec_from_file_location(
        "chip_determinism_attr", ROOT / "perf" / "wh-parity" / "chip_determinism.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verdict(monkeypatch, tmp_path, *, cross, ctrl, repeat, argv_extra=()):
    """Run main() with the two device-touching primitives replaced."""
    mod = _load()
    monkeypatch.setattr(mod, "fold", lambda model, inp, card, seed, out_dir, extra, python,
                        timeout: {"card": card, "seed": seed, "rc": 0, "wall_s": 1.0,
                                  "out_dir": str(out_dir), "log": str(out_dir / "fold.log"),
                                  "cmd": ""})

    # Key the stub on the directory names main() chose, so the test asserts the harness
    # compares the pairs it claims to and not, say, card A against itself.
    def fake_compare(a: Path, b: Path):
        pair = (a.name, b.name)
        if pair == ("card28_seed0", "card29_seed0"):
            return cross
        if pair == ("card28_seed0", "card28_seed1"):
            return ctrl
        if pair == ("card29_seed0", "card29_seed0_repeat"):
            return repeat
        raise AssertionError(f"unexpected comparison {pair}")

    monkeypatch.setattr(mod, "compare", fake_compare)
    out = tmp_path / "out"
    jf = tmp_path / "v.json"
    argv = ["chip_determinism.py", "--model", "openbind", "--input", "x.yaml",
            "--cards", "28,29", "--out-dir", str(out), "--json", str(jf), *argv_extra]
    monkeypatch.setattr(mod.sys, "argv", argv)
    mod.main()
    return json.loads(jf.read_text())


def test_same_chip_disagrees_with_itself_is_not_a_chip_difference(monkeypatch, tmp_path):
    """The case that matters: the repeat already explains the cross-chip delta."""
    r = _verdict(monkeypatch, tmp_path, cross=_differ(0.47), ctrl=_differ(47.6),
                 repeat=_differ(0.51))
    assert r["verdict"] == "NONDETERMINISTIC"
    assert "NOT a chip difference" in r["detail"]
    assert "host-side" in r["detail"]


def test_chip_difference_is_only_claimed_when_the_chip_reproduces_itself(monkeypatch, tmp_path):
    r = _verdict(monkeypatch, tmp_path, cross=_differ(0.47), ctrl=_differ(47.6), repeat=MATCH)
    assert r["verdict"] == "DIVERGENT"
    assert "IS chip-dependent" in r["detail"]


def test_clean_match_with_both_controls(monkeypatch, tmp_path):
    r = _verdict(monkeypatch, tmp_path, cross=MATCH, ctrl=_differ(47.6), repeat=MATCH)
    assert r["verdict"] == "BIT-IDENTICAL"
    assert "NOT attributed" not in r["detail"]


def test_a_match_on_top_of_an_unstable_repeat_is_incoherent_not_a_pass(monkeypatch, tmp_path):
    """Two runs that cannot reproduce themselves have no business agreeing across chips."""
    r = _verdict(monkeypatch, tmp_path, cross=MATCH, ctrl=_differ(47.6), repeat=_differ(0.3))
    assert r["verdict"] == "INCOHERENT"


def test_insensitive_control_still_voids_the_run(monkeypatch, tmp_path):
    r = _verdict(monkeypatch, tmp_path, cross=MATCH, ctrl=MATCH, repeat=MATCH)
    assert r["verdict"] == "VOID"


def test_skipping_the_repeat_downgrades_the_claim_instead_of_asserting_it(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.setattr(mod, "fold", lambda model, inp, card, seed, out_dir, extra, python,
                        timeout: {"card": card, "seed": seed, "rc": 0, "wall_s": 1.0,
                                  "out_dir": str(out_dir), "log": "", "cmd": ""})
    monkeypatch.setattr(mod, "compare", lambda a, b: (
        _differ(0.47) if b.name == "card29_seed0" else _differ(47.6)))
    out, jf = tmp_path / "o", tmp_path / "v.json"
    monkeypatch.setattr(mod.sys, "argv", ["x", "--model", "openbind", "--input", "x.yaml",
                                          "--cards", "28,29", "--out-dir", str(out),
                                          "--json", str(jf), "--skip-repeat"])
    mod.main()
    r = json.loads(jf.read_text())
    assert r["verdict"] == "UNATTRIBUTED"
    assert "not established" in r["detail"]


def test_the_repeat_fold_is_actually_run(monkeypatch, tmp_path):
    """Negative control on the harness: the extra fold exists, on card B at the same seed."""
    r = _verdict(monkeypatch, tmp_path, cross=MATCH, ctrl=_differ(47.6), repeat=MATCH)
    seen = {(f["card"], f["seed"]) for f in r["folds"]}
    assert seen == {(28, 0), (29, 0), (28, 1)}
    assert len(r["folds"]) == 4, "card29/seed0 must be folded twice, not once"
    assert "control_same_chip_same_seed_repeat" in r
