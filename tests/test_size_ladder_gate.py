"""The size-ladder gate arm must fail on a scaling cliff and on a lever going dark.

`scripts/release_gate.py --model size-ladder` is the standing check that a perf lever did
not get tuned at one sequence length and left dark at every other one. Its two comparison
legs are worth a host-only test each, because the arm is in the default arm set and a
release runs it unattended: a leg that silently never fires looks exactly like a leg that
passes.

The exponent leg had in particular never executed against a multi-rung baseline — the
RED/GREEN proof for the arm was taken at a single rung, where there is no consecutive pair
to exponent over, so the tolerance arithmetic shipped unexercised.

Host-only: no device, no network. Both legs are pure functions of a recorded baseline and a
measurement, so the measurement is synthesised here instead of folded.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import socket

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rg():
    spec = importlib.util.spec_from_file_location(
        "release_gate_under_test", REPO_ROOT / "scripts" / "release_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FIRING = {"resolved": "True", "served": 10, "declined": 0, "frac": 1.0, "how": "stats"}
RUNGS = (256, 512, 640, 768)
# The shape of a real boltz2 row on a p150a, rounded: see docs/size_ladder_baseline.json.
BASE_RUNTIME = {"256": 8.0, "512": 19.0, "640": 28.0, "768": 39.0}


def _baseline(levers=None):
    lv = dict(levers or FIRING)
    return {
        "runtime_s": dict(BASE_RUNTIME),
        "levers": {str(r): {"K2": dict(lv)} for r in RUNGS},
        "reps": 1,
        "sigma_runtime_512": 0.05,
        "exponents": {"256->512": {"k": 1.248, "tol": 0.50},
                      "512->768": {"k": 1.767, "tol": 0.539}},
    }


def _check(rg, runtime_s, levers=None, base=None, grid="13x10"):
    lv = dict(levers or FIRING)
    meas = {"levers": {str(r): {"K2": dict(lv)} for r in RUNGS},
            "runtime_s": runtime_s, "sigma": 0.05, "census_jsons": {}, "grid": grid}
    rg._size_ladder_measure_model = lambda *a, **k: meas
    return rg._size_ladder_check_model("boltz2", RUNGS, base or _baseline(),
                                       pathlib.Path("/tmp"))


def test_unchanged_ladder_passes(rg):
    assert _check(rg, dict(BASE_RUNTIME))["gate"] is True


def test_scaling_cliff_at_the_large_rung_fails(rg):
    """N^1.8 -> N^3.8 over 512->768, the magnitude the 2026-08-13 sweep measured."""
    r = _check(rg, {**BASE_RUNTIME, "768": 90.0})
    assert r["gate"] is False
    assert any("512->768" in f and "exponent" in f for f in r["findings"])


def test_uniform_slowdown_does_not_fail_the_exponent_leg(rg):
    """A flat factor cancels in a ratio, on purpose: it keeps the baseline portable across
    same-type machines and across thermal state. A uniform regression is perf_regression.py's
    job, not this arm's, and asserting it here would make the arm red on arrival elsewhere."""
    assert _check(rg, {k: v * 1.2 for k, v in BASE_RUNTIME.items()})["gate"] is True


def test_lever_going_dark_fails_and_names_the_clause(rg):
    """The defect the arm exists for: the guard still resolves True and silently declines."""
    dark = {"resolved": "True", "served": 0, "declined": 10, "frac": 0.0,
            "how": "stats", "rejects": {"fill_preconditions": 10}}
    r = _check(rg, dict(BASE_RUNTIME), levers=dark)
    assert r["gate"] is False
    assert any("went dark on fill_preconditions" in f for f in r["findings"])


def test_lever_starting_to_fire_also_fails(rg):
    """Both directions: a threshold quietly widening into a size it was never measured at
    is a change too, and the baseline is what says whether anyone signed off on it."""
    dark = {"resolved": "True", "served": 0, "declined": 10, "frac": 0.0, "how": "stats",
            "reason": "measured: declines every call at this size"}
    r = _check(rg, dict(BASE_RUNTIME), base=_baseline(levers=dark))
    assert r["gate"] is False
    assert any("started firing" in f for f in r["findings"])


def test_partial_darkness_fails(rg):
    """K2 read 560/0 at 512 aa and 560/560 at 768 on main: a fired-SET comparison calls that
    'still firing' and passes. The fraction rule is what catches the half of the defect that
    has no on/off edge."""
    half = {"resolved": "True", "served": 5, "declined": 5, "frac": 0.5, "how": "stats"}
    r = _check(rg, dict(BASE_RUNTIME), levers=half)
    assert r["gate"] is False
    assert any("exceeds the" in f for f in r["findings"])


def test_new_decline_clause_fails(rg):
    """Same fired fraction, different reason: a behaviour change with no timing signature and
    no fired-fraction signature, so nothing else in the arm can see it."""
    base_lv = {**FIRING, "served": 5, "declined": 5, "frac": 0.5,
               "rejects": {"k_tiles=4": 5}, "reason": "x"}
    cur_lv = {**FIRING, "served": 5, "declined": 5, "frac": 0.5,
              "rejects": {"m_le_n": 5}}
    r = _check(rg, dict(BASE_RUNTIME), levers=cur_lv, base=_baseline(levers=base_lv))
    assert r["gate"] is False
    assert any("decline clause" in f for f in r["findings"])


def test_threshold_constant_change_fails(rg):
    """TRANSPOSE_L1_RESIDENT resolves to the constant's VALUE, so editing a threshold fails
    the arm until someone re-records at every rung. That is the standing rule enforcing
    itself instead of relying on a reviewer noticing."""
    r = _check(rg, dict(BASE_RUNTIME), levers={**FIRING, "resolved": "2.5"})
    assert r["gate"] is False
    assert any("resolved" in f for f in r["findings"])


def test_dark_lever_without_an_exemption_reason_fails(rg):
    """A dark lever is a pass only if somebody wrote down why, once."""
    dark = {"resolved": "True", "served": 0, "declined": 10, "frac": 0.0, "how": "stats"}
    r = _check(rg, dict(BASE_RUNTIME), levers=dark, base=_baseline(levers=dark))
    assert r["gate"] is False
    assert any("no exemption reason in the baseline" in f for f in r["findings"])


def test_a_todo_is_not_an_exemption_reason(rg):
    """Recording seeds every newly dark lever with a TODO carrying the measured clause. The
    TODO must not satisfy the gate, or the record step becomes the sign-off."""
    dark = {"resolved": "True", "served": 0, "declined": 10, "frac": 0.0, "how": "stats",
            "reason": "TODO: say why this is legitimate at this size (declines on k_tiles=4 x10)"}
    r = _check(rg, dict(BASE_RUNTIME), levers=dark, base=_baseline(levers=dark))
    assert r["gate"] is False
    assert any("no exemption reason in the baseline" in f for f in r["findings"])


def test_a_real_exemption_reason_passes(rg):
    dark = {"resolved": "True", "served": 0, "declined": 10, "frac": 0.0, "how": "stats",
            "reason": "declines every call on k_tiles=4: F1_BLOCK_KEYS allow-lists only (8, 8)"}
    assert _check(rg, dict(BASE_RUNTIME), levers=dark,
                  base=_baseline(levers=dark))["gate"] is True


def test_off_lattice_rung_is_in_the_ladder_but_not_the_timing_chain(rg):
    """256/512/768/1024 all have a padded length the SDPA chunk divides, so a ladder of
    256-multiples cannot see a kernel that declines at 448/576/640/704/832/896/960 only.
    640 is the off-lattice control; it carries no exponent because a 3-sigma band over
    ln(640/512) is wider than the cliff signal it would be gating."""
    assert 640 in rg.SIZE_LADDER_RUNGS
    assert 640 not in rg.SIZE_LADDER_EXP_RUNGS


def test_cross_grid_comparison_is_refused_not_reported_as_drift(rg):
    """A guard sized against the core grid flips with the grid (protenix-v2's K2 is admitted
    on 11x10 and refused on 13x10), and board type does not pin the grid because harvesting
    means one board type presents several. Comparing across grids would report levers as newly
    dark that never went dark, which is how an arm gets switched off."""
    base = {**_baseline(), "grid": "11x10"}
    r = _check(rg, dict(BASE_RUNTIME), base=base, grid="13x10")
    assert r["gate"] is False
    assert "grid" in r["error"] and "re-record" in r["error"]


def test_same_grid_still_compares(rg):
    base = {**_baseline(), "grid": "13x10"}
    assert _check(rg, dict(BASE_RUNTIME), base=base, grid="13x10")["gate"] is True


def test_size_ladder_is_in_the_default_arm_set(rg):
    """The whole point: a release runs it without anyone remembering to."""
    src = (REPO_ROOT / "scripts" / "release_gate.py").read_text()
    default = src.split("models = args.model or", 1)[1].split("fold_models", 1)[0]
    assert '"size-ladder"' in default


def test_subset_record_keeps_the_other_models_own_provenance(rg, tmp_path, monkeypatch):
    """A subset record must not restamp the models it did not measure.

    `--size-ladder-models rf3` rewrites the card-level recorded/host/commit. Before every
    entry carried its own stamp, that made the file claim the five pc-recorded models came
    from whichever host recorded rf3 — and rf3 has to be recorded on qb1, because pc cannot
    hold the box for the ~80 min this leg needs. So this is the live case, not a hypothetical.
    """
    baseline = tmp_path / "size_ladder_baseline.json"
    baseline.write_text(json.dumps({"cards": {"p150a": {
        "recorded": "2026-08-22", "host": "pc", "commit": "9bc86a5a",
        "models": {"boltz2": _baseline()}}}}))
    meas = {"levers": {str(r): {"K2": dict(FIRING)} for r in RUNGS},
            "runtime_s": dict(BASE_RUNTIME), "sigma": 0.05, "census_jsons": {},
            "grid": "13x10"}
    monkeypatch.setattr(rg, "_size_ladder_measure_model", lambda *a, **k: meas)
    monkeypatch.setattr(rg, "_size_ladder_card_type", lambda: "p150a")
    monkeypatch.setattr(rg, "_repo_commit", lambda: "cafe1234")

    row = rg.run_size_ladder(keep=False, record=True, baseline_path=baseline, models=["rf3"])
    assert row["gate"], row

    card = json.loads(baseline.read_text())["cards"]["p150a"]
    assert card["models"]["boltz2"]["host"] == "pc"
    assert card["models"]["boltz2"]["recorded"] == "2026-08-22"
    assert card["models"]["boltz2"]["commit"] == "9bc86a5a"
    assert card["models"]["rf3"]["host"] == socket.gethostname()
    assert card["models"]["rf3"]["commit"] == "cafe1234"
    # and the card-level stamp still describes the last pass, so nothing is lost
    assert card["commit"] == "cafe1234"


def test_rf3_is_in_the_size_ladder(rg):
    """RF3 shipped as a `predict --model rf3` choice in v0.6.6 with no correctness coverage in
    either gate leg. It carries RF3-scoped perf levers and it has already had one L1 gate go
    dark above a tuned size (state/rf3-perf.md's 768->1024 aa exponent jump), which is exactly
    what this arm exists to catch."""
    assert "rf3" in rg.SIZE_LADDER_MODELS
    assert "rf3" in rg.MODELS
def test_absent_decline_clause_is_not_measured_not_no_clause(rg):
    """A baseline recorded before the census could report a clause has none, and comparing that
    against today's census read as three guards changing their mind on every model at every rung
    (95033b2f landed after the baseline, so the arm was failing on main for an instrument change).
    served/declined identical plus an unrecorded clause is not a behaviour change."""
    base_lv = {**FIRING, "served": 0, "declined": 560, "frac": 0.0, "reason": "x"}
    cur_lv = {**FIRING, "served": 0, "declined": 560, "frac": 0.0,
              "rejects": {"k_tiles=4:(4,1)": 560}}
    assert rg._size_ladder_clause_finding(base_lv, cur_lv, "F", "m/256") is None


def test_a_recorded_clause_that_really_changes_still_fails(rg):
    """The narrowing above must not blunt the rule it narrows."""
    base_lv = {**FIRING, "served": 0, "declined": 5, "frac": 0.0,
               "rejects": {"k_tiles=4": 5}, "reason": "x"}
    cur_lv = {**FIRING, "served": 0, "declined": 5, "frac": 0.0, "rejects": {"m_le_n": 5}}
    assert rg._size_ladder_clause_finding(base_lv, cur_lv, "F", "m/256") is not None


def test_a_guard_with_no_declines_has_no_clause_to_compare(rg):
    """REBLOCK_PERMUTE_GATED carried a clause it inherited from the REJECTS dict it used to
    share, on an entry with 0 declines. Dropping it is not a behaviour change either."""
    base_lv = {**FIRING, "served": 0, "declined": 0, "frac": 0.0,
               "rejects": {"window_BufferType.L1": 3}, "reason": "x"}
    cur_lv = {**FIRING, "served": 0, "declined": 0, "frac": 0.0}
    assert rg._size_ladder_clause_finding(base_lv, cur_lv, "F", "m/256") is None


def test_nesso1_precondition_is_checked_before_a_fold_not_by_one(rg, monkeypatch, tmp_path):
    """Without the checkpoint's uncommitted ccd.pkl every nesso1 rung would fail from inside a
    subprocess: twelve wasted model loads and an arm that reads as broken rather than
    unconfigured. The message has to name what to set."""
    monkeypatch.setenv("NESSO_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    # HOME too: find_ccd always searches ~/.cache/huggingface, and it now PUTS the file
    # there on a miss, so on any machine that has run `tt-bio affinity` this passed by
    # finding the real file rather than by exercising the precondition.
    monkeypatch.setenv("HOME", str(tmp_path))

    # The only failure left that a user must act on: no file on disk and no way to fetch
    # one. Simulated, so the test needs neither network nor 413 MB.
    import huggingface_hub

    def _no_network(*a, **k):
        raise OSError("simulated: no route to huggingface.co")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_network)
    pre = rg._size_ladder_precondition("nesso1")
    assert pre and "NESSO_CACHE" in pre
    assert rg._size_ladder_precondition("boltz2") is None


def test_nesso1_folds_through_affinity_not_predict(rg):
    """`predict` cannot fold this model, and the shared apo fixture has no ligand and no
    affinity property, so the leg needs both its own CLI and its own ladder."""
    assert "nesso1" in rg.SIZE_LADDER_MODELS
    f = rg._size_ladder_fixture("nesso1", 640)
    assert f.name == "cdk2_640.yaml" and "nesso1" in str(f) and f.exists()
    for rung in rg.SIZE_LADDER_RUNGS:
        assert rg._size_ladder_fixture("nesso1", rung).exists()
