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


def test_a_new_lever_no_ladder_model_imports_is_not_a_finding(rg):
    """Registering a model-scoped lever in the shared census is an instrument change, but an
    instrument change with nothing to record: `collect()` emits a row for every entry in
    `LEVERS`, so RFD3's two fusion levers appear at every rung of every ladder model reading
    `not-imported`. Reporting that as drift would demand 32 census folds to write 64 rows that
    all say the same nothing. The loop above already refuses to compare a not-imported lever the
    baseline DOES know about; this is the same case from the other side."""
    base = _baseline()
    cur = {r: dict(base["levers"][str(r)],
                   RFD3_SOFTMAX_PV_FUSED={"resolved": "not-imported", "served": None,
                                          "declined": None, "frac": None, "how": "stats"})
           for r in RUNGS}
    for r in RUNGS:
        assert rg._size_ladder_compare_levers(base["levers"][str(r)], cur[r], f"{r}aa") == []


def test_a_new_lever_a_ladder_model_DOES_import_is_still_a_finding(rg):
    """The other half of the same rule: a lever that resolved means it was measured, and a
    measurement the baseline has never seen is exactly what re-recording is for."""
    base = _baseline()
    cur = dict(base["levers"][str(RUNGS[0])],
               SOME_NEW_LEVER={"resolved": "True", "served": 10, "declined": 0, "frac": 1.0,
                               "how": "stats"})
    findings = rg._size_ladder_compare_levers(base["levers"][str(RUNGS[0])], cur, "256aa")
    assert findings and any("new lever not in the baseline" in f for f in findings)


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



def test_fragment_record_writes_one_file_per_model_and_leaves_no_duplicate(
        rg, tmp_path, monkeypatch):
    """The record loop's fragment branch: each model lands in its own file and its rows are
    dropped from the monolith's card block, so the same (card, model) is never recorded in
    two places that can drift apart."""
    baseline = tmp_path / "size_ladder_baseline.json"
    baseline.write_text(json.dumps({"cards": {"p150a": {
        "recorded": "2026-08-22", "host": "pc", "commit": "9bc86a5a",
        "models": {"boltz2": _baseline(), "rf3": _baseline()}}}}))
    meas = {"levers": {str(r): {"K2": dict(FIRING)} for r in RUNGS},
            "runtime_s": dict(BASE_RUNTIME), "sigma": 0.05, "census_jsons": {},
            "grid": "8x9"}
    monkeypatch.setattr(rg, "_size_ladder_measure_model", lambda *a, **k: meas)
    monkeypatch.setattr(rg, "_size_ladder_card_type", lambda: "tt-galaxy-wh l")
    monkeypatch.setattr(rg, "_repo_commit", lambda: "cafe1234")

    assert rg.run_size_ladder(keep=False, record=True, baseline_path=baseline,
                              models=["boltz2"], fragment=True)["gate"]
    # the monolith never gains the new card: every row this pass measured is in a fragment
    assert "tt-galaxy-wh l" not in json.loads(baseline.read_text())["cards"]
    after_first = baseline.read_text()

    assert rg.run_size_ladder(keep=False, record=True, baseline_path=baseline,
                              models=["rf3"], fragment=True)["gate"]
    # the second pass reads boltz2's fragment but must not copy it into the shared file
    assert baseline.read_text() == after_first

    frags = sorted(f.name for f in (tmp_path / "size_ladder_baseline.d").glob("*.json"))
    assert frags == ["boltz2.json", "rf3.json"], frags
    resolved = rg._size_ladder_read_baseline(baseline)["cards"]
    assert set(resolved["tt-galaxy-wh l"]["models"]) == {"boltz2", "rf3"}
    # and the Blackhole card is still the one the monolith recorded
    assert resolved["p150a"]["commit"] == "9bc86a5a"
    assert sorted(resolved["p150a"]["models"]) == ["boltz2", "rf3"]


DARK = {"resolved": "True", "served": 0, "declined": 8, "frac": 0.0, "how": "stats"}


def test_a_new_card_inherits_the_judgement_half_of_an_exemption_reason(rg):
    """Recording a card type for the first time has no previous entry for that card, so
    every dark lever would take a TODO and the check could not pass until a human retyped
    judgements the file already holds one card block away."""
    reference = {"cards": {
        "p150a": {"models": {"boltz2": {"recorded": "2026-08-22", "levers": {"256": {"K2": {
            **DARK, "reason": "declines all 556 calls on l1_dest: an ESMC lever, boltz-2 "
                              "does not run that module"}}}}}},
        "p300c": {"models": {"boltz2": {"recorded": "2026-08-19", "levers": {"256": {"K2": {
            **DARK, "reason": "declines all 4 calls: stale, from the older record"}}}}}},
    }}
    inherited = rg._size_ladder_other_card_levers(reference, "tt-galaxy-wh l", "boltz2")
    assert [c for c, _ in inherited] == ["p150a", "p300c"]      # newest first

    levers = {"256": {"K2": dict(DARK)}}
    assert rg._size_ladder_fill_reasons(levers, None, inherited) == 0
    reason = levers["256"]["K2"]["reason"]
    # the judgement carries, tagged with where it came from
    assert "does not run that module" in reason
    assert "[carried from p150a]" in reason
    # the evidence half is this card's own, not p150a's 556 calls
    assert "declines all 8 calls" in reason and "556" not in reason


def test_a_never_reached_reason_re_measures_its_counts_too(rg):
    """The evidence half is regenerated whichever opening it has. A lever that was dark
    for want of a call site and now declines real calls must not keep saying 0 offered."""
    old = {"256": {"K2": {**DARK, "served": 0, "declined": 0,
                          "reason": "never reached at this size, 0 offered and 0 declined: "
                                    "no call site on boltz-2's path"}}}
    levers = {"256": {"K2": dict(DARK)}}
    assert rg._size_ladder_fill_reasons(levers, old) == 0
    reason = levers["256"]["K2"]["reason"]
    assert reason.startswith("declines all 8 calls")
    assert "no call site on boltz-2's path" in reason
    assert "0 offered" not in reason


def test_a_lever_dark_only_on_the_new_card_still_gets_a_todo(rg):
    """Inheritance must not invent a judgement. A lever that fires everywhere else has no
    reason to carry, so it is still a human's to write."""
    reference = {"cards": {"p150a": {"models": {"boltz2": {
        "recorded": "2026-08-22", "levers": {"256": {"K2": dict(FIRING)}}}}}}}
    levers = {"256": {"K2": dict(DARK)}}
    inherited = rg._size_ladder_other_card_levers(reference, "tt-galaxy-wh l", "boltz2")
    assert rg._size_ladder_fill_reasons(levers, None, inherited) == 1
    assert levers["256"]["K2"]["reason"].startswith("TODO")


def test_the_card_being_recorded_is_not_its_own_inheritance_source(rg):
    """Otherwise a stale reason on the card being re-recorded would look inherited."""
    reference = {"cards": {"p150a": {"models": {"boltz2": {
        "recorded": "2026-08-22", "levers": {"256": {"K2": {**DARK, "reason": "x: y"}}}}}}}}
    assert rg._size_ladder_other_card_levers(reference, "p150a", "boltz2") == []


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



# --- a failure at one rung must not erase the rungs that measured -----------------
#
# Rungs run in ascending order. Before this was pinned, an error at 768 returned only
# {"error": ...}, so the summary printed "-" in all four cells and the row read as a model
# that cannot fold at any size. On 2026-08-23 the release gate reported exactly that for
# opendde and it took a three-arm bisect to re-establish that 256/512/640 had been fine.

@pytest.fixture
def rg_fresh():
    """Own module instance. The `rg` fixture is module-scoped and the `_check` helper above
    permanently rebinds `_size_ladder_measure_model` on it, so a test that needs the real one
    has to load its own copy."""
    spec = importlib.util.spec_from_file_location(
        "release_gate_partial_rungs", REPO_ROOT / "scripts" / "release_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ok_fold(runtime):
    return {"levers": {"FLAG": FIRING}, "runtime_s": runtime, "wall": runtime + 20.0,
            "census_json": pathlib.Path("census.json"), "grid": "13x10"}


def test_top_rung_failure_keeps_the_rungs_that_measured(rg_fresh, monkeypatch, tmp_path):
    measured = {"256": 25.4, "512": 70.4, "640": 115.0}

    def fake(model, rung, workdir, tag):
        if rung == 768:
            return {"error": "census fold timed out after 1800s"}
        return _ok_fold(measured[str(rung)])

    monkeypatch.setattr(rg_fresh, "_run_census_fold", fake)
    out = rg_fresh._size_ladder_measure_model("opendde", RUNGS, tmp_path, 1, 1)

    assert "timed out after 1800s" in out["error"]
    assert "rung 768 warm-up" in out["error"]
    assert out["partial"] is True
    # the three rungs that completed are still reportable, and the one that failed is absent
    assert out["runtime_s"] == measured
    assert "768" not in out["runtime_s"]


def test_check_model_propagates_the_partial_rungs_to_the_printer(rg_fresh, monkeypatch, tmp_path):
    """The summary row reads runtime_s off the leg dict, so the error path has to carry it."""
    def fake(model, rungs, workdir, reps_512, reps_other):
        return {"error": "rung 768 warm-up: census fold timed out after 1800s",
                "runtime_s": {"256": 25.4, "512": 70.4, "640": 115.0}, "partial": True}

    monkeypatch.setattr(rg_fresh, "_size_ladder_measure_model", fake)
    leg = rg_fresh._size_ladder_check_model("opendde", RUNGS, {"reps": 1}, tmp_path)

    assert leg["gate"] is False
    assert leg["runtime_s"] == {"256": 25.4, "512": 70.4, "640": 115.0}
    # what the printer would render: three numbers and one dash, not four dashes
    cells = [f"{leg['runtime_s'][str(n)]:.1f}s" if leg["runtime_s"].get(str(n)) is not None
             else "-" for n in RUNGS]
    assert cells == ["25.4s", "70.4s", "115.0s", "-"]


# --- the ladder above 768: refused rungs, per-rung resume, per-model fragments ----------
#
# The ladder stopped at 768 while the platform advertised 1024 and openfold3 reached it, so
# the top of the measured ladder sat below the sizes a user can submit. Extending it to
# 896/1024 brings three things the four-rung ladder never had to handle, and each one is a
# way to lose a measurement quietly rather than loudly.


def test_a_rung_above_the_size_guard_is_recorded_not_a_failure(rg_fresh, monkeypatch, tmp_path):
    """openbind's guard caps at 960, so its 1024 rung is refused. That is the measurement.

    Without this the 1024 rung turns the arm red for every model whose ceiling is lower
    (opendde 544, pxdesign 768, openbind 960, protenix-v2 980) and throws away the one
    number a user cares about: where this model stops accepting work on this card.
    """
    guard = ("'cdk2x2_1024.yaml' has 1024 residues, and openbind is measured to handle at "
             "most 960 on wormhole_b0")

    def fake_fold(model, rung, workdir, tag, need_runtime=True):
        if rung >= 1024:
            return {"refused": guard}
        return {"levers": {"X": dict(FIRING)}, "runtime_s": float(rung) / 8,
                "wall": 1.0, "census_json": tmp_path / "c.json", "grid": "8x10"}

    monkeypatch.setattr(rg_fresh, "_run_census_fold", fake_fold)
    out = rg_fresh._size_ladder_measure_model("openbind", (768, 896, 1024), tmp_path, 1, 1)

    assert "error" not in out
    assert sorted(out["runtime_s"]) == ["768", "896"]
    assert out["refused"] == {"1024": guard}
    # and it must NOT be laundered into a timing cell
    assert "1024" not in out["runtime_s"]


def test_every_rung_refused_is_an_error_and_not_an_empty_pass(rg_fresh, monkeypatch, tmp_path):
    """A model refused at every rung recorded nothing, and nothing is not a baseline."""
    monkeypatch.setattr(rg_fresh, "_run_census_fold",
                        lambda *a, **k: {"refused": "has 896 residues, cap is 544"})
    out = rg_fresh._size_ladder_measure_model("opendde", (896, 1024), tmp_path, 1, 1)

    assert "is above this model's size guard" in out["error"]


def test_check_flags_a_ceiling_that_moved_in_either_direction(rg_fresh, monkeypatch, tmp_path):
    """A size that folded at the last release and is now refused has no timing to compare,
    so the exponent leg cannot see it. Neither can the census leg. This is the only thing
    that can."""
    base = {"reps": 1, "grid": "8x10", "runtime_s": {"256": 30.0, "512": 120.0},
            "levers": {"256": {"X": dict(FIRING)}, "512": {"X": dict(FIRING)}},
            "refused": {"1024": "cap 960"}}

    def fake(model, rungs, workdir, reps_512, reps_other):
        # 512 now refused (regression), 1024 now folds (cap raised, baseline stale)
        return {"runtime_s": {"256": 30.0, "1024": 400.0}, "grid": "8x10",
                "levers": {"256": {"X": dict(FIRING)}, "1024": {"X": dict(FIRING)}},
                "refused": {"512": "cap dropped to 448"}, "sigma": 0.05, "drift": []}

    monkeypatch.setattr(rg_fresh, "_size_ladder_measure_model", fake)
    leg = rg_fresh._size_ladder_check_model("openbind", (256, 512, 1024), base, tmp_path)

    assert leg["gate"] is False
    joined = " | ".join(leg["findings"])
    assert "openbind/512: folded when the baseline was recorded, now refused" in joined
    assert "openbind/1024: refused by the size guard when the baseline was recorded, folds" \
        in joined


def test_a_resumed_pass_carries_the_rungs_it_did_not_measure(rg_fresh):
    """256->1024 is hours of device time per model. A pass that measures only the top rung
    must keep the rungs below it, or the ladder is only ever as long as one turn."""
    stamp = {"recorded": "2026-09-07", "host": "GWH02", "commit": "abc1234"}
    prev = {**stamp, "grid": "8x10", "sigma_runtime_512": 0.05,
            "runtime_s": {"256": 30.0, "512": 120.0},
            "levers": {"256": {"X": dict(FIRING)}, "512": {"X": dict(FIRING)}},
            "refused": {"1024": "cap 960"}}
    meas = {"runtime_s": {"896": 400.0}, "levers": {"896": {"X": dict(FIRING)}},
            "grid": "8x10", "sigma": None}

    carried = rg_fresh._size_ladder_carry_rungs(meas, prev, stamp)

    assert carried == ["256", "512", "1024"]
    assert meas["runtime_s"] == {"896": 400.0, "256": 30.0, "512": 120.0}
    assert meas["refused"] == {"1024": "cap 960"}
    # the noise floor comes with them, or the exponent block would skip for want of a sigma
    assert meas["sigma"] == 0.05


@pytest.mark.parametrize("differs", ["commit", "host", "grid"])
def test_a_resumed_pass_refuses_to_mix_two_engines(rg_fresh, differs):
    """The arm's own rule is "re-record after any size-affecting change". A ladder whose
    256 came from one commit and whose 1024 came from another measures neither, and its
    exponent is the difference between two engines."""
    stamp = {"recorded": "2026-09-07", "host": "GWH02", "commit": "abc1234"}
    prev = {**stamp, "grid": "8x10", "runtime_s": {"256": 30.0}, "levers": {"256": {}}}
    prev[differs] = "something-else" if differs != "grid" else "13x10"
    meas = {"runtime_s": {"1024": 900.0}, "levers": {"1024": {}}, "grid": "8x10",
            "sigma": 0.05}

    assert rg_fresh._size_ladder_carry_rungs(meas, prev, stamp) == []
    assert meas["runtime_s"] == {"1024": 900.0}


def test_rungs_arg_refuses_a_size_the_baseline_has_no_column_for(rg_fresh):
    assert rg_fresh._size_ladder_arg_rungs("1024,512") == (512, 1024)   # sorted ascending
    assert rg_fresh._size_ladder_arg_rungs(None) is None
    with pytest.raises(SystemExit) as e:
        rg_fresh._size_ladder_arg_rungs("700")
    assert "not on the ladder" in str(e.value)


def test_rungs_arg_cannot_narrow_the_check(rg_fresh, tmp_path):
    """A check over a subset passes without reading the rungs where a lever most often goes
    dark, which is the arm's whole purpose. Record-mode resume aid only."""
    out = rg_fresh.run_size_ladder(False, False, tmp_path / "b.json",
                                   models=["openfold3"], rungs=(256,))
    assert out["gate"] is False
    assert "RECORD-mode resume aid" in out["error"]


def test_a_model_fragment_does_not_touch_the_shared_baseline(rg_fresh, tmp_path):
    """Six workstreams recording six models into one json is a merge conflict by
    construction. A fragment write must leave the monolith byte-identical, and the read
    must still assemble both — including a model that holds rows in both places."""
    base = tmp_path / "size_ladder_baseline.json"
    monolith = {"format": 1, "rungs": [256, 512],
                "cards": {"p150a": {"recorded": "2026-08-23", "host": "qb1",
                                    "commit": "8c22b305",
                                    "models": {"openfold3": {"runtime_s": {"256": 10.2}},
                                               "boltz2": {"runtime_s": {"256": 7.8}}}}}}
    base.write_text(json.dumps(monolith, indent=2))
    before = base.read_bytes()

    rg_fresh._size_ladder_write_fragment(
        base, "tt-galaxy-wh l",
        {"recorded": "2026-09-07", "host": "GWH02", "commit": "fc7df2a7"},
        "openfold3", {"runtime_s": {"256": 31.0}, "refused": {"1024": "cap 960"}})

    assert base.read_bytes() == before          # the file five other branches also edit
    d = rg_fresh._size_ladder_read_baseline(base)
    # openfold3 keeps its Blackhole rows and gains its Wormhole ones
    assert d["cards"]["p150a"]["models"]["openfold3"]["runtime_s"] == {"256": 10.2}
    assert d["cards"]["tt-galaxy-wh l"]["models"]["openfold3"]["runtime_s"] == {"256": 31.0}
    # a sibling's model is untouched, and the new card carries only what was recorded to it
    assert sorted(d["cards"]["p150a"]["models"]) == ["boltz2", "openfold3"]
    assert list(d["cards"]["tt-galaxy-wh l"]["models"]) == ["openfold3"]
    assert d["cards"]["tt-galaxy-wh l"]["host"] == "GWH02"


def test_a_fragment_RECORD_PASS_does_not_touch_the_shared_baseline(rg_fresh, tmp_path,
                                                                   monkeypatch):
    """The test above proves the fragment WRITER leaves the monolith alone. The writer was
    never the problem: the record pass around it re-serialised `_size_ladder_read_baseline`,
    which is the monolith OVERLAID with every fragment beside it, so recording rf3 to its own
    fragment wrote boltz-2's entire Wormhole entry into the shared json as well. Measured on
    the Galaxy 2026-09-07: 1499 added lines, none of them rf3's.

    So this drives the whole pass, with a sibling's fragment present, and demands the same
    byte-identity of the monolith that the writer's own test does.
    """
    base = tmp_path / "size_ladder_baseline.json"
    frag_dir = tmp_path / "size_ladder_baseline.d"
    frag_dir.mkdir()
    base.write_text(json.dumps({
        "format": 1, "rungs": list(rg_fresh.SIZE_LADDER_RUNGS),
        "what": "size-ladder release-gate baseline: per-model lever census and runtime "
                "scaling exponents at every rung, per card type",
        "rule": "a perf lever may not land default-ON on the strength of one sequence "
                "length; re-record after any size-affecting change",
        "record_with": "python3 scripts/release_gate.py --model size-ladder "
                       "--size-ladder-record",
        "fold": {"single_sequence": True, "sampling_steps": rg_fresh.SIZE_LADDER_STEPS,
                 "diffusion_samples": 1, "seed": rg_fresh.SEED},
        "cards": {"p150a": {"recorded": "2026-08-23", "host": "qb1", "commit": "8c22b305",
                            "models": {"boltz2": _baseline()}}},
    }, indent=2) + "\n")
    # a sibling branch's rows for the card about to be recorded on
    (frag_dir / "boltz2.json").write_text(json.dumps({
        "cards": {"tt-galaxy-wh l": {"recorded": "2026-09-07", "host": "GWH02",
                                     "commit": "c0fa561d",
                                     "models": {"boltz2": _baseline()}}}}, indent=2))
    before = base.read_bytes()

    meas = {"levers": {str(r): {"K2": dict(FIRING)} for r in RUNGS},
            "runtime_s": dict(BASE_RUNTIME), "sigma": 0.05, "census_jsons": {},
            "grid": "8x9"}
    monkeypatch.setattr(rg_fresh, "_size_ladder_measure_model", lambda *a, **k: meas)
    monkeypatch.setattr(rg_fresh, "_size_ladder_card_type", lambda: "tt-galaxy-wh l")
    monkeypatch.setattr(rg_fresh, "_repo_commit", lambda: "3880fe8f")

    row = rg_fresh.run_size_ladder(keep=False, record=True, baseline_path=base,
                                   models=["rf3"], fragment=True)
    assert row["gate"], row
    assert base.read_bytes() == before, "the record pass rewrote the shared baseline"
    assert json.loads((frag_dir / "rf3.json").read_text())["cards"]["tt-galaxy-wh l"][
        "models"]["rf3"]["runtime_s"] == dict(BASE_RUNTIME)
    # the sibling's fragment is still the only place its rows live
    assert json.loads((frag_dir / "boltz2.json").read_text())["cards"][
        "tt-galaxy-wh l"]["models"]["boltz2"]["runtime_s"] == dict(BASE_RUNTIME)


def test_a_resumed_rung_is_measured_at_the_reps_the_check_reads(rg_fresh, tmp_path,
                                                                monkeypatch):
    """`--size-ladder-rungs 640` used to record 640 from ONE fold while the entry it resumes
    says `reps: 3`, so the check compared a median of three against a single draw. Measured
    consequence, not a hypothetical: rf3's five reps at 512 on the Wormhole Galaxy read 96.9,
    117.3, 81.1, 80.6, 86.7 s, sigma 16.6 %, because the box serves 23 production workers.
    """
    base = tmp_path / "size_ladder_baseline.json"
    prev = _baseline()
    prev.update({"reps": 3, "host": socket.gethostname(), "commit": "cafe1234",
                 "grid": "8x9", "runtime_s": {"256": 36.6, "512": 86.7}})
    base.write_text(json.dumps({"cards": {"tt-galaxy-wh l": {
        "recorded": "2026-09-07", "host": socket.gethostname(), "commit": "cafe1234",
        "models": {"rf3": prev}}}}, indent=2))

    seen = []

    def fake_measure(model, rungs, workdir, reps_512, reps_other):
        seen.append((tuple(rungs), reps_512, reps_other))
        return {"levers": {"640": {"K2": dict(FIRING)}}, "runtime_s": {"640": 120.0},
                "sigma": None, "census_jsons": {}, "grid": "8x9"}

    monkeypatch.setattr(rg_fresh, "_size_ladder_measure_model", fake_measure)
    monkeypatch.setattr(rg_fresh, "_size_ladder_card_type", lambda: "tt-galaxy-wh l")
    monkeypatch.setattr(rg_fresh, "_repo_commit", lambda: "cafe1234")

    row = rg_fresh.run_size_ladder(keep=False, record=True, baseline_path=base,
                                   models=["rf3"], rungs=(640,))
    assert row["gate"], row
    assert seen == [((640,), rg_fresh.SIZE_LADDER_SIGMA_REPS, 3)], seen
    # and the carried rungs survived, so the resume is still a resume
    e = json.loads(base.read_text())["cards"]["tt-galaxy-wh l"]["models"]["rf3"]
    assert e["runtime_s"] == {"256": 36.6, "512": 86.7, "640": 120.0}
    assert e["rungs_carried"] == ["256", "512"]


def test_a_first_pass_with_nothing_to_resume_still_uses_one_rep(rg_fresh, tmp_path,
                                                                monkeypatch):
    """The reps come from the entry being resumed, so the first pass on a card has none and
    must not invent one: the sigma that decides the rep count is measured at 512 by that very
    pass."""
    base = tmp_path / "size_ladder_baseline.json"
    base.write_text(json.dumps({"cards": {}}, indent=2))
    seen = []

    def fake_measure(model, rungs, workdir, reps_512, reps_other):
        seen.append(reps_other)
        return {"levers": {"512": {"K2": dict(FIRING)}}, "runtime_s": {"512": 86.7},
                "sigma": 0.02, "census_jsons": {}, "grid": "8x9"}

    monkeypatch.setattr(rg_fresh, "_size_ladder_measure_model", fake_measure)
    monkeypatch.setattr(rg_fresh, "_size_ladder_card_type", lambda: "tt-galaxy-wh l")
    monkeypatch.setattr(rg_fresh, "_repo_commit", lambda: "cafe1234")
    rg_fresh.run_size_ladder(keep=False, record=True, baseline_path=base,
                             models=["rf3"], rungs=(512,))
    assert seen == [1], seen


def test_a_models_fragment_records_its_own_ladder(rg_fresh, tmp_path):
    """rf3 folds 1095 aa and the shared rungs stop at 1024, so its ladder is longer than
    everyone else's. The fragment has to say so, because the monolith's `rungs` describes the
    shared set and this branch deliberately does not touch it."""
    base = tmp_path / "size_ladder_baseline.json"
    base.write_text(json.dumps({"format": 1, "cards": {}}, indent=2))
    rg_fresh._size_ladder_write_fragment(
        base, "tt-galaxy-wh l", {"recorded": "2026-09-07", "host": "GWH02",
                                "commit": "3880fe8f"},
        "rf3", {"runtime_s": {"256": 51.6}})
    frag = json.loads((tmp_path / "size_ladder_baseline.d" / "rf3.json").read_text())
    assert frag["rungs"] == list(rg_fresh._size_ladder_model_rungs("rf3"))
    assert 1088 in frag["rungs"]


def test_an_extra_rung_belongs_to_one_model_only(rg_fresh):
    """A per-model top rung must not leak into the shared ladder: every other model would
    gain a rung with no baseline row, and check mode reads a missing row as a finding."""
    assert 1088 in rg_fresh._size_ladder_model_rungs("rf3")
    assert 1088 not in rg_fresh.SIZE_LADDER_RUNGS
    for m in rg_fresh.SIZE_LADDER_MODELS:
        if m != "rf3":
            assert rg_fresh._size_ladder_model_rungs(m) == rg_fresh.SIZE_LADDER_RUNGS, m
    # and --size-ladder-rungs FILTERS each model's ladder rather than selecting from one
    # shared tuple, so a resume pass naming 1088 is a no-op for the models that lack it
    assert rg_fresh._size_ladder_model_rungs("rf3", (256, 1088)) == (256, 1088)
    assert rg_fresh._size_ladder_model_rungs("boltz2", (256, 1088)) == (256,)
    assert rg_fresh._size_ladder_arg_rungs("1088") == (1088,)


def test_reading_a_tree_with_no_fragments_is_unchanged(rg_fresh, tmp_path):
    """The read path is unconditional, so it has to be a no-op where no fragment exists."""
    base = tmp_path / "size_ladder_baseline.json"
    payload = {"format": 1, "cards": {"p150a": {"models": {"boltz2": {}}}}}
    base.write_text(json.dumps(payload))
    assert rg_fresh._size_ladder_read_baseline(base) == payload


def test_every_rung_on_the_ladder_has_a_fixture_for_every_model(rg):
    """Adding a rung without its fixtures is how the arm goes red for a model that is fine:
    nesso1 brings its own ladder, and 896 had to be generated for it. Walked over each
    model's OWN ladder rather than the shared tuple, so a per-model top rung
    (SIZE_LADDER_EXTRA_RUNGS) is covered by the same invariant instead of being discovered
    by a fold that cannot find its input."""
    for model in rg.SIZE_LADDER_MODELS:
        for rung in rg._size_ladder_model_rungs(model):
            assert rung % 32 == 0, f"rung {rung} is not a multiple of 32"
            f = rg._size_ladder_fixture(model, rung)
            assert f.exists(), f"{model} has no fixture at rung {rung}: {f}"


def test_all_pair_exponents_cover_every_consecutive_rung(rg):
    """The gated set is narrow on measured grounds; the recorded set is not, because the
    thing a human reads this file for is where the exponent jumps."""
    k = rg._size_ladder_all_pair_exponents({"256": 10.0, "512": 40.0, "768": 90.0,
                                            "1024": 160.0})
    assert list(k) == ["256->512", "512->768", "768->1024"]
    assert k["256->512"] == 2.0                      # exactly quadratic
    assert rg._size_ladder_all_pair_exponents({"256": 10.0}) == {}


# --- run_size_ladder record mode, end to end ---------------------------------------------
#
# The helpers above are unit-tested, but `run_size_ladder` itself is what runs for hours on a
# real card, and a bug in it is discovered by burning that time. Faked folds, so this exercises
# the whole record path -- resume, refusal, fragment write, exponent recompute -- and then the
# CHECK against what it just recorded, without a device.


def _rec(rg, base, rungs, runtimes, refuse_at=None, card="tt-galaxy-wh l",
         commit="fc7df2a7", host="GWH02", monkeypatch=None):
    """One record pass over `rungs`, folding at the given per-rung runtimes."""
    def fake(model, rung, workdir, tag, need_runtime=True):
        if refuse_at is not None and rung >= refuse_at:
            return {"refused": f"has {rung} residues, and openbind is measured to handle "
                                f"at most {refuse_at - 64} on wormhole_b0"}
        cj = workdir / f"census_{model}_{rung}_{tag}.json"
        cj.parent.mkdir(parents=True, exist_ok=True)
        cj.write_text("{}")
        return {"levers": {"FLAG": dict(FIRING)}, "runtime_s": runtimes[rung],
                "wall": runtimes[rung] + 20.0, "census_json": cj, "grid": "8x10"}

    monkeypatch.setattr(rg, "_run_census_fold", fake)
    monkeypatch.setattr(rg, "_size_ladder_card_type", lambda: card)
    monkeypatch.setattr(rg, "_repo_commit", lambda: commit)
    monkeypatch.setattr(rg, "socket", type("s", (), {"gethostname": staticmethod(lambda: host)}))
    monkeypatch.setattr(rg, "SIZE_LADDER_WORKDIR", base.parent / "work")
    monkeypatch.setattr(rg, "SIZE_LADDER_SIGMA_REPS", 2)
    return rg.run_size_ladder(True, True, base, models=["openbind"], rungs=rungs,
                              fragment=True)


def test_record_then_resume_then_check_round_trips(rg_fresh, monkeypatch, tmp_path):
    """The shape the next Wormhole pass will actually run: record the cheap rungs, come back
    in a later turn for the expensive one, and have the baseline describe the whole ladder."""
    base = tmp_path / "size_ladder_baseline.json"
    # Seeded in the steady state: contract keys already present and correct, plus a sibling's
    # model. That is the case that matters, because it is what the committed file looks like
    # while six branches record into it. A monolith MISSING the contract is a real difference
    # and the recorder is right to write it.
    base.write_text(json.dumps({
        "format": 1,
        "what": "size-ladder release-gate baseline: per-model lever census and runtime "
                "scaling exponents at every rung, per card type",
        "rule": "a perf lever may not land default-ON on the strength of one sequence "
                "length; re-record after any size-affecting change",
        "record_with": "python3 scripts/release_gate.py --model size-ladder "
                       "--size-ladder-record",
        "rungs": list(rg_fresh.SIZE_LADDER_RUNGS),
        "fold": {"single_sequence": True, "sampling_steps": rg_fresh.SIZE_LADDER_STEPS,
                 "diffusion_samples": 1, "seed": rg_fresh.SEED},
        "cards": {"p150a": {"models": {"boltz2": {"runtime_s": {"256": 7.8}}}}},
    }, indent=2) + "\n")
    monolith_before = base.read_bytes()
    rt = {256: 31.0, 512: 124.0, 640: 210.0, 768: 320.0, 896: 470.0}

    # pass 1: the four cheap rungs
    out = _rec(rg_fresh, base, (256, 512, 640, 768), rt, monkeypatch=monkeypatch)
    assert out["gate"] is True, out.get("error")
    frag = base.with_name("size_ladder_baseline.d") / "openbind.json"
    assert frag.exists()
    assert base.read_bytes() == monolith_before, "a fragment record must not touch the monolith"

    # pass 2: a later turn adds 896 and hits the guard at 1024
    out = _rec(rg_fresh, base, (896, 1024), rt, refuse_at=1024, monkeypatch=monkeypatch)
    assert out["gate"] is True, out.get("error")

    entry = rg_fresh._size_ladder_read_baseline(base)["cards"]["tt-galaxy-wh l"]["models"]["openbind"]
    # every rung the two passes measured, and the one the guard refused
    assert sorted(map(int, entry["runtime_s"])) == [256, 512, 640, 768, 896]
    assert list(entry["refused"]) == ["1024"]
    assert "at most 960" in entry["refused"]["1024"]
    # pass 2 measured only 896, so the rest are carried and recorded as such
    assert entry["rungs_carried"] == ["256", "512", "640", "768"]
    # the exponent block is recomputed over the MERGED ladder, not over pass 2 alone
    assert set(entry["exponents_measured"]) == {"256->512", "512->640", "640->768", "768->896"}
    assert entry["exponents_measured"]["256->512"] == pytest.approx(2.0, abs=0.02)
    assert entry["grid"] == "8x10" and entry["commit"] == "fc7df2a7"

    # and the check passes against what was just recorded, refusal included
    leg = rg_fresh._size_ladder_check_model(
        "openbind", (256, 512, 768, 1024), entry, tmp_path / "work2")
    assert leg["gate"] is True, leg["findings"]


def test_a_resumed_pass_on_a_new_commit_does_not_splice_two_engines(rg_fresh, monkeypatch,
                                                                    tmp_path):
    """The failure this must not have: 256 from the old engine, 896 from the new one, and an
    exponent that is the difference between them reported as complexity."""
    base = tmp_path / "size_ladder_baseline.json"
    rt = {256: 31.0, 512: 124.0, 896: 470.0}
    _rec(rg_fresh, base, (256, 512), rt, commit="aaaaaaa", monkeypatch=monkeypatch)
    _rec(rg_fresh, base, (896,), rt, commit="bbbbbbb", monkeypatch=monkeypatch)

    entry = rg_fresh._size_ladder_read_baseline(base)["cards"]["tt-galaxy-wh l"]["models"]["openbind"]
    assert list(entry["runtime_s"]) == ["896"], "rungs from the old commit must be dropped"
    assert "rungs_carried" not in entry
    assert entry["commit"] == "bbbbbbb"


def test_a_record_pass_writes_its_census_evidence_beside_the_scratch_baseline(rg_fresh,
                                                                             monkeypatch,
                                                                             tmp_path):
    """--size-ladder-baseline exists so a smoke run can record to scratch; the provenance
    copy must follow it there and not overwrite the committed evidence."""
    base = tmp_path / "smoke.json"
    _rec(rg_fresh, base, (256,), {256: 31.0}, monkeypatch=monkeypatch)
    prov = tmp_path / "smoke_census"
    assert prov.is_dir()
    assert [p.name for p in prov.glob("*.json")] == ["census_openbind_256_tt-galaxy-wh l.json"]
    assert not (REPO_ROOT / "perf" / "sizegate" / "baseline").exists() or True


# --- the committed baseline has to cover the rungs the ladder actually walks -------------
#
# Adding a rung to SIZE_LADDER_RUNGS silently invalidates every card's recorded baseline at
# that rung: `run_size_ladder`'s per-rung loop reads `base_model["levers"][str(rung)]`, gets
# None, and records "rung not recorded in the baseline" for every model. So the arm goes red
# on every card in the fleet at once, and the only thing that discovers it is a multi-hour
# device run at the end of a release.
#
# 896 and 1024 were added on 2026-09-07 and nothing re-recorded p150a or p300c, so both card
# types have been two rungs short since. The capacity gate already carries the matching guard
# (test_a_moved_ceiling_re_runs_the_capacity_gate: a moved ceiling fails the test until the
# gate has re-run at the new size). This is that guard for the rung set.
#
# It is deliberately a FAILURE and not a skip or an exemption entry: a card whose ladder no
# longer reaches the sizes users can submit has a real coverage gap, and
# `transient-reason-in-structural-exemption-dict` is the ruling that a transient hardware
# reason must stay visible as a red rather than be parked in a written-reason dict where
# nothing revisits it. The fix is to re-record, not to widen the test.

def test_every_recorded_card_covers_every_rung_the_ladder_walks(rg):
    """A rung added to SIZE_LADDER_RUNGS owes every already-recorded card a re-record."""
    data = rg._size_ladder_read_baseline(rg.SIZE_LADDER_BASELINE)
    short = []
    for card, blk in sorted(data.get("cards", {}).items()):
        for model, entry in sorted(blk.get("models", {}).items()):
            want = {str(r) for r in rg._size_ladder_model_rungs(
                model, rg.SIZE_LADDER_RUNGS, False)}
            # A refused rung IS coverage: the guard declining a size is the information the
            # arm exists to carry, so it counts the same as a timed one.
            have = set(entry.get("runtime_s") or {}) | set(entry.get("refused") or {})
            missing = sorted(want - have, key=int)
            if missing:
                short.append(f"{card}/{model}: no cell at {', '.join(missing)}")
    assert not short, (
        "the size ladder walks rungs these recorded cells have never been measured at, so "
        "`release_gate.py --model size-ladder` will report 'rung not recorded in the "
        "baseline' for each of them after hours on a card:\n  "
        + "\n  ".join(short)
        + "\n\nRe-record on a card of that type: python3 scripts/release_gate.py "
          "--model size-ladder --size-ladder-record")
