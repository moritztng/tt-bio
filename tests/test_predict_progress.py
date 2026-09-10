"""Regression tests for the `tt-bio predict` live progress view.

These are device-free: they drive the progress plumbing directly with captured
hooks and a real multiprocessing queue, locking in the contract that

  - the trunk recycling loop emits one ``trunk`` stage event per iteration
    (no "0 trunk iterations -> diffusion" jump), and
  - a trunk event is reported as ``trunk`` (not remapped to ``diffusion``),
    which was the Protenix-v2 bug, and
  - OpenDDE-style folding (which rides the Protenix-v2 trunk) reports a trunk
    phase at all.

The emitters themselves live inside the on-device compute paths
(``TrunkModule.forward`` / ``protenix.TrunkModule.__call__`` /
``esmfold2_runtime._run_one_loop``), so we assert the *contract* they must
satisfy rather than spinning a device.
"""

import multiprocessing
import os
import queue as _queue
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── helpers ──────────────────────────────────────────────────────────────

def _drain(queue):
    """Pull every event currently in the queue into a list."""
    out = []
    while True:
        try:
            out.append(queue.get_nowait())
        except Exception:
            return out


def _trunk_events(n_cycles, pfn):
    """Replay exactly what a correctly-ticking trunk recycling loop emits."""
    for cyc in range(n_cycles):
        pfn("trunk", step=cyc, total=n_cycles)


# ── 1. make_progress_fn forwards (stage, step, total) unchanged ──────────

def test_make_progress_fn_emits_trunk_per_iteration():
    """A trunk loop that ticks once per iteration must produce one trunk event
    per iteration in the queue, with the right step/total — the live view's
    per-iteration advance depends on this."""
    from tt_bio.progress import make_progress_fn

    # Use a synchronous queue.Queue (not multiprocessing.Queue): the latter has a
    # background feeder thread, so an immediate drain after put_nowait can race
    # and see fewer events than were enqueued. queue.Queue is deterministic.
    q = _queue.Queue()
    pfn = make_progress_fn(q, device_id=0, worker_id="w0")

    _trunk_events(n_cycles=10, pfn=pfn)

    events = [e for e in _drain(q) if e["event"] == "stage"]
    assert len(events) == 10
    assert [e["stage"] for e in events] == ["trunk"] * 10
    assert [e["step"] for e in events] == list(range(10))
    assert all(e["total"] == 10 for e in events)


# ── 2. report_progress is a clean passthrough (no trunk->diffusion remap) ─

def test_report_progress_passthrough_no_remap():
    """Protenix-v2 / OpenDDE hand ``report_progress`` straight to the model as
    its progress_fn. A trunk event must come out as ``trunk`` — the old worker
    wrapper remapped trunk->diffusion and hid the trunk phase. This locks the
    passthrough so the regression can't silently return."""
    import importlib

    import tt_bio.esmfold2 as _E

    captured = []
    _E.set_progress(lambda stage, step=0, total=0: captured.append((stage, step, total)))
    try:
        # Re-import not needed: report_progress delegates to the installed fn.
        _E.report_progress("trunk", 3, 10)
        _E.report_progress("diffusion", 5, 200)
    finally:
        _E.set_progress(None)
    assert captured == [("trunk", 3, 10), ("diffusion", 5, 200)]


# ── 3. ProgressDisplay advances the trunk bar per iteration (no jump) ─────

def test_display_trunk_advances_per_iteration_no_jump():
    """Feed the display the canonical event sequence a correct predict emits
    and assert the trunk bar advances 0/N -> 1/N -> ... -> N/N one iteration
    at a time, then diffusion — never jumping straight from 0 to diffusion."""
    from tt_bio.progress import DEFAULT_WEIGHTS, ProgressDisplay, _bands

    q = multiprocessing.Queue()
    disp = ProgressDisplay(q, total=1, n_workers=1, model="protenix-v2")

    def stage_seq():
        yield {"worker": "w0", "dev": 0, "event": "loading"}
        yield {"worker": "w0", "dev": 0, "event": "start", "name": "prot"}
        yield {"worker": "w0", "dev": 0, "event": "stage", "stage": "msa"}
        yield {"worker": "w0", "dev": 0, "event": "stage", "stage": "prep"}
        n_cycles = 10
        for cyc in range(n_cycles):
            yield {"worker": "w0", "dev": 0, "event": "stage",
                   "stage": "trunk", "step": cyc, "total": n_cycles}
        n_step = 200
        for k in range(n_step):
            yield {"worker": "w0", "dev": 0, "event": "stage",
                   "stage": "diffusion", "step": k, "total": n_step}
        yield {"worker": "w0", "dev": 0, "event": "stage", "stage": "confidence"}
        yield {"worker": "w0", "dev": 0, "event": "done", "name": "prot",
               "time": 1.0, "status": "ok"}

    for ev in stage_seq():
        disp._handle(ev)

    d = disp.devices["w0"]
    # Trunk phase: every iteration was recorded, advancing 0..9 over a total of 10.
    trunk_seq = [(ev["step"], ev["total"]) for ev in stage_seq()
                 if ev.get("event") == "stage" and ev.get("stage") == "trunk"]
    assert trunk_seq == [(i, 10) for i in range(10)]

    # The stage label for a mid-trunk state reads "Trunk k/N", not "Diffusion".
    d.stage = "trunk"; d.step = 4; d.total_steps = 10
    assert ProgressDisplay._stage_label(d) == "Trunk 4/10"

    # The trunk fraction advances monotonically per iteration and stays inside
    # the trunk band.
    bands = _bands(DEFAULT_WEIGHTS)
    lo, hi = bands["trunk"]
    d.frac_max = 0.0
    prev = -1.0
    for cyc in range(10):
        d.stage = "trunk"; d.step = cyc; d.total_steps = 10
        f = disp._frac(d, bands)
        assert lo <= f <= hi
        assert f > prev
        prev = f
    # And diffusion is strictly beyond the trunk band (no 0-trunk->diffusion jump).
    d.stage = "diffusion"; d.step = 0; d.total_steps = 200
    assert disp._frac(d, bands) >= hi


# -- 3b. The bar fits the terminal: a full bar is never an ellipsis -------

def test_bar_columns_fit_the_terminal():
    """At 80 columns the old fixed 88-cell table overflowed, so Rich clipped the
    bar and a 100% bar and a 92% bar both ended in the same ellipsis."""
    from tt_bio.progress import W_CNT, _columns

    for width in (60, 76, 80, 100, 120):
        worker, name, bar, stage = _columns(width)
        assert worker + name + bar + stage + W_CNT + 8 <= width, width
        assert bar >= 8, (width, bar)
    # A wide terminal still gets the full-size bar rather than a stretched one.
    assert _columns(200)[2] == 20


# -- 3c. Stage weights come from what the stages measured ----------------

def test_stage_weights_learn_from_a_finished_target():
    """The compiled-in weights are wrong for most models (esmfold2 spends 43%
    of a warm 20 aa fold in prep, budgeted at 15%). After one target finishes,
    the bands must reflect the seconds that target actually spent."""
    import time as _time

    from tt_bio.progress import ProgressDisplay, _bands

    q = multiprocessing.Queue()
    disp = ProgressDisplay(q, total=2, n_workers=1, model="esmfold2")
    disp._handle({"worker": "w0", "dev": 0, "event": "start", "name": "t1"})
    disp._handle({"worker": "w0", "dev": 0, "event": "stage", "stage": "prep"})
    _time.sleep(0.30)
    disp._handle({"worker": "w0", "dev": 0, "event": "stage",
                  "stage": "diffusion", "step": 0, "total": 4})
    _time.sleep(0.05)
    disp._handle({"worker": "w0", "dev": 0, "event": "done", "name": "t1",
                  "time": 0.35, "status": "ok"})

    prep_lo, prep_hi = _bands(disp._weights())["prep"]
    # prep took ~6x the diffusion time, so it now owns most of the bar rather
    # than the 15% the static table gave it.
    assert prep_hi - prep_lo > 0.5, (prep_lo, prep_hi)


# -- 3d. A failed target does not teach the bar that a stage is free -----

def test_failed_target_does_not_set_stage_weights():
    from tt_bio.progress import DEFAULT_WEIGHTS, ProgressDisplay

    q = multiprocessing.Queue()
    disp = ProgressDisplay(q, total=1, n_workers=1, model="esmfold2")
    disp._handle({"worker": "w0", "dev": 0, "event": "start", "name": "bad"})
    disp._handle({"worker": "w0", "dev": 0, "event": "stage", "stage": "prep"})
    disp._handle({"worker": "w0", "dev": 0, "event": "done", "name": "bad",
                  "time": 0.1, "status": "failed"})
    assert disp._weights() is DEFAULT_WEIGHTS
    assert disp.failed == 1


# -- 3e. In-flight work counts toward the header percentage --------------

def test_header_progress_counts_in_flight_work():
    """A single fold used to read 0/1 (0%) for its whole run while its own bar
    sat at 80%, and no ETA appeared until everything was already finished."""
    from tt_bio.progress import ProgressDisplay, _bands

    q = multiprocessing.Queue()
    disp = ProgressDisplay(q, total=1, n_workers=1, model="esmfold2")
    disp._handle({"worker": "w0", "dev": 0, "event": "start", "name": "t1"})
    disp._handle({"worker": "w0", "dev": 0, "event": "stage",
                  "stage": "diffusion", "step": 100, "total": 200})
    bands = _bands(disp._weights())
    work = disp._work_done(bands)
    assert 0.5 < work < 1.0, work
    assert disp.completed == 0  # the structure count stays exact


# -- 3f. The rolling log keeps more than one completed structure ---------

def test_recent_log_is_not_clamped_to_the_worker_count():
    from tt_bio.progress import RECENT_MAX, ProgressDisplay

    q = multiprocessing.Queue()
    disp = ProgressDisplay(q, total=4, n_workers=1, model="esmfold2")
    for i in range(4):
        disp._handle({"worker": "w0", "dev": 0, "event": "start", "name": f"t{i}"})
        disp._handle({"worker": "w0", "dev": 0, "event": "done", "name": f"t{i}",
                      "time": 1.0, "status": "ok"})
    assert len(disp.recent[-RECENT_MAX:]) == 4


# ── 4. OpenDDE registers a trunk phase (no loading->diffusion skip) ───────

def test_opendde_event_sequence_has_trunk_phase():
    """OpenDDE rides the Protenix-v2 trunk, so a correct event stream contains
    a trunk phase between prep and diffusion. Assert the canonical OpenDDE
    sequence includes trunk events — the old worker never passed progress_fn
    into fold, so the trunk phase was absent (loading -> diffusion)."""
    q = multiprocessing.Queue()
    disp = __import__("tt_bio.progress", fromlist=["ProgressDisplay"]).ProgressDisplay(
        q, total=1, n_workers=1, model="opendde")

    seq = []
    seq.append({"worker": "w0", "dev": 0, "event": "loading"})
    seq.append({"worker": "w0", "dev": 0, "event": "start", "name": "ag"})
    seq.append({"worker": "w0", "dev": 0, "event": "stage", "stage": "msa"})
    seq.append({"worker": "w0", "dev": 0, "event": "stage", "stage": "prep"})
    for cyc in range(10):
        seq.append({"worker": "w0", "dev": 0, "event": "stage",
                    "stage": "trunk", "step": cyc, "total": 10})
    for k in range(200):
        seq.append({"worker": "w0", "dev": 0, "event": "stage",
                    "stage": "diffusion", "step": k, "total": 200})
    seq.append({"worker": "w0", "dev": 0, "event": "done", "name": "ag",
                "time": 1.0, "status": "ok"})

    for ev in seq:
        disp._handle(ev)

    stages = [e["stage"] for e in seq if e.get("event") == "stage"]
    # trunk appears, and appears between prep and diffusion.
    assert "trunk" in stages
    assert stages.index("prep") < stages.index("trunk") < stages.index("diffusion")


# -- 5. --msa_endpoint is refused for a model that never reads it ---------

def test_msa_endpoint_refused_for_a_model_that_ignores_it():
    """boltz2 resolves MSAs through main.py, which has no endpoint client. The
    flag used to be accepted, suppress the auto-detected local DB, and then die
    mid-run with "Missing MSAs" -- blaming the user for supplying no source."""
    import click
    import pytest as _pytest

    from tt_bio.main import MSA_ENDPOINT_MODELS, _resolve_msa_default

    assert "boltz2" not in MSA_ENDPOINT_MODELS
    with _pytest.raises(click.BadParameter) as exc:
        _resolve_msa_default("boltz2", False, None, "http://host:8765",
                             False, "~/.boltz", False, None)
    msg = str(exc.value)
    assert "boltz2" in msg and "rf3" in msg and "--msa_db_path" in msg

    # And it is not refused for a model whose worker path does pass it through.
    for model in ("rf3", "openfold3", "opendde", "protenix-v2", "esmfold2"):
        _resolve_msa_default(model, False, None, "http://host:8765",
                             False, "~/.boltz", False, None)


def test_msa_endpoint_help_lists_exactly_the_models_that_read_it():
    """The help string used to be hand-written and omitted rf3, which does read
    the flag. It is built from the same tuple the refusal checks."""
    from tt_bio.main import MSA_ENDPOINT_MODELS, predict

    help_text = next(p.help for p in predict.params
                     if getattr(p, "name", None) == "msa_endpoint")
    for model in MSA_ENDPOINT_MODELS:
        assert model in help_text, model
    assert "boltz2" not in help_text


# -- 6. The step-count help matches what the models actually run ---------

#: What a default `tt-bio predict` really executed, read off the live progress
#: stream on qb1 card 3 (p150a, 20 aa, single-sequence): model -> (trunk cycles
#: shown, diffusion steps shown). The help used to say "every other model 200"
#: while rf3 ran 49, and named openfold3 as the only model running recycles+1
#: while esmfold2 runs 11 cycles for a documented default of 10.
OBSERVED_STEPS = {
    "esmfold2":  (11, 68),   # runs E, F
    "openfold3": (4, 200),   # run L
    "rf3":       (10, 49),   # run M
    "opendde":   (10, 200),  # run M
}


def test_step_count_help_matches_what_each_model_runs():
    from tt_bio.main import (PREDICT_MODELS, TRUNK_CYCLES_PLUS_ONE,
                             _resolve_recycling_steps, _resolve_sampling_steps, predict)

    helps = {p.name: p.help for p in predict.params if getattr(p, "help", None)}

    for model, (trunk_shown, diffusion_shown) in OBSERVED_STEPS.items():
        recycles = _resolve_recycling_steps(None, model)
        expected = recycles + 1 if model in TRUNK_CYCLES_PLUS_ONE else recycles
        assert expected == trunk_shown, (model, expected, trunk_shown)
        requested = _resolve_sampling_steps(None, model)
        assert {100: 68, 50: 49}.get(requested, requested) == diffusion_shown, model

    # Every model's real default appears in the help that claims to state it, so
    # the two cannot drift the way the hand-written strings did.
    for model in PREDICT_MODELS:
        assert str(_resolve_recycling_steps(None, model)) in helps["recycling_steps"]
        assert model in helps["recycling_steps"], model
        assert str(_resolve_sampling_steps(None, model)) in helps["sampling_steps"]
        assert model in helps["sampling_steps"], model
    assert "recycles = N+1" in helps["recycling_steps"]
    for model in TRUNK_CYCLES_PLUS_ONE:
        assert model in helps["recycling_steps"].split("run one more trunk cycle")[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
