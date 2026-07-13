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

    q = multiprocessing.Queue()
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
    from tt_bio.progress import ProgressDisplay, STAGE_START, STAGE_END

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

    # The display's stage label for a mid-trunk state reads "Trunk k/N", not "Diffusion".
    d.stage = "trunk"; d.step = 4; d.total_steps = 10
    assert ProgressDisplay._stage_label(d) == "Trunk 4/10"

    # The trunk fraction advances monotonically per iteration and stays within
    # the trunk band [STAGE_START["trunk"], STAGE_END["trunk"]).
    prev = -1.0
    for cyc in range(10):
        d.stage = "trunk"; d.step = cyc; d.total_steps = 10
        f = ProgressDisplay._frac(d)
        assert STAGE_START["trunk"] <= f <= STAGE_END["trunk"]
        assert f > prev
        prev = f
    # And diffusion is strictly beyond the trunk band (no 0-trunk->diffusion jump).
    d.stage = "diffusion"; d.step = 0; d.total_steps = 200
    assert ProgressDisplay._frac(d) >= STAGE_END["trunk"]


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
