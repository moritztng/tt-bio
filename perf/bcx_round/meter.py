#!/usr/bin/env python3
"""Time one BindCraft 2 gradient ROUND from the loop's own side.

A round is one iteration of `run_gradient_design_stage`'s while loop
(`bindcraft/trajectory.py:124-155`): select the active states, read the optimiser's
schedule parameters, call `design_model.sequence_gradients`, take the loss, maybe update
the sequence, maybe refresh the reference predictions. So the round's wall is the interval
between two consecutive entries into `sequence_gradients`, and everything BindCraft 2 does
between them is inside it, including anything it folds for its own bookkeeping.

Nothing here changes what the campaign computes. Every patch records a timestamp and calls
through; the only control is `--rounds`, which stops collecting after N rounds have run IN
FULL. Rounds are not shortened, recycles are not reduced and no stage is skipped.
"""
import json
import os
import threading
import time

EVENTS = []
_T0 = time.time()

#: The campaign's own residue counts, per round, read off the `protein_states` the predictor is
#: actually handed. A top-level `initialize_design_trajectory` draw is NOT this: the campaign
#: advances its key per trajectory, so probing the settings gave binder 148 for a run whose
#: trajectory 1 was binder 71. Nor is the FIRST call this: the first entry of the exact arm read
#: binder 160 and was followed by a 0.002 s boundary, so something before the first round is
#: handed a different length. n has to come from each round, like every other number here.
STATE = {}


def ev(kind, phase, t0, t1, **kw):
    EVENTS.append({"kind": kind, "phase": phase, "t0": t0, "t1": t1,
                   "dt": round(t1 - t0, 6), **kw})


class Clock:
    """AICLK straight off the card's own sysfs node, 1 s, for the whole run.

    Per round is 10+ samples at a 50 s round, which is what lets each round carry the
    clock it was measured at rather than a run-wide median.
    """

    def __init__(self, dt=1.0):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "bcx_stack"))
        from stack import sysfs_node
        node, self.pci = sysfs_node()
        self.path, self.dt, self.samples = f"{node}/tt_aiclk", dt, []
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.dt):
            try:
                self.samples.append((time.time(),
                                     int(open(self.path).read().split()[0]),
                                     os.getloadavg()[0]))
            except Exception:
                pass

    def start(self):
        self._th.start()
        return self

    def stop(self):
        self._stop.set()

    def window(self, t0, t1):
        got = [(c, l) for t, c, l in self.samples if t0 <= t <= t1]
        if not got:
            return {"n": 0}
        clk = sorted(c for c, _ in got)
        return {"n": len(clk), "aiclk_min": clk[0], "aiclk_med": clk[len(clk) // 2],
                "aiclk_max": clk[-1], "load1": round(sum(l for _, l in got) / len(got), 1)}


CLOCK = None


class StopAfterRounds(BaseException):
    """Collection is complete. Raised from the round boundary, never mid-round.

    BaseException, not Exception: BindCraft 2 catches Exception around the compile
    (campaign.py:107) and a swallowed stop would leave the run going.
    """


#: `(path, stamp)` once the caller has somewhere to write. Set it and every round boundary
#: flushes the whole event log, so a run that dies at round 40 of 100 -- or one that aborts in
#: ttnn's close_device at teardown, which qb2 does -- still leaves every round it finished.
#: A dump at the end of main() is lost by every death there is.
DUMP = None


class Meter:
    def __init__(self, rounds):
        self.rounds = rounds
        self.entries = 0

    # --- the round boundary -------------------------------------------------
    def on_sequence_gradients_enter(self):
        self.entries += 1
        if self.entries > self.rounds:
            # Stamp the boundary BEFORE unwinding: it closes the last round's wall, and
            # the campaign's own finally blocks run between the raise and the dump.
            EVENTS.append({"kind": "round_stop", "phase": "round", "t0": time.time(),
                           "round": self.entries})
            raise StopAfterRounds(f"{self.rounds} rounds collected")
        EVENTS.append({"kind": "round_start", "phase": "round", "t0": time.time(),
                       "round": self.entries, "load1": os.getloadavg()[0]})
        if DUMP:
            dump(*DUMP)


def _shapes(args):
    """The shapes of whatever tensor-like positional arguments a seam was handed."""
    out = []
    for a in args:
        shape = getattr(a, "shape", None)
        if shape is not None:
            try:
                out.append(list(shape))
            except TypeError:
                out.append(str(shape))
    return out


def install(meter, splice_mod, predictor_cls, trajectory_mod, seqopt_mod):
    """Patch the four surfaces a round is made of. All call through."""

    # 1. the device seam: every call the card runs, tagged by which of the three entry
    #    points took it. `_taped` is a forward under the tape and runs once per recycle,
    #    `_backward` is the taped backward, `_primal` is a forward-only fold (a validation
    #    or reference refold). Both device-side stacks carry the same three seams, so the
    #    `module` field is what separates the 48-block Evoformer from the 4-block extra-MSA
    #    stack. `analyze.py` sums device time across both, which is what makes `host_in_sg`
    #    right on either arm without knowing the extra-MSA swap exists.
    for module, cls in (("evoformer", splice_mod.EvoformerOnDevice),
                        ("extra_msa", splice_mod.ExtraMsaOnDevice)):
        for name in ("_primal", "_taped", "_backward"):
            orig = getattr(cls, name)

            def make(name, orig, module):
                def wrapper(self, *a, **kw):
                    t0 = time.time()
                    try:
                        return orig(self, *a, **kw)
                    finally:
                        ev("device", name.lstrip("_"), t0, time.time(),
                           round=meter.entries, module=module, shapes=_shapes(a))
                return wrapper
            setattr(cls, name, make(name, orig, module))

    # 2. the predictor's two entry points. `sequence_gradients` is the round's own call;
    #    `predict` inside a round is a fold BindCraft 2 asked for on top of it.
    sg = predictor_cls.sequence_gradients

    def sequence_gradients(self, protein_states, *a, **kw):
        try:
            STATE[str(meter.entries + 1)] = {
                state: {chain: len(protein) for chain, protein in complex_.items()}
                for state, complex_ in protein_states.items()}
        except Exception as exc:
            STATE[str(meter.entries + 1)] = repr(exc)
        meter.on_sequence_gradients_enter()
        t0 = time.time()
        try:
            return sg(self, protein_states, *a, **kw)
        finally:
            ev("predictor", "sequence_gradients", t0, time.time(),
               round=meter.entries)
    predictor_cls.sequence_gradients = sequence_gradients

    pr = predictor_cls.predict

    def predict(self, *a, **kw):
        t0 = time.time()
        try:
            return pr(self, *a, **kw)
        finally:
            ev("predictor", "predict", t0, time.time(), round=meter.entries)
    predictor_cls.predict = predict

    # 3. the sequence optimiser's own step.
    for cls_name in dir(seqopt_mod):
        cls = getattr(seqopt_mod, cls_name)
        if isinstance(cls, type) and "update_sequence" in cls.__dict__:
            orig = cls.__dict__["update_sequence"]

            def make(orig, cls_name):
                def wrapper(self, *a, **kw):
                    t0 = time.time()
                    try:
                        return orig(self, *a, **kw)
                    finally:
                        ev("optimizer", f"update_sequence:{cls_name}", t0, time.time(),
                           round=meter.entries)
                return wrapper
            setattr(cls, "update_sequence", make(orig, cls_name))

    # 4. stage boundaries, so a round can be attributed to the stage it ran in and the
    #    first round of a stage (which pays for a jit compile) is separable.
    for fn_name in ("run_gradient_design_stage", "run_sequence_mutation_stage"):
        orig = getattr(trajectory_mod, fn_name, None)
        if orig is None:
            continue

        def make(fn_name, orig):
            def wrapper(*a, **kw):
                t0 = time.time()
                EVENTS.append({"kind": "stage_start", "phase": fn_name, "t0": t0,
                               "round": meter.entries})
                try:
                    return orig(*a, **kw)
                finally:
                    ev("stage", fn_name, t0, time.time(), round=meter.entries)
            return wrapper
        setattr(trajectory_mod, fn_name, make(fn_name, orig))


def dump(path, stamp):
    with open(path, "w") as fh:
        json.dump({"stamp": {**stamp, "state_shape": dict(STATE),
                             "dumped_utc": time.strftime("%FT%TZ", time.gmtime()),
                             "rounds_dumped": sum(1 for e in EVENTS
                                                  if e["kind"] == "round_start")},
                   "state_shape": STATE, "events": EVENTS,
                   "aiclk": [[t, c, l] for t, c, l in (CLOCK.samples if CLOCK else [])]},
                  fh)
