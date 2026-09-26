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

#: How many device callbacks are in flight RIGHT NOW. A sampler reads it to decide whether the
#: card's stream is busy for the millisecond it is looking at. It is a depth and not a flag
#: because XLA:CPU is allowed to run independent thunks concurrently -- if it ever does, this
#: goes above 1 and that is itself the finding. Guarded, because a concurrent enter/exit pair
#: on two threads is exactly the case being measured and `+= 1` is not atomic across them.
DEV_DEPTH = 0
DEV_MAX_DEPTH = 0
_DEV_LOCK = threading.Lock()


def dev_enter():
    global DEV_DEPTH, DEV_MAX_DEPTH
    with _DEV_LOCK:
        DEV_DEPTH += 1
        if DEV_DEPTH > DEV_MAX_DEPTH:
            DEV_MAX_DEPTH = DEV_DEPTH


def dev_exit():
    global DEV_DEPTH
    with _DEV_LOCK:
        DEV_DEPTH -= 1


class Timeline:
    """For every wall millisecond: is the device stream busy, and is a host thread running.

    Two independent reads, neither inferred from the other. `DEV_DEPTH` is the device side.
    `time.process_time_ns()` is CLOCK_PROCESS_CPUTIME_ID -- CPU nanoseconds summed over every
    thread of this process -- so the host side is measured CPU and not "the main thread is not
    in a callback". Differencing two consecutive samples gives cores-busy over the interval,
    which is what separates a host that is computing from one that is blocked on the card.

    Ticks are the wrong instrument for this: /proc/self/stat is 10 ms granular and gap 14 of
    the round is 24 ms. process_time is nanosecond granular and one read costs ~0.4 us.

    The sampler charges its own CPU to `self_cpu_s` off `time.thread_time_ns()`, so the
    instrument's cost is stated rather than differenced out of a round wall -- differencing
    walls across runs on a loaded box is what this campaign has paid for twice.
    """

    def __init__(self, dt_ms=2.0):
        self.dt = dt_ms / 1000.0
        self.t, self.cpu, self.dev = [], [], []
        self.t0 = None
        self.self_cpu_s = 0.0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        c0 = time.thread_time_ns()
        t, cpu, dev = self.t, self.cpu, self.dev
        pt, now, dt = time.process_time_ns, time.time, self.dt
        self.t0 = now()
        t0 = self.t0
        nxt = t0
        while not self._stop.is_set():
            # us since t0, us of process CPU, depth. Ints, because 50k samples a round go to
            # JSON and floats there triple the file for no resolution.
            t.append(int((now() - t0) * 1e6))
            cpu.append(pt() // 1000)
            dev.append(DEV_DEPTH)
            nxt += dt
            slack = nxt - now()
            if slack > 0:
                self._stop.wait(slack)
            else:
                nxt = now()          # overrun: re-base rather than spin to catch up
        self.self_cpu_s = round((time.thread_time_ns() - c0) / 1e9, 4)

    def start(self):
        self._th.start()
        while self.t0 is None:
            time.sleep(0.001)
        return self

    def stop(self):
        self._stop.set()
        self._th.join(timeout=2.0)

    def blob(self):
        # The sampler thread is still appending while a round boundary dumps this. Take the
        # common prefix: the three lists only ever grow and a slice is atomic under the GIL,
        # so a reader gets a consistent prefix rather than three different lengths. Reading a
        # blob whose writer is alive is what cost `bcx-p10-devgap` a pass.
        n = min(len(self.t), len(self.cpu), len(self.dev))
        return {"t0": self.t0, "dt_ms": self.dt * 1000.0, "n": n,
                "self_cpu_s": self.self_cpu_s, "dev_max_depth": DEV_MAX_DEPTH,
                "t_us": self.t[:n], "cpu_us": self.cpu[:n], "dev": self.dev[:n]}


#: Set by the runner when --timeline is on. Dumped beside the events.
TIMELINE = None


def reach():
    """The fused triangle-attention backward's counters, snapshotted at a round boundary.

    Cumulative, so consecutive boundaries subtract to a per-round reach. `bw_calls` is every
    entry into `autograd.triangle_attention`'s backward and is counted before any gate;
    `served` is the fused kernel actually running; `declined` is a gate refusing and falling
    through to the chunked recompute. A round with `bw_calls` high and `served` zero is a
    routing problem, a round with `declined` high is a shape gate, and those point at
    opposite fixes -- which is why all three are stamped and not just the one.
    """
    try:
        from tt_bio import triatt_bw
    except Exception as exc:
        return {"error": repr(exc)}
    return {k: triatt_bw.STATS.get(k, 0) for k in ("bw_calls", "served", "declined")}


def _mm_reach():
    """`TT_BIO_MM_LAYOUT`'s counters at a round boundary, same contract as `reach()`."""
    try:
        from tt_bio import mm_layout
    except Exception as exc:
        return {"error": repr(exc)}
    return mm_layout.reach()


class StopAfterRounds(BaseException):
    """Collection is complete. Raised from the round boundary, never mid-round.

    BaseException, not Exception: BindCraft 2 catches Exception around the compile
    (campaign.py:107) and a swallowed stop would leave the run going.
    """


#: Callables returning a dict of reach counters, stamped into every `round_start` and every
#: `round_stop`. A lever's counters are process totals everywhere else, and a process total
#: cannot tell a round that served 432 from two rounds that served 216 and 648 -- so a lever
#: that fires on half the rounds reads as a lever that fired, and its seconds are a blend of
#: two arms. Differencing consecutive boundaries gives each round its own reach.
REACH: list = []


def _reach():
    out = {}
    for fn in REACH:
        try:
            out.update(fn())
        except Exception as exc:                 # an instrument must not kill a round
            out["reach_error"] = repr(exc)
    return out


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
                           "round": self.entries, "triatt_bw": reach(),
                           "mm_layout": _mm_reach(), "reach": _reach()})
            raise StopAfterRounds(f"{self.rounds} rounds collected")
        # The reach of TT_BIO_MM_LAYOUT at the boundary, cumulative. A per-round count is the
        # difference of two of these, so an arm whose lever serves 0 calls says so per round
        # and not only in a total that a compile round could have carried.
        EVENTS.append({"kind": "round_start", "phase": "round", "t0": time.time(),
                       "round": self.entries, "load1": os.getloadavg()[0],
                       "triatt_bw": reach(), "mm_layout": _mm_reach(),
                       "reach": _reach()})
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
    #    stack. `analyze.py` sums device time across all of them, which is what makes
    #    `host_in_sg` right on any arm without knowing which swaps are on.
    #
    #    `TemplateOnDevice` was missing from this tuple until 2026-09-26 and the composed
    #    arm runs three of its callbacks a round, so its card time was charged to the HOST
    #    column of every reading taken with the template lever on. Enumerate the classes
    #    the module actually defines rather than listing two of three by hand: a stack
    #    that is on the card and not in this tuple reads as host time, and that is the
    #    one failure mode this loop has.
    for module, cls in (("evoformer", splice_mod.EvoformerOnDevice),
                        ("extra_msa", splice_mod.ExtraMsaOnDevice),
                        ("template", splice_mod.TemplateOnDevice)):
        for name in ("_primal", "_taped", "_backward"):
            orig = getattr(cls, name)

            def make(name, orig, module):
                def wrapper(self, *a, **kw):
                    t0 = time.time()
                    dev_enter()
                    try:
                        return orig(self, *a, **kw)
                    finally:
                        dev_exit()
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
    # The timeline goes in its own file: it is 50k samples a round against the event log's few
    # hundred, and a reader that only wants the round walls should not have to parse it.
    if TIMELINE is not None:
        tl = os.path.join(os.path.dirname(path), "timeline.json")
        with open(tl, "w") as fh:
            json.dump(TIMELINE.blob(), fh, separators=(",", ":"))
    with open(path, "w") as fh:
        json.dump({"stamp": {**stamp, "state_shape": dict(STATE),
                             "dumped_utc": time.strftime("%FT%TZ", time.gmtime()),
                             "rounds_dumped": sum(1 for e in EVENTS
                                                  if e["kind"] == "round_start")},
                   "state_shape": STATE, "events": EVENTS,
                   "aiclk": [[t, c, l] for t, c, l in (CLOCK.samples if CLOCK else [])]},
                  fh)
