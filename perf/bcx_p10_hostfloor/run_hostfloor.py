#!/usr/bin/env python3
"""Split the COMPOSED round`s host seconds into intervals that must sum to its wall.

The arm is `perf/bcx_round/run_round.py` with the campaign`s three levers on, unchanged.
This file imports it and calls its `main()`, so the campaign, the pool, the predictor and
the round meter are `bcx-mainstep`s. Everything added here is measurement.

`bcx-p10-hostmap` attributed main`s host side by XLA op_name and left 1.68 s under the
label "outside XLA: Python + JAX dispatch". A residual label is not a measurement, so this
row splits `sequence_gradients` (`bindcraft/af2.py:380-418`) by wall clock instead:

    sg enter -> the compiled call        the Python that builds the arguments
    the compiled call                    XLA`s own execution, callbacks inside it
    the compiled call -> sg exit         the Python that takes the answer apart
    sg exit -> the next round`s entry    BindCraft 2`s own loop body

Those four are disjoint and cover the round by construction, so the table cannot fail to
sum; what it can do is put the seconds in the wrong place, and the named helpers inside
each interval are what guards against that.

Every interval also carries the PROCESS CPU seconds burnt inside it (`os.times`, all
threads). That is what leg 2 needs: host work can only be hidden behind a device call if
the host is not already busy during it, and 3.55 s of the device column is the ttnn
dispatch loop, which is host CPU.
"""
import argparse
import contextlib
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
for _p in (str(_ROOT / "perf" / "bcx_round"), str(_ROOT / "perf" / "bcx_predictor"),
           str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meter as M                                                      # noqa: E402

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--out", required=True)
_ap.add_argument("--profile", default="", help="A,B: jax.profiler over rounds A..B")
_known, _rest = _ap.parse_known_args()
PROFILE = tuple(int(x) for x in _known.profile.split(",")) if _known.profile else ()
TRACE_DIR = os.path.join(_known.out, "trace")
sys.argv = [sys.argv[0], "--out", _known.out] + _rest

MET = [None]
LOCKED = [None]


def cpu():
    """Process CPU seconds, every thread, user + system."""
    t = os.times()
    return t.user + t.system


def stamp(kind, phase, t0, c0, **kw):
    t1, c1 = time.time(), cpu()
    M.EVENTS.append({"kind": kind, "phase": phase, "t0": t0, "t1": t1,
                     "dt": round(t1 - t0, 6), "cpu": round(c1 - c0, 6),
                     "round": MET[0].entries if MET[0] else 0, **kw})


def wrap_fn(mod, name, kind="pyfn"):
    """Time a module-level helper where its CALLER looks it up."""
    orig = getattr(mod, name, None)
    if orig is None:
        return False

    def wrapper(*a, **kw):
        t0, c0 = time.time(), cpu()
        try:
            return orig(*a, **kw)
        finally:
            stamp(kind, name, t0, c0)
    setattr(mod, name, wrapper)
    return True


class Timed:
    """The compiled gradient program, with its dispatch and its completion separated.

    JAX dispatches asynchronously, so the call can return before the program has run and
    the wait then lands wherever the first `float()` of a result happens to be. Splitting
    it here moves nothing: `run_gradient_design_stage` takes `float(graph_design_loss)` two
    statements later, so the block was always inside the round.
    """

    def __init__(self, fn):
        self._fn = fn

    def __getattr__(self, k):
        return getattr(self._fn, k)

    def __call__(self, *a, **kw):
        import jax
        t0, c0 = time.time(), cpu()
        out = self._fn(*a, **kw)
        t1, c1 = time.time(), cpu()
        jax.block_until_ready(out)
        t2, c2 = time.time(), cpu()
        M.EVENTS.append({"kind": "host", "phase": "xla_dispatch", "t0": t0, "t1": t1,
                         "dt": round(t1 - t0, 6), "cpu": round(c1 - c0, 6),
                         "round": MET[0].entries})
        M.EVENTS.append({"kind": "host", "phase": "xla_block", "t0": t1, "t1": t2,
                         "dt": round(t2 - t1, 6), "cpu": round(c2 - c1, 6),
                         "round": MET[0].entries})
        return out


def install_hostfloor():
    import bindcraft.af2 as bc2_af2
    import bindcraft.trajectory as bc2_traj

    # 1. the compiled program. `_compiled_sequence_gradients` is the cache lookup, its
    #    return value is the program, and `.lower().compile()` runs on it inside
    #    `one_worker_compiles` before it is called.
    real_csg = bc2_af2.AlphaFoldDesignModel._compiled_sequence_gradients

    def csg(self, *a, **kw):
        t0, c0 = time.time(), cpu()
        try:
            return Timed(real_csg(self, *a, **kw))
        finally:
            stamp("host", "jit_cache_lookup", t0, c0)
    bc2_af2.AlphaFoldDesignModel._compiled_sequence_gradients = csg

    # BindCraft 2 lowers and compiles on EVERY call (`bindcraft/af2.py:402-403`) inside a
    # per-compile-shape `fcntl.flock`. The lock is replaced by a plain timer here, not
    # wrapped, and that is a bug fix rather than a shortcut: an flock belongs to the OPEN
    # FILE DESCRIPTION, `one_worker_compiles` releases it by closing the file, and
    # `tt_bio/main.py`'s stderr-filter fork leaves a child holding an inherited copy of
    # that description. The lock then outlives the round that took it and the FIRST later
    # round whose compile shape hashes to the same file blocks on it forever -- the
    # process deadlocks against itself, `wchan = locks_lock_inode_wait` at 0 % CPU, which
    # reads exactly like the co-tenant contention the campaign brief warns about and is
    # not that. Seen here on round 10 of 10; `/proc/locks` named 2588686 as both the
    # holder and the blocked waiter and killing the forked child released it.
    # The lock coordinates concurrent BindCraft 2 processes over a SHARED cache directory.
    # This arm has a private one and is the only writer, so there is nothing to coordinate
    # with, and in steady state the whole bracket measures 0.004 s a round.
    real_owc = bc2_af2.one_worker_compiles
    LOCKED[0] = real_owc

    @contextlib.contextmanager
    def owc(shape):
        t0, c0 = time.time(), cpu()
        yield
        stamp("host", "lower_compile", t0, c0)
    bc2_af2.one_worker_compiles = owc

    # 2. the named Python `sequence_gradients` runs either side of the call. Patched in
    #    `bindcraft.af2`s namespace because that is where the caller resolves them.
    got = [n for n in ("collect_shared_chains", "canonical_state_names",
                       "canonical_chain_names", "pad_design_chains",
                       "renamed_protein_states", "renamed_state_losses",
                       "protein_state_shapes", "concatenate_chain_arrays",
                       "frozen_interface_arguments", "split_residue_arrays_by_chain",
                       "trim_padded_metrics", "real_residue_positions")
           if wrap_fn(bc2_af2, n)]

    # 3. BindCraft 2`s own loop body, outside `sequence_gradients` entirely.
    got += [n for n in ("losses_for_active_states", "transfer_binder_sequences",
                        "weighted_design_loss", "passes_design_filters")
            if wrap_fn(bc2_traj, n, kind="loop")]
    return got


def cpu_seams(bindcraft2):
    """CPU burnt inside each device callback, beside the wall the meter already takes."""
    for cls_name in ("EvoformerOnDevice", "ExtraMsaOnDevice"):
        cls = getattr(bindcraft2, cls_name, None)
        if cls is None:
            continue
        for name in ("_primal", "_taped", "_backward"):
            orig = getattr(cls, name, None)
            if orig is None:
                continue

            def make(orig, name, cls_name):
                def wrapper(self, *a, **kw):
                    t0, c0 = time.time(), cpu()
                    try:
                        return orig(self, *a, **kw)
                    finally:
                        stamp("devcpu", f"{cls_name}.{name.lstrip(chr(95))}", t0, c0)
                return wrapper
            setattr(cls, name, make(orig, name, cls_name))


class FloorMeter(M.Meter):
    """`M.Meter` plus the per-round CPU reading and the profiler window.

    The profiled rounds and the unprofiled ones run in the same process, on the same card
    and the same compiled program, so the instrument's own cost is measured, not assumed.
    """

    tracing = False

    def on_sequence_gradients_enter(self):
        import jax
        r = self.entries + 1
        M.EVENTS.append({"kind": "sched", "phase": "round", "t0": time.time(),
                         "round": r, "cpu_total": round(cpu(), 3),
                         "load1": os.getloadavg()[0]})
        if PROFILE and self.tracing and r > PROFILE[1]:
            jax.profiler.stop_trace()
            self.tracing = False
            M.EVENTS.append({"kind": "trace_stop", "phase": "round", "t0": time.time(),
                             "round": r})
        super().on_sequence_gradients_enter()          # may raise StopAfterRounds
        if PROFILE and r == PROFILE[0]:
            jax.profiler.start_trace(TRACE_DIR)
            self.tracing = True
            M.EVENTS.append({"kind": "trace_start", "phase": "round", "t0": time.time(),
                             "round": r})


def main():
    import run_round
    M.Meter = FloorMeter

    from tt_bio import bindcraft2
    real_install = M.install

    def install(meter, splice_mod, predictor_cls, trajectory_mod, seqopt_mod):
        MET[0] = meter
        # inside the meter`s `device` event, so its wall stays the number the campaign quotes
        cpu_seams(bindcraft2)
        real_install(meter, splice_mod, predictor_cls, trajectory_mod, seqopt_mod)
    M.install = install

    wrapped = install_hostfloor()
    print(f"hostfloor wrapped: {wrapped}", flush=True)
    try:
        run_round.main()
    finally:
        try:
            import jax
            jax.profiler.stop_trace()
        except Exception:
            pass


if __name__ == "__main__":
    main()
