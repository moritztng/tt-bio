#!/usr/bin/env python3
"""Attribute a BindCraft 2 gradient round's HOST seconds, at the public configuration.

The arm is `perf/bcx_round/run_round.py`, unchanged. This file imports it and calls its
`main()`, so the campaign, the model pool, the predictor and the round meter are the same
ones `bcx-mainstep` read 32.909 s on. Everything added here is measurement:

* `--profile A,B` wraps rounds A..B in `jax.profiler` and stamps `trace_start`/`trace_stop`
  into the meter's event log. `attrib.py` maps the thunks in that window to AF2 modules.
  Rounds outside the window are the unprofiled control, in the same process, on the same
  card and the same compiled program, so the instrument's own cost is measured, not assumed.
* per round: this process's CPU seconds and run-queue wait over every thread
  (`/proc/self/task/*/schedstat`). The round is 63 % host, so waiting for a CPU is a thing
  that has to be visible.
* `lower_compile`: BindCraft 2 lowers and compiles its gradient program on EVERY call
  (`bindcraft/af2.py:402-403`) and leans on JAX's caches to make that cheap. Cheap is a
  claim, so it gets a timestamp.
* the seam census: for each `pure_callback` crossing, the bytes handed across at the JAX
  face in each direction, and the split of the callback's own wall into marshal-in,
  dispatch, `synchronize_device` and marshal-out.

Nothing here changes what the model computes. Every wrapper records a timestamp and calls
through; `--rounds` stops collection after N rounds have run in full, which is
`run_round.py`'s own control.
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
_ap.add_argument("--profile", default="", help="A,B: jax.profiler over rounds A..B")
_ap.add_argument("--out", required=True)
_known, _rest = _ap.parse_known_args()
PROFILE = tuple(int(x) for x in _known.profile.split(",")) if _known.profile else ()
TRACE_DIR = os.path.join(_known.out, "trace")
sys.argv = [sys.argv[0], "--out", _known.out] + _rest

#: which callback is on the stack, so a marshalling event knows what it is marshalling for.
CALL = ["-"]


def tids():
    return [int(t) for t in os.listdir("/proc/self/task")]


def sched():
    """(CPU seconds, run-queue wait seconds) summed over every thread alive now."""
    run = wait = 0
    for t in tids():
        try:
            a, b, _ = open(f"/proc/self/task/{t}/schedstat").read().split()
        except OSError:
            continue
        run, wait = run + int(a), wait + int(b)
    return run / 1e9, wait / 1e9


class HostmapMeter(M.Meter):
    """`M.Meter` plus the profiler window and the per-round scheduler reading."""

    tracing = False

    def on_sequence_gradients_enter(self):
        import jax
        r = self.entries + 1
        run, wait = sched()
        M.EVENTS.append({"kind": "sched", "phase": "round", "t0": time.time(), "round": r,
                         "cpu_s": round(run, 3), "wait_s": round(wait, 3),
                         "threads": len(tids())})
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


def _nbytes(x):
    try:
        import numpy as np
        return int(np.asarray(x).nbytes)
    except Exception:
        return 0


def _tensor_bytes(t, width):
    """`width` bytes an element, which is what the tensor becomes on the other side."""
    try:
        return int(t.numel()) * width
    except Exception:
        return 0


def seam_census(meter, bindcraft2):
    """Time and size every crossing of the JAX/device seam. Measurement only.

    The meter already times the whole callback and charges it to `device`. That charge
    includes host marshalling -- `_inputs` copies four numpy arrays and pads them,
    `_Trunk.up` casts to bfloat16 and uploads, `_Trunk.down` brings the answer back and
    casts to float32 -- none of which is device arithmetic. This splits the callback's wall
    so the marshalling can be named on its own row.
    """
    trunk_cls = bindcraft2._Trunk
    evo_cls = bindcraft2.EvoformerOnDevice

    def stamp(tag, t0, nbytes=0, **kw):
        M.EVENTS.append({"kind": "seam", "phase": tag, "t0": t0, "t1": time.time(),
                         "dt": round(time.time() - t0, 6), "bytes": nbytes,
                         "call": CALL[0], "round": meter.entries, **kw})

    def wrap(cls, name, tag, size=lambda a, out: 0):
        orig = getattr(cls, name)

        def wrapper(self, *a, **kw):
            t0 = time.time()
            out = None
            try:
                out = orig(self, *a, **kw)
                return out
            finally:
                stamp(tag, t0, size(a, out))
        setattr(cls, name, wrapper)

    # host -> device: a float32 torch tensor becomes a bfloat16 device tensor.
    wrap(trunk_cls, "up", "marshal_in:up", lambda a, out: _tensor_bytes(a[0], 2))
    wrap(trunk_cls, "seed", "marshal_in:seed", lambda a, out: _tensor_bytes(a[0], 2))
    # device -> host.
    wrap(trunk_cls, "down", "marshal_out:down", lambda a, out: _tensor_bytes(out, 2))
    # numpy -> torch -> padded, entirely on host.
    wrap(evo_cls, "_inputs", "marshal_in:inputs",
         lambda a, out: sum(_nbytes(x) for x in a))
    # the dispatch loop, and the wait for the card.
    wrap(trunk_cls, "evoformer", "dispatch:evoformer")
    wrap(trunk_cls, "sync", "dispatch:sync")

    # the crossing itself: what JAX hands over and what it gets back.
    for name in ("_primal", "_taped", "_backward"):
        orig = getattr(evo_cls, name)

        def make(name, orig):
            def wrapper(self, *a, **kw):
                t0, prev = time.time(), CALL[0]
                CALL[0] = name.lstrip("_")
                out = None
                try:
                    out = orig(self, *a, **kw)
                    return out
                finally:
                    CALL[0] = prev
                    into = sum(_nbytes(x) for x in a)
                    back = sum(_nbytes(x) for x in (out or ()))
                    M.EVENTS.append({"kind": "crossing", "phase": name.lstrip("_"),
                                     "t0": t0, "t1": time.time(),
                                     "dt": round(time.time() - t0, 6),
                                     "bytes_to_host": into, "bytes_to_jax": back,
                                     "round": meter.entries})
            return wrapper
        setattr(evo_cls, name, make(name, orig))


def main():
    import run_round

    # `run_round.main` builds its meter as `M.Meter(args.rounds)`, so the class it picks up
    # is whatever `meter.Meter` names at that moment.
    M.Meter = HostmapMeter

    from tt_bio import bindcraft2
    real_install = M.install

    def install(meter, splice_mod, predictor_cls, trajectory_mod, seqopt_mod):
        # The census wraps first so the meter's `device` event contains it, which keeps the
        # callback's wall the same number `bcx-mainstep` reported.
        seam_census(meter, bindcraft2)
        real_install(meter, splice_mod, predictor_cls, trajectory_mod, seqopt_mod)
    M.install = install

    import bindcraft.af2 as bc2_af2
    real_owc = bc2_af2.one_worker_compiles

    @contextlib.contextmanager
    def owc(shape):
        t0 = time.time()
        with real_owc(shape):
            yield
        M.EVENTS.append({"kind": "host", "phase": "lower_compile", "t0": t0,
                         "t1": time.time(), "dt": round(time.time() - t0, 6)})
    bc2_af2.one_worker_compiles = owc

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
