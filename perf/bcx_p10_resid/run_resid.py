#!/usr/bin/env python3
"""bcx-p10-resid leg 1: what is inside the device column that is not a block.

`bcx-p10-devmap` attributed one block and multiplied, and left a 2.603 s residual it named
three candidates for: the seam inside the device column, the pair masks, and the five-trunk
rotation. This runner measures two of the three on the round itself.

The arm is `perf/bcx_round/run_round.py`, unchanged: same campaign, same pool, same predictor,
same meter. Everything here records a timestamp and calls through.

* the seam census, on BOTH device stacks. `bcx-p10-hostmap` wrote this for
  `EvoformerOnDevice` only and the extra-MSA stack was not on card in its arm. The swapped
  program crosses the seam twice per phase, so the extra stack gets its own rows.
* the trunk identity of every device call, and whether that trunk was on card before the call.
  The pool loads lazily (`TrunkPool._load`), so the FIRST fold on each of the five
  `multimer_v3` checkpoints pays a weight upload inside a device call. That is the
  "five trunks, not one" candidate, and it is a per-model one-off or it is not.
"""
import argparse
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
_known, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0], "--out", _known.out] + _rest

#: which crossing is on the stack, so a seam event knows what it is marshalling for.
CALL = ["-"]
POOL = [None]


def _nbytes(x):
    try:
        import numpy as np
        return int(np.asarray(x).nbytes)
    except Exception:
        return 0


def _tensor_bytes(t, width):
    try:
        return int(t.numel()) * width
    except Exception:
        return 0


def _outs(out):
    """Whatever a crossing returned, as a tuple. A bare ndarray is not falsy-testable."""
    if out is None:
        return ()
    return out if isinstance(out, tuple) else (out,)


def census(meter, bindcraft2):
    """Split every device call wall into marshalling, dispatch, sync and trunk load."""
    trunk_cls = bindcraft2._Trunk
    pool_cls = bindcraft2.TrunkPool

    def stamp(tag, t0, nbytes=0, **kw):
        t1 = time.time()
        M.EVENTS.append({"kind": "seam", "phase": tag, "t0": t0, "t1": t1,
                         "dt": round(t1 - t0, 6), "bytes": nbytes,
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

    wrap(trunk_cls, "up", "marshal_in:up", lambda a, out: _tensor_bytes(a[0], 2))
    wrap(trunk_cls, "seed", "marshal_in:seed", lambda a, out: _tensor_bytes(a[0], 2))
    wrap(trunk_cls, "down", "marshal_out:down", lambda a, out: _tensor_bytes(out, 2))
    wrap(trunk_cls, "evoformer", "dispatch:evoformer")
    wrap(trunk_cls, "extra_msa", "dispatch:extra_msa")
    wrap(trunk_cls, "sync", "dispatch:sync")

    # The lazy trunk load: which checkpoint, and whether the card already held it.
    orig_load = pool_cls._load

    def _load(self, name):
        POOL[0] = self
        fresh = name not in self._trunks
        t0 = time.time()
        try:
            return orig_load(self, name)
        finally:
            if fresh:
                stamp("trunk_load", t0, 0, model=name)
    pool_cls._load = _load

    for module, cls in (("evoformer", bindcraft2.EvoformerOnDevice),
                        ("extra_msa", bindcraft2.ExtraMsaOnDevice)):
        wrap(cls, "_inputs", "marshal_in:inputs:" + module,
             lambda a, out: sum(_nbytes(x) for x in a))
        for name in ("_primal", "_taped", "_backward"):
            orig = getattr(cls, name)

            def make(name, orig, module):
                def wrapper(self, *a, **kw):
                    t0, prev = time.time(), CALL[0]
                    CALL[0] = module + ":" + name.lstrip("_")
                    pool = self.pool
                    POOL[0] = pool
                    model = pool._current
                    held = model in pool._trunks
                    out = None
                    try:
                        out = orig(self, *a, **kw)
                        return out
                    finally:
                        CALL[0] = prev
                        t1 = time.time()
                        M.EVENTS.append({
                            "kind": "crossing", "phase": name.lstrip("_"), "module": module,
                            "t0": t0, "t1": t1, "dt": round(t1 - t0, 6),
                            "model": model, "trunk_held": held,
                            "bytes_to_host": sum(_nbytes(x) for x in a),
                            "bytes_to_jax": sum(_nbytes(x) for x in _outs(out)),
                            "round": meter.entries})
                return wrapper
            setattr(cls, name, make(name, orig, module))


def main():
    import run_round
    from tt_bio import bindcraft2

    real_install = M.install

    def install(meter, splice_mod, predictor_cls, trajectory_mod, seqopt_mod):
        # The census wraps FIRST so the meter's `device` event still contains all of it and
        # the device column stays the same number the anchor reported.
        census(meter, bindcraft2)
        real_install(meter, splice_mod, predictor_cls, trajectory_mod, seqopt_mod)
    M.install = install

    real_dump = M.dump

    def dump(path, stamp):
        pool = POOL[0]
        if pool is not None:
            stamp["pool_selections"] = dict(pool.selections)
            stamp["pool_resident"] = pool.resident
            stamp["pool_on_card"] = sorted(pool._trunks)
        return real_dump(path, stamp)
    M.dump = dump

    run_round.main()


if __name__ == "__main__":
    main()
