"""Device kernel time of one RFD4 network call's head and tail, around the trunk, at the fair path's shape.

    perf/rfd4/p512/call_prof.sh <out_dir> <card> [--n 512 --batch 4 --lane 1 --nsteps 2]

Runs `DesignEngine.design` on the frozen pdl1 spec (real features, real lanes) for `--nsteps` sampler steps, under
the device profiler, with signposts around what a call does outside the trunk passes: `upload` (host tensor to the
card, tilized there when that pays), `masks`, `pair_build` (the pair built on the card), `trunk` (Trunk.__call__,
the passes themselves), `readback` (s back to the host, untilized on the card). An op is charged to the innermost
open signpost; `launch` alone is the launch's own glue (padding casts, the masked copies). Only DEVICE KERNEL
DURATION is read. A short `--nsteps` is a census, not a design: every call has the same head and tail, and the
per-design figure is the per-call one times the calls a design makes (60 at thinking_medium, lane width 1).
"""
from __future__ import annotations

import argparse
import functools
import json
import os

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402,F401
from tracy import signpost  # noqa: E402

import clocksample  # noqa: E402


def _bracket(fn, name):
    @functools.wraps(fn)
    def w(*a, **k):
        signpost(f"B|{name}")
        try:
            return fn(*a, **k)
        finally:
            signpost(f"E|{name}")
    return w


def instrument():
    from tt_bio import tenstorrent
    from tt_bio.rfd4 import pair, trunk as T
    T.upload = _bracket(T.upload, "upload")
    for mod, name in ((T, "download"), (T, "download_later"), (tenstorrent, "download")):
        if hasattr(mod, name):
            setattr(mod, name, _bracket(getattr(mod, name), "readback"))
    for cls, meth, name in ((T.Trunk, "launch", "launch"), (T.Trunk, "collect", "collect"),
                            (T.Trunk, "masks_for", "masks"), (T.Trunk, "__call__", "trunk"),
                            (pair.PairOnCard, "__call__", "pair_build")):
        setattr(cls, meth, _bracket(getattr(cls, meth), name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.expanduser("~/rfd4-bundle"))
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lane", type=int, default=1)
    ap.add_argument("--nsteps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E

    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    engine.max_parallel_samples = a.lane
    instrument()
    ex = rfd4.prepare_examples(os.path.expanduser(f"~/rfd4-inputs/pdl1-{a.n}.yaml"), a.out + "/inputs")[0]
    features, _ = rfd4.featurize(ex.cif, seed=1000)
    run = lambda: engine.design(features, num_designs=a.batch, nsteps=a.nsteps, seed=1000, rng="per_design")  # noqa
    run()  # compile
    ttnn.ReadDeviceProfiler(engine._device)
    signpost("elt_prof")
    with clocksample.during(period=0.5) as clk:
        run()
    signpost("elt_end")
    ttnn.synchronize_device(engine._device)
    json.dump({"n": a.n, "batch": a.batch, "lane": a.lane, "nsteps": a.nsteps, "aiclk": clk.summary(),
               "load": os.getloadavg()[0], "card": os.environ.get("TT_VISIBLE_DEVICES")},
              open(a.out + "/run.json", "w"), indent=2)
    ttnn.ReadDeviceProfiler(engine._device)


if __name__ == "__main__":
    main()
