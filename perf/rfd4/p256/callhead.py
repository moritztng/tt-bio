"""Where a lane call goes beyond the trunk kernels: every piece of a design, synced, exclusive times.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/p256/callhead.py --spec <yaml> --weights <bundle> --out <dir> \
        [--width 4] [--reps 2] [--seed 0]

One lane of `--width` designs (`max_parallel_samples = width`), one warm-up batch, then `--reps` timed batches. Each
piece is wrapped with `ttnn.synchronize_device` before and after, so its time is its own host work plus the device
work it enqueued, and nothing overlaps: pieces sum to the wall. Nested pieces are charged exclusively (the pair build
uploads its inputs; those count as upload, not pair). Pieces: prepare, cond (FlowNetwork), heads (FlowGlue), pair
(PairOnCard, the pair built on the card), upload (every host->device tensor), trunk (Trunk.__call__: what the
census profiles, plus its own masked copies and clones), download (readback + untilize), decode (autoencoder);
`rest` is the wall minus all of them (sampler step, padding, glue). The syncs cost a round trip each, so the wall
here is not the lane wall; the per-piece numbers are what count. Compare `trunk` with the census at this clock.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402

import clocksample  # noqa: E402

ACC = defaultdict(float)
CALLS = defaultdict(int)
STACK: list = []
DEV = []


def timed(owner, name, label):
    fn = getattr(owner, name)

    def wrap(*a, **k):
        ttnn.synchronize_device(DEV[0])
        STACK.append(0.0)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            ttnn.synchronize_device(DEV[0])
            dt = time.perf_counter() - t0
            child = STACK.pop()
            ACC[label] += dt - child
            CALLS[label] += 1
            if STACK:
                STACK[-1] += dt
    setattr(owner, name, wrap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from tt_bio import rfd4, tenstorrent as TT
    from tt_bio.rfd4 import engine as E, network as N, pair as P, trunk as T

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    engine.max_parallel_samples = a.width
    DEV.append(engine._device)
    timed(N.FlowNetwork, "prepare", "prepare")
    timed(N.FlowNetwork, "_conditioning", "cond")
    timed(N.FlowGlue, "heads", "heads")
    timed(P.PairOnCard, "__call__", "pair")
    timed(TT, "upload", "upload")
    timed(T, "upload", "upload")
    timed(T, "download", "download")
    timed(T.Trunk, "__call__", "trunk")
    timed(E.DesignEngine, "decode", "decode")

    ex = rfd4.prepare_examples(a.spec, out / "inputs")[0]
    features, _ = rfd4.featurize(ex.cif, seed=a.seed)
    nsteps = engine.spec.nsteps
    sha = os.popen("git rev-parse --short HEAD").read().strip()
    for rep in range(a.reps + 1):
        ACC.clear()
        CALLS.clear()
        with clocksample.during(period=1.0) as clk:
            t0 = time.perf_counter()
            engine.design(features, num_designs=a.width, nsteps=nsteps, seed=a.seed + rep, rng="per_design")
            wall = time.perf_counter() - t0
        cs = clk.summary().get(0) or {}
        per = {k: round(v / a.width, 4) for k, v in sorted(ACC.items())}
        per["rest"] = round((wall - sum(ACC.values())) / a.width, 4)
        rec = {"kind": "warmup" if rep == 0 else "timed", "size": int(features["padding_mask"].shape[-1]),
               "width": a.width, "wall_s_per_design": round(wall / a.width, 4), "s_per_design": per,
               "calls": dict(CALLS), "aiclk_mhz": cs.get("median"), "aiclk_min_mhz": cs.get("min"),
               "load1": round(os.getloadavg()[0], 2), "sha": sha,
               "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec), flush=True)
        with open(out / "callhead.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
