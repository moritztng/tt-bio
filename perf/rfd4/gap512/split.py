"""(wk/rfd4-perf-lanes4 split.py, plus designs.pt for byte identity.)
Where the wall goes between batched trunk calls: a timeline of `run_lanes` at one width.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/lanes4/split.py --spec <yaml> --weights <bundle> --out <dir> \
        --width 4 [--designs 32] [--seed 0]

Same call as `perf/rfd4/b128/designs.py` (one warm-up call excluded, then `--designs` designs at `--width`), with every
stage of every network call timed on the host: prepare (glue + init repr), upload (Trunk.launch up to the first trunk
op), enqueue (the trunk's op dispatch), cond (next step's conditioning, inside launch), wait (collect blocked on the
card + readback), heads, and the sampler's step (the rest). The card is idle from one call's readback until the next
call's first op is enqueued; `gaps` sums exactly that. Writes `timeline.json` and prints a per-call summary.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402

EV: list = []


def timed(owner, name, label):
    fn = getattr(owner, name)

    def wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            EV.append((label, t0, time.perf_counter()))
    setattr(owner, name, wrap)


def summarize(ev, designs, wall):
    ev = sorted(ev, key=lambda e: e[1])
    tot = {}
    for lab, a, b in ev:
        tot[lab] = tot.get(lab, 0.0) + (b - a)
    # card idle: from the end of each `wait` (readback returned) to the start of the next `enqueue`
    gaps, last_wait_end = [], None
    for lab, a, b in ev:
        if lab == "wait":
            last_wait_end = b
        elif lab == "enqueue" and last_wait_end is not None:
            gaps.append(a - last_wait_end)
            last_wait_end = None
    calls = sum(1 for e in ev if e[0] == "enqueue")
    out = {"designs": designs, "wall_s": round(wall, 3), "calls": calls,
           "s_per_design": round(wall / designs, 4),
           "stage_s_per_design": {k: round(v / designs, 4) for k, v in sorted(tot.items())},
           "gap_s_per_design": round(sum(gaps) / designs, 4),
           "gap_ms_per_call_median": round(1e3 * sorted(gaps)[len(gaps) // 2], 2) if gaps else None,
           "wait_s_per_design": round(tot.get("wait", 0) / designs, 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--designs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E, network as N, trunk as T

    timed(N.FlowNetwork, "prepare", "prepare")
    timed(N.FlowNetwork, "_conditioning", "cond")
    timed(N.FlowGlue, "heads", "heads")
    timed(T.Trunk, "launch", "launch")
    timed(T.Trunk, "__call__", "enqueue")
    timed(T.Trunk, "collect", "wait")
    timed(E.DesignEngine, "decode", "decode")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    engine.max_parallel_samples = a.width
    ex = rfd4.prepare_examples(a.spec, out / "inputs")[0]
    features, _ = rfd4.featurize(ex.cif, seed=a.seed)
    nsteps = engine.spec.nsteps
    engine.design(features, num_designs=a.width, nsteps=nsteps, seed=a.seed + 1000, rng="per_design")
    EV.clear()
    with clocksample.during(period=1.0) as clk:
        t0 = time.perf_counter()
        output = engine.design(features, num_designs=a.designs, nsteps=nsteps, seed=a.seed, rng="per_design")
        wall = time.perf_counter() - t0
    torch.save(output, out / "designs.pt")  # byte identity across arms
    cs = clk.summary().get(0) or {}
    s = summarize(EV, a.designs, wall)
    s.update(width=a.width, width_achieved=engine.chunk_width, aiclk_mhz=cs.get("median"),
             aiclk_min_mhz=cs.get("min"), load1=round(os.getloadavg()[0], 2),
             aiclk_mean_mhz=round(sum(clk.clocks.get(0) or [0]) / max(1, len(clk.clocks.get(0) or [])), 1),
             t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(json.dumps(s), flush=True)
    (out / "timeline.json").write_text(json.dumps({"summary": s, "t0": t0,
                                                   "events": [(l, x - t0, y - t0) for l, x, y in EV],
                                                   "aiclk": clk.clocks.get(0), "load": clk.load}))


if __name__ == "__main__":
    main()
