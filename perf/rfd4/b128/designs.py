"""Whole RFD4 designs at one sample width: s/design end to end, AICLK sampled during, every CIF written.

    TT_VISIBLE_DEVICES=<card> [TT_BIO_LEVERS=<set>] python perf/rfd4/b128/designs.py --spec <yaml> \
        --weights <bundle> --out <dir> --width 16 [--designs 32] [--seed 0] [--calls 1]

One warm-up call of `--width` designs on seed + 1000 pays the compile and is excluded. Each timed call designs
`--designs` designs on `--seed` with `rng="per_design"`, so design i draws the same noise at every width and the CIFs of
two arms pair up seed for seed (`<id>_d<i>.cif`). Width 1 runs the designs as interleaved lanes, the user's path.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--designs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calls", type=int, default=1)
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    engine.max_parallel_samples = a.width
    ex = rfd4.prepare_examples(a.spec, out / "inputs")[0]
    features, atom_array = rfd4.featurize(ex.cif, seed=a.seed)
    nsteps = engine.spec.nsteps
    t0 = time.perf_counter()
    engine.design(features, num_designs=a.width, nsteps=nsteps, seed=a.seed + 1000, rng="per_design")
    rows = [{"kind": "warmup", "designs": a.width, "wall_s": round(time.perf_counter() - t0, 3)}]
    print(json.dumps(rows[-1]), flush=True)
    for c in range(a.calls):
        with clocksample.during(period=1.0) as clk:
            t0 = time.perf_counter()
            o = engine.design(features, num_designs=a.designs, nsteps=nsteps, seed=a.seed, rng="per_design")
            t_design = time.perf_counter() - t0
        if c == 0:
            for k in range(a.designs):
                rfd4.write_design(o, features, atom_array, out / f"{ex.example_id}_d{k}.cif", row=k)
        cs = clk.summary().get(0) or {}
        rows.append({"kind": "timed", "call": c, "width": a.width, "width_achieved": engine.chunk_width,
                     "designs": a.designs, "design_s": round(t_design, 3),
                     "s_per_design": round(t_design / a.designs, 4), "aiclk_mhz": cs.get("median"),
                     "aiclk_min_mhz": cs.get("min"), "aiclk_samples": cs.get("n", 0),
                     "load1": round(os.getloadavg()[0], 2), "levers": os.environ.get("TT_BIO_LEVERS"),
                     "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(json.dumps(rows[-1]), flush=True)
    (out / "designs.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
