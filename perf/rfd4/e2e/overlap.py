"""Designs per second for several designs in one engine call, with the clock and the host load.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/e2e/overlap.py --spec <spec.yaml> --weights <bundle> \
        --out <dir> --tag <side> [--designs 4] [--width 1] [--reps 2] [--seed 0]

One engine, `--reps` calls of `engine.design(num_designs=--designs)` at a chunk width of `--width`
(`max_parallel_samples`), per-design noise, every design written. The first call compiles and is
recorded as such. Run from another tree with PYTHONPATH pointing at it to time that tree's engine:
the harness reads nothing but the public engine and writer, so the same file times both sides of
an A/B. Each design's output tensors are saved (`<tag>_r<rep>.pt`) so the two sides can be compared
bit for bit.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402

CLOCK_FLOOR_MHZ = 1200
FIELDS = ("coords_nm", "residue_type", "atom_mask", "insertion_logits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--designs", type=int, default=4)
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", action="store_true",
                    help="time FlowNetwork.prepare/launch/finish (host wall in each stage, summed)")
    a = ap.parse_args()

    import tt_bio
    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "overlap.jsonl", "a")
    tree = Path(tt_bio.__file__).resolve().parents[1]
    pinned = tree / ".sha"  # a `git archive` tree sits inside another checkout; it names its own sha
    sha = pinned.read_text().strip() if pinned.exists() else \
        os.popen(f"git -C {tree} rev-parse --short HEAD 2>/dev/null").read().strip()
    stages = _split_stages() if a.split else None

    designs = rfd4.prepare_examples(a.spec, out / "inputs")
    features, atom_array = rfd4.featurize(designs[0].cif, seed=a.seed)
    engine = E.DesignEngine(a.weights, recipe="thinking_medium", max_parallel_samples=a.width)

    for rep in range(a.reps):
        if stages is not None:
            stages.clear()
        load0 = os.getloadavg()[0]
        with clocksample.during(period=1.0) as clk:
            t0 = time.perf_counter()
            output = engine.design(features, num_designs=a.designs, nsteps=engine.spec.nsteps,
                                   seed=a.seed, rng="per_design")
            t_design = time.perf_counter() - t0
            t1 = time.perf_counter()
            for row in range(a.designs):  # one design at a time: the writer reads features row 0
                one = E.DesignOutput(*(getattr(output, k)[row:row + 1] for k in FIELDS))
                rfd4.write_design(one, features, atom_array, out / f"{a.tag}_s{a.seed}_r{rep}_d{row}.cif")
            t_write = time.perf_counter() - t1
        torch.save({k: getattr(output, k) for k in FIELDS},
                   out / f"{a.tag}_r{rep}.pt")
        summ = clk.summary()
        dev = (summ.get(0) or summ.get("0") or next(iter(summ.values()), None)) if summ else None
        mhz = dev.get("median") if dev else None
        wall = t_design + t_write
        rec = {"kind": "compile" if rep == 0 else "warm", "tag": a.tag, "tree": str(tree), "sha": sha,
               "size": int(features["padding_mask"].shape[-1]), "designs": a.designs, "width": a.width,
               "recipe": "thinking_medium", "steps": engine.spec.nsteps,
               "wall_s": round(wall, 3), "design_s": round(t_design, 3), "write_s": round(t_write, 3),
               "s_per_design": round(wall / a.designs, 3), "designs_per_s": round(a.designs / wall, 4),
               "aiclk_mhz": mhz, "aiclk": summ, "clock_artifact": bool(mhz is not None and mhz < CLOCK_FLOOR_MHZ),
               "load1": round(load0, 2), "load1_end": round(os.getloadavg()[0], 2), "nproc": os.cpu_count(),
               "card": os.environ.get("TT_VISIBLE_DEVICES"), "seed": a.seed,
               **({"stages_s": {k: round(v, 3) for k, v in stages.items()}} if stages is not None else {}),
               "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps({k: v for k, v in rec.items() if k not in ("aiclk", "tree")}), flush=True)


def _split_stages():
    from collections import defaultdict

    from tt_bio.rfd4 import network as N

    acc = defaultdict(float)
    for name in ("prepare", "launch", "finish"):
        fn = getattr(N.FlowNetwork, name, None)
        if fn is None:
            continue

        def wrapper(self, *args, _fn=fn, _name=name, **kwargs):
            t = time.perf_counter()
            r = _fn(self, *args, **kwargs)
            acc[_name] += time.perf_counter() - t
            return r
        setattr(N.FlowNetwork, name, wrapper)
    return acc


if __name__ == "__main__":
    main()
