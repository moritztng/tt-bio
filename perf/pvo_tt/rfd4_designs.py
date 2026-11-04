"""The TT RFD4 design arm for pvo-accuracy: N finished designs at the PVO target, one shard per chip.

    TT_VISIBLE_DEVICES=<chip> python perf/pvo_tt/rfd4_designs.py --spec DIR/9TH6-A-b120.yaml \
        --weights nvidia/RFD4-Proteina --out DIR --n 16 --seed0 1234

Not a timing harness: `perf/rfd4/fair/run.py` already priced this cell (168.156 s a design on WH, batch 1).
This writes the designs state/pvo/RFD4-ACC.md grades, named `*_generated.cif` so `perf/rfd4/quality.py`
finds them, one seed a design so the set is reproducible design by design. TT fast = TT exact for RFD4
(`FAST_LEVERS = NORMAL_LEVERS`), so there is one TT distribution, not two.

Shards: give each chip its own `--seed0`/`--n` slice of the seed block and its own `--out`; the seeds are
what make the shards one set. Re-running a shard is free -- a seed whose CIF is already on disk is skipped,
so a chip recalled mid-shard resumes where it stopped.

Clock and contention are recorded per design the way the fair harness records them per batch, because a
design taken at a suppressed clock is still a valid design but its seconds are not comparable.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # perf/ for clocksample

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
from fairbox import Contention  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16, help="designs this shard must land")
    ap.add_argument("--seed0", type=int, default=1234, help="design i runs on seed0 + i (GPU side: DESIGN_SEED 1234)")
    ap.add_argument("--batch", type=int, default=1, help="designs a call; 1 is RFD4 own-best width at this size on WH")
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import sampler

    out = Path(a.out)
    (out / "designs").mkdir(parents=True, exist_ok=True)
    log = open(out / "designs.jsonl", "a")
    seeds = [a.seed0 + i for i in range(a.n)]
    todo = [s for s in seeds if not list((out / "designs").glob(f"*_s{s}_*_generated.cif"))]

    def emit(rec):
        rec = {**rec, "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
               "spec": a.spec, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    if not todo:
        emit({"kind": "skip", "why": "every seed in this shard already has a design", "seeds": seeds})
        return 0

    with clocksample.during(period=1.0) as clk:
        t0 = time.perf_counter()
        engine = E.DesignEngine(a.weights, recipe="thinking_medium")
        t_build = time.perf_counter() - t0
    emit({"kind": "load", "wall_s": round(t_build, 2), "aiclk": clk.summary(), "todo": todo})
    passes = int(sum(sampler.recycle_depths(engine.spec, engine.spec.nsteps)))
    engine.max_parallel_samples = a.batch

    for seed in todo:
        d = out / "work" / f"s{seed}"
        with clocksample.during(period=1.0) as clk, Contention() as con:
            t0 = time.perf_counter()
            ex = rfd4.prepare_examples(a.spec, d / "inputs")[0]
            features, atom_array = rfd4.featurize(ex.cif, seed=seed)
            c0 = engine.counted()
            o = engine.design(features, num_designs=a.batch, nsteps=engine.spec.nsteps, seed=seed, rng="per_design")
            paths = [out / "designs" / f"{ex.example_id}_s{seed}_d{k}_generated.cif" for k in range(a.batch)]
            rfd4.write_designs(o, features, atom_array, paths)
            wall = time.perf_counter() - t0
        c1 = engine.counted()
        ran = {k: c1[k] - c0[k] for k in ("trunk_passes", "trunk_blocks", "decoded")}
        cs = clk.summary().get(0) or {}
        want = {"trunk_passes": passes * a.batch, "trunk_blocks": passes * a.batch * c1["blocks_per_pass"],
                "decoded": a.batch}
        short = "; ".join(f"{k} counted {ran[k]}, scheduled {v}" for k, v in want.items() if ran[k] != v)
        digests = o.digests()
        emit({"kind": "design", "seed": seed, "batch": a.batch, "designs": a.batch,
              "size": int(features["padding_mask"].shape[-1]), "wall_s": round(wall, 3),
              "recipe": "thinking_medium", "steps": engine.spec.nsteps, "trunk_passes": passes, "rng": "per_design",
              "files": [p.name for p in paths], "design_digests": digests,
              "digest": hashlib.sha256("".join(digests).encode()).hexdigest(),
              "aiclk_mhz": cs.get("median"), "aiclk_min_mhz": cs.get("min"), "aiclk_samples": cs.get("n", 0),
              "power_w": round(statistics.median(clk.power.get(0)), 1) if clk.power.get(0) else None,
              **con.summary(), **({"work_mismatch": short} if short else {})})
        if short:  # a design that ran other work than the recipe is not this model s design
            raise SystemExit(f"rfd4 designs: {short} (seed {seed})")

    have = len(list((out / "designs").glob("*_generated.cif")))
    emit({"kind": "summary", "seeds": seeds, "designs_on_disk": have, "wanted": a.n})
    return 0 if have >= a.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
