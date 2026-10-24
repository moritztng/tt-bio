"""Paired whole-design A/B of the batch CIF write on the card: A writes the batch serially (`write_design` per row,
what main does), B through `write_designs`. One engine; every round runs the same seed once per arm, the order
alternating, and times what perf/rfd4/fair/run.py times (featurize + design + write) with AICLK sampled throughout.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/gap224/write_ab.py --spec <yaml> --weights <bundle> --batch 8 \
        --rounds 4 --out <dir>

Appends one JSON line per batch to <dir>/ab.jsonl and prints a summary line. A and B CIFs of the same seed are
byte-compared (minus timestamps).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402


def body(p):
    return [line for line in Path(p).read_text().splitlines() if "time" not in line.split(" ")[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "ab.jsonl", "a")
    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    engine.max_parallel_samples = a.batch
    rfd4.writer_pool()

    def batch(arm, seed):
        d = out / f"{arm}-s{seed}"
        with clocksample.during(period=1.0) as clk:
            t0 = time.perf_counter()
            ex = rfd4.prepare_examples(a.spec, d / "inputs")[0]
            features, atom_array = rfd4.featurize(ex.cif, seed=seed)
            t1 = time.perf_counter()
            o = engine.design(features, num_designs=a.batch, nsteps=engine.spec.nsteps, seed=seed, rng="per_design")
            t2 = time.perf_counter()
            paths = [d / f"{ex.example_id}_s{seed}_d{k}.cif" for k in range(a.batch)]
            if arm == "A":
                for k, p in enumerate(paths):
                    rfd4.write_design(o, features, atom_array, p, row=k)
            else:
                rfd4.write_designs(o, features, atom_array, paths)
            t3 = time.perf_counter()
        cs = clk.summary().get(0) or {}
        rec = {"arm": arm, "seed": seed, "size": int(features["padding_mask"].shape[-1]), "batch": a.batch,
               "width_achieved": engine.chunk_width, "wall_s": round(t3 - t0, 3), "featurize_s": round(t1 - t0, 3),
               "design_s": round(t2 - t1, 3), "write_s": round(t3 - t2, 3),
               "aiclk_mhz": cs.get("median"), "aiclk_min_mhz": cs.get("min"),
               "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        return rec, paths

    batch("B", a.seed - 1)  # cold: compile, caches, the pool's first use
    recs = {"A": [], "B": []}
    same = total = 0
    for r in range(a.rounds):
        seed = a.seed + r
        got = {}
        for arm in ("AB" if r % 2 == 0 else "BA"):
            rec, paths = batch(arm, seed)
            recs[arm].append(rec)
            got[arm] = paths
        same += sum(body(x) == body(y) for x, y in zip(got["A"], got["B"]))
        total += a.batch

    def med(arm, k):
        return statistics.median(r[k] for r in recs[arm])

    s = {"summary": True, "size": recs["A"][0]["size"], "batch": a.batch, "rounds": a.rounds,
         "designs_a_side": a.rounds * a.batch,
         **{f"{arm}_{k}": round(med(arm, k), 3) for arm in "AB" for k in ("wall_s", "design_s", "write_s")},
         "speedup_wall": round(med("A", "wall_s") / med("B", "wall_s"), 4),
         "paired_ratio": [round(x["wall_s"] / y["wall_s"], 4) for x, y in zip(recs["A"], recs["B"])],
         "aiclk_mhz": sorted({r["aiclk_mhz"] for v in recs.values() for r in v if r["aiclk_mhz"]}),
         "aiclk_min_mhz": min((r["aiclk_min_mhz"] for v in recs.values() for r in v if r["aiclk_min_mhz"]),
                              default=None),
         "cif_identical_a_vs_b": f"{same}/{total}"}
    log.write(json.dumps(s) + "\n")
    print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
