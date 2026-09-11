#!/usr/bin/env python3
"""Step 2: does the in-projection's DRAM round trip come back as time when it goes to L1?

Arms, all in one process so they interleave:

  A    incumbent, both flags off
  B    step 1 only (mask after the channel move, E6 eligible)
  C<R> step 1 + step 2 at row width R (projection row block in L1, gated move reads it there)
  A2   the incumbent again, as this session's own A/A floor

Falsifier for step 2, from the design: the row-blocked projection is SLOWER than the dual-NOC
one it replaces, at R = 128 and at R = 64. `mm_dualnoc.in_proj` refuses a non-DRAM memory_config,
so going to L1 gives up the dual-NOC drain; the bet is that a whole DRAM round trip is worth more
than a quarter of a DRAM write.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))
sys.path.insert(0, str(ROOT / "perf" / "b2x_trimul"))

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.reblock_permute as RB
from real_traffic import counts
from trimul_ab import weights, make_inputs, sha


def arm(name, r=None):
    """Set the two flags for an arm. Returns nothing; arms are set immediately before each call."""
    T.set_trimul_mask_after_move(name != "A" and name != "A2")
    T.set_trimul_inproj_rowblock(name.startswith("C"), r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rs", default="64,128,256")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rs = [int(v) for v in a.rs.split(",")]
    arms = [("A", None), ("B", None)] + [(f"C{r}", r) for r in rs] + [("A2", None)]

    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    ck = T.trunk_compute_kernel_config(compute_kernel_config())
    mods = {e: T.TriangleMultiplication(e, weights(), ck) for e in (False, True)}
    res = {"n": a.n, "reps": a.reps, "arms": [x[0] for x in arms], "host": "qb2", "card": 2,
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- parity: every arm against A, both mask kinds, both variants ----
    par = {}
    for mk in ("ones", "random"):
        z, m = make_inputs(dev, a.n, mk)
        for ending in (False, True):
            row = {}
            for nm, r in arms:
                arm(nm, r)
                out = mods[ending](z, m)
                ttnn.synchronize_device(dev)
                row[nm] = sha(out)
                ttnn.deallocate(out)
            # reuse control: the row-block path must not free anything the caller owns either
            arm("C%d" % rs[-1], rs[-1])
            reuse = []
            for _ in range(3):
                out = mods[ending](z, m)
                ttnn.synchronize_device(dev)
                reuse.append(sha(out))
                ttnn.deallocate(out)
            arm("A")
            row["reuse_stable"] = len(set(reuse)) == 1 and reuse[0] == row["A"]
            row["all_bitexact"] = all(v == row["A"] for k, v in row.items()
                                      if k != "reuse_stable" and isinstance(v, str))
            par[f"{mk}/{'ending' if ending else 'starting'}"] = row
            print(json.dumps({f"{mk}/{'ending' if ending else 'starting'}": row}), flush=True)
        ttnn.deallocate(z); ttnn.deallocate(m)
    res["parity"] = par
    # A different mask must give a different hash, else the control reads nothing.
    res["control_fires"] = (par["ones/starting"]["A"] != par["random/starting"]["A"]
                            and par["ones/ending"]["A"] != par["random/ending"]["A"])
    print("control_fires:", res["control_fires"], flush=True)

    z, m = make_inputs(dev, a.n, "ones")

    # ---- bytes ----
    by = {}
    for ending in (False, True):
        cell = {}
        for nm, r in arms[:-1]:
            arm(nm, r)
            ttnn.deallocate(mods[ending](z, m))
            ttnn.synchronize_device(dev)
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            out = mods[ending](z, m)
            ttnn.synchronize_device(dev)
            g = ttnn.graph.end_graph_capture()
            ttnn.deallocate(out)
            c = counts({"sig": "trimul", "nodes": g})
            cell[nm] = {q: round(c[q], 3) if isinstance(c[q], float) else c[q]
                        for q in ("real_MB", "real_w_MB", "real_r_MB", "n_ops")}
        base = cell["A"]["real_MB"]
        cell["deleted_MB"] = {k: round(base - v["real_MB"], 3)
                              for k, v in cell.items() if isinstance(v, dict)}
        cell["deleted_Z"] = {k: round(v / 67.108864, 3) for k, v in cell["deleted_MB"].items()}
        by["ending" if ending else "starting"] = cell
        arm("A")
        print(json.dumps({("ending" if ending else "starting"): cell["deleted_Z"]}), flush=True)
    res["bytes"] = by

    # ---- time ----
    tm = {}
    for ending in (False, True):
        for nm, r in arms:                       # compile every arm before timing any of them
            arm(nm, r)
            for _ in range(2):
                ttnn.deallocate(mods[ending](z, m))
        ttnn.synchronize_device(dev)
        samples = {nm: [] for nm, _ in arms}
        for _ in range(a.reps):
            for nm, r in arms:
                arm(nm, r)
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                out = mods[ending](z, m)
                ttnn.synchronize_device(dev)
                samples[nm].append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(out)
        arm("A")
        med = {k: round(statistics.median(v), 4) for k, v in samples.items()}
        tm["ending" if ending else "starting"] = {
            "median_ms": med,
            "raw_ms": {k: [round(x, 4) for x in v] for k, v in samples.items()},
            "AA_floor_pct": round(100 * (med["A2"] - med["A"]) / med["A"], 3),
            "x_vs_A": {k: round(med["A"] / v, 4) for k, v in med.items()},
        }
        print(json.dumps(med | {"floor%": tm["ending" if ending else "starting"]["AA_floor_pct"]}),
              flush=True)
        print(json.dumps(tm["ending" if ending else "starting"]["x_vs_A"]), flush=True)
    res["time"] = tm
    res["rejects"] = {"|".join(str(x) for x in k): v for k, v in RB.REJECTS.items()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
