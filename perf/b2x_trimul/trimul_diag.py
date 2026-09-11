#!/usr/bin/env python3
"""Two checks the first pass could not make.

1. NEGATIVE CONTROL. The step-1 parity run compared arm-off to arm-on under two different masks
   and the `ending` trimul returned the SAME hash under both. A parity check whose input the
   change is supposed to be sensitive to does not move is not reading what it claims to read
   (memory `negative-control-must-break-what-check-reads`), so: does the ending trimul's output
   depend on the mask at all, and where does the dependence go?

2. BYTES UNDER A FULL CAPTURE. The byte pass ran with ttnn's fast runtime mode on, which the
   capture itself warns records no per-operation sub-graph: it charged 4 top-level ops for a
   whole trimul. Re-count with TTNN_CONFIG_OVERRIDES={"enable_fast_runtime_mode": false} and
   report both, because the difference between them is how much the corrected counter was
   seeing in the first place.
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

import torch
import ttnn
import tt_bio.tenstorrent as T
from real_traffic import counts
from trimul_ab import weights, make_inputs, sha, CZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    ck = T.trunk_compute_kernel_config(compute_kernel_config())
    mods = {e: T.TriangleMultiplication(e, weights(), ck) for e in (False, True)}
    res = {"n": a.n, "fast_runtime": ttnn.CONFIG.enable_fast_runtime_mode}
    print("enable_fast_runtime_mode =", res["fast_runtime"], flush=True)

    # --- 1. negative control -------------------------------------------------
    ctl = {}
    for ending in (False, True):
        z, m_ones = make_inputs(dev, a.n, "ones")
        _, m_rand = make_inputs(dev, a.n, "random")
        row = {}
        for mk, mm in (("ones", m_ones), ("random", m_rand)):
            for arm in (False, True):
                prev = T.set_trimul_mask_after_move(arm)
                r = mods[ending](z, mm)
                ttnn.synchronize_device(dev)
                t = ttnn.to_torch(r).float()
                row[f"{mk}/{'on' if arm else 'off'}"] = {
                    "sha": sha(r), "absmax": round(float(t.abs().max()), 6),
                    "mean": round(float(t.mean()), 8),
                    "nonfinite": int((~torch.isfinite(t)).sum()),
                }
                ttnn.deallocate(r)
                T.set_trimul_mask_after_move(prev)
        row["mask_sensitive"] = row["ones/off"]["sha"] != row["random/off"]["sha"]
        row["bitexact_ones"] = row["ones/off"]["sha"] == row["ones/on"]["sha"]
        row["bitexact_random"] = row["random/off"]["sha"] == row["random/on"]["sha"]
        ctl["ending" if ending else "starting"] = row
        ttnn.deallocate(z); ttnn.deallocate(m_ones); ttnn.deallocate(m_rand)
        print(json.dumps({k: v for k, v in row.items() if not isinstance(v, dict)}
                         | {k: v["sha"] for k, v in row.items() if isinstance(v, dict)}), flush=True)
    res["negative_control"] = ctl

    # --- 2. bytes under this capture mode ------------------------------------
    z, m = make_inputs(dev, a.n, "ones")
    by = {}
    for ending in (False, True):
        cell = {}
        for arm in (False, True):
            prev = T.set_trimul_mask_after_move(arm)
            ttnn.deallocate(mods[ending](z, m))
            ttnn.synchronize_device(dev)
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            r = mods[ending](z, m)
            ttnn.synchronize_device(dev)
            g = ttnn.graph.end_graph_capture()
            ttnn.deallocate(r)
            c = counts({"sig": "trimul", "nodes": g})
            k = "on" if arm else "off"
            cell[k] = {q: c[q] for q in ("real_MB", "real_w_MB", "real_r_MB", "n_ops")}
            cell[k + "_top"] = [[round(t / 1e6, 3), nm, round(w / 1e6, 3), round(rr / 1e6, 3)]
                                for t, nm, w, rr in c["per_op"][:16]]
            T.set_trimul_mask_after_move(prev)
        d = cell["off"]["real_MB"] - cell["on"]["real_MB"]
        cell["deleted_MB"] = round(d, 3)
        cell["deleted_Z"] = round(d / 67.108864, 3)
        by["ending" if ending else "starting"] = cell
        print(json.dumps({k: v for k, v in cell.items() if not k.endswith("_top")}), flush=True)
        for r_ in cell["off_top"][:10]:
            print("   off %9.3f MB %-42s" % (r_[0], r_[1]), flush=True)
        for r_ in cell["on_top"][:10]:
            print("   on  %9.3f MB %-42s" % (r_[0], r_[1]), flush=True)
    res["bytes"] = by
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
