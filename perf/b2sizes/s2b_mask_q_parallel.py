"""S2b screen: restore K2 at 768/1024 aa by splitting the q chunks across cores.

K2 declines every call above ~640 aa on `fill_preconditions`, and the specific condition is
`q_per_core == 1`: it asks for `split = (cores // H, H, 1)`, so `q_per_core == q_num_chunks`, and the
widest q_chunk stops spanning the sequence once it overflows L1 (q768 at 768 aa, q1024 at 1024 aa).
`_Q_PARALLEL` asks for `q_pf = q_num_chunks` instead. No kernel edit: the hoisted fill already bases
its read on `local_q_start`.

Arm A is the shipped path at the production shape, arm B the same call with the lever on. Median of
`--reps` after 2 warm, `torch.equal` for parity, one process, synced around every timed call.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="768,1024")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dh", type=int, default=32)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="perf/b2sizes/s2b_mask_q_parallel.json")
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as PM

    dev = T.get_device()
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(T.COMPUTE_GRID_MAIN), "heads": a.heads, "dh": a.dh, "reps": a.reps,
           "runs": []}

    for N in [int(s) for s in a.sizes.split(",")]:
        H, DH = a.heads, a.dh
        torch.manual_seed(0)
        parts = []
        for _ in range(3):
            parts.append(torch.randn(N, H, N, DH, dtype=torch.float32).to(torch.bfloat16))
        tb = torch.randn(1, H, N, N, dtype=torch.float32).to(torch.bfloat16)

        def up(t):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

        q, k, v = (up(t) for t in parts)
        b = up(tb)
        del parts, tb
        scale = float(DH) ** -0.5
        rec = {"N": N, "shape": [N, H, N, DH], "arms": {}}

        for arm, qpar in (("A_shipped", False), ("B_q_parallel", True)):
            PM._Q_PARALLEL = qpar
            PM.STATS[0] = PM.STATS[1] = 0
            PM.REJECTS.clear()
            for _ in range(2):                                    # warm: also caches L1 refusals
                o = T._tri_att_sdpa_at(q, k, v, b, scale)
                ttnn.deallocate(o)
            PM.STATS[0] = PM.STATS[1] = 0
            PM.REJECTS.clear()
            ms = []
            keep = None
            for i in range(a.reps):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                o = T._tri_att_sdpa_at(q, k, v, b, scale)
                ttnn.synchronize_device(dev)
                ms.append((time.perf_counter() - t0) * 1e3)
                if i == a.reps - 1:
                    keep = ttnn.to_torch(o)
                ttnn.deallocate(o)
            rec["arms"][arm] = {
                "q_parallel": qpar, "ms_median": round(statistics.median(ms), 3),
                "ms_min": round(min(ms), 3), "ms_all": [round(x, 3) for x in ms],
                "k2_served": PM.STATS[0], "k2_declined": PM.STATS[1],
                "k2_rejects": {f"{r}:{sh}": n for (r, sh), n in PM.REJECTS.items()},
                "sdpa_q_chunk_over_l1": sorted(str(x) for x in T._SDPA_Q_CHUNK_OVER_L1),
            }
            rec.setdefault("_out", {})[arm] = keep
            print(f"  N={N:5d} {arm:14s} q_par={int(qpar)} median {statistics.median(ms):9.3f} ms "
                  f"served={PM.STATS[0]} declined={PM.STATS[1]} {rec[arms][arm][k2_rejects]}",
                  flush=True)

        oa, ob = rec["_out"].pop("A_shipped"), rec["_out"].pop("B_q_parallel")
        rec.pop("_out")
        rec["torch_equal"] = bool(torch.equal(oa, ob))
        d = (oa.float() - ob.float()).abs()
        rec["max_abs_diff"] = float(d.max())
        rec["speedup"] = round(rec["arms"]["A_shipped"]["ms_median"]
                               / rec["arms"]["B_q_parallel"]["ms_median"], 4)
        print(f"  N={N:5d} speedup {rec[speedup]:.4f}x  torch.equal={rec[torch_equal]} "
              f"max_abs_diff={rec[max_abs_diff]:.3e}", flush=True)
        res["runs"].append(rec)
        for t in (q, k, v, b):
            ttnn.deallocate(t)
        del oa, ob, d

    from pathlib import Path
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
