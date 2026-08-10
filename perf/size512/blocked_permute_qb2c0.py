#!/usr/bin/env python3
"""Deliverable 4 — price the size-INDEPENDENT alternative to the fit test, at 512 aa.

The fit test is not a fix; it is a fix that only works on small inputs. Row-blocking is the
alternative the charter prefers, and `TriangleAttention.__call__` already contains the blocked
transpose (`tenstorrent.py:1846-1851`) -- it slices a column strip of the pair tensor, permutes it
with the same `_transpose_memory_config(blk)` call, and norms it per block. The branch is unreachable
at these sizes (`S > SEQ_LEN_MORE_CHUNKING = 1536`, and `TRIANGLE_ATT_CHUNK_SIZE = 512` is over the
row bound anyway), so the question is what it would cost if it were reachable.

Blocking trades one big op for `N/R` small ones: the L1 residency becomes a property of R rather than
of N, but every extra call pays its own dispatch, its own circular-buffer setup and its own kernel
launch. That trade is what this measures, at the 512 aa shape, against the full-tensor DRAM permute
that production actually takes there.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import ttnn  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=2, pipe=3, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="384,448,512")
    ap.add_argument("--blocks", default="32,64,128,256")
    a = ap.parse_args()

    import importlib.metadata as im
    from tt_bio.tenstorrent import get_device, COMPUTE_GRID_MAIN, _l1_memory_config_if_it_fits

    dev = get_device()
    gx, gy = COMPUTE_GRID_MAIN
    budget = int(ttnn.get_max_worker_l1_unreserved_size()) * gx * gy
    res = {"host": "qb2", "chip": 0, "ttnn": im.version("ttnn"),
           "core_grid_main": f"{gx}x{gy}", "l1_budget_bytes": budget, "runs": []}

    for N in [int(v) for v in a.sizes.split(",")]:
        S = -(-N // 32) * 32
        x = ttnn.from_torch(torch.randn(N, S, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=DRAM)
        full_dram = timed(dev, lambda: ttnn.deallocate(ttnn.permute(x, (1, 0, 2), memory_config=DRAM)))
        rec0 = {"N": N, "R": None, "calls": 1, "dest": "dram", "ms": round(full_dram * 1e3, 4),
                "note": "full-tensor DRAM permute -- what production takes at this size"}
        res["runs"].append(rec0)
        print("  " + json.dumps(rec0), flush=True)

        for R in [int(v) for v in a.blocks.split(",")]:
            if R > S:
                continue
            blk_shape = [N, R, 256]
            blk_bytes = N * R * 256 * 2
            fits = 2.5 * blk_bytes <= budget

            def run():
                outs = []
                for s in range(0, S, R):
                    e = min(s + R, S)
                    b = x[:, s:e, :]
                    outs.append(ttnn.permute(b, (1, 0, 2), memory_config=L1))
                    ttnn.deallocate(b)
                for o in outs:
                    ttnn.deallocate(o)

            rec = {"N": N, "R": R, "calls": -(-S // R), "dest": "l1",
                   "block_shape": blk_shape, "block_MB": round(blk_bytes / 1e6, 2),
                   "fit_test_2p5x": bool(fits)}
            try:
                t = timed(dev, run, warm=1, pipe=2, reps=3)
                rec["ms"] = round(t * 1e3, 4)
                rec["vs_full_dram"] = round(full_dram / t, 3)
                rec["per_call_us"] = round(t * 1e6 / rec["calls"], 1)
            except Exception as e:                                              # noqa: BLE001
                rec["error"] = str(e)[:160]
            res["runs"].append(rec)
            print("  " + json.dumps(rec), flush=True)
        ttnn.deallocate(x)

    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
