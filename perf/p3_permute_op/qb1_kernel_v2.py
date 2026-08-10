#!/usr/bin/env python3
"""Deliverable 4: the one authorised optimisation axis — instructions per NOC transaction.

The transaction COUNT is a closed question: 64 per source tile is a floor over all kernel
structures (P4's bf16 -> fp32 test left the time unchanged while the byte-bound control rose 1.36x),
so v2 issues exactly the same transactions as v1. What changes is the instruction stream around them:

  * the gather loop's NOC coordinates and 32-byte length are written once per kernel invocation
    (`noc_async_read_one_packet_set_state`) instead of once per transaction inside `tt_memmove`;
  * every address in the gather loop is an induction variable, and the loop is split at the
    face-row boundary so no multiply, divide or branch survives in the body;
  * the reader's per-row `row < D1` test becomes two loops with `page += Nt`.

X6 measured a single `if` in that gather loop at 10.7 us on a 97 us op, which is what says this axis
is worth pricing at all.

Both variants are selected by pointing `reblock_permute.KERNEL_DIR` at a directory, so this is a
probe with no production change until the result says to make one.

    TT_VISIBLE_DEVICES=0 python3 perf/p3_permute_op/qb1_kernel_v2.py
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn

MC = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}


def timeit(device, fn, reps=21, warmup=5):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def throughput_us(device, fn, k=40, warmup=5):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(k):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "qb1_kernel_v2.json"))
    a = ap.parse_args()

    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T
    RP.set_enabled(True)

    V1 = REPO / "tt_bio" / "kernels" / "reblock_permute"
    V2 = REPO / "tt_bio" / "kernels" / "reblock_permute_v2"

    device = T.get_device()
    R = {"wheel": "0.67.4", "host": "qb1", "card": 0, "rounds": [], "parity": []}

    def use(variant):
        RP.KERNEL_DIR = V1 if variant == "v1" else V2
        RP._CACHE.clear()

    # ---- does v2 compile, and is it bit-exact including the tile padding ------------------------
    tensors = {}
    for N in (298, 320):
        ref = torch.randn(1, N, N, 32, dtype=torch.bfloat16)
        gold = ref.permute(0, 3, 1, 2).contiguous()
        for where in ("l1", "dram"):
            tensors[(N, where)] = (
                ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                                memory_config=MC[where]), gold)

    for variant in ("v1", "v2"):
        use(variant)
        try:
            for (N, where), (x, gold) in tensors.items():
                mc = MC[where]
                r = RP.reblock_permute(x, mc, device)
                st = ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)
                R["parity"].append({
                    "variant": variant, "N": N, "buf": where,
                    "torch_equal_vs_torch": bool(torch.equal(ttnn.to_torch(r), gold)),
                    "torch_equal_vs_ttnn_permute_incl_padding":
                        bool(torch.equal(ttnn.to_torch(r), ttnn.to_torch(st))),
                })
                ttnn.deallocate(r); ttnn.deallocate(st)
        except Exception:
            R["parity"].append({"variant": variant, "compile_or_run_error": traceback.format_exc()})
            print(traceback.format_exc())
            Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
            return 1
    for p in R["parity"]:
        print("parity:", p, flush=True)

    # ---- the A/B, variants alternating -----------------------------------------------------------
    for rnd in range(a.rounds):
        for (N, where), (x, _g) in sorted(tensors.items()):
            mc = MC[where]
            row = {"round": rnd, "N": N, "buf": where}
            for variant in ("v1", "v2"):
                use(variant)
                fn = lambda x=x, mc=mc: ttnn.deallocate(RP.reblock_permute(x, mc, device))
                row[f"{variant}_synced_us"] = round(timeit(device, fn), 2)
                row[f"{variant}_thru_us"] = round(throughput_us(device, fn), 2)
            stock = lambda x=x, mc=mc: ttnn.deallocate(
                ttnn.permute(x, (0, 3, 1, 2), memory_config=mc))
            row["stock_thru_us"] = round(throughput_us(device, stock), 2)
            row["v2_over_v1_thru"] = round(row["v1_thru_us"] / row["v2_thru_us"], 4)
            row["v2_over_stock_thru"] = round(row["stock_thru_us"] / row["v2_thru_us"], 4)
            row["v1_over_stock_thru"] = round(row["stock_thru_us"] / row["v1_thru_us"], 4)
            R["rounds"].append(row)
            print("ab:", row, flush=True)

    def med(v):
        v = sorted(v)
        return v[len(v) // 2]

    R["summary"] = {}
    for (N, where) in sorted(tensors.keys()):
        rs = [r for r in R["rounds"] if r["N"] == N and r["buf"] == where]
        R["summary"][f"N{N}_{where}"] = {
            "v1_thru_us": med([r["v1_thru_us"] for r in rs]),
            "v2_thru_us": med([r["v2_thru_us"] for r in rs]),
            "stock_thru_us": med([r["stock_thru_us"] for r in rs]),
            "v2_over_v1": round(med([r["v1_thru_us"] for r in rs])
                                / med([r["v2_thru_us"] for r in rs]), 4),
            "v2_over_stock": round(med([r["stock_thru_us"] for r in rs])
                                   / med([r["v2_thru_us"] for r in rs]), 4),
            "v1_over_stock": round(med([r["stock_thru_us"] for r in rs])
                                   / med([r["v1_thru_us"] for r in rs]), 4),
            "us_saved_per_call": round(med([r["v1_thru_us"] for r in rs])
                                       - med([r["v2_thru_us"] for r in rs]), 2),
        }
        print("summary:", f"N{N}_{where}", R["summary"][f"N{N}_{where}"], flush=True)

    Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
