#!/usr/bin/env python3
"""Deliverables 1 and 4 at the shape the fold on THIS card actually constructs, [1, 298, 298, 64].

Two things are measured here.

**Channel generality.** X6's kernel required exactly 32 channels. On qb1's 13x10 grid the trunk's
`_trimul_chunk_size` doubles the chunk to 64 (the L1 budget scales with the grid, and 11x10 does not
clear the doubling threshold), so a 298 aa fold on this card issues 4352 calls at
`[1, 298, 298, 64]` and X6's gate refused every one of them. Both kernels here take Ct = C/32.

**The instruction-count axis, bounded.** The transaction count is closed: 64 NOC transactions per
source tile is a floor over all kernel structures (P4's bf16 -> fp32 test left the time unchanged
while the byte-bound control rose 1.36x), so v2 issues exactly the transactions v1 does. What
changes is the instruction stream: the gather's NOC coordinates and 32-byte length are written once
per invocation (`noc_async_read_one_packet_set_state`) rather than per transaction inside
`tt_memmove`, every address is an induction variable, and the loops are split so no multiply, divide
or branch survives in the body. X6 measured one `if` in that loop at 10.7 us on a 97 us op, which is
what makes this worth pricing.

Variants are selected by pointing `reblock_permute.KERNEL_DIR` at a directory, so nothing about the
production module changes between arms. v2 won and is now the production kernel, so the v1 arm is
materialised out of git rather than kept as a second copy in the tree:

    git worktree add /tmp/rp_v1 a14d1647     # the commit where reblock_permute/ still held v1
    TT_VISIBLE_DEVICES=0 python3 perf/p3_permute_op/qb1_kernel_v2.py --v1-dir /tmp/rp_v1/tt_bio/kernels/reblock_permute
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn

MC = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}
# (N, C, buffer). 298/64 on L1 is what a 298 aa fold on qb1's 13x10 grid runs.
CONFIGS = [(298, 64, "l1"), (298, 64, "dram"), (298, 32, "l1"), (320, 64, "l1"), (320, 32, "l1")]


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
    """Per-call cost with K calls enqueued back to back and ONE sync at the end.

    This is the production-relevant rate: inside a fold, calls are enqueued back to back, so a host
    cost below the device time hides behind the previous call's execution. Syncing per call charges
    host and device serially and, on this op, inverts the sign of the ratio against stock.
    """
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(k):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / k


def med(v):
    v = sorted(v)
    return v[len(v) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--v1-dir", default=None,
                    help="kernel dir for the v1 arm; default is the tree's own (now v2), which makes "
                         "the A/B a null control. See the module docstring for the git worktree.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "qb1_kernel_v2.json"))
    a = ap.parse_args()

    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T
    RP.set_enabled(True)

    V = {"v1": Path(a.v1_dir) if a.v1_dir else REPO / "tt_bio" / "kernels" / "reblock_permute",
         "v2": REPO / "tt_bio" / "kernels" / "reblock_permute"}

    device = T.get_device()
    g = device.compute_with_storage_grid_size()
    R = {"wheel": "0.67.4", "host": "qb1", "card": 0,
         "device_grid": [int(g.x), int(g.y)], "rounds": [], "parity": []}
    print("grid", R["device_grid"], flush=True)

    def use(variant):
        RP.KERNEL_DIR = V[variant]
        RP._CACHE.clear()

    # The chunk width the trunk would pick on this card, read from production, not assumed.
    R["trimul_chunk_size_298"] = int(T._trimul_chunk_size(298, 128))
    R["compute_grid_main"] = list(T.COMPUTE_GRID_MAIN)
    print("trimul_chunk_size(298)", R["trimul_chunk_size_298"], flush=True)

    tensors = {}
    for (N, C, where) in CONFIGS:
        if (N, C) not in [(k[0], k[1]) for k in tensors]:
            pass
        ref = torch.randn(1, N, N, C, dtype=torch.bfloat16)
        tensors[(N, C, where)] = (
            ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                            memory_config=MC[where]),
            ref.permute(0, 3, 1, 2).contiguous())

    # ---- parity: both variants, every config, against torch and against stock incl. padding ------
    for variant in ("v1", "v2"):
        use(variant)
        for (N, C, where), (x, gold) in tensors.items():
            mc = MC[where]
            try:
                r = RP.reblock_permute(x, mc, device)
                st = ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)
                R["parity"].append({
                    "variant": variant, "N": N, "C": C, "buf": where,
                    "torch_equal_vs_torch": bool(torch.equal(ttnn.to_torch(r), gold)),
                    "torch_equal_vs_ttnn_permute_incl_padding":
                        bool(torch.equal(ttnn.to_torch(r), ttnn.to_torch(st))),
                })
                ttnn.deallocate(r); ttnn.deallocate(st)
            except Exception:
                R["parity"].append({"variant": variant, "N": N, "C": C, "buf": where,
                                    "error": traceback.format_exc()})
                print(traceback.format_exc())
            print("parity:", R["parity"][-1], flush=True)
    if any("error" in p for p in R["parity"]):
        Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
        return 1
    R["parity_all_true"] = all(p["torch_equal_vs_torch"]
                               and p["torch_equal_vs_ttnn_permute_incl_padding"]
                               for p in R["parity"])
    print("parity_all_true:", R["parity_all_true"], flush=True)

    # ---- the A/B, variants alternating -----------------------------------------------------------
    for rnd in range(a.rounds):
        for (N, C, where), (x, _g) in tensors.items():
            mc = MC[where]
            row = {"round": rnd, "N": N, "C": C, "buf": where}
            for variant in ("v1", "v2"):
                use(variant)
                fn = lambda x=x, mc=mc: ttnn.deallocate(RP.reblock_permute(x, mc, device))
                fn()
                row[f"{variant}_synced_us"] = round(timeit(device, fn), 2)
                row[f"{variant}_thru_us"] = round(throughput_us(device, fn), 2)
            stock = lambda x=x, mc=mc: ttnn.deallocate(
                ttnn.permute(x, (0, 3, 1, 2), memory_config=mc))
            row["stock_thru_us"] = round(throughput_us(device, stock), 2)
            row["stock_synced_us"] = round(timeit(device, stock), 2)
            row["v2_over_v1_thru"] = round(row["v1_thru_us"] / row["v2_thru_us"], 4)
            row["v1_over_stock_thru"] = round(row["stock_thru_us"] / row["v1_thru_us"], 4)
            row["v2_over_stock_thru"] = round(row["stock_thru_us"] / row["v2_thru_us"], 4)
            R["rounds"].append(row)
            print("ab:", row, flush=True)

    R["summary"] = {}
    for (N, C, where) in tensors:
        rs = [r for r in R["rounds"] if (r["N"], r["C"], r["buf"]) == (N, C, where)]
        R["summary"][f"N{N}_C{C}_{where}"] = {
            "v1_thru_us": med([r["v1_thru_us"] for r in rs]),
            "v2_thru_us": med([r["v2_thru_us"] for r in rs]),
            "stock_thru_us": med([r["stock_thru_us"] for r in rs]),
            "v1_synced_us": med([r["v1_synced_us"] for r in rs]),
            "v2_synced_us": med([r["v2_synced_us"] for r in rs]),
            "stock_synced_us": med([r["stock_synced_us"] for r in rs]),
            "v2_over_v1": round(med([r["v1_thru_us"] for r in rs])
                                / med([r["v2_thru_us"] for r in rs]), 4),
            "v1_over_stock": round(med([r["stock_thru_us"] for r in rs])
                                   / med([r["v1_thru_us"] for r in rs]), 4),
            "v2_over_stock": round(med([r["stock_thru_us"] for r in rs])
                                   / med([r["v2_thru_us"] for r in rs]), 4),
            "us_saved_v2_over_v1": round(med([r["v1_thru_us"] for r in rs])
                                         - med([r["v2_thru_us"] for r in rs]), 2),
        }
        print("summary:", f"N{N}_C{C}_{where}", R["summary"][f"N{N}_C{C}_{where}"], flush=True)

    # ---- why the two instruments disagree: a queue-depth sweep -----------------------------------
    # If the difference is a fixed per-call dispatch cost that pipelines behind device execution,
    # total(K) = a + K*b with a larger `a` for the wired op and a smaller `b`. Fit it, don't assert.
    x, _g = tensors[(298, 64, "l1")]
    mc = ttnn.L1_MEMORY_CONFIG
    sweep = {}
    for name in ("v1", "v2", "stock"):
        if name == "stock":
            f = lambda: ttnn.deallocate(ttnn.permute(x, (0, 3, 1, 2), memory_config=mc))
        else:
            use(name)
            f = lambda: ttnn.deallocate(RP.reblock_permute(x, mc, device))
        f()
        sweep[name] = {k: round(min(throughput_us(device, f, k=k) for _ in range(3)), 2)
                       for k in (1, 2, 4, 8, 16, 32)}
        print("sweep:", name, sweep[name], flush=True)
    R["queue_depth_sweep_N298_C64_l1_us_per_call"] = sweep

    Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
