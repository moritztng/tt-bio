#!/usr/bin/env python3
"""Op-level A/B of the legal bank spread on the shipped `reblock_permute`.

`trix-transaction` measured 1.2797x through `page ^ (il & 7)`, a proxy that spreads banks and reads
the wrong data. This times the real reorder, which reads the same pages in a different order and is
`torch.equal` against `ttnn.permute` at every rung (`cb_allocator.py`).

Arms are (CT_STREAM, OUT_CB buffers). `s1b2` is byte-for-byte the shipped walk and the shipped CB.
Arms are interleaved round by round and `s1b2` is timed first and last, so the A/A floor brackets
the session.

Two brackets, because they answer different questions:
  `synced`  one `synchronize_device` per call. Comparable with `bank_ablate.py`, and charges the
            ~0.05 ms host launch/drain floor that `fixterm-decompose` measured, on every arm.
  `batched` `--batch` calls issued back to back before one sync. This is how a fold issues them,
            so it is the bracket the fold-level prediction should be carried out from.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import torch
import ttnn

ARMS = [("s1b2", 1, 2), ("s2b2", 2, 2), ("s4b1", 4, 1), ("s4b2", 4, 2),
        ("s8b1", 8, 1), ("s8b2", 8, 2)]


class Aiclk(threading.Thread):
    def __init__(self, card):
        super().__init__(daemon=True)
        self.path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
        self.samples, self.stop = [], threading.Event()

    def read(self):
        try:
            return int(self.path.read_text().strip())
        except Exception:
            return None

    def run(self):
        while not self.stop.wait(0.05):
            v = self.read()
            if v is None:
                return
            self.samples.append(v)

    @staticmethod
    def stat(v):
        return None if not v else {"n": len(v), "min": min(v), "max": max(v),
                                   "median": int(statistics.median(v))}

    def report(self):
        return self.stat(list(self.samples))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--card", type=int, default=2)
    ap.add_argument("--out", default="op_ab.json")
    a = ap.parse_args()

    from tt_bio import reblock_permute as rbp
    from tt_bio import tenstorrent as tt_dev

    device = tt_dev.get_device()
    clk = Aiclk(a.card)
    clk.start()
    mainthread_clk = []
    try:
        x = ttnn.from_torch(torch.randn(1, a.n, a.n, a.c, dtype=torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        want = ttnn.to_torch(ttnn.permute(x, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG))
        nbytes = a.n * a.n * a.c * 2

        def select(S, bufs):
            rbp._CT_STREAM_PIN, rbp._CT_BUFS_PIN = str(S), str(bufs)
            rbp._CACHE.clear()

        def run_arm(S, bufs):
            select(S, bufs)
            for _ in range(3):
                ttnn.deallocate(rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG))
            ttnn.synchronize_device(device)
            synced = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                r = rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.synchronize_device(device)
                synced.append(time.perf_counter() - t0)
                ttnn.deallocate(r)
                mainthread_clk.append(clk.read())
            batched = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                rs = [rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG) for _ in range(a.batch)]
                ttnn.synchronize_device(device)
                batched.append((time.perf_counter() - t0) / a.batch)
                for r in rs:
                    ttnn.deallocate(r)
                mainthread_clk.append(clk.read())
            return statistics.median(synced) * 1e3, statistics.median(batched) * 1e3

        order = []
        for r in range(a.rounds):
            order += [(f"{nm}_r{r}", S, b) for nm, S, b in ARMS]
        order.append(("s1b2_last", 1, 2))

        res = {}
        for label, S, bufs in order:
            s, b = run_arm(S, bufs)
            res[label] = {"synced_ms": round(s, 4), "batched_ms": round(b, 4)}
            print(label, res[label], flush=True)

        def med(nm, key):
            return statistics.median([v[key] for lbl, v in res.items()
                                      if lbl.startswith(nm + "_r")])

        base_s, base_b = med("s1b2", "synced_ms"), med("s1b2", "batched_ms")
        summary = {}
        for nm, S, bufs in ARMS:
            s, b = med(nm, "synced_ms"), med(nm, "batched_ms")
            summary[nm] = {
                "ct_stream": S, "bufs": bufs, "out_cb_kb": 32 * S * bufs * 2048 / 1024,
                "synced_ms": round(s, 4), "batched_ms": round(b, 4),
                "synced_ratio": round(base_s / s, 4), "batched_ratio": round(base_b / b, 4),
                "batched_gbs_each_way": round(nbytes / (b * 1e-3) / 1e9, 2),
                "saved_ms_per_op_batched": round(base_b - b, 4),
            }

        # correctness of the derived default, last, on the same tensor
        rbp._CT_STREAM_PIN = rbp._CT_BUFS_PIN = None
        rbp._CACHE.clear()
        got = rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG)
        exact = bool(torch.equal(ttnn.to_torch(got), want))

        aa_s = abs(res["s1b2_r0"]["synced_ms"] - res["s1b2_last"]["synced_ms"]) \
            / res["s1b2_r0"]["synced_ms"] * 100
        aa_b = abs(res["s1b2_r0"]["batched_ms"] - res["s1b2_last"]["batched_ms"]) \
            / res["s1b2_r0"]["batched_ms"] * 100
        mt = [v for v in mainthread_clk if v]
        out = {
            "shape": [1, a.n, a.n, a.c], "bytes_each_way": nbytes,
            "arms_in_order": res, "summary": summary,
            "aa_floor_pct_synced": round(aa_s, 4), "aa_floor_pct_batched": round(aa_b, 4),
            "derived_default_bit_exact_vs_ttnn_permute": exact,
            "aiclk_during_sysfs_20hz": clk.report(),
            "aiclk_during_measuring_thread": Aiclk.stat(mt),
            "grid": [device.compute_with_storage_grid_size().x,
                     device.compute_with_storage_grid_size().y],
            "batch": a.batch, "iters": a.iters, "rounds": a.rounds,
        }
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps({k: out[k] for k in
                          ("summary", "aa_floor_pct_synced", "aa_floor_pct_batched",
                           "derived_default_bit_exact_vs_ttnn_permute",
                           "aiclk_during_sysfs_20hz", "aiclk_during_measuring_thread")}, indent=1))
    finally:
        clk.stop.set()
        tt_dev.cleanup()


if __name__ == "__main__":
    main()
