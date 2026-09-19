#!/usr/bin/env python3
"""Parity and op-level A/B of the bank spread on `reblock_permute_gated`, the op the fold calls.

`trix-transaction` measured `reblock_permute`. `perf/trix_bankspread/firing_512_qb2c2.json` found
that op serves 0 calls per 512 aa fold and the gated one serves 1120 (2096 on the campaign's
10-recycle protocol), so this re-runs the whole question where the traffic is.

Shape is the production call read off `trix-scaffold-attribute`'s in-fold tape:
`[1, 512, 512, 1024] -> [1, 256, 512, 512]`, value slice 512, gate slice 0, matching
`gp_roles()`'s shipped column order.

Correctness is `torch.equal` against the two-op ttnn sequence the kernel replaces,
`permute(chunk(xw,4,-1)[p] * sigmoid(chunk(xw,4,-1)[g]), (0,3,1,2))`, which is the check the
shipped kernel already carries.

Arms are (CT_STREAM, OUT_CB buffers), interleaved round by round, `s1b2` first and last.
Timed in the batched bracket, `--batch` calls before one sync, which is how a fold issues them.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--cw", type=int, default=1024, help="wide projection channels")
    ap.add_argument("--slice-c", type=int, default=256)
    ap.add_argument("--p-slice", type=int, default=512)
    ap.add_argument("--g-slice", type=int, default=0)
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--out", default="gated_ab.json")
    a = ap.parse_args()

    from tt_bio import reblock_permute as rbp
    from tt_bio import tenstorrent as tt_dev

    device = tt_dev.get_device()
    clk = Aiclk(a.card)
    clk.start()
    mt = []
    try:
        xt = torch.randn(1, a.n, a.n, a.cw, dtype=torch.bfloat16)
        xw = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # the two-op ttnn sequence the kernel replaces, on device, as the reference
        ch = ttnn.chunk(xw, 4, -1)
        pi, gi = a.p_slice // a.slice_c, a.g_slice // a.slice_c
        ref = ttnn.permute(ttnn.multiply(ch[pi], ttnn.sigmoid(ch[gi])), (0, 3, 1, 2),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        want = ttnn.to_torch(ref)
        for c in ch:
            ttnn.deallocate(c)
        ttnn.deallocate(ref)
        nbytes_out = a.n * a.n * a.slice_c * 2

        def call():
            return rbp.reblock_permute_gated(xw, a.p_slice, a.g_slice, a.slice_c,
                                             memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def select(S, bufs):
            rbp._CT_STREAM_PIN, rbp._CT_BUFS_PIN = str(S), str(bufs)
            rbp._CACHE_GATED.clear()

        exact = {}
        for nm, S, bufs in ARMS:
            select(S, bufs)
            got = call()
            ttnn.synchronize_device(device)
            exact[nm] = bool(torch.equal(ttnn.to_torch(got), want))
            ttnn.deallocate(got)
            print("parity", nm, exact[nm], flush=True)

        def run_arm(S, bufs):
            select(S, bufs)
            for _ in range(3):
                ttnn.deallocate(call())
            ttnn.synchronize_device(device)
            ts = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                rs = [call() for _ in range(a.batch)]
                ttnn.synchronize_device(device)
                ts.append((time.perf_counter() - t0) / a.batch)
                for r in rs:
                    ttnn.deallocate(r)
                mt.append(clk.read())
            return statistics.median(ts) * 1e3

        order = [(f"{nm}_r{r}", S, b) for r in range(a.rounds) for nm, S, b in ARMS]
        order.append(("s1b2_last", 1, 2))
        res = {}
        for label, S, bufs in order:
            res[label] = round(run_arm(S, bufs), 4)
            print(label, res[label], flush=True)

        def med(nm):
            return statistics.median([v for lbl, v in res.items() if lbl.startswith(nm + "_r")])

        base = med("s1b2")
        summary = {}
        for nm, S, bufs in ARMS:
            m = med(nm)
            summary[nm] = {
                "ct_stream": S, "bufs": bufs, "out_cb_kb": 32 * S * bufs * 2048 / 1024,
                "ms": round(m, 4), "ratio": round(base / m, 4),
                "saved_ms_per_call": round(base - m, 4),
                "bit_exact_vs_ttnn_chunk_sigmoid_mul_permute": exact[nm],
            }

        rbp._CT_STREAM_PIN = rbp._CT_BUFS_PIN = None
        rbp._CACHE_GATED.clear()
        got = call()
        default_exact = bool(torch.equal(ttnn.to_torch(got), want))
        S_def, depth_def = rbp._ct_stream_gated(a.slice_c // 32, 32 * 32 * 2)

        out = {
            "in_shape": [1, a.n, a.n, a.cw], "out_shape": [1, a.slice_c, a.n, a.n],
            "p_slice": a.p_slice, "g_slice": a.g_slice,
            "arms_in_order": res, "summary": summary,
            "aa_floor_pct": round(abs(res["s1b2_r0"] - res["s1b2_last"]) / res["s1b2_r0"] * 100, 4),
            "derived_default": {"ct_stream": S_def, "out_cb_tiles": depth_def,
                                "bit_exact": default_exact},
            "aiclk_during_sysfs_20hz": Aiclk.stat(list(clk.samples)),
            "aiclk_during_measuring_thread": Aiclk.stat([v for v in mt if v]),
            "grid": [device.compute_with_storage_grid_size().x,
                     device.compute_with_storage_grid_size().y],
            "out_bytes": nbytes_out, "batch": a.batch, "iters": a.iters, "rounds": a.rounds,
        }
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps({k: out[k] for k in
                          ("summary", "aa_floor_pct", "derived_default",
                           "aiclk_during_measuring_thread")}, indent=1))
    finally:
        clk.stop.set()
        tt_dev.cleanup()


if __name__ == "__main__":
    main()
