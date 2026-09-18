#!/usr/bin/env python3
"""Op-level bank ablation on the SHIPPED forward channel move.

`noc_probe.py` measures the read side in isolation and can only bound the op: it says the reader is
48.9 % of the op's wall and that spreading its banks is 1.703x on that leg. This runs the real
`reblock_permute` -- its own compute kernel, its own writer, its own CB structure -- with one line of
its reader edited, so the number is the op ratio rather than a bound on it.

Two arms, both k10-p1's rig pattern: kernel variants are generated from the shipped sources by
asserted string edits into a temp dir, and the shipped files are never touched.

  `bank8`   `page` -> `page ^ (il & 7)`. The XOR stays inside the same aligned 8-page block, so
            every read is in bounds; it spreads the inner run over all 8 DRAM banks. WRONG DATA,
            timing only, never a correctness claim.
  `noread`  deletes the DRAM read and keeps the CB accounting, which prices the reader's share of
            the wall the way `k10-p1-trimul-critpath` priced its own.

`base` is timed first and last under two labels, so the A/A floor brackets the session.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import threading
import time
from pathlib import Path

import torch
import ttnn

EDITS = {
    "base": [],
    "bank8": [("noc_async_read_page(page, s, get_write_ptr(cb_id_in));",
               "noc_async_read_page(page ^ (il & 7), s, get_write_ptr(cb_id_in));")],
    "noread": [("noc_async_read_page(page, s, get_write_ptr(cb_id_in));", "")],
}


class Aiclk(threading.Thread):
    def __init__(self, card):
        super().__init__(daemon=True)
        self.path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.wait(0.05):
            try:
                self.samples.append(int(self.path.read_text().strip()))
            except Exception:
                return

    def report(self):
        v = list(self.samples)
        return None if not v else {"n": len(v), "min": min(v), "max": max(v),
                                   "median": int(statistics.median(v))}


def variant(src_dir: Path, tmp: Path, name: str) -> Path:
    d = tmp / name
    shutil.copytree(src_dir, d)
    f = d / "reader_reblock_permute.cpp"
    t = f.read_text()
    for old, new in EDITS[name]:
        assert t.count(old) == 1, (name, old, t.count(old))
        t = t.replace(old, new)
    f.write_text(t)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--card", type=int, default=2)
    ap.add_argument("--out", default="bank_ablate.json")
    a = ap.parse_args()

    from tt_bio import tenstorrent as tt_dev
    from tt_bio import reblock_permute as rbp

    device = tt_dev.get_device()
    clk = Aiclk(a.card)
    clk.start()
    try:
        shipped = rbp.KERNEL_DIR
        tmp = Path(tempfile.mkdtemp(prefix="trix_bank_"))
        dirs = {k: variant(shipped, tmp, k) for k in EDITS}

        x = ttnn.from_torch(torch.randn(1, a.n, a.n, a.c, dtype=torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        nbytes = a.n * a.n * a.c * 2

        def run_arm(name):
            rbp.KERNEL_DIR = dirs[name]
            rbp._CACHE.clear()
            for _ in range(3):
                ttnn.deallocate(rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG))
            ttnn.synchronize_device(device)
            ts = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                r = rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.synchronize_device(device)
                ts.append(time.perf_counter() - t0)
                ttnn.deallocate(r)
            return statistics.median(ts) * 1e3

        # Interleave: base, bank8, noread, base, bank8, noread, ...
        order = []
        for r in range(a.rounds):
            order += [("base", f"base_r{r}"), ("bank8", f"bank8_r{r}"),
                      ("noread", f"noread_r{r}")]
        res = {}
        for arm, label in order:
            res[label] = round(run_arm(arm), 4)

        # Correctness of the shipped path, on the shipped kernels, after all the swapping.
        rbp.KERNEL_DIR = shipped
        rbp._CACHE.clear()
        got = rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG)
        want = ttnn.permute(x, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        exact = bool(torch.equal(ttnn.to_torch(got), ttnn.to_torch(want)))

        med = {k: statistics.median([v for lbl, v in res.items() if lbl.startswith(k + "_")])
               for k in EDITS}
        out = {
            "shape": [1, a.n, a.n, a.c], "bytes_each_way": nbytes,
            "arms_in_order": res, "median_by_arm": med,
            "aa_floor_pct": round(abs(res["base_r0"] - res[f"base_r{a.rounds-1}"])
                                  / res["base_r0"] * 100, 4) if a.rounds > 1 else None,
            "bank8_ratio": round(med["base"] / med["bank8"], 4),
            "reader_share_of_wall": round((med["base"] - med["noread"]) / med["base"], 4),
            "reader_ms": round(med["base"] - med["noread"], 4),
            "base_gbs_each_way": round(nbytes / (med["base"] * 1e-3) / 1e9, 2),
            "bank8_gbs_each_way": round(nbytes / (med["bank8"] * 1e-3) / 1e9, 2),
            "shipped_bit_exact_vs_ttnn_permute": exact,
            "aiclk_during_sysfs_20hz": clk.report(),
            "grid": [device.compute_with_storage_grid_size().x,
                     device.compute_with_storage_grid_size().y],
        }
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        shutil.rmtree(tmp, ignore_errors=True)
    finally:
        clk.stop.set()
        tt_dev.cleanup()


if __name__ == "__main__":
    main()
