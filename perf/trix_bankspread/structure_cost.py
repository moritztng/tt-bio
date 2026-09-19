#!/usr/bin/env python3
"""Why the legal reorder is 1.1098x where the XOR proxy said 1.2797x.

The proxy changed one address and nothing else: same push order, same 64-tile OUT_CB, same writer
gather stride. The legal reorder also moves the writer's window to 32*S tiles and its L1 gather
stride to S tiles. This separates the two by running four arms through the real op:

  `base`      CT_STREAM = 1, the shipped walk.
  `xor8`      CT_STREAM = 1 with `page ^ (il & 7)` -- trix-transaction's proxy, re-run in this
              session and this bracket so its 1.2797x is comparable. WRONG DATA, timing only.
  `legal`     CT_STREAM = 4, the shipped default, correct pages and `torch.equal`.
  `onebank`   CT_STREAM = 4 with `page + k` -> `page`, so the reorder's whole structure is paid
              (bigger CB, strided gather) and the bank spread is removed. WRONG DATA, timing only.
  `noread`    the DRAM read deleted at CT_STREAM = 1, which prices the read leg.

`base / onebank` is what the structure costs; `onebank / legal` is what the banks buy.
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

READ = "noc_async_read_page(page + k, s, get_write_ptr(cb_id_in));"
ARMS = {
    "base": (1, []),
    "xor8": (1, [(READ, "noc_async_read_page((page + k) ^ (il & 7), s, get_write_ptr(cb_id_in));")]),
    "legal": (4, []),
    "onebank": (4, [(READ, "noc_async_read_page(page, s, get_write_ptr(cb_id_in));")]),
    "noread": (1, [(READ, "")]),
}


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


def variant(src_dir: Path, tmp: Path, name: str) -> Path:
    d = tmp / name
    shutil.copytree(src_dir, d)
    f = d / "reader_reblock_permute.cpp"
    t = f.read_text()
    for old, new in ARMS[name][1]:
        assert t.count(old) == 1, (name, t.count(old))
        t = t.replace(old, new)
    f.write_text(t)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--card", type=int, default=2)
    ap.add_argument("--out", default="structure_cost.json")
    a = ap.parse_args()

    from tt_bio import reblock_permute as rbp
    from tt_bio import tenstorrent as tt_dev

    device = tt_dev.get_device()
    clk = Aiclk(a.card)
    clk.start()
    mt = []
    try:
        shipped = rbp.KERNEL_DIR
        tmp = Path(tempfile.mkdtemp(prefix="trix_bankspread_"))
        dirs = {k: variant(shipped, tmp, k) for k in ARMS}

        x = ttnn.from_torch(torch.randn(1, a.n, a.n, a.c, dtype=torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        want = ttnn.to_torch(ttnn.permute(x, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG))

        def run_arm(name):
            S = ARMS[name][0]
            rbp.KERNEL_DIR = dirs[name]
            rbp._CT_STREAM_PIN, rbp._CT_BUFS_PIN = str(S), "2"
            rbp._CACHE.clear()
            for _ in range(3):
                ttnn.deallocate(rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG))
            ttnn.synchronize_device(device)
            ts = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                rs = [rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG) for _ in range(a.batch)]
                ttnn.synchronize_device(device)
                ts.append((time.perf_counter() - t0) / a.batch)
                for r in rs:
                    ttnn.deallocate(r)
                mt.append(clk.read())
            return statistics.median(ts) * 1e3

        order = [(f"{k}_r{r}", k) for r in range(a.rounds) for k in ARMS]
        order.append(("base_last", "base"))
        res = {}
        for label, arm in order:
            res[label] = round(run_arm(arm), 4)
            print(label, res[label], flush=True)

        med = {k: statistics.median([v for lbl, v in res.items() if lbl.startswith(k + "_r")])
               for k in ARMS}

        # the shipped default, on the shipped kernels, after all the swapping
        rbp.KERNEL_DIR = shipped
        rbp._CT_STREAM_PIN = rbp._CT_BUFS_PIN = None
        rbp._CACHE.clear()
        got = rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG)
        exact = bool(torch.equal(ttnn.to_torch(got), want))

        out = {
            "shape": [1, a.n, a.n, a.c], "median_ms_by_arm": med, "arms_in_order": res,
            "aa_floor_pct": round(abs(res["base_r0"] - res["base_last"]) / res["base_r0"] * 100, 4),
            "xor_proxy_ratio": round(med["base"] / med["xor8"], 4),
            "legal_ratio": round(med["base"] / med["legal"], 4),
            "structure_cost_ratio": round(med["base"] / med["onebank"], 4),
            "banks_buy_ratio": round(med["onebank"] / med["legal"], 4),
            "read_leg_ms": round(med["base"] - med["noread"], 4),
            "read_leg_share": round((med["base"] - med["noread"]) / med["base"], 4),
            "default_bit_exact_vs_ttnn_permute": exact,
            "aiclk_during_sysfs_20hz": Aiclk.stat(list(clk.samples)),
            "aiclk_during_measuring_thread": Aiclk.stat([v for v in mt if v]),
        }
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        shutil.rmtree(tmp, ignore_errors=True)
    finally:
        clk.stop.set()
        tt_dev.cleanup()


if __name__ == "__main__":
    main()
