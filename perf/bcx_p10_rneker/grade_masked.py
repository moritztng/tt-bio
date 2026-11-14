"""Grade `rne_add_masked` (AF2 dropout folded into the residual) on a card, and time it.

    out = round_rne_bf16(x + round_rne_bf16(u * m)),   m a per-row keep mask [n, c]

Reference: torch on the host, where a bfloat16 multiply and add each compute in float32 and round
to nearest even. A bf16 x bf16 product and the sum of two bf16 values are exact in float64, so the
reference is checked against float64 once (`ref_is_exact`) and every device element must match it
exactly. The unmasked `rne_add` is graded the same way, because the masked arm changed its program
arguments. Timing: the masked kernel against the two calls it replaces (`ttnn.multiply` then
`rne_add`), warm medians, with the AICLK read off the card during the timed loop.

Run with TT_VISIBLE_DEVICES=<card>. Writes JSON to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn                                                             # noqa: E402
from tt_bio import rne_add                                              # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                     # noqa: E402


def rne(t64):
    return t64.to(torch.float32).to(torch.bfloat16)


def aiclk():
    card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        return int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read().split()[0])
    except OSError:
        return None


def timed(fn, dev, reps):
    for _ in range(3):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ts, clk = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(1e3 * (time.perf_counter() - t0))
        clk.append(aiclk())
        ttnn.deallocate(o)
    return statistics.median(ts), clk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[224, 195])
    ap.add_argument("--c", type=int, default=128)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ensure_p300_mesh_descriptor()
    rne_add.MASKED = True
    dev = ttnn.open_device(device_id=0)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rep = {"cells": [], "aiclk_before": aiclk()}
    try:
        for n in a.n:
            g = torch.Generator().manual_seed(n)
            x = (torch.randn(1, n, n, a.c, generator=g) * 4).to(torch.bfloat16)
            u = (torch.randn(1, n, n, a.c, generator=g)).to(torch.bfloat16)
            # AF2's keep mask (0 or 1/0.75) on half the slab, arbitrary values on the rest so the
            # product actually rounds; ties are frequent with 8-bit mantissas.
            keep = (torch.rand(1, 1, n, a.c, generator=g) > 0.25).to(torch.float32) / 0.75
            m = torch.where(torch.arange(a.c) < a.c // 2, keep,
                            torch.randn(1, 1, n, a.c, generator=g)).to(torch.bfloat16)
            prod64 = u.double() * m.double()
            ref = (x + u * m)                                   # torch bf16: RNE each step
            ref64 = rne(x.double() + rne(prod64).double())
            cell = {"n": n, "c": a.c, "ref_is_exact": bool(torch.equal(ref, ref64))}
            dx, du, dm = up(x), up(u), up(m)
            cfg = ttnn.DRAM_MEMORY_CONFIG
            cell["eligible"] = rne_add.masked_eligible(dx, du, dm, cfg)
            if cell["eligible"]:
                got = ttnn.to_torch(rne_add.rne_add_masked(dx, du, dm, memory_config=cfg))
                cell["masked_wrong"] = int((got.reshape(ref.shape) != ref).sum())
                inputs_ok = (torch.equal(ttnn.to_torch(dx).reshape(x.shape), x)
                             and torch.equal(ttnn.to_torch(du).reshape(u.shape), u))
                cell["inputs_untouched"] = bool(inputs_ok)
                cell["masked_ms"], cell["masked_aiclk"] = timed(
                    lambda: rne_add.rne_add_masked(dx, du, dm, memory_config=cfg), dev, a.reps)
            # The two calls it replaces, as af2 runs them today, and their accuracy.
            prod = ttnn.multiply(du, dm)
            two = ttnn.to_torch(rne_add.rne_add(dx, prod, memory_config=cfg))
            cell["chain_wrong"] = int((two.reshape(ref.shape) != ref).sum())
            plain_ref = x + u
            cell["unmasked_wrong"] = int(
                (ttnn.to_torch(rne_add.rne_add(dx, du, memory_config=cfg)).reshape(x.shape)
                 != plain_ref).sum())
            ttnn.deallocate(prod)
            # widen_add is the same program at ROUND_MODE=0: it must still build and be exact.
            wide = ttnn.to_torch(rne_add.widen_add(dx, du)).reshape(x.shape)
            cell["widen_wrong"] = int((wide.double() != (x.double() + u.double()).float().double()).sum())

            def chain():
                p = ttnn.multiply(du, dm)
                o = rne_add.rne_add(dx, p, memory_config=cfg)
                ttnn.deallocate(p)
                return o
            cell["chain_ms"], cell["chain_aiclk"] = timed(chain, dev, a.reps)
            cell["elements"] = x.numel()
            rep["cells"].append(cell)
            print(json.dumps({k: v for k, v in cell.items() if not k.endswith("aiclk")}), flush=True)
    finally:
        rep["rejects"] = {str(k): v for k, v in rne_add.REJECTS.items()}
        pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
