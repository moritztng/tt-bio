"""The dual-NOC in-projection with a bias, against the stock biased `minimal_matmul`, on a card.

The fused tri-mul gate asks `mm_dualnoc.in_proj(x, [w_gp | w_gout], bias=[b_gp | b_gout],
split=(512, 128))`. Every output element must equal the stock op's bit for bit: same kernels,
same block config, the bias added by the same compute kernel. Checked unsplit and split, at
BC2's 224-token pair shape and at 195, with timings and the AICLK read during the loop.

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
from tt_bio import mm_dualnoc as DN                                     # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                     # noqa: E402


def aiclk():
    card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        return int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read().split()[0])
    except OSError:
        return None


def timed(fn, dev, reps=20):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    ts, clk = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append(1e3 * (time.perf_counter() - t0))
        clk.append(aiclk())
    return statistics.median(ts), min(c for c in clk if c) if any(clk) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ensure_p300_mesh_descriptor()
    dev = ttnn.open_device(device_id=0)
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    ckc = cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    dram = ttnn.DRAM_MEMORY_CONFIG
    res = {"arch": str(dev.arch()), "cases": []}
    try:
        for S in (224, 195):
            g = torch.Generator().manual_seed(S)
            up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                           dtype=ttnn.bfloat16)
            x = up(torch.randn(1, S, S, 128, generator=g))
            w = up(torch.randn(128, 640, generator=g) / 11.3)
            b = up(torch.randn(640, generator=g))
            ref = ttnn.to_torch(ttnn.experimental.minimal_matmul(
                x, w, bias_tensor=b, memory_config=dram, dtype=ttnn.bfloat16,
                compute_kernel_config=ckc)).float()
            nob = ttnn.to_torch(ttnn.experimental.minimal_matmul(
                x, w, memory_config=dram, dtype=ttnn.bfloat16, compute_kernel_config=ckc)).float()
            for split in (None, (512, 128)):
                DN.REJECTS.clear()
                o = DN.in_proj(x, w, ckc, ttnn.bfloat16, dram, split, b)
                case = {"S": S, "split": split, "served": o is not None,
                        "rejects": {str(k): v for k, v in DN.REJECTS.items()}}
                if o is not None:
                    got = (ttnn.to_torch(o).float() if split is None else
                           torch.cat([ttnn.to_torch(t).float() for t in o], dim=-1))
                    got = got.reshape(ref.shape)
                    case.update(wrong=int((got != ref).sum()), n=ref.numel(),
                                max_abs=float((got - ref).abs().max()),
                                bias_moved=int((nob != ref).sum()))
                    case["ms_dualnoc"], case["aiclk"] = timed(
                        lambda: DN.in_proj(x, w, ckc, ttnn.bfloat16, dram, split, b), dev)
                    case["ms_stock"], _ = timed(lambda: ttnn.experimental.minimal_matmul(
                        x, w, bias_tensor=b, memory_config=dram, dtype=ttnn.bfloat16,
                        compute_kernel_config=ckc), dev)
                res["cases"].append(case)
                print(json.dumps(case), flush=True)
    finally:
        ttnn.close_device(dev)
    res["ok"] = all(c.get("served") and c.get("wrong") == 0 and c.get("bias_moved", 0) > 0
                    for c in res["cases"])
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print("OK" if res["ok"] else "FAIL")


if __name__ == "__main__":
    main()
