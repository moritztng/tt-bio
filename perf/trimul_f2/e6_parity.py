#!/usr/bin/env python3
"""E6 parity: does the gate folded into the forward channel move match the two ops it replaces?

The plain reblock_permute is bit-exact by construction, a permute being a pure index reordering.
This one is not: it does arithmetic, so the acceptance is `torch.equal` against the SAME sequence
running on device -- `ttnn.chunk` + `ttnn.multiply_(p, g, [SIGMOID])` + `reblock_permute` -- and not
against torch on host, which would only prove both are near the same real number.

The compute config is the variable. `ttnn.multiply_` is called in the trimul with no
compute_kernel_config, so it runs the wheel's default; the fused kernel names its own. Sweep it
until one is exact, and record every one that is not, because a near-miss is the failure mode that
ships silently (perfwar-ttnn-silu-approx-mode-dropped).

Shapes: the CB ring-wrap bug in this kernel family passes at N=128 and N=256 and produces silent
garbage at N=512, so a single-shape check is worthless.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.reblock_permute as RB
from tt_bio.tenstorrent import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="288,320,352,384,448,512,576,640")
    ap.add_argument("--cs", default="32,64,256")
    ap.add_argument("--sweep", action="store_true", help="sweep the compute config, not just the pin")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = get_device()
    res = {"host": "qb2", "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "cells": []}
    torch.manual_seed(0)

    configs = [(ttnn.MathFidelity.HiFi4, False)]
    if a.sweep:
        configs = [(f, acc) for f in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2,
                                      ttnn.MathFidelity.LoFi) for acc in (False, True)]

    try:
        for fid, acc in configs:
            RB.GATE_FIDELITY, RB.GATE_FP32_ACC = fid, acc
            RB._CACHE_GATED.clear()
            for N in [int(v) for v in a.ns.split(",")]:
                for C in [int(v) for v in a.cs.split(",")]:
                    cell = {"N": N, "C": C, "fidelity": str(fid), "fp32_dest_acc": acc}
                    try:
                        h = torch.randn(1, N, N, 4 * C, dtype=torch.bfloat16)
                        xw = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                             device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        # the production sequence, on device
                        g_a, g_b, p_a, p_b = ttnn.chunk(xw, chunks=4, dim=-1)
                        ref_a = ttnn.multiply_(
                            p_a, g_a, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                        ref_mv = RB.reblock_permute(ref_a, ttnn.DRAM_MEMORY_CONFIG)
                        ref = ttnn.to_torch(ref_mv)
                        for t in (g_a, g_b, p_b, ref_a, ref_mv):
                            ttnn.deallocate(t)
                        # the fused kernel
                        got_t = RB.reblock_permute_gated(xw, 2 * C, 0, C, ttnn.DRAM_MEMORY_CONFIG)
                        got = ttnn.to_torch(got_t)
                        ttnn.deallocate(got_t)
                        ttnn.deallocate(xw)
                        cell["shape_ok"] = list(got.shape) == list(ref.shape)
                        cell["equal"] = bool(torch.equal(got, ref))
                        if not cell["equal"] and cell["shape_ok"]:
                            d = (got.float() - ref.float()).abs()
                            cell["max_abs"] = float(d.max())
                            cell["n_diff"] = int((d > 0).sum())
                            cell["frac_diff"] = round(float((d > 0).float().mean()), 6)
                    except Exception as e:                                        # noqa: BLE001
                        cell["error"] = f"{type(e).__name__}: {e}"[:300]
                    res["cells"].append(cell)
                    print(f"  N={N:4d} C={C:3d} {str(fid).split('.')[-1]:6s} acc={int(acc)} "
                          f"-> {cell.get('equal', cell.get('error', '?'))}"
                          f"{'' if cell.get('equal') else '  ' + str(cell.get('max_abs', ''))}",
                          flush=True)
                    a.out.write_text(json.dumps(res, indent=1))
    finally:
        pass
    a.out.write_text(json.dumps(res, indent=1))
    ok = [c for c in res["cells"] if c.get("equal")]
    print(f"\n{len(ok)}/{len(res['cells'])} cells bit-exact")


if __name__ == "__main__":
    main()
