#!/usr/bin/env python3
"""Trimul matmul K block: which Kt the shared CB pricer narrows, and what narrowing costs.

Prints the allocator's bank bytes and grid, then per Kt the band's block, the priced one and
the CB bytes of each. At the Kt given by --run it runs the triangle product both ways,
[1, C, S, S] x [1, C, S, S] with transpose_b as the trimul issues it, and scores both against
a float64 torch product of the same bf16 operands.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kt", type=int, nargs="+", default=[49, 65, 70, 77, 79, 80, 81, 84, 88, 97])
ap.add_argument("--run", type=int, nargs="*", default=[81])
ap.add_argument("--channels", type=int, default=4)
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = T.get_device()
gx, gy = T.COMPUTE_GRID_MAIN
res = {"arch": str(dev.arch()), "grid": [gx, gy], "bank": T._l1_bank_bytes(),
       "budget": T._matmul_cb_budget(), "rows": [], "runs": []}
for kt in a.kt:
    m, n = -(-kt // gy), -(-kt // gx)
    band = T._trimul_in0_block_w(kt)
    w = T._triangle_mul_program_config(kt).in0_block_w
    res["rows"].append({"kt": kt, "tokens": kt * 32, "band_w": band, "w": w,
                        "cb_band": T._matmul_cb_bytes(band, m, n, 2), "cb_w": T._matmul_cb_bytes(w, m, n, 2)})
    print(res["rows"][-1], flush=True)

ckc = ttnn.types.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                             fp32_dest_acc_en=True, packer_l1_acc=True) \
    if dev.arch() == ttnn.Arch.WORMHOLE_B0 else ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
for kt in a.run:
    s = kt * 32
    g = torch.Generator().manual_seed(kt)
    A = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
    B = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
    ref = torch.matmul(A.double(), B.double().transpose(-1, -2))
    ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tb = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shipped = T._triangle_mul_program_config(kt)
    for label, w in (("band", T._trimul_in0_block_w(kt)), ("priced", shipped.in0_block_w)):
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=w, out_subblock_h=1, out_subblock_w=1,
            out_block_h=shipped.per_core_M, out_block_w=shipped.per_core_N, per_core_M=shipped.per_core_M,
            per_core_N=shipped.per_core_N, transpose_mcast=False, fused_activation=None, fuse_batch=False)
        row = {"kt": kt, "arm": label, "w": w}
        try:
            out = ttnn.matmul(ta, tb, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                              program_config=pc, dtype=ttnn.bfloat16, transpose_b=True)
            o = ttnn.to_torch(out).double()
            ttnn.deallocate(out)
            err = (o - ref).norm() / ref.norm()
            row.update(rel_l2=float(err), max_abs=float((o - ref).abs().max()), ref_max=float(ref.abs().max()))
            row["_o"] = o
        except Exception as e:
            row["error"] = str(e).splitlines()[0][:300]
        res["runs"].append(row)
    arms = [r for r in res["runs"] if r["kt"] == kt and "_o" in r]
    if len(arms) == 2:
        d = (arms[0]["_o"] - arms[1]["_o"]).abs()
        res["runs"][-1]["band_vs_priced_max_abs"] = float(d.max())
    for r in res["runs"]:
        r.pop("_o", None)
        print(r, flush=True)
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
