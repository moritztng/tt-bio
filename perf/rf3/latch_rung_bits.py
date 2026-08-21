#!/usr/bin/env python3
"""Is every rung of the two narrowing ladders bit-identical to the rung above it?

`_L1_OUT_RUNG` walks `(out_block_h, out_block_w)` down and `_BMM_CFG_RUNG` walks `per_core_M` down.
Neither moves `in0_block_w`, so the contraction is accumulated in the same order on every rung and
the walk should be `torch.equal`, not merely close. That is the claim this script tests, with
`packer_l1_acc` ON -- the setting that makes K-block order visible in the first place, so a false
pass is not hiding behind it.

The L1-output class is the one MEASURED to refuse: the RF3 768 aa triangle attention
out-projection, `(1, 768, 768, 64) x (64, 64)` bf16. The batched-matmul classes are rank-4
attention shapes with more than one legal `per_core_M`; that latch took 0 refusals on RF3 at 768
and 1024 aa, so there is no firing class to copy and these are in-class shapes rather than a
reproduction.

Every comparison carries an A/A control: rung 0 run twice. A control above 0 means the op is not
deterministic at this shape and the rung comparison says nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import ttnn                                                                     # noqa: E402
from tt_bio import tenstorrent as T                                             # noqa: E402


def ckc():
    """HiFi4 with `packer_l1_acc` on, which is what the pair-track projections run."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=True)


def rand(shape, dev, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(*shape, generator=g, dtype=torch.float32).to(torch.bfloat16)
    return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)


def l1_out_ladder(dev, x_shape, w_shape, bw_cap, block_w):
    """`ttnn.linear` to an L1 output at every rung of the drain-block ladder."""
    _, x = rand(x_shape, dev, 11)
    _, w = rand(w_shape, dev, 12)
    rows = []
    ref = None
    for rung in range(8):
        cfg = T._pair_proj_config(x, w, bw_cap=bw_cap, out_l1=True, block_w=block_w, rung=rung)
        if cfg is None:
            break
        try:
            o = ttnn.linear(x, w, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                            compute_kernel_config=ckc(), program_config=cfg)
        except Exception as e:                                                   # noqa: BLE001
            rows.append({"rung": rung, "obh": cfg.out_block_h, "obw": cfg.out_block_w,
                         "error": str(e)[:160]})
            continue
        h = ttnn.to_torch(o).float()
        ttnn.deallocate(o)
        if ref is None:
            ref = h
            # A/A: the same rung twice, so a nonzero rung diff can be read as a rung diff.
            o2 = ttnn.linear(x, w, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                             compute_kernel_config=ckc(), program_config=cfg)
            aa = ttnn.to_torch(o2).float()
            ttnn.deallocate(o2)
            rows.append({"rung": "0 (A/A control)", "obh": cfg.out_block_h,
                         "obw": cfg.out_block_w, "in0_block_w": cfg.in0_block_w,
                         "torch_equal": bool(torch.equal(ref, aa)),
                         "max_abs": float((ref - aa).abs().max())})
        rows.append({"rung": rung, "obh": cfg.out_block_h, "obw": cfg.out_block_w,
                     "in0_block_w": cfg.in0_block_w,
                     "torch_equal": bool(torch.equal(ref, h)),
                     "max_abs": float((ref - h).abs().max())})
    ttnn.deallocate(x)
    ttnn.deallocate(w)
    return rows


def bmm_ladder(dev, a_shape, b_shape):
    """`ttnn.matmul` with the batched program config at every rung of the per_core_M ladder."""
    _, a = rand(a_shape, dev, 21)
    _, b = rand(b_shape, dev, 22)
    batch = 1
    for d in a_shape[:-2]:
        batch *= d
    args = (batch, -(-a_shape[-2] // 32), -(-a_shape[-1] // 32), -(-b_shape[-1] // 32), 2)
    rows, ref = [], None
    for rung in range(16):
        cfg = T._batched_matmul_config(*args, rung)
        if cfg is None:
            break
        try:
            o = ttnn.matmul(a, b, compute_kernel_config=ckc(), program_config=cfg)
        except Exception as e:                                                   # noqa: BLE001
            rows.append({"rung": rung, "per_core_M": cfg.per_core_M, "error": str(e)[:160]})
            continue
        h = ttnn.to_torch(o).float()
        ttnn.deallocate(o)
        if ref is None:
            ref = h
            o2 = ttnn.matmul(a, b, compute_kernel_config=ckc(), program_config=cfg)
            aa = ttnn.to_torch(o2).float()
            ttnn.deallocate(o2)
            rows.append({"rung": "0 (A/A control)", "per_core_M": cfg.per_core_M,
                         "in0_block_w": cfg.in0_block_w,
                         "torch_equal": bool(torch.equal(ref, aa)),
                         "max_abs": float((ref - aa).abs().max())})
        rows.append({"rung": rung, "per_core_M": cfg.per_core_M,
                     "in0_block_w": cfg.in0_block_w,
                     "torch_equal": bool(torch.equal(ref, h)),
                     "max_abs": float((ref - h).abs().max())})
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--aa", type=int, default=768)
    args = ap.parse_args()

    dev = T.get_device()
    n = args.aa
    rep = {"grid": list(T.COMPUTE_GRID_MAIN), "l1_bank_bytes": T._l1_bank_bytes(), "aa": n,
           "cases": {}}

    # The class the census caught refusing, at the size it refused at.
    rep["cases"]["rf3_triatt_out_proj"] = l1_out_ladder(
        dev, (1, n, n, 64), (64, 64), T._PAIR_PROJ_L1_BW, None)
    # The pair FFN fc1 class, where the L1 leg is worth a 2.15 GB/call round trip.
    rep["cases"]["pair_ffn_fc1"] = l1_out_ladder(
        dev, (1, 32, n, 256), (256, 1024), T._PAIR_FFN_FC1_BW, T._PAIR_FFN_FC1_BLOCK_W)
    # Batched attention shapes with more than one legal per_core_M.
    rep["cases"]["bmm_attn_qk"] = bmm_ladder(dev, (32, 8, 256, 64), (32, 8, 64, 256))
    rep["cases"]["bmm_attn_av"] = bmm_ladder(dev, (32, 8, 256, 256), (32, 8, 256, 64))

    bad = []
    for name, rows in rep["cases"].items():
        for r in rows:
            if "error" in r:
                bad.append((name, r["rung"], r["error"]))
            elif not r["torch_equal"]:
                bad.append((name, r["rung"], f"max_abs {r['max_abs']}"))
    rep["all_bit_exact"] = not bad
    for name, rows in rep["cases"].items():
        print(f"== {name}")
        for r in rows:
            print("  " + json.dumps(r))
    print("ALL BIT EXACT" if not bad else f"NOT BIT EXACT: {bad}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
