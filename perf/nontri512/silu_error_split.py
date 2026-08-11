#!/usr/bin/env python3
"""Step C: `_UNFUSED_SILU` changes TWO things at once. Which one costs the plDDT?

The flag moves the silu off the fp32 dest accumulator and onto the bf16-packed matmul output, AND
it flips the sigmoid algorithm from Cody-Waite to exp_21f (because ttnn.silu on a bf16 tensor always
takes the fast branch). The fold measured the combined effect: plDDT 0.828628 -> 0.802729.

This splits it, at the real fc1 shape, with the fp32 accumulator as the common origin:

    acc32     ttnn.linear(x_norm, fc1_w, dtype=float32)      the accumulator itself
    prod      ttnn.linear(..., activation="silu", dtype=bf16) production, accurate silu on the acc
    matbf     ttnn.linear(..., dtype=bf16)                    the bf16-packed matmul output
    usilu     ttnn.silu(matbf)                                the _UNFUSED_SILU route
    accsilu   ttnn.silu(acc32) packed to bf16                 accurate silu, fp32 input, on device

and two host references built from acc32 in torch fp32:

    ref_exact  bf16(silu(acc32))                              what production should be
    ref_inrnd  bf16(silu(fp32(bf16(acc32))))                   input rounding ONLY, exact algorithm

so that

    ref_inrnd vs ref_exact   = the input-rounding half, alone
    usilu     vs ref_inrnd   = the algorithm half, alone
    usilu     vs prod        = the total the fold saw
    accsilu   vs prod        = a control: same algorithm, same input, must be 0

A third arm is priced here too, because it is the one a custom kernel would build: keep the fp32
accumulator as the silu input and change only the algorithm. That is `ref_inrnd`'s complement and
its error is read off the same table.
"""

import argparse
import json

import torch

import ttnn
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


def cmp(name, a, b):
    """a, b: torch fp32 tensors holding bf16-representable values."""
    n = a.numel()
    d = (a != b)
    k = int(d.sum())
    ae = (a - b).abs()
    # relative error against the reference b, only where b is non-zero
    nz = b != 0
    rel = (ae[nz] / b[nz].abs()).max() if int(nz.sum()) else torch.tensor(0.0)
    return {
        "name": name,
        "elems": n,
        "differ": k,
        "differ_frac": round(k / n, 6),
        "max_abs": float(ae.max()),
        "max_rel": float(rel),
        "rms": float((a - b).pow(2).mean().sqrt()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--cin", type=int, default=256)
    ap.add_argument("--chid", type=int, default=1024)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    dev = get_device()
    N, h, cin, chid = args.n, args.h, args.cin, args.chid
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    print(f"fc1 leg: x [1,{h},{N},{cin}] @ w [{cin},{chid}], HiFi4 + fp32_dest_acc, "
          f"grid {CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}", flush=True)

    # A layer-norm output is unit-variance by construction, so that is the input distribution.
    torch.manual_seed(0)
    xt = torch.randn(1, h, N, cin)
    xt = (xt - xt.mean(-1, keepdim=True)) / xt.std(-1, keepdim=True)
    wt = torch.randn(cin, chid) / (cin ** 0.5)

    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)

    # Outputs go to DRAM and come back to host one at a time. Holding several of these in L1 at
    # once clashes the next program's static circular buffers -- the same trap transition_gemm.py
    # documented. Memory config cannot reach the dest-mode branch, so the numerics are unaffected.
    def lin(dtype, activation=None):
        return ttnn.linear(x, w, activation=activation, compute_kernel_config=ckc,
                           memory_config=DRAM, dtype=dtype, core_grid=CORE_GRID_MAIN)

    acc32_t = lin(ttnn.float32)
    acc32 = ttnn.to_torch(acc32_t).to(torch.float32)
    accsilu_t = ttnn.silu(acc32_t, memory_config=DRAM)   # accurate branch, fp32 input, on device
    # accsilu comes back fp32; the production comparison is at bf16 output, so round it there.
    accsilu = ttnn.to_torch(accsilu_t).to(torch.bfloat16).to(torch.float32)
    ttnn.deallocate(acc32_t)
    ttnn.deallocate(accsilu_t)

    prod_t = lin(ttnn.bfloat16, "silu")
    prod = ttnn.to_torch(prod_t).to(torch.float32)
    ttnn.deallocate(prod_t)

    matbf_t = lin(ttnn.bfloat16)
    matbf = ttnn.to_torch(matbf_t).to(torch.float32)
    usilu_t = ttnn.silu(matbf_t, memory_config=DRAM)
    usilu = ttnn.to_torch(usilu_t).to(torch.float32)
    ttnn.deallocate(matbf_t)
    ttnn.deallocate(usilu_t)

    silu = torch.nn.functional.silu
    ref_exact = silu(acc32).to(torch.bfloat16).to(torch.float32)
    acc_bf = acc32.to(torch.bfloat16).to(torch.float32)
    ref_inrnd = silu(acc_bf).to(torch.bfloat16).to(torch.float32)

    rows = [
        # --- controls: the chain is only readable if these hold ---
        cmp("control: matmul bf16 output == bf16(acc32)", matbf, acc_bf),
        cmp("control: production fused == correctly-rounded torch on acc32", prod, ref_exact),
        cmp("control: device accurate silu on fp32 input == production fused", accsilu, prod),
        # --- the split ---
        cmp("INPUT ROUNDING alone (exact algo on bf16 input vs on acc32)", ref_inrnd, ref_exact),
        cmp("ALGORITHM alone (fast silu on bf16 input vs exact algo on same)", usilu, ref_inrnd),
        cmp("TOTAL, what the fold saw (_UNFUSED_SILU vs production)", usilu, prod),
    ]

    print("", flush=True)
    for r in rows:
        print(f"  {r['name']:62s} differ {r['differ_frac']*100:7.4f} %  "
              f"max_abs {r['max_abs']:.6g}  rms {r['rms']:.6g}", flush=True)

    if args.out:
        json.dump({"shape": [1, h, N, cin], "c_hid": chid, "rows": rows,
                   "grid": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}"},
                  open(args.out, "w"), indent=2)
        print("\nwrote " + args.out, flush=True)


if __name__ == "__main__":
    main()
