#!/usr/bin/env python3
"""Does a leading batch dim amortise per-op cost at the 117-aa trunk shape?

The multi-prediction plan rests on one number: at N=128 (117 aa padded), how much of a
trunk op's wall time is fixed per-op cost that B independent predictions would share,
and how much is work that scales with B?

Every timed region is bracketed by ttnn.synchronize_device on both sides (the queued-work
rule). Three legs per shape:

  serial  sync; t0; op(); sync; t1     -> full cost incl. device drain
  issue   sync; t0; op(); t1           -> upper bound on host-side issue cost
  pipe    sync; t0; K x op(); sync; t1 -> per-call wall with dispatch overlapped

Batching B predictions makes a row-wise pair op see (B, N, N, c), which for a row-local
op is the same tensor as (1, B*N, N, c). So the scaling can be measured without touching
the model.
"""

import argparse
import json
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

N = 128
C_Z = 256


def med(xs):
    return sorted(xs)[len(xs) // 2]


def timed(dev, fn, warm=3, iters=7, pipe=8):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)

    serial, issue = [], []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ttnn.synchronize_device(dev)
        t2 = time.perf_counter()
        serial.append((t2 - t0) * 1e3)
        issue.append((t1 - t0) * 1e3)

    pipes = []
    for _ in range(3):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        pipes.append((time.perf_counter() - t0) * 1e3 / pipe)
    return dict(serial_ms=round(med(serial), 4), issue_ms=round(med(issue), 4),
                pipe_ms=round(med(pipes), 4))


def make(dev, shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, action="append", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    bs = args.b or [1, 2, 4, 8]

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    torch.manual_seed(0)

    # weights, shared across B
    ln_w = make(dev, (C_Z,))
    ln_b = make(dev, (C_Z,))
    w1 = make(dev, (C_Z, 4 * C_Z))
    w2 = make(dev, (C_Z, 4 * C_Z))
    w3 = make(dev, (4 * C_Z, C_Z))
    w_qkv = make(dev, (C_Z, 3 * 8 * 32))
    w_o = make(dev, (8 * 32, C_Z))

    out = []
    for B in bs:
        rec = {"B": B, "N": N, "c_z": C_Z}
        x = make(dev, (1, B * N, N, C_Z))

        # 1. pair transition (swiglu): ~36% of block FLOPs, pure row-wise matmul
        def transition():
            xn = ttnn.layer_norm(x, weight=ln_w, bias=ln_b, epsilon=1e-5,
                                 compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            a = ttnn.linear(xn, w1, activation="silu", compute_kernel_config=ckc,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, core_grid=CORE_GRID_MAIN)
            b = ttnn.linear(xn, w2, compute_kernel_config=ckc,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, core_grid=CORE_GRID_MAIN)
            ttnn.deallocate(xn)
            a = ttnn.multiply_(a, b)
            ttnn.deallocate(b)
            y = ttnn.linear(a, w3, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(a)
            ttnn.deallocate(y)
        rec["transition"] = timed(dev, transition)

        # 2. one row-local attention: qkv proj + SDPA over the N axis + out proj
        def attn():
            qkv = ttnn.linear(x, w_qkv, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q, k, v = (ttnn.reshape(qkv[:, :, :, i * 256:(i + 1) * 256], (B * N, 8, N, 32))
                       for i in range(3))
            ttnn.deallocate(qkv)
            o = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
            for t in (q, k, v):
                ttnn.deallocate(t)
            o = ttnn.reshape(o, (1, B * N, N, 256))
            y = ttnn.linear(o, w_o, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            ttnn.deallocate(y)
        rec["attn"] = timed(dev, attn)

        # 3. elementwise add on the pair tensor (the residual adds; 5 per block)
        y0 = make(dev, (1, B * N, N, C_Z))

        def eltwise():
            ttnn.deallocate(ttnn.add(x, y0))
        rec["eltwise"] = timed(dev, eltwise)
        ttnn.deallocate(y0)
        ttnn.deallocate(x)

        # 4. the trimul layout permute, done per-prediction on a (B,N,N,c) view
        xb = make(dev, (B, N, N, 128))

        def permute():
            p = ttnn.permute(xb, (0, 3, 1, 2))
            t = ttnn.transpose(p, -2, -1)
            ttnn.deallocate(p)
            ttnn.deallocate(t)
        rec["permute"] = timed(dev, permute)
        ttnn.deallocate(xb)

        out.append(rec)
        print(json.dumps(rec), flush=True)

    print("\n--- per-prediction cost (pipe_ms / B), lower is better ---")
    hdr = f"{'op':12s}" + "".join(f"{('B=' + str(b)):>12s}" for b in bs) + f"{'B8 speedup':>12s}"
    print(hdr)
    for op in ("transition", "attn", "eltwise", "permute"):
        row = f"{op:12s}"
        per = []
        for r in out:
            v = r[op]["pipe_ms"] / r["B"]
            per.append(v)
            row += f"{v:12.4f}"
        row += f"{per[0] / per[-1]:12.2f}x"
        print(row)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
