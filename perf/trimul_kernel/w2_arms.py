#!/usr/bin/env python3
"""A/B the two candidate trimul changes on the real 298 aa module, and on the block.

Arms (both are independent toggles in tt_bio.tenstorrent):
  mm_out    the two output projections through ttnn.experimental.minimal_matmul instead of
            ttnn.linear(core_grid=). NOT bit-exact: the two kernels block the contraction
            differently, so bf16 accumulates in a different order.
  dram_move the output channel move writes straight to DRAM, dropping the separate clone
            that used to move the chunk there. Index-only, so it must be bit-exact.

Every arm is compared against the arm-off baseline on the SAME weights and input, and the
block is timed too so the op-level number is never quoted alone.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:perfwar-trimul-kernel \
        python3 -u perf/trimul_kernel/w2_arms.py --n 320
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

ARMS = [
    ("base", dict(mm=False, dram=False)),
    ("dram_move", dict(mm=False, dram=True)),
    ("mm_out", dict(mm=True, dram=False)),
    ("mm_out+dram_move", dict(mm=True, dram=True)),
]


def set_arm(cfg):
    T._TRIMUL_MM_OUT = cfg["mm"]
    T._TRIMUL_OUT_MOVE_DRAM = cfg["dram"]


def free(out):
    for t in (out if isinstance(out, (tuple, list)) else (out,)):
        ttnn.deallocate(t)


def timeit(dev, fn, warm=3, iters=7, pipe=10):
    """Median of `iters` synced calls, plus the pipelined per-call cost.

    Both regions are synced on entry and exit; the pipe number is the one to quote for a
    block that runs back to back, the serial one for a single op in isolation.
    """
    for _ in range(warm):
        free(fn())
    ttnn.synchronize_device(dev)
    ser = []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ser.append((time.perf_counter() - t0) * 1e3)
        free(r)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(pipe)]
    ttnn.synchronize_device(dev)
    pip = (time.perf_counter() - t0) * 1e3 / pipe
    for o in outs:
        free(o)
    return sorted(ser)[len(ser) // 2], pip


def parity(a, b):
    eq = bool(torch.equal(a, b))
    af, bf = a.float(), b.float()
    d = af - bf
    return dict(
        bit_exact=eq,
        rmsd=round(float(d.pow(2).mean().sqrt()), 7),
        max_abs=round(float(d.abs().max()), 7),
        ref_std=round(float(af.std()), 5),
        pcc=round(float(torch.corrcoef(torch.stack([af.flatten(), bf.flatten()]))[0, 1]), 8),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    N = args.n

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build_layer(ckc)
    mc = T._triangle_mul_memory_config(N)
    print(f"N={N} c_z={c_z} hidden={layer.triangle_multiplication_start._hidden} "
          f"chunk={T._trimul_chunk_size(N, layer.triangle_multiplication_start._hidden)} "
          f"memcfg={'L1' if mc.buffer_type == ttnn.BufferType.L1 else 'DRAM'}", flush=True)
    torch.manual_seed(0)
    s0 = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)

    rows = []
    for which in ("tri_mul_start", "tri_mul_end"):
        tm = (layer.triangle_multiplication_start if which == "tri_mul_start"
              else layer.triangle_multiplication_end)
        ref = None
        for name, cfg in ARMS:
            set_arm(cfg)
            o = tm(z, None)
            h = ttnn.to_torch(o)
            free(o)
            if ref is None:
                ref = h
            ser, pip = timeit(dev, lambda: tm(z, None))
            p = parity(ref, h)
            rows.append(dict(kind="op", module=which, arm=name, serial_ms=round(ser, 3),
                             pipe_ms=round(pip, 3), **p))
            print(f"{which:14s} {name:18s} serial {ser:7.3f}  pipe {pip:7.3f}  "
                  f"exact={p['bit_exact']} rmsd={p['rmsd']:.6f} (std {p['ref_std']}) "
                  f"pcc={p['pcc']}", flush=True)

    ref = None
    for name, cfg in ARMS:
        set_arm(cfg)
        o = layer(s0, z)
        h = ttnn.to_torch(o[1])
        free(o)
        if ref is None:
            ref = h
        ser, pip = timeit(dev, lambda: layer(s0, z), warm=2, iters=5, pipe=5)
        p = parity(ref, h)
        rows.append(dict(kind="block", module="pairformer_block", arm=name,
                         serial_ms=round(ser, 3), pipe_ms=round(pip, 3), **p))
        print(f"{'BLOCK':14s} {name:18s} serial {ser:7.3f}  pipe {pip:7.3f}  "
              f"exact={p['bit_exact']} rmsd={p['rmsd']:.6f} pcc={p['pcc']}", flush=True)

    set_arm(dict(mm=True, dram=True))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
