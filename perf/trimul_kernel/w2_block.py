#!/usr/bin/env python3
"""Block-level A/B of the trimul output projection at 298 aa: ttnn.linear vs minimal_matmul.

The op-level A/B (w2_arms.py) put this at 7.122 -> 6.378 ms pipe per trimul. Two trimuls per
Pairformer block, so the block should move by ~1.49 ms of 40.28. This measures that on the real
block instead of adding it up, and reports the parity of the block output, which is what the
trunk actually propagates.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:perfwar-trimul-kernel \
        python3 -u perf/trimul_kernel/w2_block.py --n 320
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402


def parity(a, b):
    af, bf = a.float(), b.float()
    d = af - bf
    return dict(
        bit_exact=bool(torch.equal(a, b)),
        rmsd=round(float(d.pow(2).mean().sqrt()), 7),
        max_abs=round(float(d.abs().max()), 7),
        ref_std=round(float(af.std()), 5),
        pcc=round(float(torch.corrcoef(torch.stack([af.flatten(), bf.flatten()]))[0, 1]), 8),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    N = args.n

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build_layer(ckc)
    torch.manual_seed(0)
    s_host = torch.randn(1, N, 384)
    # The block consumes its z (in-place residual adds deallocate it), so every call gets a
    # freshly uploaded copy of the SAME host tensor. The upload sits outside the timed region,
    # which is synced on both sides.
    z_host = torch.randn(1, N, N, c_z)
    print(f"N={N} c_z={c_z}", flush=True)

    def fresh(h):
        return ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    rows, ref = [], None
    for name, mm in (("linear", False), ("minimal_matmul", True)):
        T._TRIMUL_MM_OUT = mm
        T._TRIMUL_OUT_MOVE_DRAM = False
        try:
            # Warm, then one call kept for parity, then timed synced calls one at a time so
            # the block's own peak allocation is never multiplied by a pipe depth.
            for _ in range(3):
                s, z = layer(fresh(s_host), fresh(z_host))
                ttnn.deallocate(s)
                ttnn.deallocate(z)
            ttnn.synchronize_device(dev)
            s, z = layer(fresh(s_host), fresh(z_host))
            h = ttnn.to_torch(z)
            ttnn.deallocate(s)
            ttnn.deallocate(z)
            if ref is None:
                ref = h
            ts = []
            for _ in range(args.iters):
                si, zi = fresh(s_host), fresh(z_host)
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                s, z = layer(si, zi)
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(s)
                ttnn.deallocate(z)
            med = sorted(ts)[len(ts) // 2]
            p = parity(ref, h)
            rows.append(dict(arm=name, block_ms=round(med, 3),
                             series=[round(t, 2) for t in ts], **p))
            print(f"BLOCK {name:16s} median {med:7.3f} ms  series "
                  f"{[round(t, 2) for t in ts]}  exact={p['bit_exact']} "
                  f"rmsd={p['rmsd']:.6f} (std {p['ref_std']}) pcc={p['pcc']}", flush=True)
        except Exception:
            traceback.print_exc()
            rows.append(dict(arm=name, error=traceback.format_exc()[-600:]))

    if len(rows) == 2 and "block_ms" in rows[0] and "block_ms" in rows[1]:
        print(f"speedup {rows[0]['block_ms'] / rows[1]['block_ms']:.4f}x", flush=True)
    T._TRIMUL_MM_OUT = True
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
