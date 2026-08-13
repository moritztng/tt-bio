"""The S1 kernel's first question, asked before any kernel exists: does the atom attention chain
actually get cheaper in proportion to the key width?

S1 measured a mean tile-list density of 0.1673 over a real R4 trajectory, and the atom decoder's
score-buffer chain is 61.6 % of its region. The predicted landing -- ~314 ms/step -> ~52 -- assumes
a block-sparse kernel that touches 16.73 % of the tiles costs 16.73 % of the time. Nothing measures
that, and it is the assumption the whole build rests on.

This screens it without writing the kernel, by shrinking the key axis instead of sparsifying it. The
dense chain at n_key = 1024 does exactly the arithmetic a perfect block-sparse kernel would do at
density 1024/6080, on contiguous tiles, with every op already tuned. So it is an UPPER BOUND on what
the kernel can reach, and if the chain does not scale here it cannot scale there.

The chain, verbatim from RFD3AtomBlock.__call__ (model.py:1481-1500):

    scores = matmul(q, k^T)            -> typecast fp32
    scores = add(scores, bias, MUL_UNARY_SFPU scale)
    attention = softmax(scores, -1)    -> typecast bf16
    out = attention @ v
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

B, H, L, HD = 2, 4, 6051, 32
FULL = 6080  # 6052 aligned up to a tile multiple


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="*",
                    default=[6080, 3040, 1536, 1024, 768])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--roof", type=float, default=385.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
                                           math_approx_mode=False, fp32_dest_acc_en=False,
                                           packer_l1_acc=True)
    g = torch.Generator().manual_seed(7)
    q = ttnn.from_torch(torch.randn(B, H, L, HD, generator=g) * 0.1, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)

    rows = {}
    for nk in a.widths:
        kt = ttnn.from_torch(torch.randn(B, H, HD, nk, generator=g) * 0.1, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev)
        v = ttnn.from_torch(torch.randn(B, H, nk, HD, generator=g) * 0.1, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev)
        bias = ttnn.from_torch(torch.randn(B, H, L, nk, generator=g), dtype=ttnn.float32,
                               layout=ttnn.TILE_LAYOUT, device=dev)

        def chain():
            s = ttnn.matmul(q, kt, compute_kernel_config=ckc)
            s = ttnn.typecast(s, ttnn.float32, memory_config=s.memory_config())
            s = ttnn.add(s, bias, input_tensor_a_activations=[
                ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, HD ** -0.5)])
            at = ttnn.softmax(s, dim=-1)
            ttnn.deallocate(s)
            ab = ttnn.typecast(at, ttnn.bfloat16, memory_config=at.memory_config())
            ttnn.deallocate(at)
            o = ttnn.matmul(ab, v, compute_kernel_config=ckc)
            ttnn.deallocate(ab)
            return o

        ttnn.deallocate(chain())
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            o = chain()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(o)
        ms = statistics.median(ts) * 1e3
        # the score buffer is written/read as fp32 several times; count one fp32 pass as the unit
        gb = B * H * L * nk * 4 / 1e9
        rows[nk] = {"ms": ms, "score_gb_fp32": gb, "implied_gbs": gb / (ms * 1e-3)}
        for t in (kt, v, bias):
            ttnn.deallocate(t)
        print(f"n_key {nk:5d}  ({nk / FULL:5.3f} of full)  {ms:8.2f} ms   "
              f"score buf {gb:6.3f} GB fp32   {gb / (ms * 1e-3):7.1f} GB/s per pass", flush=True)

    base = rows[a.widths[0]]["ms"]
    print(f"\n{'n_key':>7s} {'width frac':>11s} {'time frac':>10s} {'speedup':>9s} "
          f"{'scaling eff':>12s}")
    for nk in a.widths:
        wf = nk / a.widths[0]
        tf = rows[nk]["ms"] / base
        print(f"{nk:7d} {wf:11.4f} {tf:10.4f} {base / rows[nk]['ms']:8.2f}x "
              f"{wf / tf:11.2f}x", flush=True)
    print("\nscaling eff = (width fraction) / (time fraction). 1.00 means the chain is perfectly "
          "proportional to key width, which is the ceiling a block-sparse kernel could reach.")

    ttnn.close_device(dev)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"B": B, "H": H, "L": L, "head_dim": HD, "rows": rows},
                                    indent=2))
        print(f"[done] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
