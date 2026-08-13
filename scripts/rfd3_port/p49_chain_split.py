"""Is the atom attention chain compute-bound or byte-bound? It decides what the S1 kernel must do.

p46 showed the chain costs 15.06 % of its dense time when the key axis is 1024 instead of 6080. But
that shrinks the BYTES and the COMPUTE together, and the two are collectable by different kernels:

  * skipping the arithmetic of empty tiles, survivors left at their original positions, is bit-exact
    (p48/p49: padding the key axis with masked tiles is torch.equal at every width up to 2x);
  * shrinking the materialised buffer requires compacting it, which relocates survivors, which is
    NOT bit-exact.

So if the chain is byte-bound, an in-place skip collects almost nothing and the exact form of S1 is
worthless. If it is compute-bound, the exact form gets most of the 6.64x.

This times each op of the chain separately at the real full width and labels it.
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

B, H, L, HD, NK = 2, 4, 6051, 32, 6080


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
                                           math_approx_mode=False, fp32_dest_acc_en=False,
                                           packer_l1_acc=True)
    g = torch.Generator().manual_seed(7)
    q = ttnn.from_torch(torch.randn(B, H, L, HD, generator=g) * 0.1, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    kt = ttnn.from_torch(torch.randn(B, H, HD, NK, generator=g) * 0.1, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    v = ttnn.from_torch(torch.randn(B, H, NK, HD, generator=g) * 0.1, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    bias = ttnn.from_torch(torch.randn(B, H, L, NK, generator=g), dtype=ttnn.float32,
                           layout=ttnn.TILE_LAYOUT, device=dev)

    def t(fn):
        r = fn()
        ttnn.synchronize_device(dev)
        ttnn.deallocate(r)
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            r = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(r)
        return statistics.median(ts) * 1e3

    scale = HD ** -0.5
    s_bf = ttnn.matmul(q, kt, compute_kernel_config=ckc)
    s32 = ttnn.typecast(s_bf, ttnn.float32, memory_config=s_bf.memory_config())
    sb = ttnn.add(s32, bias, input_tensor_a_activations=[
        ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, scale)])
    at = ttnn.softmax(sb, dim=-1)
    ab = ttnn.typecast(at, ttnn.bfloat16, memory_config=at.memory_config())

    rows = [
        ("matmul q@kT", "COMPUTE", lambda: ttnn.matmul(q, kt, compute_kernel_config=ckc)),
        ("typecast->fp32", "BYTES", lambda: ttnn.typecast(s_bf, ttnn.float32,
                                                          memory_config=s_bf.memory_config())),
        ("add+scale", "BYTES", lambda: ttnn.add(s32, bias, input_tensor_a_activations=[
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, scale)])),
        ("softmax", "BYTES", lambda: ttnn.softmax(sb, dim=-1)),
        ("typecast->bf16", "BYTES", lambda: ttnn.typecast(at, ttnn.bfloat16,
                                                          memory_config=at.memory_config())),
        ("matmul attn@v", "COMPUTE", lambda: ttnn.matmul(ab, v, compute_kernel_config=ckc)),
    ]

    gb = B * H * L * NK * 4 / 1e9
    print(f"score buffer fp32 = {gb:.3f} GB   (bf16 {gb / 2:.3f} GB)")
    tot = comp = 0.0
    res = {}
    for name, kind, fn in rows:
        ms = t(fn)
        tot += ms
        if kind == "COMPUTE":
            comp += ms
        res[name] = {"ms": ms, "kind": kind}
        print(f"{name:16s} {kind:8s} {ms:8.2f} ms", flush=True)
    byt = tot - comp
    print(f"\n{'TOTAL':16s} {'':8s} {tot:8.2f} ms")
    print(f"{'  compute':16s} {'':8s} {comp:8.2f} ms  ({100 * comp / tot:4.1f} %)  "
          f"<- what an in-place, bit-exact tile skip can delete")
    print(f"{'  byte passes':16s} {'':8s} {byt:8.2f} ms  ({100 * byt / tot:4.1f} %)  "
          f"<- needs compacted storage, which is not bit-exact")

    ttnn.close_device(dev)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"B": B, "H": H, "L": L, "n_key": NK, "score_gb_fp32": gb,
                                     "total_ms": tot, "compute_ms": comp, "byte_ms": byt,
                                     "rows": res}, indent=2))
        print(f"[done] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
