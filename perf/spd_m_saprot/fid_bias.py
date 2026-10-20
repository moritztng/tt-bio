"""Which operand a HiFi2 matmul truncates, and whether a block-fp8 activation removes the bias it leaves.

At HiFi2 the --fast projections read |y| ~0.3 % low against float64 (ls16/ls17, bias -0.003), which is what failed
qkv + out at HiFi2 (L7b). If the multiplier keeps 7 mantissa bits of in0 and drops bf16's eighth, an activation
already rounded to 7 bits (bf16 storage, host RNE) or stored as block-fp8 should read unbiased. Per shape and
fidelity: x as bf16, as bf16 rounded to 7 bits, as block-fp8 (device typecast from bf16, the op the model would add),
each on minimal_matmul at the shipped blocks, ms a call and rel_l2 / signed bias against float64 of the bf16 x.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/fid_bias.py --out fb.jsonl
"""
import argparse
import json
import math
import time

import torch
import ttnn

SHAPES = {"qkv": (1280, 3840, (8, 8, 8, 1, 4)), "out": (1280, 1280, (16, 4, 4, 1, 4)),
          "fc1": (1280, 5120, (4, 8, 8, 1, 4))}


def round7(x):
    """bf16 values rounded to nearest-even at 7 significant bits (bf16 carries 8)."""
    i = x.float().view(torch.int32)
    lsb = (i >> 17) & 1
    return ((i + 0xFFFF + lsb) & ~0x1FFFF).view(torch.float32).to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=7 * 544)
    ap.add_argument("--shapes", default="qkv,out,fc1")
    ap.add_argument("--fids", default="HiFi2,HiFi3")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T
    T.set_fast_mode(True)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    out = open(a.out, "a")
    gen = torch.Generator().manual_seed(0)
    M = a.rows

    def timed(fn):
        o = fn(); ttnn.synchronize_device(dev)
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            for _ in range(a.calls):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t) / a.calls * 1e3)
        return o, sorted(ts)[1]

    def emit(rec):
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n"); out.flush()

    for name in a.shapes.split(","):
        K, N, (mb, kb, nb, sh, sw) = SHAPES[name]
        # LN-output-like activations: unit variance with a few wide channels
        x = (torch.randn(M, K, generator=gen) * (1 + 3 * (torch.rand(K, generator=gen) < 0.02))).to(torch.bfloat16)
        w = (torch.randn(K, N, generator=gen) / math.sqrt(K)).to(torch.bfloat16)
        b = (torch.randn(N, generator=gen) * 0.1).to(torch.bfloat16)
        wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat8_b)
        wq = ttnn.to_torch(wd).double()
        bd = ttnn.from_torch(b.reshape(1, N), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        ref = x.double() @ wq + b.double()
        L1 = ttnn.L1_MEMORY_CONFIG
        xs = {"bf16": ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=L1),
              "r7": ttnn.from_torch(round7(x), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=L1)}
        x8, tc_ms = timed(lambda: ttnn.typecast(xs["bf16"], ttnn.bfloat8_b, memory_config=L1))
        xs["b8"] = x8
        x8h = ttnn.to_torch(x8).double()
        emit(dict(shape=name, arm="typecast", ms=round(tc_ms, 4),
                  rel_l2=float((x8h - x.double()).norm() / x.double().norm()),
                  bias=float(((x8h - x.double()) * x.double().sign()).mean() / x.double().abs().mean())))
        cfg = ttnn.MinimalMatmulConfig(M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh, subblock_w=sw,
                                       compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y))
        for fid in a.fids.split(","):
            ck = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                                  fp32_dest_acc_en=True, packer_l1_acc=True)
            for tag, xd in xs.items():
                try:
                    o, ms = timed(lambda: ttnn.experimental.minimal_matmul(
                        xd, wd, bias_tensor=bd, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, config=cfg))
                    got = ttnn.to_torch(o).double().reshape(ref.shape); ttnn.deallocate(o)
                    d = got - ref
                    emit(dict(shape=name, fid=fid, x=tag, ms=round(ms, 4), tflops=round(2 * M * K * N / ms / 1e9, 1),
                              rel_l2=float(d.norm() / ref.norm()),
                              bias=float((d * ref.sign()).mean() / ref.abs().mean())))
                except Exception as e:  # noqa: BLE001 -- a refused arm is a data point
                    emit(dict(shape=name, fid=fid, x=tag, error=str(e).splitlines()[0][:200]))
        for t in xs.values():
            ttnn.deallocate(t)
        ttnn.deallocate(wd); ttnn.deallocate(bd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
