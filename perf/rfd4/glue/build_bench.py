"""The card's pair build (`PairOnCard`) at one size: synced wall per build and error against float64, per variant.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/glue/build_bench.py --n 512 --out build.json

Variants of the step after the gather, z = x * s * (w gamma) + beta (x = const + W_x[bin], s per pair):
* `ship`: the shipped `PairOnCard.__call__`
* `flat`: the same ops on the [1, 1, n*n, pair] view, so the per-channel operands are plain row broadcasts
* `addcmul`: beta + y * wg as one ternary op
* `onehot*`: the gather as one matmul, onehot(bin) x [hi; mid; lo] with K = 3 x 32, fp32 accumulation; each output is
  the three bf16 parts summed in the fp32 dest, which is the fp32 weight exactly (pair.split_bf16)
The census (call_prof) has the per-op device times; this has the wall a step pays, host uploads included.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/, for clocksample

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import torch  # noqa: E402
import ttnn  # noqa: E402
import clocksample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/ttuser/rfd4-bundle")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import pair as P
    from tt_bio.tenstorrent import upload

    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    dev, b = engine._device, engine.pair
    split = b.split
    n, g = a.n, torch.Generator().manual_seed(0)
    c = int(split.w_step.shape[-1])
    nb = split.nbins
    const = P.PairConstant(torch.randn(1, n, n, c, generator=g), torch.rand(1, n, n, nb, generator=g) + 0.5)
    p = P.PairInputs(const, torch.randint(0, nb, (1, n, n), generator=g), torch.rand(1, n, n, generator=g) + 0.5,
                     torch.randn(1, c, generator=g), torch.randn(1, c, generator=g))
    ref = split.assemble(p)

    from tt_bio.rfd4.trunk import compute_kernel_config
    assert nb <= 32
    eye = torch.zeros(nb, 96)
    wst = torch.zeros(96, c)
    for k, part in enumerate(P.split_bf16(split.w_step).float()):
        eye[torch.arange(nb), 32 * k + torch.arange(nb)] = 1.0
        wst[32 * k:32 * k + nb] = part
    eye_d = ttnn.from_torch(eye.bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    wst_d = ttnn.from_torch(wst.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ckc = compute_kernel_config()

    def onehot_rows(bins):
        idx = ttnn.from_torch(bins.reshape(1, n * n).to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                              dtype=ttnn.uint32)
        oh = ttnn.embedding(idx, eye_d, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(idx)
        rows = ttnn.matmul(oh, wst_d, dtype=ttnn.float32, compute_kernel_config=ckc)
        ttnn.deallocate(oh)
        return ttnn.reshape(rows, (1, n, n, c))

    def affine(flat, ternary, gather=None):
        def build():
            cst = b._constant(p.constant, 0)
            rows = (gather or b.rows)(p.bins)
            x = ttnn.add(rows, cst)
            ttnn.deallocate(rows)
            s = upload(p.scale[..., None].float().contiguous(), dtype=ttnn.float32, device=dev)
            wg = (split.ln_weight.double() * p.gamma.double()).float()
            wg, beta = (ttnn.from_torch(v[:, None, None], layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
                        for v in (wg, p.beta))
            y = ttnn.multiply(x, s)
            ttnn.deallocate(x)
            ttnn.deallocate(s)
            if flat:
                y = ttnn.reshape(y, (1, 1, n * n, c))
            if ternary:
                z = ttnn.addcmul(beta, y, wg)
            else:
                x = ttnn.multiply(y, wg)
                z = ttnn.add(x, beta)
                ttnn.deallocate(x)
            for t in (y, wg, beta):
                ttnn.deallocate(t)
            return ttnn.reshape(z, (1, n, n, c)) if flat else z
        return build

    variants = {"ship": lambda: b(p, 0), "flat": affine(True, False), "addcmul": affine(False, True),
                "flat_addcmul": affine(True, True), "onehot": affine(False, False, onehot_rows),
                "onehot_flat_addcmul": affine(True, True, onehot_rows)}
    rec = {"n": n, "reps": a.reps, "variants": {}}
    with clocksample.during(period=0.5) as clk:
        for name, fn in variants.items():
            try:
                z = fn()
                got = ttnn.to_torch(z).double()
                ttnn.deallocate(z)
                err = {"rel_rms": float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt()),
                       "max_abs": float((got - ref).abs().max())}
                ms = []
                for _ in range(a.reps):
                    ttnn.synchronize_device(dev)
                    t = time.perf_counter()
                    z = fn()
                    ttnn.synchronize_device(dev)
                    ms.append(1e3 * (time.perf_counter() - t))
                    ttnn.deallocate(z)
                rec["variants"][name] = {"median_ms": round(statistics.median(ms), 3), "min_ms": round(min(ms), 3),
                                         **err}
            except Exception as e:  # a variant ttnn refuses is a result, not a crash
                rec["variants"][name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(name, rec["variants"][name], flush=True)
    rec["aiclk"], rec["load1"] = clk.summary(), os.getloadavg()[0]
    Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
