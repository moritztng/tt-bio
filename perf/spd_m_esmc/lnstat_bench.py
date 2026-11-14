"""What an input LayerNorm folded into its consumer would leave behind, at the ESMC page shapes.

Device ms a call (median of --repeat, --inner calls per synchronize) for: the shipped affine LayerNorm to DRAM and to
L1, the same without gamma / beta, the row statistics alone (layer_norm_pre_all_gather: sum x and sum x^2 per row),
and a per-row broadcast multiply (the scale a consumer without an epilogue would pay).

    python3 perf/spd_m_esmc/lnstat_bench.py --out lnstat.jsonl
"""
import argparse
import json
import time

import torch
import ttnn

SHAPES = {"esmc-300m": (7, 544, 960), "esmc-600m": (7, 544, 1152), "esmc-6b": (1, 544, 2560)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    out = open(a.out, "w")
    gen = torch.Generator().manual_seed(0)
    for model, shape in SHAPES.items():
        d = shape[-1]
        x = ttnn.from_torch(torch.randn(*shape, generator=gen).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
        w = ttnn.from_torch((1 + 0.1 * torch.randn(d, generator=gen)).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
        b = ttnn.from_torch((0.1 * torch.randn(d, generator=gen)).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
        col = ttnn.from_torch(torch.rand(*shape[:-1], 1).to(torch.bfloat16) + 0.5, layout=ttnn.TILE_LAYOUT, device=dev)
        l1 = ttnn.L1_MEMORY_CONFIG
        arms = {
            "ln_affine_dram": lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck),
            "ln_affine_l1": lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck, memory_config=l1),
            "ln_plain_dram": lambda: ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=ck),
            "rms_dram": lambda: ttnn.rms_norm(x, weight=w, epsilon=1e-5, compute_kernel_config=ck),
            "stats": lambda: ttnn.layer_norm_pre_all_gather(x, compute_kernel_config=ck),
            "stats_l1": lambda: ttnn.layer_norm_pre_all_gather(x, compute_kernel_config=ck, memory_config=l1),
            "row_mul_dram": lambda: ttnn.multiply(x, col),
        }
        for name, fn in arms.items():
            try:
                t = fn()
                rec = {"model": model, "arm": name, "out_shape": [int(s) for s in t.padded_shape]}
                ttnn.deallocate(t)
                ts = []
                for _ in range(a.repeat):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(a.inner):
                        ttnn.deallocate(fn())
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) / a.inner * 1000)
                rec["ms"] = round(sorted(ts)[len(ts) // 2], 4)
            except Exception as exc:  # an op the device refuses at this shape is a result too
                rec = {"model": model, "arm": name, "error": str(exc).splitlines()[0][:300]}
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
