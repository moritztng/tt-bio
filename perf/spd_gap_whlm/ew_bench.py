"""The protein LM's elementwise ops at the WH page shape, by where the activation lives: DRAM, L1 interleaved, L1
height-sharded over the grid.

The WH census puts ~40 % of an ESMC-300M page batch in elementwise ops (layer_norm 18.5, gate multiply 13.4, rope 12.1,
add 8.1 ms of 160.7) at ~70-100 GB/s against 248 GB/s of DRAM. Before redesigning the block around an L1-resident
residual stream, this says what each op costs per placement. Per arm: device ms (median of --repeat, --inner calls per
synchronize), GB/s of operand + output bytes, and the max abs difference from the DRAM arm.

    python3 perf/spd_gap_whlm/ew_bench.py --model esmc-300m --out ew.jsonl
"""
import argparse
import json
import math
import time

import torch
import ttnn

# (d_model, d_ff, rows of the page batch)
MODELS = {"esmc-300m": (960, 2560, 7 * 544), "esmc-600m": (1152, 3072, 7 * 544), "saprot-650m": (1280, 5120, 7 * 544)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=sorted(MODELS))
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    d, ff, m = MODELS[args.model]
    out = open(args.out, "w")
    gen = torch.Generator().manual_seed(0)

    def placements(width):
        mt = m // 32 if m % 32 == 0 else math.ceil(m / 32)
        n = min(g.x * g.y, mt)
        h = math.ceil(mt / n)
        n = math.ceil(mt / h)
        crs = ttnn.num_cores_to_corerangeset(n, g, row_wise=True)
        hs = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(crs, [h * 32, width], ttnn.ShardOrientation.ROW_MAJOR))
        return {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG, f"l1_hs{n}x{h}": hs}

    for width in (d, ff):
        a = torch.randn(1, 1, m, width, generator=gen).to(torch.bfloat16)
        b = torch.randn(1, 1, m, width, generator=gen).to(torch.bfloat16)
        w = (1 + 0.1 * torch.randn(width, generator=gen)).to(torch.bfloat16)
        ops = {
            "layer_norm": (lambda x, y, wt, mc: ttnn.layer_norm(x, weight=wt, epsilon=1e-5, compute_kernel_config=ck,
                                                                memory_config=mc), 2),
            "add": (lambda x, y, wt, mc: ttnn.add(x, y, memory_config=mc), 3),
            "multiply": (lambda x, y, wt, mc: ttnn.multiply(x, y, memory_config=mc), 3),
        }
        base = {}
        for place, mc in placements(width).items():
            try:
                xd = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
                yd = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
                wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)  # as Module.torch_to_tt
            except Exception as e:
                print(json.dumps({"model": args.model, "width": width, "place": place, "error": str(e).splitlines()[0][:200]}))
                continue
            for name, (fn, n_tensors) in ops.items():
                rec = {"model": args.model, "op": name, "shape": [m, width], "place": place}
                try:
                    o = fn(xd, yd, wd, mc)
                    got = ttnn.to_torch(o).float()
                    ttnn.deallocate(o)
                    ts = []
                    for _ in range(args.repeat):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        for _ in range(args.inner):
                            ttnn.deallocate(fn(xd, yd, wd, mc))
                        ttnn.synchronize_device(dev)
                        ts.append((time.perf_counter() - t0) / args.inner * 1e3)
                    ms = sorted(ts)[len(ts) // 2]
                    rec.update(ms=round(ms, 4), gbs=round(n_tensors * m * width * 2 / ms / 1e6, 1))
                    if name not in base:
                        base[name] = got
                    rec["max_diff_vs_dram"] = float((got - base[name]).abs().max())
                except Exception as e:  # a placement the op refuses is a result, not a crash
                    rec["error"] = str(e).splitlines()[0][:200]
                print(json.dumps(rec), flush=True)
                out.write(json.dumps(rec) + "\n")
                out.flush()
            for t in (xd, yd, wd):
                ttnn.deallocate(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
