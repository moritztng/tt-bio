#!/usr/bin/env python3
"""Achievable matmul rate at the shapes a fold really runs, with the shape list taken from a census.

perf/ceiling/shape_roof.py hard-codes eight 298 aa shapes. Re-typing that list for a second size is
how a ceiling picks up a transcription error, so this reads the classes out of a
`perf/ceiling/flopfold.py` census, pads them the way the tile does, keeps every class carrying at
least `--min-frac` of the fold's executed arithmetic, and measures each one at the padded shape it
really executes. Every backend the codebase has is tried and the best is kept, which is what makes
the result a roof for this model rather than for ttnn's default.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:<slug> \
      python3 perf/moonshot512/shape_roof_census.py \
        --census perf/moonshot512/census_fold_p512.json \
        --out perf/moonshot512/shape_roof_512_qb2c1.json
"""
import argparse
import json
import math
import statistics as st
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/perf/", 1)[0])
from tt_bio.tenstorrent import get_device  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
c32 = lambda x: int(math.ceil(x / 32) * 32)


def timed(dev, fn, warm=3, pipe=4, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / pipe)
    return st.median(out)


def dims(s):
    return [int(x) for x in s.split("x")]


def pad_exec(a, b):
    """Padded (a_shape, b_shape) and the (K, N) the rate model keys on.

    Same rule as perf/ceiling/ceiling_model.py: only tiled dims pad. For `[..., M, K] @ [K, N]`
    that is M, K and N; leading dims are batch and do not pad.
    """
    A, B = dims(a), dims(b)
    if len(B) == 2 and A[-1] == B[0]:
        Ap = A[:-2] + [c32(A[-2]), c32(A[-1])] if len(A) >= 2 else [c32(A[-1])]
        Bp = [c32(B[0]), c32(B[1])]
        return Ap, Bp, c32(A[-1]), c32(B[1])
    if len(B) > 2 and A[-1] == B[-2]:
        Ap = A[:-2] + [c32(A[-2]), c32(A[-1])]
        Bp = B[:-2] + [c32(B[-2]), c32(B[-1])]
        return Ap, Bp, c32(A[-1]), c32(B[-1])
    Ap = A[:-2] + [c32(A[-2]), c32(A[-1])] if len(A) >= 2 else [c32(A[-1])]
    return Ap, dims(b), c32(A[-1]), c32(A[-1])


def flops_of(A, B):
    if len(B) == 2:
        m = 1
        for d in A[:-1]:
            m *= d
        return 2 * m * B[0] * B[1]
    batch = 1
    for d in A[:-2]:
        batch *= d
    return 2 * batch * A[-2] * A[-1] * B[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True, help="flopfold.py output for the size being priced")
    ap.add_argument("--min-frac", type=float, default=0.01,
                    help="keep classes carrying at least this fraction of executed FLOPs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cen = json.load(open(args.census))
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    print(f"grid {g.x}x{g.y} = {g.x*g.y} cores", flush=True)
    KC = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    MM = getattr(ttnn.experimental, "minimal_matmul", None)

    # ---- build the class list: executed FLOPs per (op, padded a, padded b)
    classes, sdpa = {}, {}
    for e in cen["top_matmul_shapes"]:
        if e["op"] == "scaled_dot_product_attention":
            A = dims(e["a"])
            f = (c32(A[-2]) / A[-2]) ** 2                 # q and k sequence dims both pad
            key = tuple(A[:-2] + [c32(A[-2]), c32(A[-1])])
            d = sdpa.setdefault(key, {"exec_flops": 0.0, "stages": set()})
            d["exec_flops"] += e["flops"] * f
            d["stages"].add(e["stage"])
            continue
        Ap, Bp, K, N = pad_exec(e["a"], e["b"])
        lg = flops_of(dims(e["a"]), dims(e["b"]))
        ex = flops_of(Ap, Bp)
        key = (e["op"], tuple(Ap), tuple(Bp))
        d = classes.setdefault(key, {"exec_flops": 0.0, "K": K, "N": N, "stages": set()})
        d["exec_flops"] += e["flops"] * (ex / lg if lg else 1.0)
        d["stages"].add(e["stage"])

    tot = sum(v["exec_flops"] for v in classes.values()) + sum(v["exec_flops"] for v in sdpa.values())
    keep = {k: v for k, v in classes.items() if v["exec_flops"] / tot >= args.min_frac}
    keep_sdpa = {k: v for k, v in sdpa.items() if v["exec_flops"] / tot >= args.min_frac}
    covered = sum(v["exec_flops"] for v in keep.values()) + sum(v["exec_flops"] for v in keep_sdpa.values())
    print(f"executed {tot/1e12:.1f} TFLOP over {len(classes)+len(sdpa)} classes; measuring "
          f"{len(keep)+len(keep_sdpa)} that carry {100*covered/tot:.1f}%\n", flush=True)

    res = {"census": args.census, "grid": [g.x, g.y], "cores": g.x * g.y,
           "exec_tflop_total": round(tot / 1e12, 3), "coverage_frac": round(covered / tot, 4),
           "classes": {}, "sdpa": {}}

    for (op, Ap, Bp), v in sorted(keep.items(), key=lambda kv: -kv[1]["exec_flops"]):
        label = f"{op} {'x'.join(map(str,Ap))}@{'x'.join(map(str,Bp))}"
        gf = flops_of(list(Ap), list(Bp)) / 1e9
        print(f"  {label}   ({gf:.1f} GFLOP/call, {v['exec_flops']/1e12:.1f} TFLOP/fold, "
              f"{','.join(sorted(v['stages']))})", flush=True)
        try:
            a = ttnn.from_torch(torch.randn(*Ap) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
            b = ttnn.from_torch(torch.randn(*Bp) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
        except Exception as e:                                            # noqa: BLE001
            print(f"    ALLOC ERR {str(e)[:90]}", flush=True)
            res["classes"][label] = {"alloc_error": str(e)[:300],
                                     "exec_tflop_per_fold": round(v["exec_flops"] / 1e12, 3)}
            continue
        variants = {}

        def add(name, fn):
            try:
                ms = timed(dev, fn)
            except Exception as e:                                        # noqa: BLE001
                print(f"    {name:24s} ERR {str(e)[:70]}", flush=True)
                return
            variants[name] = round(gf / (ms / 1e3) / 1e3, 2)
            print(f"    {name:24s} {ms:9.4f} ms  {gf/(ms/1e3)/1e3:7.2f} TFLOP/s", flush=True)

        add("matmul->DRAM", lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=KC, memory_config=DRAM)))
        add("matmul->L1", lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=KC, memory_config=L1)))
        add("matmul+grid->DRAM", lambda: ttnn.deallocate(ttnn.matmul(
            a, b, compute_kernel_config=KC, memory_config=DRAM, core_grid=ttnn.CoreGrid(x=g.x, y=g.y))))
        if MM is not None and len(Bp) == 2:
            add("minimal_matmul->DRAM", lambda: ttnn.deallocate(MM(a, b, memory_config=DRAM)))
            add("minimal_matmul->L1", lambda: ttnn.deallocate(MM(a, b, memory_config=L1)))
        best = max(variants.items(), key=lambda kv: kv[1]) if variants else ("none", 0.0)
        res["classes"][label] = {
            "op": op, "a": list(Ap), "b": list(Bp), "K": v["K"], "N": v["N"],
            "stages": sorted(v["stages"]), "gflop_per_call": round(gf, 2),
            "exec_tflop_per_fold": round(v["exec_flops"] / 1e12, 3),
            "variants": variants, "best": best[0], "best_tflops": best[1]}
        print(f"    -> best {best[0]} {best[1]} TFLOP/s", flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)

    for key, v in sorted(keep_sdpa.items(), key=lambda kv: -kv[1]["exec_flops"]):
        b_, h, s, d_ = key
        label = f"sdpa {'x'.join(map(str,key))}"
        gf = 4 * b_ * h * s * s * d_ / 1e9
        print(f"  {label}   ({gf:.1f} GFLOP/call, {v['exec_flops']/1e12:.1f} TFLOP/fold)", flush=True)
        try:
            q, k, vv = (ttnn.from_torch(torch.randn(b_, h, s, d_) * 0.1, layout=ttnn.TILE_LAYOUT,
                                        device=dev, dtype=ttnn.bfloat16) for _ in range(3))
            bias = ttnn.from_torch(torch.randn(1, h, s, s) * 0.1, layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
            prog = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=g, q_chunk_size=s,
                                          k_chunk_size=s, exp_approx_mode=False)
            ms = timed(dev, lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
                q, k, vv, attn_mask=bias, is_causal=False, program_config=prog,
                compute_kernel_config=KC)))
            res["sdpa"][label] = {"shape": list(key), "ms": round(ms, 4),
                                  "gflop_per_call": round(gf, 2),
                                  "exec_tflop_per_fold": round(v["exec_flops"] / 1e12, 3),
                                  "tflops": round(gf / (ms / 1e3) / 1e3, 2),
                                  "q_chunk": s, "k_chunk": s}
            print(f"    {ms:9.4f} ms  {gf/(ms/1e3)/1e3:7.2f} TFLOP/s", flush=True)
        except Exception as e:                                            # noqa: BLE001
            print(f"    ERR {str(e)[:120]}", flush=True)
            res["sdpa"][label] = {"shape": list(key), "error": str(e)[:300],
                                  "exec_tflop_per_fold": round(v["exec_flops"] / 1e12, 3)}

    rates = [(v["exec_tflop_per_fold"], v["best_tflops"]) for v in res["classes"].values()
             if v.get("best_tflops")]
    rates += [(v["exec_tflop_per_fold"], v["tflops"]) for v in res["sdpa"].values() if v.get("tflops")]
    if rates:
        t = sum(f / r for f, r in rates)
        res["flop_weighted_tflops_measured_classes"] = round(sum(f for f, _ in rates) / t, 2)
        print(f"\nFLOP-weighted achievable rate over the measured classes: "
              f"{res['flop_weighted_tflops_measured_classes']} TFLOP/s", flush=True)
    json.dump(res, open(args.out, "w"), indent=2, default=str)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
