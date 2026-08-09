#!/usr/bin/env python3
"""In-model A/B for the tall-narrow matmul program config, interleaved in one process.

The op-level win overstates the in-model win, so the deliverable number is the Pairformer
block, not the linear. A and B alternate inside one process against the same layer and the
same input, so neither can drift away from the other between runs.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--warm", type=int, default=6)
ap.add_argument("--iters", type=int, default=7)
ap.add_argument("--pipe", type=int, default=10)
ap.add_argument("--out", default=None)
args = ap.parse_args()


def med(xs):
    return sorted(xs)[len(xs) // 2]


def timed(dev, fn):
    for _ in range(args.warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(args.iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(args.pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) * 1e3 / args.pipe)
    return med(o)


dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
layer, c_z = build_layer(ckc)
N = args.n
torch.manual_seed(0)
s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
z0 = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
print(f"N={N} c_z={c_z} COMPUTE_GRID_MAIN={T.COMPUTE_GRID_MAIN}", flush=True)

new_placement = T._matmul_placement
cfg = T._tall_narrow_matmul_program_config(N * N // 32, c_z // 32, 3 * 8 * 32 // 32, True)
print(f"config chosen for the qkv shape: {cfg}", flush=True)


def old_placement(a, w, fp32_dest_acc=True):
    return {"core_grid": T.CORE_GRID_MAIN}


sn = ttnn.layer_norm(s, weight=layer.pre_norm_s_weight, bias=layer.pre_norm_s_bias,
                     epsilon=1e-5, compute_kernel_config=ckc)

WORK = [("tri_att start", lambda: layer.triangle_attention_start(z0, None)),
        ("tri_att end", lambda: layer.triangle_attention_end(z0, None)),
        ("s attention_pair_bias", lambda: layer.attention_pair_bias(sn, z0, seq_mask=None)),
        ("BLOCK (s,z)", None)]

res, outs = {}, {}
for arm in ("OFF", "ON", "OFF", "ON"):   # interleaved, twice each
    T._matmul_placement = old_placement if arm == "OFF" else new_placement
    for name, fn in WORK:
        if fn is None:
            st = {"s": s, "z": z0}

            def fn():  # noqa: F811
                st["s"], st["z"] = layer(st["s"], st["z"])
        ms = timed(dev, fn)
        res.setdefault(name, {}).setdefault(arm, []).append(round(ms, 4))
        if name != "BLOCK (s,z)" and name not in outs.get(arm, {}):
            outs.setdefault(arm, {})[name] = ttnn.to_torch(fn()).float()
    print(f"  arm {arm}: " + "  ".join(f"{n}={res[n][arm][-1]:.4f}" for n, _ in WORK), flush=True)

print("\n=== in-model A/B (median of the repeats) ===", flush=True)
summary = {}
for name, _ in WORK:
    off, on = med(res[name]["OFF"]), med(res[name]["ON"])
    summary[name] = {"off_ms": off, "on_ms": on, "speedup": round(off / on, 4),
                     "saved_ms": round(off - on, 4), "raw": res[name]}
    print(f"  {name:24s} off {off:7.4f} ms   on {on:7.4f} ms   {off/on:5.3f}x   "
          f"saved {off-on:+.4f} ms", flush=True)

print("\n=== numerics: new config vs the auto path, same inputs ===", flush=True)
num = {}
for name in outs["OFF"]:
    a, b = outs["OFF"][name], outs["ON"][name]
    p = float(torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1])
    num[name] = {"bit_exact": bool(torch.equal(a, b)), "pcc": round(p, 9),
                 "max_abs_diff": round(float((a - b).abs().max()), 6),
                 "rel_fro": round(float((a - b).norm() / b.norm()), 8)}
    print(f"  {name:24s} {num[name]}", flush=True)

out = {"n": N, "grid": list(T.COMPUTE_GRID_MAIN), "timing": summary, "numerics": num,
       "program_config": str(cfg)}
if args.out:
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)
