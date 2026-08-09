#!/usr/bin/env python3
"""In-model A/B for the L1-resident tri-attention qkv projection, interleaved in one process.

The op-level win overstates the in-model win, so the number that counts is the Pairformer block.
A and B alternate against the same layer and the same input, so neither can drift from the other.

Correctness is checked before any timing runs, on a freshly built pair tensor per capture, because
the whole-block arm mutates its input in place and contaminated an earlier version of this probe.
"""
import argparse, json, time
from pathlib import Path
import sys

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


def med(x):
    return sorted(x)[len(x) // 2]


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
s_t, z_t = torch.randn(1, N, 384), torch.randn(1, N, N, c_z)
mk_s = lambda: ttnn.from_torch(s_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
mk_z = lambda: ttnn.from_torch(z_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

on_cfg = T._l1_resident_linear_config
off_cfg = lambda x, w, dtype, full_k=True: None
print(f"N={N} c_z={c_z} grid={T.COMPUTE_GRID_MAIN}", flush=True)
print(f"config for the tri-att qkv shape: {T._l1_resident_matmul_config(N * N // 32, c_z // 32, 8 * 32 * 3 // 32, 2, True)}",
      flush=True)

# ---- correctness first, on fresh inputs, before anything has been mutated -----------------
print("\n=== bit-exactness in the model ===", flush=True)
num = {}
for name, call in (("tri_att start", lambda z: layer.triangle_attention_start(z, None)),
                   ("tri_att end", lambda z: layer.triangle_attention_end(z, None)),
                   ("BLOCK (s,z)", None)):
    got = {}
    for arm, fn in (("OFF", off_cfg), ("ON", on_cfg)):
        T._l1_resident_linear_config = fn
        z = mk_z()
        if call is None:
            so, zo = layer(mk_s(), z)
            got[arm] = (ttnn.to_torch(so), ttnn.to_torch(zo))
        else:
            got[arm] = (ttnn.to_torch(call(z)),)
    num[name] = {"bit_exact": all(bool(torch.equal(a, b)) for a, b in zip(got["OFF"], got["ON"]))}
    if not num[name]["bit_exact"]:
        a, b = got["OFF"][-1].float(), got["ON"][-1].float()
        num[name]["max_abs_diff"] = round(float((a - b).abs().max()), 6)
        num[name]["pcc"] = round(float(torch.corrcoef(
            torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1]), 9)
    print(f"  {name:16s} {num[name]}", flush=True)

# ---- then timing, interleaved ---------------------------------------------------------------
print("\n=== interleaved A/B ===", flush=True)
s, z0 = mk_s(), mk_z()
sn = ttnn.layer_norm(s, weight=layer.pre_norm_s_weight, bias=layer.pre_norm_s_bias,
                     epsilon=1e-5, compute_kernel_config=ckc)
WORK = [("tri_att start", lambda: layer.triangle_attention_start(z0, None)),
        ("tri_att end", lambda: layer.triangle_attention_end(z0, None)),
        ("BLOCK (s,z)", None)]
res = {}
for arm in ("OFF", "ON", "OFF", "ON", "OFF", "ON"):
    T._l1_resident_linear_config = off_cfg if arm == "OFF" else on_cfg
    for name, fn in WORK:
        if fn is None:
            st = {"s": mk_s(), "z": mk_z()}

            def fn():  # noqa: F811
                st["s"], st["z"] = layer(st["s"], st["z"])
        res.setdefault(name, {}).setdefault(arm, []).append(round(timed(dev, fn), 4))
    print(f"  {arm}: " + "  ".join(f"{n}={res[n][arm][-1]:.4f}" for n, _ in WORK), flush=True)

print("\n=== result (median of 3 repeats per arm) ===", flush=True)
summary = {}
for name, _ in WORK:
    off, on = med(res[name]["OFF"]), med(res[name]["ON"])
    summary[name] = {"off_ms": off, "on_ms": on, "speedup": round(off / on, 4),
                     "saved_ms": round(off - on, 4), "raw": res[name]}
    print(f"  {name:16s} off {off:7.4f} ms  on {on:7.4f} ms  {off/on:5.3f}x  saved {off-on:+.4f} ms",
          flush=True)

T._l1_resident_linear_config = on_cfg
if args.out:
    json.dump({"n": N, "grid": list(T.COMPUTE_GRID_MAIN), "timing": summary, "numerics": num},
              open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)
