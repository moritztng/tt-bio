#!/usr/bin/env python3
"""In-model A/B for the generalized L1-resident trunk projections (qkv + g + o), one process.

Extends inmodel_ab.py (qkv only) to the two new tri-attention sites: the gate projection g
(minimal_matmul -> full-K L1 config) and the output projection o (ttnn.linear auto -> same
K-blocking, result in L1). Arms interleave against the same layer and input:

  PROD  what main runs: qkv L1 only
  G     qkv + g
  GO    qkv + g + o (this branch)

Correctness is checked before any timing runs, on a freshly built pair tensor per capture,
because the whole-block arm mutates its input in place. Op-level bit-exactness for each site
is checked directly against the op the site replaces.
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

real_cfg = T._l1_resident_linear_config
g_weights = {id(layer.triangle_attention_start.g_weight), id(layer.triangle_attention_end.g_weight)}


def cfg_g(x, w, dtype, full_k=True):  # PROD + g
    if id(w) in g_weights:
        return real_cfg(x, w, dtype, full_k)
    if not full_k:
        return None
    return real_cfg(x, w, dtype, full_k)


def cfg_prod(x, w, dtype, full_k=True):  # PROD: qkv only
    if id(w) in g_weights or not full_k:
        return None
    return real_cfg(x, w, dtype, full_k)


ARMS = (("PROD", cfg_prod), ("G", cfg_g), ("GO", real_cfg))
print(f"N={N} c_z={c_z} grid={T.COMPUTE_GRID_MAIN}", flush=True)
for tag, mt, kt, nt, fk in (("qkv", N * N // 32, c_z // 32, 3 * (c_z // 32) * 32 // 32, True),
                            ("g", N * N // 32, c_z // 32, c_z // 32, True),
                            ("o", N * N // 32, c_z // 32, c_z // 32, False)):
    print(f"  {tag}: {T._l1_resident_matmul_config(mt, kt, nt, 2, fk)}", flush=True)

# ---- op-level bit-exactness: each site against the op it replaces --------------------------
print("\n=== op-level bit-exactness ===", flush=True)
tri = layer.triangle_attention_start
x3 = ttnn.from_torch(torch.randn(N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
num = {}
for tag, w, fk in (("qkv", tri.qkv_weight, True), ("g", tri.g_weight, True)):
    cfg = T._l1_resident_linear_config(x3, w, ttnn.bfloat16, full_k=fk)
    old = ttnn.experimental.minimal_matmul(input_tensor=x3, weight_tensor=w,
                                           compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    if cfg is None:
        num[tag] = {"config": None}
        continue
    new = ttnn.linear(x3, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                      memory_config=ttnn.L1_MEMORY_CONFIG, program_config=cfg)
    num[tag] = {"bit_exact": bool(torch.equal(ttnn.to_torch(old), ttnn.to_torch(new)))}
    ttnn.deallocate(old)
    ttnn.deallocate(new)
cfg_o = T._l1_resident_linear_config(x3, tri.o_weight, ttnn.bfloat16, full_k=False)
old_o = ttnn.linear(x3, tri.o_weight, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                    core_grid=T.CORE_GRID_MAIN)
if cfg_o is None:
    num["o"] = {"config": None}
else:
    new_o = ttnn.linear(x3, tri.o_weight, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                        memory_config=ttnn.L1_MEMORY_CONFIG, program_config=cfg_o)
    num["o"] = {"bit_exact": bool(torch.equal(ttnn.to_torch(old_o), ttnn.to_torch(new_o)))}
    ttnn.deallocate(new_o)
ttnn.deallocate(old_o)
ttnn.deallocate(x3)
print(" " + json.dumps(num), flush=True)

# ---- block-level bit-exactness, fresh inputs per capture ------------------------------------
print("\n=== bit-exactness in the model ===", flush=True)
for name, call in (("tri_att start", lambda z: layer.triangle_attention_start(z, None)),
                   ("tri_att end", lambda z: layer.triangle_attention_end(z, None)),
                   ("BLOCK (s,z)", None)):
    got = {}
    for arm, fn in ARMS:
        T._l1_resident_linear_config = fn
        z = mk_z()
        if call is None:
            so, zo = layer(mk_s(), z)
            got[arm] = (ttnn.to_torch(so), ttnn.to_torch(zo))
        else:
            got[arm] = (ttnn.to_torch(call(z)),)
    for arm in ("G", "GO"):
        num[f"{name} {arm}"] = {"bit_exact": all(bool(torch.equal(a, b))
                                                 for a, b in zip(got["PROD"], got[arm]))}
    print(f"  {name:16s} " + "  ".join(f"{a}={num[f'{name} {a}']['bit_exact']}" for a in ("G", "GO")),
          flush=True)

# ---- then timing, interleaved ---------------------------------------------------------------
print("\n=== interleaved A/B ===", flush=True)
s, z0 = mk_s(), mk_z()
WORK = [("tri_att start", lambda: layer.triangle_attention_start(z0, None)),
        ("tri_att end", lambda: layer.triangle_attention_end(z0, None)),
        ("BLOCK (s,z)", None)]
res = {}
for rep in range(3):
    for arm, fn in ARMS:
        T._l1_resident_linear_config = fn
        for name, wfn in WORK:
            if wfn is None:
                st = {"s": mk_s(), "z": mk_z()}

                def wfn():  # noqa: F811
                    st["s"], st["z"] = layer(st["s"], st["z"])
            res.setdefault(name, {}).setdefault(arm, []).append(round(timed(dev, wfn), 4))
        print(f"  {arm}: " + "  ".join(f"{n}={res[n][arm][-1]:.4f}" for n, _ in WORK), flush=True)

print("\n=== result (median of 3 repeats per arm) ===", flush=True)
summary = {}
for name, _ in WORK:
    summary[name] = {}
    line = f"  {name:16s}"
    for arm, _ in ARMS:
        v = med(res[name][arm])
        summary[name][arm] = {"ms": v, "raw": res[name][arm]}
        line += f"  {arm} {v:7.4f} ms"
    for arm in ("G", "GO"):
        summary[name][arm]["speedup_vs_prod"] = round(summary[name]["PROD"]["ms"] / summary[name][arm]["ms"], 4)
        line += f"  {arm}/PROD {summary[name][arm]['speedup_vs_prod']:5.3f}x"
    print(line, flush=True)

T._l1_resident_linear_config = real_cfg
if args.out:
    json.dump({"n": N, "grid": list(T.COMPUTE_GRID_MAIN), "timing": summary, "numerics": num},
              open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)
