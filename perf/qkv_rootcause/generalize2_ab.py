#!/usr/bin/env python3
"""In-model A/B for the second wave of L1-resident trunk projections, one process.

Extends generalize_ab.py (tri-attention qkv + g + o) to the remaining Pairformer-trunk
projections in the k_tiles < num_cores defect class:

  trimul tail    TriangleMultiplication p_out / g_out   ([N,N,C]x[C,C], non-chunked path)
  transition fc3 Transition swiglu fc3                   ([h,N,4C]x[4C,C], 4D chunk paths)
  minitri x4     MiniTriangularUpdate p_in/g_in/p_out/g_out (BoltzGen Miniformer, D=128)

All new sites go through _l1_resident_linear with full_k=False, i.e. the auto config's
own K blocking with only the result placement changed, so every arm must be bit-exact.

Arms, switched by monkeypatching T._l1_resident_linear_config:

  PROD    what main runs: qkv only
  HEAD    this branch before this pass: qkv + g + o
  NEW     this pass: qkv + g + o + trimul tail + transition fc3 (+ minitri, checked
          separately since the PairformerLayer does not contain one)

Correctness (torch.equal, per module and whole block) runs before any timing, on
freshly built inputs per capture because the block mutates z in place. Timing is
interleaved PROD/HEAD/NEW with a device sync before every timed region.
"""
import argparse, json, time
from pathlib import Path
import sys

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--warm", type=int, default=4)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--pipe", type=int, default=10)
ap.add_argument("--skip-timing", action="store_true")
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
tri_att_ids = {
    id(w)
    for tri in (layer.triangle_attention_start, layer.triangle_attention_end)
    for w in (tri.qkv_weight, tri.g_weight, tri.o_weight)
}
qkv_ids = {
    id(tri.qkv_weight)
    for tri in (layer.triangle_attention_start, layer.triangle_attention_end)
}


def cfg_head(x, w, dtype, full_k=True):  # HEAD: tri-att sites only
    if id(w) not in tri_att_ids:
        return None
    return real_cfg(x, w, dtype, full_k)


def cfg_prod(x, w, dtype, full_k=True):  # PROD: qkv only
    if id(w) not in qkv_ids:
        return None
    return real_cfg(x, w, dtype, full_k)


ARMS = (("PROD", cfg_prod), ("HEAD", cfg_head), ("NEW", real_cfg))
print(f"N={N} c_z={c_z} grid={T.COMPUTE_GRID_MAIN} "
      f"l1_unreserved={ttnn.get_max_worker_l1_unreserved_size()}", flush=True)

# ---- which new-site configs the guard admits at this N -------------------------------------
print("=== guard admissions ===", flush=True)
tm = layer.triangle_multiplication_start
tz = layer.transition_z
n_tiles = c_z // 32
adm = {}
z4 = mk_z()
for tag, x, w in (("trimul p_out", z4, tm.out_p_weight), ("trimul g_out", z4, tm.g_out_weight)):
    cfg = T._l1_resident_linear_config(x, w, ttnn.bfloat16, full_k=False)
    adm[tag] = cfg is not None
    print(f"  {tag:14s} k={int(w.shape[0])} n={int(w.shape[-1])} -> {'L1' if cfg else 'fallback'}", flush=True)
hidden = int(tz.fc3_weight.shape[0])
# transition fc3 is NOT a site: the 4D path accumulates per-chunk results before the
# concat, and L1-resident chunks crashed the next chunk's static CBs (measured, c_z=256
# at 256 tokens). Recorded here so the guard table is complete.
adm["transition fc3"] = False
print(f"  transition fc3  k={hidden} n={c_z} -> not eligible (chunk-accumulated result)",
      flush=True)

# ---- op-level bit-exactness of the new sites against the ops they replace -------------------
print("=== op-level bit-exactness (torch.equal vs ttnn.linear auto) ===", flush=True)
num = {}
for tag, x, w in (("trimul p_out", z4, tm.out_p_weight), ("trimul g_out", z4, tm.g_out_weight)):
    old = ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
    new = T._l1_resident_linear(x, w, ttnn.bfloat16, ckc)
    num[tag] = bool(torch.equal(ttnn.to_torch(old), ttnn.to_torch(new)))
    ttnn.deallocate(old)
    ttnn.deallocate(new)
    print(f"  {tag:14s} bit_exact={num[tag]}", flush=True)
# (transition fc3 op-check removed with the site itself)

# ---- MiniTriangularUpdate (BoltzGen Miniformer), synthetic D=128 weights --------------------
print("=== minitri (BoltzGen, D=128) ===", flush=True)
D = 128
sd = {
    "norm_in.weight": torch.ones(D), "norm_in.bias": torch.zeros(D),
    "p_in.weight": torch.randn(D, D) * 0.05, "g_in.weight": torch.randn(D, D) * 0.05,
    "norm_out.weight": torch.ones(D // 2), "norm_out.bias": torch.zeros(D // 2),
    "p_out.weight": torch.randn(D, D // 2) * 0.05, "g_out.weight": torch.randn(D, D // 2) * 0.05,
}
mini = T.MiniTriangularUpdate(sd, ckc)
zm_t = torch.randn(1, N, N, D)
got = {}
for arm, fn in (("PROD", cfg_prod), ("NEW", real_cfg)):
    T._l1_resident_linear_config = fn
    zm = ttnn.from_torch(zm_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    got[arm] = ttnn.to_torch(mini(zm, None))
num["minitri NEW"] = bool(torch.equal(got["PROD"], got["NEW"]))
print(f"  whole-minitri bit_exact={num['minitri NEW']}", flush=True)
del mini, got

# ---- boltz2-class TriangleMultiplication (c_z=128), synthetic weights -----------------------
# The protenix layer only exercises c_z=256. c=128 admits the L1 tail further out (the
# per-core budget scales with n_tiles), including the knife-edge at N=384 where the
# p_out result is still live while g_out runs. That is the case that must run clean.
print("=== tri_mul (boltz2 class, c=128) ===", flush=True)
C128 = 128
sd128 = {
    "norm_in.weight": torch.ones(C128), "norm_in.bias": torch.zeros(C128),
    "g_in.weight": torch.randn(2 * C128, C128) * 0.05,
    "p_in.weight": torch.randn(2 * C128, C128) * 0.05,
    "norm_out.weight": torch.ones(C128), "norm_out.bias": torch.zeros(C128),
    "g_out.weight": torch.randn(C128, C128) * 0.05,
    "p_out.weight": torch.randn(C128, C128) * 0.05,
}
tm128 = T.TriangleMultiplication(False, sd128, ckc)
zt128 = torch.randn(1, N, N, C128)
got = {}
for arm, fn in (("PROD", cfg_prod), ("NEW", real_cfg)):
    T._l1_resident_linear_config = fn
    z128 = ttnn.from_torch(zt128, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    got[arm] = ttnn.to_torch(tm128(z128, None))
num["tri_mul c128 NEW"] = bool(torch.equal(got["PROD"], got["NEW"]))
_cfg128 = T._l1_resident_linear_config(
    ttnn.from_torch(zt128, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16),
    tm128.out_p_weight, ttnn.bfloat16, full_k=False)
adm["trimul p_out c128"] = _cfg128 is not None
print(f"  whole-trimul bit_exact={num['tri_mul c128 NEW']} "
      f"(tail guard {'admits' if adm['trimul p_out c128'] else 'refuses'} at c=128)",
      flush=True)
del tm128, got

# ---- module- and block-level bit-exactness, fresh inputs per capture ------------------------
print("=== module/block bit-exactness in the model ===", flush=True)
SITES = (
    ("tri_mul start", lambda z: layer.triangle_multiplication_start(z, None)),
    ("tri_mul end", lambda z: layer.triangle_multiplication_end(z, None)),
    ("transition_z", lambda z: layer.transition_z(z)),
    ("BLOCK (s,z)", None),
)
for name, call in SITES:
    got = {}
    for arm, fn in ARMS:
        T._l1_resident_linear_config = fn
        z = mk_z()
        if call is None:
            so, zo = layer(mk_s(), z)
            got[arm] = (ttnn.to_torch(so), ttnn.to_torch(zo))
        else:
            got[arm] = (ttnn.to_torch(call(z)),)
    for arm in ("HEAD", "NEW"):
        num[f"{name} {arm}"] = all(bool(torch.equal(a, b)) for a, b in zip(got["PROD"], got[arm]))
    print(f"  {name:14s} " + "  ".join(f"{a}={num[f'{name} {a}']}" for a in ("HEAD", "NEW")),
          flush=True)
T._l1_resident_linear_config = real_cfg

# ---- attention_pair_bias must be UNTOUCHED (wide shape, guard refuses) ----------------------
# verified implicitly: the BLOCK check above includes it, and NEW must equal PROD there.

if args.skip_timing:
    print(json.dumps({"n": N, "admissions": adm, "numerics": num}, indent=2), flush=True)
    if args.out:
        json.dump({"n": N, "grid": list(T.COMPUTE_GRID_MAIN), "admissions": adm,
                   "numerics": num}, open(args.out, "w"), indent=2)
        print("wrote", args.out, flush=True)
    sys.exit(0)

# ---- timing, interleaved --------------------------------------------------------------------
print("=== interleaved A/B ===", flush=True)
z0 = mk_z()
WORK = [
    ("tri_mul start", lambda: layer.triangle_multiplication_start(z0, None)),
    ("transition_z", lambda: layer.transition_z(z0)),
    ("BLOCK (s,z)", None),
]
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
T._l1_resident_linear_config = real_cfg

print("=== result (median of 3 repeats per arm) ===", flush=True)
summary = {}
for name, _ in WORK:
    summary[name] = {}
    line = f"  {name:14s}"
    for arm, _ in ARMS:
        v = med(res[name][arm])
        summary[name][arm] = {"ms": v, "raw": res[name][arm]}
        line += f"  {arm} {v:7.4f}"
    for arm in ("HEAD", "NEW"):
        v_arm = summary[name][arm]["ms"]
        summary[name][arm]["speedup_vs_prod"] = round(summary[name]["PROD"]["ms"] / v_arm, 4)
        line += f"  {arm}/PROD {summary[name][arm]['speedup_vs_prod']:5.3f}x"
    print(line, flush=True)

if args.out:
    json.dump({"n": N, "grid": list(T.COMPUTE_GRID_MAIN), "admissions": adm,
               "timing": summary, "numerics": num}, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)
