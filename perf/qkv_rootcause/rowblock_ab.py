#!/usr/bin/env python3
"""A/B for row-blocked L1 residency in tri-attention, one process, one card.

Whole-tensor L1 residency dies with sequence length: the qkv result is 25 MB at 128
tokens and 157 MB at 320, so past ~128 tokens every guard in _l1_resident_matmul_config
refuses and the projection is back to writing its result to DRAM for
nlp_create_qkv_heads to read straight back. Row-blocking the pair tensor bounds the
resident result to one block, which makes residency independent of sequence length.

Arms (monkeypatch T._tri_att_row_block):
  BASE  main + the generalize branch: no row blocking, whole-tensor guard only
  ROW   this pass: row-blocked, L1-resident projections per block

Bit-exactness (torch.equal) runs before any timing on fresh inputs, since the block
mutates z in place. Timing interleaves the arms and syncs both sides of every region.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=320)
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--skip-timing", action="store_true")
ap.add_argument("--out", default=None)
args = ap.parse_args()

med = lambda x: sorted(x)[len(x) // 2]


def timed(dev, fn):
    for _ in range(args.warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(args.iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) * 1e3)
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

real_row = T._tri_att_row_block
ARMS = (("BASE", lambda x, w, d: 0), ("ROW", real_row))

print(f"N={N} c_z={c_z} grid={T.COMPUTE_GRID_MAIN} "
      f"l1_unreserved={ttnn.get_max_worker_l1_unreserved_size()}", flush=True)

# ---- what the chooser picks -------------------------------------------------------------
z_probe = mk_z()
xv = ttnn.reshape(z_probe, tuple(z_probe.shape)[1:])
tri = layer.triangle_attention_start
rows = real_row(xv, tri.qkv_weight, ttnn.bfloat16)
whole = T._l1_resident_linear_config(xv, tri.qkv_weight, ttnn.bfloat16)
info = {"row_block": rows, "whole_tensor_fits": whole is not None,
        "blocks": (N // rows) if rows else 0}
print(f"chooser: whole_tensor_fits={info['whole_tensor_fits']} row_block={rows} "
      f"({info['blocks']} blocks)", flush=True)
del xv, z_probe

# ---- bit-exactness ----------------------------------------------------------------------
print("=== bit-exactness (torch.equal, ROW vs BASE) ===", flush=True)
SITES = (
    ("tri_att start", lambda z: layer.triangle_attention_start(z, None)),
    ("tri_att end", lambda z: layer.triangle_attention_end(z, None)),
    ("BLOCK (s,z)", None),
)
num = {}
for name, call in SITES:
    got = {}
    for arm, fn in ARMS:
        T._tri_att_row_block = fn
        z = mk_z()
        if call is None:
            so, zo = layer(mk_s(), z)
            got[arm] = (ttnn.to_torch(so), ttnn.to_torch(zo))
        else:
            got[arm] = (ttnn.to_torch(call(z)),)
    num[name] = all(bool(torch.equal(a, b)) for a, b in zip(got["BASE"], got["ROW"]))
    print(f"  {name:14s} bit_exact={num[name]}", flush=True)
T._tri_att_row_block = real_row

if args.skip_timing:
    out = {"n": N, "c_z": c_z, "grid": list(T.COMPUTE_GRID_MAIN), "chooser": info, "numerics": num}
    print(json.dumps(out, indent=2), flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    sys.exit(0)

# ---- timing ------------------------------------------------------------------------------
print("=== interleaved A/B ===", flush=True)
z0 = mk_z()
WORK = (
    ("tri_att start", lambda: layer.triangle_attention_start(z0, None)),
    ("tri_att end", lambda: layer.triangle_attention_end(z0, None)),
    ("BLOCK (s,z)", None),
)
res = {}
for rep in range(args.reps):
    for arm, fn in ARMS:
        T._tri_att_row_block = fn
        for name, wfn in WORK:
            if wfn is None:
                st = {"s": mk_s(), "z": mk_z()}

                def wfn():  # noqa: F811
                    st["s"], st["z"] = layer(st["s"], st["z"])
            res.setdefault(name, {}).setdefault(arm, []).append(round(timed(dev, wfn), 4))
        print(f"  rep{rep} {arm}: " +
              "  ".join(f"{n}={res[n][arm][-1]:.4f}" for n, _ in WORK), flush=True)
T._tri_att_row_block = real_row

print("=== result (median of reps) ===", flush=True)
summary = {}
for name, _ in WORK:
    b, r = med(res[name]["BASE"]), med(res[name]["ROW"])
    summary[name] = {"BASE_ms": b, "ROW_ms": r, "speedup": round(b / r, 4),
                     "raw": res[name]}
    print(f"  {name:14s} BASE {b:8.4f}  ROW {r:8.4f}  {b / r:6.3f}x", flush=True)

out = {"n": N, "c_z": c_z, "grid": list(T.COMPUTE_GRID_MAIN), "chooser": info,
       "numerics": num, "timing": summary}
if args.out:
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)
