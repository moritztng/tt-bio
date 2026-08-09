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
ap.add_argument("--n", type=int, nargs="+", default=[320])
ap.add_argument("--synthetic-c", type=int, nargs="*", default=[],
                help="also run a synthetic TriangleAttention at these c_z (128 boltz2, 384 opendde)")
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
real_row = T._tri_att_row_block
ARMS = (("BASE", lambda x, w, d: 0), ("ROW", real_row))
print(f"grid={T.COMPUTE_GRID_MAIN} l1_unreserved={ttnn.get_max_worker_l1_unreserved_size()}",
      flush=True)


def synth_tri_att(c, ending=False):
    """TriangleAttention with random weights at a c_z the protenix checkpoint does not have."""
    h, hd = c // 32, 32
    sd = {
        "layer_norm.weight": torch.ones(c), "layer_norm.bias": torch.zeros(c),
        "linear_q.weight": torch.randn(h * hd, c) * 0.05,
        "linear_k.weight": torch.randn(h * hd, c) * 0.05,
        "linear_v.weight": torch.randn(h * hd, c) * 0.05,
        "linear_g.weight": torch.randn(h * hd, c) * 0.05,
        "linear_o.weight": torch.randn(c, h * hd) * 0.05,
        "linear.weight": torch.randn(h, c) * 0.05,
    }
    return T.TriangleAttention(hd, h, ending, sd, ckc)


def run(tag, mod, N, c, whole_block=None):
    """bit-exactness then interleaved timing for one module at one size."""
    torch.manual_seed(0)
    z_t = torch.randn(1, N, N, c)
    mk_z = lambda: ttnn.from_torch(z_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    zp = mk_z()
    xv = ttnn.reshape(zp, tuple(zp.shape)[1:])
    rows = real_row(xv, mod.qkv_weight, ttnn.bfloat16)
    whole = T._l1_resident_linear_config(xv, mod.qkv_weight, ttnn.bfloat16) is not None
    ttnn.deallocate(zp)
    got, times = {}, {}
    for arm, fn in ARMS:
        T._tri_att_row_block = fn
        got[arm] = ttnn.to_torch(mod(mk_z(), None))
        z0 = mk_z()
        times[arm] = timed(dev, lambda: mod(z0, None))
        ttnn.deallocate(z0)
    T._tri_att_row_block = real_row
    eq = bool(torch.equal(got["BASE"], got["ROW"]))
    sp = round(times["BASE"] / times["ROW"], 4)
    print(f"  {tag:22s} N={N:4d} c={c:3d} whole_fits={str(whole):5s} rows={rows:4d} "
          f"BASE {times['BASE']:8.4f}  ROW {times['ROW']:8.4f}  {sp:6.3f}x  bit_exact={eq}",
          flush=True)
    return {"tag": tag, "n": N, "c": c, "whole_fits": whole, "row_block": rows,
            "BASE_ms": round(times["BASE"], 4), "ROW_ms": round(times["ROW"], 4),
            "speedup": sp, "bit_exact": eq}


out = []
print("=== protenix-v2 layer 0 (c_z=256) ===", flush=True)
for N in args.n:
    out.append(run("tri_att start", layer.triangle_attention_start, N, c_z))
    out.append(run("tri_att end", layer.triangle_attention_end, N, c_z))
    # whole Pairformer block, both arms, fresh state each time
    torch.manual_seed(0)
    s_t, z_t = torch.randn(1, N, 384), torch.randn(1, N, N, c_z)
    res, blk = {}, {}
    for arm, fn in ARMS:
        T._tri_att_row_block = fn
        st = {"s": ttnn.from_torch(s_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16),
              "z": ttnn.from_torch(z_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)}
        so, zo = layer(st["s"], st["z"])
        blk[arm] = (ttnn.to_torch(so), ttnn.to_torch(zo))
        st = {"s": ttnn.from_torch(s_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16),
              "z": ttnn.from_torch(z_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)}

        def wfn(st=st):
            st["s"], st["z"] = layer(st["s"], st["z"])
        res[arm] = timed(dev, wfn)
    T._tri_att_row_block = real_row
    eq = all(bool(torch.equal(a, b)) for a, b in zip(blk["BASE"], blk["ROW"]))
    sp = round(res["BASE"] / res["ROW"], 4)
    print(f"  {'BLOCK (s,z)':22s} N={N:4d} c={c_z:3d} {'':18s}"
          f"BASE {res['BASE']:8.4f}  ROW {res['ROW']:8.4f}  {sp:6.3f}x  bit_exact={eq}", flush=True)
    out.append({"tag": "BLOCK", "n": N, "c": c_z, "BASE_ms": round(res["BASE"], 4),
                "ROW_ms": round(res["ROW"], 4), "speedup": sp, "bit_exact": eq})

for c in args.synthetic_c:
    print(f"=== synthetic TriangleAttention (c_z={c}) ===", flush=True)
    m = synth_tri_att(c)
    for N in args.n:
        out.append(run(f"synth c={c}", m, N, c))
    del m

print("=== summary ===", flush=True)
bad = [r for r in out if not r["bit_exact"]]
print(f"  bit-exact: {len(out) - len(bad)}/{len(out)}" + (f"  FAILURES: {bad}" if bad else ""), flush=True)
if args.out:
    json.dump({"grid": list(T.COMPUTE_GRID_MAIN),
               "l1_unreserved": ttnn.get_max_worker_l1_unreserved_size(), "runs": out},
              open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)
