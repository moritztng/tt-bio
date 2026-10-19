#!/usr/bin/env python3
"""OpenFold3's DiT adaLN, `sb + LN(a) * sc` on [1, 512, 768] bf16: where the LayerNorm's 27 us a call goes (whcen4: 16
cores, 0.23 of roof, 9.6k calls a fold on Wormhole) and what a different program costs.

Arms (each timed as host wall over --calls synchronized back-to-back calls, error against float64 on the same operands):
  ln_<fid>         interleaved ttnn.layer_norm (the shipped call) at HiFi4 / HiFi2 / LoFi, fp32 dest
  ln_default_*     LayerNormDefaultProgramConfig variants (legacy reduction / rsqrt off, welford)
  ln_block<x>x<y>  block-sharded LayerNormShardedMultiCoreProgramConfig on an x*y grid, LN alone; the interleaved->sharded
                   and sharded->interleaved copies timed separately (shard_in_us, shard_out_us)
  addcmul          the shipped addcmul after the LN, for the pair's total
Usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/of3_adaln/ln_check.py --out r.json
"""
import argparse, json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--width", type=int, default=768)
ap.add_argument("--calls", type=int, default=400)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = T.get_device()
torch.manual_seed(0)
x = (torch.randn(1, a.rows, a.width) * 3 + 0.5).bfloat16()
sc = torch.rand(1, a.rows, a.width).bfloat16()
sb = torch.randn(1, a.rows, a.width).bfloat16()
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
X, SC, SB = up(x), up(sc), up(sb)
xd = x.double()
ref_ln = torch.nn.functional.layer_norm(xd, (a.width,), eps=1e-5)
ref = sb.double() + ref_ln * sc.double()
FID = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}
ckc = lambda fid, approx=True: ttnn.WormholeComputeKernelConfig(  # noqa: E731
    math_fidelity=FID[fid], math_approx_mode=approx, fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(f):
    out = f()
    for _ in range(20):
        ttnn.deallocate(f())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.calls):
        ttnn.deallocate(f())
    ttnn.synchronize_device(dev)
    return out, (time.perf_counter() - t0) / a.calls * 1e6


res = {"rows": a.rows, "width": a.width, "calls": a.calls, "arch": "wormhole" if T.is_wormhole() else "blackhole",
       "grid": list(dev.compute_with_storage_grid_size().__repr__().split()) if hasattr(dev, "compute_with_storage_grid_size") else None}


def arm(name, f, tail=None):
    try:
        out, us = timed(f)
        o = ttnn.to_torch(tail(out) if tail else out).double()
        res[name] = {"us": round(us, 2), "ln_max_err": float((o - ref_ln).abs().max()),
                     "ln_rms_err": float((o - ref_ln).pow(2).mean().sqrt())}
    except Exception as e:  # an arm the wheel refuses is a result, not a crash
        res[name] = {"error": f"{type(e).__name__}: {e}"[:400]}
        traceback.print_exc()
    print(name, res[name], flush=True)


for fid in ("HiFi4", "HiFi2", "LoFi"):
    arm(f"ln_{fid}", lambda fid=fid: ttnn.layer_norm(X, epsilon=1e-5, compute_kernel_config=ckc(fid)))
for k in ({"legacy_reduction": False}, {"legacy_rsqrt": False}, {"legacy_reduction": False, "legacy_rsqrt": False},
          {"use_welford": True}):
    nm = "ln_default_" + "_".join(f"{n}{int(v)}" for n, v in k.items())
    try:
        pc = ttnn.LayerNormDefaultProgramConfig(**k)
    except Exception as e:
        res[nm] = {"error": f"{type(e).__name__}: {e}"[:400]}
        continue
    arm(nm, lambda pc=pc: ttnn.layer_norm(X, epsilon=1e-5, compute_kernel_config=ckc("HiFi4"), program_config=pc))

# addcmul of the shipped pair (on the HiFi4 LN output)
AN = ttnn.layer_norm(X, epsilon=1e-5, compute_kernel_config=ckc("HiFi4"))
try:
    out, us = timed(lambda: ttnn.addcmul(SB, AN, SC, value=1.0))
    o = ttnn.to_torch(out).double()
    res["addcmul"] = {"us": round(us, 2), "max_err": float((o - ref).abs().max())}
except Exception as e:
    res["addcmul"] = {"error": str(e)[:400]}
print("addcmul", res["addcmul"], flush=True)

th, tw = a.rows // 32, a.width // 32
for gx, gy in ((8, 8), (8, 4), (4, 8), (6, 8), (8, 2)):
    if th % gy or tw % gx:
        continue
    bh, bw = th // gy, tw // gx
    nm = f"ln_block{gx}x{gy}"
    try:
        mc = ttnn.create_sharded_memory_config(
            shape=(a.rows, a.width), core_grid=ttnn.CoreGrid(y=gy, x=gx), strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=False)
        sub = max(d for d in (4, 3, 2, 1) if bw % d == 0)
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(gx, gy), subblock_w=sub,
                                                         block_h=bh, block_w=bw, inplace=False)
        XS = ttnn.to_memory_config(X, mc)
        _, res[nm + "_shard_in_us"] = timed(lambda: ttnn.to_memory_config(X, mc))
        arm(nm, lambda: ttnn.layer_norm(XS, epsilon=1e-5, compute_kernel_config=ckc("HiFi4"), program_config=pc,
                                        memory_config=mc),
            tail=lambda t: ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG))
        YS = ttnn.layer_norm(XS, epsilon=1e-5, compute_kernel_config=ckc("HiFi4"), program_config=pc, memory_config=mc)
        _, res[nm + "_shard_out_us"] = timed(lambda: ttnn.to_memory_config(YS, ttnn.DRAM_MEMORY_CONFIG))
        ttnn.deallocate(XS); ttnn.deallocate(YS)
    except Exception as e:
        res.setdefault(nm, {"error": f"{type(e).__name__}: {e}"[:400]})
        traceback.print_exc()
    print(nm, res.get(nm), res.get(nm + "_shard_in_us"), res.get(nm + "_shard_out_us"), flush=True)

print(json.dumps(res), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))
