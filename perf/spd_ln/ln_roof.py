"""Pair layer norm on the roofline: ttnn.layer_norm on [1, N, N, C] bf16 (DRAM, tile) against ops that move the
same bytes (clone: read + write once; typecast to bfp8: read 2 B, write ~1 B), the norm's own variants (no
affine, fp32 DST at HiFi3 as `ln_f32`, bf16 DST, bfp8 output) and tt_bio.pair_ln (its copy floor `abl 1`, HiFi4,
HiFi2). Rows carry a per-position offset (|mean| / std up to ~10, as trunk pair rows do); rel_rms of each norm
against float64. ms a call, warm median. Run under `timeout`."""
import argparse, json, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="512x256,1024x256,512x128,736x128")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--only-pair", action="store_true", help="pair_ln arms and the ttnn reference only")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, pair_ln as PL

dev = T.get_device()
DR = ttnn.DRAM_MEMORY_CONFIG


def timed(f, reps):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        o = f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    return round(statistics.median(ms), 4)


def ckc(fid, f32):
    return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False,
                                                  fp32_dest_acc_en=f32, packer_l1_acc=False)


g = torch.Generator().manual_seed(0)
for sh in A.shapes.split(","):
    n, c = (int(v) for v in sh.split("x"))
    xt = (torch.randn(1, n, n, c, generator=g) * 0.7 + 3 * torch.randn(1, n, n, 1, generator=g)).to(torch.bfloat16)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DR)
    wt = (1 + 0.1 * torch.randn(c, generator=g)).to(torch.bfloat16)
    bt = (0.1 * torch.randn(c, generator=g)).to(torch.bfloat16)
    w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    mb = n * n * c * 2 / 1e6
    arms = {
        "clone": lambda: ttnn.clone(x, memory_config=DR),
        "typecast_bfp8": lambda: ttnn.typecast(x, ttnn.bfloat8_b),
        "ln_hifi3_f32": lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ckc(ttnn.MathFidelity.HiFi3, True)),
        "ln_hifi4_f32": lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ckc(ttnn.MathFidelity.HiFi4, True)),
        "ln_hifi2_bf16dst": lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ckc(ttnn.MathFidelity.HiFi2, False)),
        "ln_lofi_bf16dst": lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ckc(ttnn.MathFidelity.LoFi, False)),
        "ln_noaffine_hifi3_f32": lambda: ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=ckc(ttnn.MathFidelity.HiFi3, True)),
    }
    rec = {"n": n, "c": c, "MB_in": round(mb, 1)}
    if A.only_pair:
        arms = {"ln_hifi3_f32": arms["ln_hifi3_f32"]}
    # pair_ln, risk order (copy floor first), one line each so a hang names its stage
    pl = {"pair_ln_abl1": lambda: PL.pair_ln(x, wt, bt, abl=1),
          "pair_ln_hifi4": lambda: PL.pair_ln(x, wt, bt),
          "pair_ln_hifi2": lambda: PL.pair_ln(x, wt, bt, fidelity=ttnn.MathFidelity.HiFi2)}
    rec["pair_ln_eligible"] = PL.eligible(x)
    chk = n * n * c <= 736 * 736 * 128 * 2   # float64 references only where host memory stays small
    for k, f in pl.items():
        try:
            o = f(); ttnn.synchronize_device(dev)
            print(json.dumps({"stage": k, "n": n, "c": c}), flush=True)
            got = ttnn.to_torch(o).double() if chk else None; ttnn.deallocate(o)
            if not chk:
                pass
            elif k == "pair_ln_abl1":
                rec[k + "_exact_copy"] = bool(torch.equal(got, xt.double()))
            else:
                ref = torch.nn.functional.layer_norm(xt.double(), (c,), wt.double(), bt.double(), 1e-5)
                rec[k + "_relrms"] = float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
            rec[k] = timed(f, A.reps)
        except Exception as e:
            rec[k] = f"ERR {type(e).__name__}: {str(e)[:200]}"
    if chk:
      ref = torch.nn.functional.layer_norm(xt.double(), (c,), wt.double(), bt.double(), 1e-5)
      o = arms["ln_hifi3_f32"](); got = ttnn.to_torch(o).double(); ttnn.deallocate(o)
      rec["ln_hifi3_f32_relrms"] = float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
      rec["bf16_floor_relrms"] = float(((ref.to(torch.bfloat16).double() - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
      del ref, got
    for k, f in arms.items():
        try:
            rec[k] = timed(f, A.reps)
        except Exception as e:  # an unsupported variant is a fact, not a stop
            rec[k] = f"ERR {type(e).__name__}: {str(e)[:120]}"
    try:  # bfp8 output straight from the norm, if this ttnn takes dtype=
        rec["ln_hifi3_f32_out_bfp8"] = timed(lambda: ttnn.layer_norm(
            x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ckc(ttnn.MathFidelity.HiFi3, True),
            dtype=ttnn.bfloat8_b), A.reps)
    except Exception as e:
        rec["ln_hifi3_f32_out_bfp8"] = f"ERR {type(e).__name__}: {str(e)[:120]}"
    rec["clone_GBps"] = round(2 * mb / rec["clone"], 1) if isinstance(rec.get("clone"), float) else None
    print(json.dumps(rec), flush=True)
    for t in (x, w, b):
        ttnn.deallocate(t)
