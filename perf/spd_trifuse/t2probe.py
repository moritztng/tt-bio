#!/usr/bin/env python3
"""Does trimul_t2g change one ESMFold2 trunk call under a lever set? Same z, same random weights, one call per
arm (ESMFold2's form: add_to_input, no mask), arms are lever specs. Prints bit equality against the first arm,
float64 error and T2 served/declined per arm.

    python perf/spd_trifuse/t2probe.py --n 512 --arms "normal-trunk_hifi3+trimul_gin,normal-trunk_hifi3+trimul_gin+trimul_t2g"
"""
import argparse, json, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--cz", type=int, default=256)
ap.add_argument("--arms", required=True, help="comma-separated lever specs; ';' separates arms when specs hold commas")
ap.add_argument("--ending", action="store_true")
ap.add_argument("--no-ref", action="store_true", help="skip the float64 reference (large N: ~25 GB host)")
ap.add_argument("--reps", type=int, default=5, help="timed calls per arm (synced, in-place on one z)")
ap.add_argument("--fast-mode", action="store_true", help="tenstorrent.set_fast_mode(True), as Wormhole ESMFold2 --fast folds")
a = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.trimul_tail as TL
from tt_bio.af2 import compute_kernel_config

cz = h = a.cz
N = a.n
g = torch.Generator().manual_seed(0)
r = lambda *s: torch.randn(*s, generator=g) / s[-1] ** 0.5
W0 = {"norm_in.weight": 1 + 0.1 * torch.randn(cz, generator=g), "norm_in.bias": 0.1 * torch.randn(cz, generator=g),
      "norm_out.weight": 1 + 0.1 * torch.randn(h, generator=g), "norm_out.bias": 0.1 * torch.randn(h, generator=g),
      "g_in.weight": r(2 * h, cz), "p_in.weight": r(2 * h, cz), "g_out.weight": r(cz, cz), "p_out.weight": r(cz, h)}
z_host = torch.randn(1, N, N, cz, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16).float()


def reference():
    d = torch.float64
    z = z_host.to(d); Wd = {k: v.to(d) for k, v in W0.items()}
    ln = torch.nn.functional.layer_norm
    x = ln(z, (cz,), Wd["norm_in.weight"], Wd["norm_in.bias"], 1e-5)
    gp = torch.sigmoid(x @ Wd["g_in.weight"].T) * (x @ Wd["p_in.weight"].T)
    A_, B_ = gp[..., :h], gp[..., h:]
    o = torch.einsum("bkic,bkjc->bijc", A_, B_) if a.ending else torch.einsum("bikc,bjkc->bijc", A_, B_)
    o = ln(o, (h,), Wd["norm_out.weight"], Wd["norm_out.bias"], 1e-5)
    return z + (o @ Wd["p_out.weight"].T) * torch.sigmoid(x @ Wd["g_out.weight"].T)


ref = None if a.no_ref else reference()
dev = T.get_device()
if a.fast_mode:
    T.set_fast_mode(True)
outs = []
for spec in (a.arms.split(";") if ";" in a.arms else a.arms.split(",")):
    with T.lever_scope(spec):
        ckc = T.trunk_compute_kernel_config(compute_kernel_config())
        mod = T.TriangleMultiplication(a.ending, W0, ckc, gated_move=True)
        for rep in range(2):
            s0 = list(TL.T2_STATS)
            zd = ttnn.from_torch(z_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            y = mod(zd, None, add_to_input=True)
            yt = ttnn.to_torch(y).float()
            ttnn.deallocate(y)
            if zd.is_allocated():
                ttnn.deallocate(zd)
        t2 = [TL.T2_STATS[0] - s0[0], TL.T2_STATS[1] - s0[1]]
        zd = ttnn.from_torch(z_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ms = []
        for _ in range(a.reps):
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            zd = mod(zd, None, add_to_input=True)
            ttnn.synchronize_device(dev); ms.append(1e3 * (time.perf_counter() - t0))
        ttnn.deallocate(zd)
    u, ur = (yt - z_host).double(), (ref - z_host.double())
    rec = {"arm": spec, "t2": t2, "ms_median": round(statistics.median(ms), 3), "ms": [round(m, 3) for m in ms], "rel_rms_update": ((u - ur).pow(2).mean().sqrt() / ur.pow(2).mean().sqrt()).item(),
           "max_abs": (yt.double() - ref).abs().max().item(), "finite": bool(torch.isfinite(yt).all()),
           "equal_first": bool(torch.equal(yt, outs[0])) if outs else None,
           "maxdiff_first": (yt - outs[0]).abs().max().item() if outs else None}
    if ref is not None:
        u, ur = (yt - z_host).double(), (ref - z_host.double())
        rec["rel_rms_update"] = ((u - ur).pow(2).mean().sqrt() / ur.pow(2).mean().sqrt()).item()
        rec["max_abs"] = (yt.double() - ref).abs().max().item()
    outs.append(yt)
    print(json.dumps(rec), flush=True)
T.cleanup()
