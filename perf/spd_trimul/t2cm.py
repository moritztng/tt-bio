"""T2CM (`trimul_t2cm`: T2 reads the einsum's channel-major output, `reblock_permute_back` skipped) against
the incumbent T2 path, on the full TriangleMultiplication module called as the Pairformer calls it
(add_to_input=True, pair mask). Both arms run the same base lever set (default normal+trimul_t1+trimul_t2,
OpenFold3's 512-token route); the lever arm adds trimul_t2cm. Per N and variant (start / end): max abs
difference lever vs incumbent and whether it is bit-exact (it must be: a pure data move), both finite, how
many calls T2 / T2CM / the back move served in each arm, and the median ms of interleaved warm synchronized
calls. One JSON line per (N, variant).

T2CM has NEVER RUN on a device before this script: run it under `timeout`, smallest N first. Below
trimul_tail.MIN_N (160) T2 does not serve, so N = 64 is an inert A/A of the module.

usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/spd_trimul/t2cm.py [--ns 64,256,512]
"""
import argparse, glob, json, os, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="64,256,512")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--c", type=int, default=128, help="c_z = hidden")
ap.add_argument("--base", default="normal+trimul_t1+trimul_t2")
ap.add_argument("--variants", default="start,end")
ap.add_argument("--no-mask", action="store_true")
ap.add_argument("--out", help="also append each JSON line here (the pool job passes it)")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, reblock_permute as RB
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
kcfg = T.trunk_compute_kernel_config(compute_kernel_config())
BASE = T.parse_levers(A.base)
WITH = T.parse_levers(A.base + "+trimul_t2cm")
C = A.c
g = torch.Generator().manual_seed(0)
r = lambda *s: torch.randn(*s, generator=g) / s[-1] ** 0.5  # noqa: E731
sd = {"norm_in.weight": 1 + 0.1 * torch.randn(C, generator=g), "norm_in.bias": 0.1 * torch.randn(C, generator=g),
      "norm_out.weight": 1 + 0.1 * torch.randn(C, generator=g), "norm_out.bias": 0.1 * torch.randn(C, generator=g),
      "g_in.weight": r(2 * C, C), "p_in.weight": r(2 * C, C), "g_out.weight": r(C, C), "p_out.weight": r(C, C)}
sd = {k: v.to(torch.bfloat16).float() for k, v in sd.items()}


def aiclk():
    out = {}
    for f in sorted(glob.glob("/sys/class/tenstorrent/*/tt_aiclk")):
        try:
            out[f.split("/")[-2]] = open(f).read().strip()
        except OSError:
            pass
    return out


print(json.dumps({"grid": tuple(T.COMPUTE_GRID_MAIN), "arch": str(dev.arch()), "fid": str(kcfg.math_fidelity),
                  "c": C, "base": sorted(BASE), "visible": os.environ.get("TT_VISIBLE_DEVICES"),
                  "aiclk": aiclk()}), flush=True)


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def stats():
    return {"t2": list(TT.T2_STATS), "t2cm": list(TT.T2CM_STATS), "t1": list(TT.T1_STATS),
            "back": RB.STATS_BACK[0]}


def delta(a, b):
    return {k: ([y - x for x, y in zip(a[k], b[k])] if isinstance(a[k], list) else b[k] - a[k]) for k in a}


def run(mod, levers, zt, m):
    T._LEVERS = levers
    return mod(zt, m, add_to_input=True)


bad = 0
for n in [int(v) for v in A.ns.split(",") if v]:
    z0 = torch.randn(1, n, n, C, generator=g).to(torch.bfloat16)
    m_host = None if A.no_mask else (torch.rand(1, n, n, generator=g) > 0.1).float()
    m = None if m_host is None else up(m_host)
    for var in A.variants.split(","):
        rec = {"n": n, "variant": var}
        try:
            mod = T.TriangleMultiplication(var == "end", sd, kcfg)
            s0 = stats()
            inc = ttnn.to_torch(run(mod, BASE, up(z0), m)).float()
            s1 = stats()
            lev = ttnn.to_torch(run(mod, WITH, up(z0), m)).float()
            s2 = stats()
            rec["inc_served"], rec["lev_served"] = delta(s0, s1), delta(s1, s2)
            rec["finite"] = bool(torch.isfinite(inc).all() and torch.isfinite(lev).all())
            d = (lev - inc).abs()
            rec["max_abs_diff"] = float(d.max())
            rec["n_diff"] = int((d > 0).sum())
            rec["bit_exact"] = bool(torch.equal(lev, inc))
            if not rec["bit_exact"]:
                idx = (d > 0).reshape(-1).nonzero().flatten()[:8]
                rec["diff_at"] = [[int(i), float(inc.reshape(-1)[i]), float(lev.reshape(-1)[i])] for i in idx]
            rec["rejects"] = {f"{k[0]}:{k[1]}": v for k, v in TT.REJECTS.items() if k[0].startswith("t2")}
            # interleaved warm timing, each arm on its own in-place z
            zb, zl = up(z0), up(z0)
            zb = run(mod, BASE, zb, m); zl = run(mod, WITH, zl, m); ttnn.synchronize_device(dev)
            tb, tl = [], []
            for _ in range(A.reps):
                t0 = time.perf_counter(); zb = run(mod, BASE, zb, m); ttnn.synchronize_device(dev)
                tb.append((time.perf_counter() - t0) * 1e3)
                t0 = time.perf_counter(); zl = run(mod, WITH, zl, m); ttnn.synchronize_device(dev)
                tl.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(zb); ttnn.deallocate(zl)
            rec["inc_ms"], rec["lev_ms"] = round(statistics.median(tb), 4), round(statistics.median(tl), 4)
            rec["saved_ms"] = round(rec["inc_ms"] - rec["lev_ms"], 4)
            rec["aiclk"] = aiclk()
        except Exception as e:                                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            rec["err"] = str(e).splitlines()[0][:300]
        bad += "err" in rec or not rec.get("bit_exact", False) or not rec.get("finite", False)
        line = json.dumps(rec)
        print(line, flush=True)
        if A.out:
            with open(A.out, "a") as f:
                f.write(line + "\n")
    if m is not None:
        ttnn.deallocate(m)
T._LEVERS = BASE
sys.exit(1 if bad else 0)
