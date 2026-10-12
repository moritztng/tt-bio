"""Transition T3 (the pair Transition as one kernel, tt_bio/transition_t3.py) against the incumbent Transition
module, both with add_to_input=True as the Pairformer calls it: rel rms and max abs against a float64 torch
reference of x + fc3(silu(fc1(ln(x))) * fc2(ln(x))) built from the same bf16 inputs, T3 finite, T3 served,
and median ms of warm synchronized calls each. One JSON line per N. Levers: the global normal set (silu_f32
on, as OpenFold3 runs it), plus transition_t3 for the T3 arm.

T3 has NEVER RUN on a device before this script: run it under `timeout`, smallest N first.

usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/spd_trimul/t3.py [--ns 64,256,512]
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="64,256,512")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--c", type=int, default=128)
ap.add_argument("--repeat", type=int, default=0, help="rerun T3 on the same input and list its bad elements each time")
ap.add_argument("--probe", action="store_true", help="isolate a T3 defect: RES=0 direct call vs fp64 fc3 output, "
                "and W3 = 0 (output must be x exactly)")
ap.add_argument("--fid", default="", help="override the trunk fidelity for both arms (HiFi2, HiFi3, LoFi)")
ap.add_argument("--approx", type=int, default=-1, help="override math_approx_mode (0 = exact silu, no silu_f32)")
ap.add_argument("--f3s", type=int, default=1, help="transition_t3.F3_SPLIT: fc3 K chunks")
ap.add_argument("--wh", action="store_true", help="no-op (T3 serves on Wormhole since the hifi4 decline)")
ap.add_argument("--out", help="also append each JSON line here (the qb runner passes it)")
A = ap.parse_args()
os.environ.setdefault("TT_BIO_LEVERS", "normal")

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, transition_t3 as T3
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
T3.F3_SPLIT = A.f3s
kcfg = T.trunk_compute_kernel_config(compute_kernel_config())
if A.fid:
    kcfg.math_fidelity = getattr(ttnn.MathFidelity, A.fid)
if A.approx >= 0:
    kcfg.math_approx_mode = bool(A.approx)
BASE = T.parse_levers("normal")
WITH = T.parse_levers("normal+transition_t3")
C, H = A.c, 4 * A.c
g = torch.Generator().manual_seed(0)
sd = {"norm.weight": 1 + 0.1 * torch.randn(C, generator=g), "norm.bias": 0.1 * torch.randn(C, generator=g),
      "fc1.weight": torch.randn(H, C, generator=g) / C ** 0.5, "fc2.weight": torch.randn(H, C, generator=g) / C ** 0.5,
      "fc3.weight": torch.randn(C, H, generator=g) / H ** 0.5}
sd = {k: v.to(torch.bfloat16).float() for k, v in sd.items()}
T._LEVERS = BASE
tr = T.Transition(sd, kcfg)
print(json.dumps({"grid": tuple(T.COMPUTE_GRID_MAIN), "fid": str(kcfg.math_fidelity), "c": C, "hidden": H}),
      flush=True)


def ref(x):
    x = x.double()
    w = {k: v.double() for k, v in sd.items()}
    xn = torch.nn.functional.layer_norm(x, (C,), w["norm.weight"], w["norm.bias"], 1e-5)
    h = torch.nn.functional.silu(xn @ w["fc1.weight"].t()) * (xn @ w["fc2.weight"].t())
    return x + h @ w["fc3.weight"].t()


def rel(a, r):
    a = a.double()
    return (float((a - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()), float((a - r).abs().max()))


def up(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def run(levers, xt):
    T._LEVERS = levers
    return tr(xt, add_to_input=True)


def timed(levers, x0, reps):
    xt = up(x0)
    run(levers, xt); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        xt = run(levers, xt); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    ttnn.deallocate(xt)
    return round(statistics.median(ms), 4)


for n in [int(v) for v in A.ns.split(",")]:
    x0 = torch.randn(1, n, n, C, generator=g).to(torch.bfloat16)
    r = ref(x0.float())
    rec = {"n": n}
    inc = ttnn.to_torch(run(BASE, up(x0))).float()
    s0 = T3.STATS[0]
    t3 = ttnn.to_torch(run(WITH, up(x0))).float()
    rec["t3_served"] = T3.STATS[0] - s0
    rec["t3_why"] = dict(T3.WHY)
    rec["t3_finite"] = bool(torch.isfinite(t3).all())
    if rec["t3_finite"]:
        rec["inc_rel_rms_f64"], rec["inc_max_abs"] = rel(inc, r)
        rec["t3_rel_rms_f64"], rec["t3_max_abs"] = rel(t3, r)
        rec["t3_vs_inc_max_abs"] = float((t3 - inc).abs().max())
        bad = ((t3 - r.float()).abs() > 0.25).reshape(-1, 32, C)        # [row tile, row, channel]
        if bad.any():
            ut = bad.any(-1).any(-1).nonzero().flatten()
            rows = bad.any(-1).any(0).nonzero().flatten()
            chans = bad.any(1).any(0).nonzero().flatten()
            rec["bad"] = {"n": int(bad.sum()), "units": ut[:40].tolist(), "nunits": int(ut.numel()),
                          "rows": rows.tolist(), "chans": chans[:40].tolist(), "nchans": int(chans.numel())}
    if A.repeat:
        rb = []
        for _ in range(A.repeat):
            o = ttnn.to_torch(run(WITH, up(x0))).float()
            d = ((o - r.float()).abs() > 0.25).reshape(-1)
            rb.append([int(v) for v in d.nonzero().flatten()[:8]])
        rec["repeat_bad_flat_idx"] = rb
        fi = torch.tensor(rb[0][:6], dtype=torch.long)
        xf, rf, tf, nf = x0.float().reshape(-1), r.float().reshape(-1), o.reshape(-1), inc.reshape(-1)
        rec["bad_vals"] = [[round(float(v[i]), 5) for v in (xf, rf, tf, nf)] for i in fi]   # x, ref, t3, incumbent
    if A.probe:
        T._LEVERS = WITH
        tr._t3(up(x0), True)                                     # builds tr._t3_w
        w12, w3 = tr._t3_w
        ckc = T._mm_generic.ckc_args(T.silu_ckc(tr.compute_kernel_config))
        grid = tuple(T.COMPUTE_GRID_MAIN)
        y = T3.transition(up(x0), tr.norm_weight, tr.norm_bias, w12, w3, ckc, grid, add_to_input=False)
        y = ttnn.to_torch(y).float()
        yr = (r - x0.double()).float()
        d = (y - yr).abs()
        bi = (d > 0.25).reshape(-1).nonzero().flatten()
        rec["p_res0_bad"] = int(bi.numel())
        rec["p_res0_max_abs"] = float(d.max())
        rec["p_res0_vals"] = [[round(float(yr.reshape(-1)[i]), 4), round(float(y.reshape(-1)[i]), 4)] for i in bi[:6]]
        rec["p_res0_rows"] = sorted({int(i) // C for i in bi[:400]})[:20]
        rec["p_res0_nrows"] = len({int(i) // C for i in bi})
        w3z = ttnn.from_torch(torch.zeros(H, C, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.to_torch(T3.transition(up(x0), tr.norm_weight, tr.norm_bias, w12, w3z, ckc, grid,
                                        add_to_input=True)).float()
        rec["p_w3zero_max_abs_vs_x"] = float((o - x0.float()).abs().max())
        rec["p_w3zero_bad"] = int(((o - x0.float()).abs() > 0).sum())
        # the fp64 hidden activations of the bad rows: a silu outlier would show as a large |h|
        xd = x0.double().reshape(-1, C)
        w = {k: v.double() for k, v in sd.items()}
        rows = torch.tensor(sorted({int(i) // C for i in bi})[:50], dtype=torch.long)
        if rows.numel():
            xn = torch.nn.functional.layer_norm(xd[rows], (C,), w["norm.weight"], w["norm.bias"], 1e-5)
            f1 = xn @ w["fc1.weight"].t()
            rec["p_bad_rows_fc1_absmax"] = float(f1.abs().max())
            rec["p_bad_rows_fc1_min"] = float(f1.min())
        allf1 = torch.nn.functional.layer_norm(xd[:20000], (C,), w["norm.weight"], w["norm.bias"], 1e-5) @ w["fc1.weight"].t()
        rec["p_ref_fc1_absmax_20k_rows"] = float(allf1.abs().max())
    rec["inc_ms"] = timed(BASE, x0, A.reps)
    rec["t3_ms"] = timed(WITH, x0, A.reps)
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(line + "\n")
