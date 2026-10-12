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
ap.add_argument("--out", help="also append each JSON line here (the qb runner passes it)")
A = ap.parse_args()
os.environ.setdefault("TT_BIO_LEVERS", "normal")

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, transition_t3 as T3
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
kcfg = T.trunk_compute_kernel_config(compute_kernel_config())
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
    rec["inc_ms"] = timed(BASE, x0, A.reps)
    rec["t3_ms"] = timed(WITH, x0, A.reps)
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(line + "\n")
