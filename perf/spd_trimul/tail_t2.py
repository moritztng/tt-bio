"""T2 (one kernel: layer_norm -> @ Wp -> * sigmoid(g) -> + z in place) against the incumbent
four-op c_z = 128 trimul output tail: rel rms and max abs against a float64 torch reference for
both, T2 finite, and median ms of 10 warm synchronized calls each. One JSON line per N.

The incumbent is the tail's own ops: ttnn.layer_norm (ln_compute_kernel_config), the
`_trimul_out_proj` projection, multiply_ with SIGMOID on b, add_ into z.

T2 has NEVER RUN on a device before this script: run it under `timeout`, smallest N first, so a
hang costs one bounded run (synchronize_device does not return to a SIGALRM).

usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/spd_trimul/tail_t2.py [--ns 64,128,512]
"""
import argparse, json, statistics, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="64,128,512")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--c", type=int, default=128)
ap.add_argument("--gate", action="store_true",
                help="GATE form: g is the trimul's input norm and T2 computes sigmoid(g @ wg) itself")
ap.add_argument("--block", default=None,
                help="M,K,N,subblock_h,subblock_w: measurement-only _MM_BLOCK entry for (c/32, c/32)")
ap.add_argument("--out", help="also append each JSON line here (the qb runner passes it)")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
kcfg = T.trunk_compute_kernel_config(compute_kernel_config())
ckc = MG.ckc_args(kcfg)
grid = tuple(T.COMPUTE_GRID_MAIN)
TT.set_t2(True)
C = A.c
TT.T2_KEYS.add((C // 32, C // 32))   # measurement only: the model path keeps its own keys
if A.block:
    T._MM_BLOCK[(C // 32, C // 32)] = tuple(int(v) for v in A.block.split(","))
gen = torch.Generator().manual_seed(0)
up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
print(json.dumps({"grid": grid, "ckc": [str(c) for c in ckc], "c": C, "gate": A.gate,
                  "block": T._MM_BLOCK.get((C // 32, C // 32))}), flush=True)


def rel(a, r):
    a = a.double()
    return (float((a - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()),
            float((a - r).abs().max()))


def timed(f, reps):
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


bad = 0
for n in [int(s) for s in A.ns.split(",") if s]:
    rec = {"n": n}
    # Magnitudes like the tail's: x_out O(1) with a per-row offset (LN must remove it), z O(1).
    h = {"x": (torch.randn(1, n, n, C, generator=gen) * 2 + torch.randn(1, n, n, 1, generator=gen)),
         "g": torch.randn(1, n, n, C, generator=gen), "z": torch.randn(1, n, n, C, generator=gen),
         "w": torch.randn(C, C, generator=gen) / C ** 0.5,
         "gamma": 1 + 0.1 * torch.randn(C, generator=gen), "beta": 0.1 * torch.randn(C, generator=gen)}
    if A.gate:
        h["wg"] = torch.randn(C, C, generator=gen) / C ** 0.5
    h = {k: v.to(torch.bfloat16) for k, v in h.items()}
    d = {k: up(h[k]) for k in ("x", "g", "w", "gamma", "beta") + (("wg",) if A.gate else ())}
    wg = d.get("wg")
    x64 = {k: v.double() for k, v in h.items()}
    mu = x64["x"].mean(-1, keepdim=True)
    var = (x64["x"] - mu).pow(2).mean(-1, keepdim=True)
    xn = (x64["x"] - mu) / torch.sqrt(var + 1e-5) * x64["gamma"] + x64["beta"]
    g64 = x64["g"] @ x64["wg"] if A.gate else x64["g"]
    ref = x64["z"] + (xn @ x64["w"]) * torch.sigmoid(g64)
    P = n * n * C * 2

    def incumbent(z):
        xn_ = ttnn.layer_norm(d["x"], weight=d["gamma"], bias=d["beta"], epsilon=1e-5,
                              compute_kernel_config=T.ln_compute_kernel_config(kcfg))
        p = T._trimul_out_proj(xn_, d["w"], kcfg, None)
        ttnn.deallocate(xn_)
        g = T._trimul_out_proj(d["g"], wg, kcfg, None) if A.gate else d["g"]
        p = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        if A.gate:
            ttnn.deallocate(g)
        z = ttnn.add_(z, p)
        ttnn.deallocate(p)
        return z

    try:
        z = up(h["z"])
        out = ttnn.to_torch(incumbent(z)).reshape(ref.shape)
        rec["inc_rel_rms_f64"], rec["inc_max_abs"] = rel(out, ref)
        rec["inc_ms"] = timed(lambda: incumbent(z), A.reps)
        ttnn.deallocate(z)
    except Exception as e:                                                    # noqa: BLE001
        rec["inc_err"] = str(e).splitlines()[0][:300]

    try:
        z = up(h["z"])
        rec["t2_why"] = TT.t2_eligible(d["x"], d["gamma"], d["beta"], d["w"], d["g"], z, wg)
        y = TT.fused_tail_ln(d["x"], d["gamma"], d["beta"], d["w"], d["g"], z, ckc, grid, wg=wg)
        if y is None:
            rec["t2_err"] = f"declined: {rec['t2_why']}"
        else:
            ttnn.synchronize_device(dev)
            out = ttnn.to_torch(y).reshape(ref.shape)
            rec["t2_finite"] = bool(torch.isfinite(out).all())
            rec["t2_rel_rms_f64"], rec["t2_max_abs"] = rel(out, ref)
            rec["t2_ms"] = timed(lambda: TT.fused_tail_ln(d["x"], d["gamma"], d["beta"], d["w"],
                                                          d["g"], z, ckc, grid, wg=wg), A.reps)
            rec["t2_GBps"] = round(4 * P / rec["t2_ms"] / 1e6, 1)
            if "inc_ms" in rec:
                rec["speedup"] = round(rec["inc_ms"] / rec["t2_ms"], 3)
        ttnn.deallocate(z)
    except Exception as e:                                                    # noqa: BLE001
        rec["t2_err"] = str(e).splitlines()[0][:300]
        traceback.print_exc()
    rec["t2_stats"] = list(TT.T2_STATS)
    bad += "t2_err" in rec or "inc_err" in rec
    print(json.dumps(rec), flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    for t in d.values():
        ttnn.deallocate(t)
sys.exit(1 if bad else 0)
