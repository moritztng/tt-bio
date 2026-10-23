"""RFD4's triangle tail standalone (T2CM, plain form, bf16 pair stream: what `pair_tail_cm` runs at 512, and plain T2
below it), per variant: median ms of warm synchronized calls, the error against a float64 reference, and the output's
sha256 (equal across trees = the same program). Variants: base (bf16 g_out, the trunk's fidelity), h
(trimul_tail.T2_HIFI2: the p GEMM at HiFi2), g8 (g_out bfp8, as trimul_t1g8 writes it), hg8 (both); and with
--abl the base timed under T2_ABL stage skips (2 p GEMM, 8 epilogue, 1 LayerNorm, 16 sigmoid).
--xtree adds plain T2 at the other models' keys (Boltz-2 c 128, c 256, OpenDDE c 384 with trimul_t2w), base only.

usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/rfd4/tri7/tail.py [--ns 512,256] [--abl 2,8,11,27] [--xtree]
"""
import argparse, hashlib, json, statistics, sys, time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="512,256")
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--variants", default="base,h,g8,hg8")
ap.add_argument("--abl", default="")
ap.add_argument("--xtree", action="store_true")
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, _mm_generic
from tt_bio.rfd4.trunk import compute_kernel_config

dev = T.get_device()
ckc = _mm_generic.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
grid = tuple(T.COMPUTE_GRID_MAIN)
up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)


def aiclk():
    import glob
    return {f.split("/")[-2]: open(f).read().strip() for f in sorted(glob.glob("/sys/class/tenstorrent/*/tt_aiclk"))}


def emit(rec):
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(line + "\n")


def timed(f, reps):
    f(); ttnn.synchronize_device(dev)
    ms, clk = [], []
    for i in range(reps):
        t0 = time.perf_counter()
        f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
        if i % 5 == 0:
            clk.append(aiclk())
    return round(statistics.median(ms), 4), clk


def case(n, K, N, cm, gen):
    h = {"x": torch.randn(1, n, n, K, generator=gen) * 2 + torch.randn(1, n, n, 1, generator=gen),
         "g": torch.randn(1, n, n, N, generator=gen), "z": torch.randn(1, n, n, N, generator=gen),
         "w": torch.randn(K, N, generator=gen) / K ** 0.5,
         "gamma": 1 + 0.1 * torch.randn(1, K, generator=gen), "beta": 0.1 * torch.randn(1, K, generator=gen)}
    h = {k: v.to(torch.bfloat16) for k, v in h.items()}
    x64 = {k: v.double() for k, v in h.items()}
    mu = x64["x"].mean(-1, keepdim=True)
    var = (x64["x"] - mu).pow(2).mean(-1, keepdim=True)
    xn = (x64["x"] - mu) / torch.sqrt(var + 1e-5) * x64["gamma"] + x64["beta"]
    ref = x64["z"] + (xn @ x64["w"]) * torch.sigmoid(x64["g"])
    d = {k: up(h[k]) for k in ("g", "w", "gamma", "beta")}
    d["x"] = up(h["x"].permute(0, 3, 1, 2)) if cm else up(h["x"])
    return h, d, ref


def run(d, z, g, cm, stream16):
    if cm:
        return TT.fused_tail_ln_cm(d["x"], d["gamma"], d["beta"], d["w"], g, z, ckc, grid, stream16=True)
    return TT.fused_tail_ln(d["x"], d["gamma"], d["beta"], d["w"], g, z, ckc, grid, stream16=stream16)


def measure(n, K, N, cm, stream16, variants, abl, tag):
    gen = torch.Generator().manual_seed(n * 7 + K)
    h, d, ref = case(n, K, N, cm, gen)
    g8 = ttnn.typecast(d["g"], ttnn.bfloat8_b)
    for v in variants:
        TT.T2_HIFI2 = v in ("h", "hg8")
        g = g8 if v in ("g8", "hg8") else d["g"]
        rec = {"tag": tag, "n": n, "K": K, "N": N, "cm": cm, "variant": v}
        z = up(h["z"])
        y = run(d, z, g, cm, stream16)
        if y is None:
            rec["declined"] = TT.REJECTS and {":".join(k): c for k, c in TT.REJECTS.items()}
            emit(rec); ttnn.deallocate(z); continue
        out = ttnn.to_torch(y).reshape(ref.shape)
        rec["sha"] = hashlib.sha256(out.to(torch.bfloat16).view(torch.int16).numpy().tobytes()).hexdigest()[:16]
        e = out.double() - ref
        rec["rel_rms_f64"] = float(e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
        rec["upd_rel_rms"] = float(e.pow(2).mean().sqrt() / (ref - h["z"].double()).pow(2).mean().sqrt())
        rec["finite"] = bool(torch.isfinite(out).all())
        rec["ms"], rec["aiclk"] = timed(lambda: run(d, z, g, cm, stream16), A.reps)
        if v == "base":
            for m in abl:
                TT.T2_ABL = m
                rec[f"ms_abl{m}"] = timed(lambda: run(d, z, g, cm, stream16), A.reps)[0]
            TT.T2_ABL = 0
        TT.T2_HIFI2 = False
        emit(rec)
        ttnn.deallocate(z)


TT.T2 = TT.T2CM = True
emit({"start": True, "ckc": [str(c) for c in ckc], "grid": grid, "aiclk": aiclk()})
abl = [int(v) for v in A.abl.split(",") if v]
for n in [int(s) for s in A.ns.split(",") if s]:
    cm = TT.t2cm_stream16_fits(n, grid[0] * grid[1])
    with T.lever_scope(frozenset({"pair_tail_cm", "pair_bf16z", "pair_tail"})):
        measure(n, 256, 384, cm, True, A.variants.split(","), abl, "rfd4")
if A.xtree:
    for K, lev in ((128, ()), (256, ()), (384, ("trimul_t2w",))):
        with T.lever_scope(frozenset(lev)):
            measure(512, K, K, False, False, ["base"], [], f"x{K}")
