"""T2CM in AF2's form against float64: `out = z + mask * (layer_norm(x) @ Wp + bp) * sigmoid(xb @ Wg + bg)`,
x read channel-major [1, K, S, S] as the trimul einsum leaves it, mask a per-row dropout mask [1, 1, S, N]
shared over i, the result written to a separate tensor (z untouched). The incumbent composed ops (permute
back, layer_norm, two biased linears, multiply with SIGMOID on b, mask multiply, add) are graded against the
same float64 reference, so the fused kernel's error sits beside the error main already ships. Also the
shipped T2CM form (no extras, z in place) as an A/A of the unchanged program. One JSON line per case.

usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/bc4_fuse/t2cm_af2.py [--n 224] [--reps 10]
"""
import argparse, glob, json, os, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=224)
ap.add_argument("--c", type=int, default=128)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
kcfg = T.trunk_compute_kernel_config(compute_kernel_config())
ckc, grid = MG.ckc_args(kcfg), tuple(T.COMPUTE_GRID_MAIN)
TT.set_t2(True)
TT.set_t2cm(True)
N, C = A.n, A.c
g = torch.Generator().manual_seed(A.seed)
bf = lambda t: t.to(torch.bfloat16).double()  # noqa: E731
xcm = bf(torch.randn(1, C, N, N, generator=g))
xb = bf(torch.randn(1, N, N, C, generator=g))
z = bf(torch.randn(1, N, N, C, generator=g))
lnw, lnb = bf(1 + 0.1 * torch.randn(C, generator=g)), bf(0.1 * torch.randn(C, generator=g))
wp, wg = bf(torch.randn(C, C, generator=g) / C ** 0.5), bf(torch.randn(C, C, generator=g) / C ** 0.5)
bp, bgv = bf(0.1 * torch.randn(C, generator=g)), bf(0.1 * torch.randn(C, generator=g))
mask = bf((torch.rand(1, 1, N, C, generator=g) > 0.25).double() / 0.75)


def aiclk():
    out = {}
    for f in sorted(glob.glob("/sys/class/tenstorrent/*/tt_aiclk")):
        try:
            out[f.split("/")[-2]] = open(f).read().strip()
        except OSError:
            pass
    return out


def up(t):
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def ref(af2):
    x = xcm.permute(0, 2, 3, 1)
    xn = torch.nn.functional.layer_norm(x, (C,), lnw, lnb, 1e-5)
    if not af2:
        return z + (xn @ wp) * torch.sigmoid(xb)        # shipped form: g is the gate pre-activation
    u = (xn @ wp + bp) * torch.sigmoid(xb @ wg + bgv)
    return z + u * mask


def grade(dev_out, r):
    o = ttnn.to_torch(dev_out).double().reshape(r.shape)
    d = o - r
    return {"rel_l2": float(d.norm() / (r - z).norm()), "max_abs": float(d.abs().max()),
            "finite": bool(torch.isfinite(o).all())}


print(json.dumps({"n": N, "c": C, "grid": grid, "arch": str(dev.arch()), "fid": str(kcfg.math_fidelity),
                  "visible": os.environ.get("TT_VISIBLE_DEVICES"), "aiclk": aiclk()}), flush=True)
X, XB, Z = up(xcm), up(xb), up(z)
LW, LB = up(lnw.reshape(1, C)), up(lnb.reshape(1, C))
WP, WG = up(wp), up(wg)
BP, BG = up(bp.reshape(1, C)), up(bgv.reshape(1, C))
BPR, BGR = TT.bias_rows(BP), TT.bias_rows(BG)
M = up(mask)
z_before = ttnn.to_torch(Z).double()


def fused():
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, N, N, C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                         ttnn.DRAM_MEMORY_CONFIG)
    r = TT.fused_tail_ln_cm(X, LW, LB, WP, XB, Z, ckc, grid, wg=WG, bp=BPR, bg=BGR, mask=M, out=out)
    if r is None:
        raise SystemExit(f"declined: {TT.REJECTS}")
    return r


def composed():
    xp = ttnn.permute(X, (0, 2, 3, 1))
    xn = ttnn.layer_norm(xp, weight=LW, bias=LB, epsilon=1e-5,
                         compute_kernel_config=T.ln_compute_kernel_config(kcfg))
    p = ttnn.linear(xn, WP, bias=BP, compute_kernel_config=kcfg)
    gl = ttnn.linear(XB, WG, bias=BG, compute_kernel_config=kcfg)
    u = ttnn.multiply(p, gl, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    um = ttnn.multiply(u, M)
    o = ttnn.add(Z, um)
    for t in (xp, xn, p, gl, u, um):
        ttnn.deallocate(t)
    return o


recs = []
for name, fn, af2 in (("t2cm_af2", fused, True), ("composed_af2", composed, True)):
    o = fn()
    rec = {"case": name, **grade(o, ref(af2))}
    ttnn.deallocate(o)
    recs.append(rec)
recs[0]["z_untouched"] = bool(torch.equal(ttnn.to_torch(Z).double(), z_before))
# the shipped form, z in place, as before this change (g = xb as the gate pre-activation)
Zs = up(z)
o = TT.fused_tail_ln_cm(X, LW, LB, WP, XB, Zs, ckc, grid)
recs.append({"case": "t2cm_shipped", **grade(o, ref(False))} if o is not None else {"case": "t2cm_shipped",
                                                                                    "declined": str(TT.REJECTS)})
ttnn.deallocate(Zs)
# warm interleaved timing, fused vs composed
ttnn.synchronize_device(dev)
tf, tc = [], []
for _ in range(A.reps):
    for fn, acc in ((fused, tf), (composed, tc)):
        t0 = time.perf_counter(); o = fn(); ttnn.synchronize_device(dev)
        acc.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
recs.append({"case": "timing", "fused_ms": round(statistics.median(tf), 4), "composed_ms": round(statistics.median(tc), 4),
             "fused_range": [round(min(tf), 4), round(max(tf), 4)], "composed_range": [round(min(tc), 4), round(max(tc), 4)],
             "aiclk": aiclk(), "loadavg": os.getloadavg()})
for r in recs:
    print(json.dumps(r), flush=True)
