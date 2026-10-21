"""pair_relu (AF2's pair Transition, LN prologue in the kernel) per call at BC2's 224 shapes: the Evoformer pair
(c 128, hidden 512) and the template pair (c 64, hidden 128), forward and backward, against float64. Then what binds:
the production fidelity against HiFi2 and LoFi, and phase ablations (wrong on purpose, timing only): no fc1/fc2
matmuls (WS_NO_FC12), no fc3, both, no x read, no write, and the same program with no LayerNorm prologue.

usage: TT_VISIBLE_DEVICES=<chip> python perf/bc4_swiglu/relu_probe.py --out x.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=224)
ap.add_argument("--shapes", default="128x512,64x128")
ap.add_argument("--calls", type=int, default=50)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--drops", default="", help="comma list of WS_NO_ ablations, only these")
ap.add_argument("--quick", action="store_true", help="production arm only")
ap.add_argument("--out", required=True)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, pair_swiglu_ws as W
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
DR = ttnn.DRAM_MEMORY_CONFIG
CHIP = os.environ.get("TT_VISIBLE_DEVICES", "0")
FID = compute_kernel_config().math_fidelity
out = open(A.out, "a")


def emit(rec):
    rec["aiclk"] = int(open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read().split()[0])
    rec["load"] = os.getloadavg()[0]
    print(json.dumps(rec), flush=True)
    out.write(json.dumps(rec) + "\n")
    out.flush()


def timed(fn):
    runs = []
    for _ in range(A.reps):
        fn().deallocate()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        keep = [fn() for _ in range(A.calls)]
        ttnn.synchronize_device(dev)
        runs.append((time.perf_counter() - t0) / A.calls * 1e3)
        for o in keep:
            o.deallocate()
    runs.sort()
    return runs[0], runs[len(runs) // 2], runs[-1]


def up(t):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DR)


def rr(d, r):
    return float(((d.double() - r).pow(2).mean() / r.pow(2).mean()).sqrt())


emit({"fidelity": str(FID), "sha": os.popen("git rev-parse HEAD").read().strip(),
      "dirty": os.popen("git status --porcelain --untracked-files=no").read().strip()})
n = A.n
for shape in A.shapes.split(","):
    C, H = map(int, shape.split("x"))
    g = torch.Generator().manual_seed(C)
    bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
    x = bf(torch.randn(1, n, n, C, generator=g) * 2 + 0.3)
    dy = bf(torch.randn(1, n, n, C, generator=g))
    w1, w2 = bf(torch.randn(C, H, generator=g) / C ** 0.5), bf(torch.randn(H, C, generator=g) / H ** 0.5)
    b1, b2 = bf(torch.randn(H, generator=g) * 0.1), bf(torch.randn(C, generator=g) * 0.1)
    gam, bet = bf(1 + 0.1 * torch.randn(C, generator=g)), bf(0.1 * torch.randn(C, generator=g))
    # float64 reference on the same bf16 operands
    x64 = x.double()
    mu, var = x64.mean(-1, keepdim=True), x64.var(-1, unbiased=False, keepdim=True)
    xn = (x64 - mu) / (var + 1e-5).sqrt() * gam.double() + bet.double()
    pre = xn @ w1.double() + b1.double()
    ref_f = pre.clamp_min(0) @ w2.double() + b2.double()
    ref_b = ((pre > 0).double() * (dy.double() @ w2.double().t())) @ w1.double().t()
    dx, ddy = up(x), up(dy)
    rw = W.relu_weights(w1.float(), b1.float(), w2.float(), b2.float(), dev)
    bw12, bw3 = W.relu_bw_weights(w1.float(), w2.float(), dev)
    gl, bl = (ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for v in (gam, bet))
    LN = (gl, bl, 1e-5)
    assert W.relu_eligible(dx, rw[0], rw[2], LN[:2])
    fwd = lambda fid=FID, d=(), ln=LN: W._run(dx, rw[0], rw[2], fid, d, W.SILU_POLY, ln, None, (rw[1], rw[3]))  # noqa
    bwd = lambda fid=FID, d=(), ln=LN: W._run(dx, bw12, bw3, fid, d, W.SILU_POLY, ln, None, (rw[1], ddy), bw=True)  # noqa
    for name, fn, ref in (("fwd", fwd, ref_f), ("bw", bwd, ref_b)):
        o = fn()
        err = rr(ttnn.to_torch(o).double(), ref)
        o.deallocate()
        lo, med, hi = timed(fn)
        gflop = 2 * n * n * C * H * (2 if name == "fwd" else 3) / 1e9
        emit({"c": C, "h": H, "dir": name, "arm": "prod", "rel_vs_f64": err, "ms": lo, "ms_med": med, "ms_max": hi,
              "gflop": gflop, "tflops": gflop / lo})
        if A.quick:
            continue
        arms = [("HiFi2", dict(fid=ttnn.MathFidelity.HiFi2)), ("LoFi", dict(fid=ttnn.MathFidelity.LoFi)),
                ("no_ln", dict(ln=None))]
        drops = A.drops.split(",") if A.drops else ("FC12", "FC3", "FC12+FC3", "XREAD", "WRITE", "XREAD+WRITE",
                                                    "FC12+FC3+XREAD+WRITE", "WREAD", "WREAD+FC12+FC3")
        if A.drops:
            arms = []
        for drop in drops:
            arms.append(("no_" + drop.lower(), dict(d=tuple(("WS_NO_" + k, "1") for k in drop.split("+")))))
        arms.append(("no_ln_no_fc12+fc3", dict(ln=None, d=(("WS_NO_FC12", "1"), ("WS_NO_FC3", "1")))))
        for arm, kw in arms:
            f = lambda kw=kw, fn=fn: fn(**kw)  # noqa: E731
            lo, med, hi = timed(f)
            emit({"c": C, "h": H, "dir": name, "arm": arm, "ms": lo, "ms_med": med, "ms_max": hi})
    for t in (dx, ddy, gl, bl, bw12, bw3, *rw):
        t.deallocate()
emit({"done": True})
