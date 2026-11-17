"""RFD4's pair transition as the trunk runs it (`pair_block`: pair_relu with its layer_norm prologue, residual in place
into the bf16 z, bfp8 weights, HiFi2), c_z 384, hidden 768: its time per call, what each LayerNorm stage costs
(LN_ABL), and LN_FUSE (the finish in DST, kernels/common/ln_prologue.hpp) with its error against float64.

Arms, one JSON line per N:
  base     production
  rows32   gamma / beta replicated to 32 rows, no fuse: must equal base bit for bit (the fuse's input)
  lnf      LN_FUSE=1 on the rows32 vectors
  abl<m>   LN_ABL=m (1 skips the finish, 2 the square): timing only, the output is wrong on purpose
Error: rel rms of each arm's update (out - z) against a float64 reference on the same bf16 z and the device's own
bfp8 weights.

usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/spd_trimul/pair_relu_ln.py --ns 352,256,128 --out F.jsonl
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="352,256,128")
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--c", type=int, default=384)
ap.add_argument("--hidden", type=int, default=768)
ap.add_argument("--abl", default="1,2")
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, pair_swiglu_ws as WS
import clocksample

dev = T.get_device()
EPS, C, H = 1e-5, A.c, A.hidden
FID = ttnn.MathFidelity.HiFi2
torch.manual_seed(0)
_run = WS._run
EXTRA: list = []


def run_with_defines(x, w12, w3, fidelity, defines, *a, **k):
    return _run(x, w12, w3, fidelity, tuple(defines) + tuple(EXTRA), *a, **k)


WS._run = run_with_defines
up = lambda t, d=ttnn.bfloat16: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
back = lambda t: ttnn.to_torch(t).double()  # noqa: E731
w1, b1, w2, b2 = (torch.randn(C, H) / C ** .5, torch.randn(H) * .1, torch.randn(H, C) / H ** .5, torch.randn(C) * .1)
W = WS.relu_weights(w1, b1, w2, b2, dev, dtype=ttnn.bfloat8_b)
gam, bet = 1 + .1 * torch.randn(C), .1 * torch.randn(C)
g1, b1t = up(gam.reshape(1, C)), up(bet.reshape(1, C))
g32, b32 = up(gam.reshape(1, C).expand(32, C).contiguous()), up(bet.reshape(1, C).expand(32, C).contiguous())
rw1, rw2 = back(W[0])[:C, :H], back(W[2])[:H, :C]
rb1, rb2 = back(W[1])[0, :H], back(W[3])[0, :C]


def rel(a, r):
    return float((a - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt())


def call(z, g, b):
    return WS.pair_relu(z, *W, fidelity=FID, ln=(g, b, EPS), resid=z)


def timed(f, reps):
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


for n in [int(v) for v in A.ns.split(",")]:
    z0 = torch.randn(1, n, n, C).bfloat16()
    zf = z0.double()
    ref = torch.relu(torch.nn.functional.layer_norm(zf, (C,), gam.double(), bet.double(), EPS) @ rw1 + rb1) @ rw2 + rb2
    rec = {"n": n, "c": C, "hidden": H}
    clk = clocksample.during(period=0.5).__enter__()
    outs = {}
    arms = [("base", g1, b1t, []), ("rows32", g32, b32, []), ("lnf", g32, b32, [("LN_FUSE", "1")])]
    arms += [(f"abl{m}", g1, b1t, [("LN_ABL", m)]) for m in A.abl.split(",") if m]
    for name, g, b, d in arms:
        EXTRA[:] = d
        try:
            z = up(z0)
            call(z, g, b)
            o = back(z)
            outs[name] = o
            if not name.startswith("abl"):
                rec[f"{name}_rel_rms_f64"] = rel(o - zf, ref)
                rec[f"{name}_finite"] = bool(torch.isfinite(o).all())
            rec[f"{name}_ms"] = timed(lambda: call(z, g, b), A.reps)
            ttnn.deallocate(z)
        except Exception as e:  # noqa: BLE001
            rec[f"{name}_error"] = repr(e)[:400]
    EXTRA[:] = []
    clk.__exit__(None, None, None)
    if "base" in outs and "rows32" in outs:
        rec["rows32_equal_base"] = bool(torch.equal(outs["base"], outs["rows32"]))
    rec["aiclk"] = clk.summary()
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(line + "\n")
