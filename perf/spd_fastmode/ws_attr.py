"""Where pair_swiglu_ws spends its time: the kernel with one phase dropped at a time, plus LN and the bfp8 cast alone.

    TT_VISIBLE_DEVICES=<card> python perf/spd_fastmode/ws_attr.py [--n 512] [--c 256] [--hidden 1024] [--reps 20] [--rms]

Arms: `fast` (bfp8 x and weights, HiFi2), `normal` (bf16, HiFi4) and `lofi` (bfp8, LoFi), each once with metal's silu_tile, then with the
default polynomial silu plain and with one of the kernels' WS_NO_* defines (FC12, SILU, GATE, FC3 in compute; XREAD in the reader; WRITE in the writer), or several joined by `+`. A dropped phase makes the
output wrong by design, so this prints time only: ms/call, median of --reps after 3 warm calls. One JSON line per arm.
--rms runs every pair_swiglu call with the RMSNorm prologue (WS_RMS), as RFdiffusion3's pair Transition calls it.
--ln runs it with the LayerNorm prologue (WS_LN) over a bf16 x, as the `ws_ln` lever (Protenix-v2 --fast) calls it.
"""
import argparse, json, os, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--c", type=int, default=256)
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--rms", action="store_true")
ap.add_argument("--ln", action="store_true")
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import pair_swiglu_ws as W  # noqa: E402

g = torch.Generator().manual_seed(0)
c, h = a.c, a.hidden
w1, w2 = (torch.randn(c, h, generator=g) / c ** 0.5 for _ in range(2))
w3 = torch.randn(h, c, generator=g) / h ** 0.5
dev = T.get_device()
xt = ttnn.from_torch(torch.randn(1, a.n, a.n, c, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=dev)
nw = ttnn.from_torch(torch.ones(c), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
nb = ttnn.from_torch(torch.randn(c, generator=g) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


if a.rms:
    _ps = W.pair_swiglu
    W.pair_swiglu = lambda *args, **kw: _ps(*args, ln=(nw, None, 1e-6), **kw)
if a.ln:
    _ps = W.pair_swiglu
    W.pair_swiglu = lambda *args, **kw: _ps(*args, ln=(nw, nb, 1e-5), **kw)


def timed(fn):
    for _ in range(3):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
    return round(1e3 * statistics.median(ts), 3)


def emit(**kw):
    print(json.dumps(dict(n=a.n, c=c, hidden=h, rms=a.rms, card=os.environ.get("TT_VISIBLE_DEVICES"), **kw)), flush=True)


emit(arm="layer_norm", ms=timed(lambda: ttnn.layer_norm(xt, weight=nw, epsilon=1e-5,
                                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)))
emit(arm="typecast_bfp8", ms=timed(lambda: ttnn.typecast(xt, ttnn.bfloat8_b)))
MODES = (("fast", ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2), ("normal", ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
         ("lofi", ttnn.bfloat8_b, ttnn.MathFidelity.LoFi))   # ws_lofi, the Wormhole --fast pair SwiGLU
if a.rms:  # the prologue reads a bf16 x; RFdiffusion3 runs bf16 at its own fidelity in both modes
    MODES = (("normal", ttnn.bfloat16, ttnn.MathFidelity.HiFi4), ("hifi2", ttnn.bfloat16, ttnn.MathFidelity.HiFi2))
for mode, dt, fid in MODES:
    parts = W.weights(w1, w2, w3, dev, dt)
    if len(parts) > 1:  # attribution is per program; a chunked call is several
        emit(arm=mode, ms=None, note=f"{len(parts)} hidden chunks (L1)")
        continue
    (w12, w3t), = parts
    x = ttnn.typecast(xt, dt) if dt != ttnn.bfloat16 and not a.ln else xt   # the LN prologue reads the raw bf16 x
    if not W.eligible(x, w12, w3t, ln=(nw, nb) if a.ln else ()):
        emit(arm=mode, ms=None, note="not eligible (L1)")
        continue
    emit(arm=mode + "-metal_silu", fidelity=str(fid), ms=timed(lambda: W.pair_swiglu(x, w12, w3t, fid, (), 0)))
    emit(arm=mode + "-poly6_silu", fidelity=str(fid), ms=timed(lambda: W.pair_swiglu(x, w12, w3t, fid, (), 2)))
    for tag, xin in (("", x), ("-bf16x", xt)):   # ws_psilu: the silu on the packer, must equal poly6 bit for bit
        if tag and (dt == ttnn.bfloat16 or a.ln or not W.eligible(xt, w12, w3t)):
            continue
        ref = ttnn.to_torch(W.pair_swiglu(xin, w12, w3t, fid, (), 2))
        got = ttnn.to_torch(W.pair_swiglu(xin, w12, w3t, fid, (), 2, pack_silu=True))
        emit(arm=mode + tag + "-pack_silu", fidelity=str(fid), equal_poly6=bool(torch.equal(ref, got)),
             ms=timed(lambda: W.pair_swiglu(xin, w12, w3t, fid, (), 2, pack_silu=True)))
    if dt != ttnn.bfloat16 and W.eligible(xt, w12, w3t):  # is the bfp8 cast of x (typecast_bfp8 above) worth it?
        emit(arm=mode + "-bf16x", fidelity=str(fid), ms=timed(lambda: W.pair_swiglu(xt, w12, w3t, fid)))
        emit(arm=mode + "-bf16x-poly6_silu", fidelity=str(fid), ms=timed(lambda: W.pair_swiglu(xt, w12, w3t, fid, (), 2)))
    for drop in ("", "FC12", "SILU", "GATE", "FC3", "XREAD", "WRITE", "FC12+FC3", "FC12+FC3+SILU"):
        defs = [("WS_NO_" + d, "1") for d in drop.split("+") if d]
        emit(arm=mode + ("-" + drop.lower() if drop else ""), fidelity=str(fid),
             ms=timed(lambda: W.pair_swiglu(x, w12, w3t, fid, defs)))
