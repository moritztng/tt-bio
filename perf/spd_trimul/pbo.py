"""Pair-bias fusion, step 1: RFD4's pair transition (pair_relu, LayerNorm prologue, residual in place into the bf16 z,
bfp8 weights, HiFi2, c_z 384, hidden 768) in pair-bias order (`pbo`: whole (b, ib, jb) groups of 32 row tiles a core,
i-first) against the default order, and the standalone `rms_pair_bias` call the fused epilogue would remove.

Per N one JSON line: base_ms, pbo_ms (the order's own cost), pbo_equal (must be true: only the order moves),
rms_ms (dit_pair_bias.rms_pair_bias on the result, n_out 2), aiclk sampled during.

usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/spd_trimul/pbo.py --ns 352,256 --out F.jsonl
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="352,256")
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--c", type=int, default=384)
ap.add_argument("--hidden", type=int, default=768)
ap.add_argument("--heads", type=int, default=16)
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, pair_swiglu_ws as WS, dit_pair_bias as PB
import clocksample

dev = T.get_device()
EPS, C, H = 1e-5, A.c, A.hidden
FID = ttnn.MathFidelity.HiFi2
torch.manual_seed(0)
up = lambda t, d=ttnn.bfloat16: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
W = WS.relu_weights(torch.randn(C, H) / C ** .5, torch.randn(H) * .1, torch.randn(H, C) / H ** .5,
                    torch.randn(C) * .1, dev, dtype=ttnn.bfloat8_b)
g, b = up((1 + .1 * torch.randn(C)).reshape(1, C)), up((.1 * torch.randn(C)).reshape(1, C))
wt = PB.rms_weight([torch.randn(C, A.heads) / C ** .5 for _ in range(2)], dev)


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
    rec = {"n": n, "c": C, "hidden": H, "heads": A.heads}
    clk = clocksample.during(period=0.5).__enter__()
    outs = {}
    for name, pbo in (("base", False), ("pbo", True)):
        try:
            z = up(z0)
            WS.pair_relu(z, *W, fidelity=FID, ln=(g, b, EPS), resid=z, pbo=pbo)
            outs[name] = ttnn.to_torch(z)
            rec[f"{name}_ms"] = timed(lambda: WS.pair_relu(z, *W, fidelity=FID, ln=(g, b, EPS), resid=z, pbo=pbo),
                                      A.reps)
            if name == "base":
                def rms():
                    for t in PB.rms_pair_bias(z, wt, EPS, A.heads, n_out=2, fidelity=FID):
                        ttnn.deallocate(t)
                rec["rms_ms"] = timed(rms, A.reps)
            ttnn.deallocate(z)
        except Exception as e:  # noqa: BLE001
            rec[f"{name}_error"] = repr(e)[:400]
    clk.__exit__(None, None, None)
    if len(outs) == 2:
        rec["pbo_equal"] = bool(torch.equal(outs["base"], outs["pbo"]))
    rec["aiclk"] = clk.summary()
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(line + "\n")
