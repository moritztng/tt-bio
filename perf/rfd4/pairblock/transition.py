"""RFD4's pair transition, the trunk's composed ops against one `pair_relu` program: ms per call and error vs float64.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/pairblock/transition.py --weights ~/rfd4-bundle --out <json> \
        [--n 128 256 512] [--arms normal fast fused_hifi4 fused_hifi2 fused_lofi fused_hifi2lofi] [--tree <tt-bio checkout>]

`normal` / `fast` run `PairUpdate._transition` (LayerNorm, fc1 + relu, fc2, the pair-mask multiply) built and called
under the trunk's NORMAL_LEVERS / FAST_LEVERS of the tree on the path (`--tree`: a checkout whose tt_bio is imported
instead of this one's, for a lever set this branch does not carry). `fused_*` run `pair_swiglu_ws.pair_relu` with the
LayerNorm prologue and bfp8 weights at that fidelity (`fused_hifi2lofi`: fc1 HiFi2, fc2 LoFi); with every token real the pair mask is all ones and is skipped.
Input: a bf16 pair stream with layer 0's checkpoint weights. Error is relative RMS on a 16384-row sample against the
float64 transition of the same bf16 input. Time is the median over `--reps` of `--inner` back-to-back calls between
syncs. AICLK and host load are sampled during the run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, nargs="+", default=[128, 256, 512])
ap.add_argument("--arms", nargs="+", default=["normal", "fused_hifi4", "fused_hifi2"])
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--inner", type=int, default=10)
ap.add_argument("--tree", default=str(Path(__file__).resolve().parents[3]))
a = ap.parse_args()
sys.path.insert(0, a.tree)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # perf/: clocksample

import torch  # noqa: E402

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
import ttnn  # noqa: E402

from tt_bio.rfd4 import trunk as T  # noqa: E402
from tt_bio.tenstorrent import WeightScope, get_device, lever_scope  # noqa: E402

ROWS = 16384
PREFIX = "nn.pair_update_layers.0."


def layer0(weights):
    from safetensors import safe_open
    f = safe_open(str(Path(weights) / "model.safetensors"), "pt")
    return {k[len(PREFIX):]: f.get_tensor(k) for k in f.keys() if k.startswith(PREFIX)}


def reference(x, w):
    g = lambda k: w["transition_out." + k].double()  # noqa: E731
    h = torch.nn.functional.layer_norm(x.double(), (x.shape[-1],), g("layer_norm.weight"), g("layer_norm.bias"), T.EPS)
    h = torch.relu(h @ g("linear_1.weight").t() + g("linear_1.bias"))
    return h @ g("linear_2.weight").t() + g("linear_2.bias")


def timed(fn, dev, reps, inner):
    out = fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.deallocate(out)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for i in range(inner):
            out = fn()
            if i < inner - 1:
                ttnn.deallocate(out)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / inner)
    ts.sort()
    return out, 1e3 * ts[len(ts) // 2], 1e3 * ts[0]


def main():
    dev = get_device()
    w = layer0(a.weights)
    ckc = T.compute_kernel_config()
    rows = []
    with clocksample.during() as clk:
        for n in a.n:
            gen = torch.Generator().manual_seed(n)
            z = (torch.randn(1, n, n, 384, generator=gen) * 1.5 + 0.2).to(torch.bfloat16)
            ref = reference(z.reshape(-1, 384)[:ROWS], w)
            zb = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            mask_c = ttnn.from_torch(torch.ones(1, n, n, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for arm in a.arms:
                if arm in ("normal", "fast"):
                    levers = T.NORMAL_LEVERS if arm == "normal" else getattr(T, "FAST_LEVERS")
                    with lever_scope(levers):
                        pu = T.PairUpdate(WeightScope(dict(w)), ckc)
                        fn = lambda: pu._transition(zb, mask_c)  # noqa: E731
                        out, ms, ms_min = timed(fn, dev, a.reps, a.inner)
                    extra = {"levers": sorted(levers)}
                else:
                    from tt_bio import pair_swiglu_ws as W
                    fid = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2,
                           "lofi": ttnn.MathFidelity.LoFi, "hifi2lofi": ttnn.MathFidelity.HiFi2}[arm.split("_")[1]]
                    lofi2 = arm == "fused_hifi2lofi"
                    g = lambda k: w["transition_out." + k]  # noqa: E731
                    w1, b1, w2, b2 = W.relu_weights(g("linear_1.weight").t().float(), g("linear_1.bias").float(),
                                                    g("linear_2.weight").t().float(), g("linear_2.bias").float(),
                                                    dev, dtype=ttnn.bfloat8_b)
                    up = lambda t: ttnn.from_torch(t.reshape(1, -1).float(), dtype=ttnn.bfloat16,  # noqa: E731
                                                   layout=ttnn.TILE_LAYOUT, device=dev)
                    ln = (up(g("layer_norm.weight")), up(g("layer_norm.bias")))
                    if not W.relu_eligible(zb, w1, w2, ln):
                        rows.append(dict(n=n, arm=arm, declined=True))
                        print(json.dumps(rows[-1]), flush=True)
                        continue
                    fn = lambda: W.pair_relu(zb, w1, b1, w2, b2, fidelity=fid, ln=ln + (T.EPS,), fc2_lofi=lofi2)  # noqa: E731
                    out, ms, ms_min = timed(fn, dev, a.reps, a.inner)
                    extra = {"weights": "bfp8", "fidelity": str(fid), "fc2_lofi": lofi2}
                got = ttnn.to_torch(out).reshape(-1, 384)[:ROWS].double()
                ttnn.deallocate(out)
                err = float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
                r = dict(n=n, arm=arm, ms=round(ms, 4), ms_min=round(ms_min, 4), rel_rms_vs_f64=err,
                         digest=float(got.float().sum()), t=time.strftime("%H:%M:%S"), **extra)
                rows.append(r)
                print(json.dumps(r), flush=True)
            ttnn.deallocate(zb)
            ttnn.deallocate(mask_c)
    out = dict(rows=rows, aiclk=clk.summary(), load=clk.load and max(clk.load), tree=a.tree,
               script="perf/rfd4/pairblock/transition.py")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("aiclk", clk.summary(), "load", out["load"])


if __name__ == "__main__":
    main()
