"""RFD4's pair biases for one two-block group: the trunk's chain (rms_norm, linear, permute, slices) against one
`rms_pair_bias` program. ms per call and error vs float64.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/pairbias/bias.py --weights ~/rfd4-bundle --out <json> [--n 128 256 512]

Weights: blocks 1 and 2's folded bias projections (`PairBiasAttention.bias_w_host`), 12 heads each. Input: a random bf16
pair stream. Error is relative RMS over the whole [32, n, n] output against float64 rms0(z) @ W of the same bf16
input and bf16 weights. Time is the median over `--reps` of `--inner` back-to-back calls between syncs; AICLK and
host load sampled during the run.
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
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--inner", type=int, default=10)
a = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # perf/: clocksample

import torch  # noqa: E402

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
import ttnn  # noqa: E402

from tt_bio import dit_pair_bias as D  # noqa: E402
from tt_bio.rfd4 import trunk as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402


def bias_w(weights):
    from safetensors import safe_open
    f = safe_open(str(Path(weights) / "model.safetensors"), "pt")
    ws = []
    for i in (1, 2):
        p = f"nn.transformer_layers.{i}.mhba.mha."
        tb, pn = f.get_tensor(p + "to_bias.weight").double(), f.get_tensor(p + "pair_norm.weight").double()
        hd = f.get_tensor(p + "to_qkv.weight").shape[0] // 3 // tb.shape[0]
        ws.append(tb.t() * pn[:, None] * hd ** 0.5)
    return torch.cat(ws, dim=1)   # [C, 2 * heads]


def timed(fn, dev, reps, inner):
    free = lambda o: [ttnn.deallocate(t) for t in (o if isinstance(o, list) else [o])]  # noqa: E731
    out = fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        free(out)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for i in range(inner):
            out = fn()
            if i < inner - 1:
                free(out)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / inner)
    ts.sort()
    return out, 1e3 * ts[len(ts) // 2], 1e3 * ts[0]


def main():
    dev = get_device()
    w = bias_w(a.weights).to(torch.bfloat16)
    ckc = T.compute_kernel_config()
    w_tt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    h = w.shape[1] // 2
    wt_tt = D.rms_weight([w[:, :h], w[:, h:]], dev)
    rows = []
    with clocksample.during() as clk:
        for n in a.n:
            gen = torch.Generator().manual_seed(n)
            z = (torch.randn(1, n, n, 384, generator=gen) * 1.5 + 0.2).to(torch.bfloat16)
            zd = z.double()
            ref = ((zd * (zd.pow(2).mean(-1, keepdim=True) + T.EPS).rsqrt()) @ w.double()).permute(0, 3, 1, 2)
            zb = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            arms = {"chain": lambda: T.pair_biases(zb, w_tt, None, ckc, ttnn.bfloat16)}
            if D.rms_eligible(zb, wt_tt):
                arms["fused"] = lambda: D.rms_pair_bias(zb, wt_tt, T.EPS, h, n_out=2, fidelity=ckc.math_fidelity)
            for arm, fn in arms.items():
                out, ms, ms_min = timed(fn, dev, a.reps, a.inner)
                outs = out if isinstance(out, list) else [out]
                got = torch.cat([ttnn.to_torch(o).double() for o in outs], dim=1)
                for o in outs:
                    ttnn.deallocate(o)
                err = float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
                r = dict(n=n, arm=arm, ms=round(ms, 4), ms_min=round(ms_min, 4), rel_rms_vs_f64=err,
                         max_abs=float((got - ref).abs().max()), t=time.strftime("%H:%M:%S"))
                rows.append(r)
                print(json.dumps(r), flush=True)
            ttnn.deallocate(zb)
    out = dict(rows=rows, aiclk=clk.summary(), load=clk.load and max(clk.load),
               script="perf/rfd4/pairbias/bias.py")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("aiclk", clk.summary(), "load", out["load"])


if __name__ == "__main__":
    main()
