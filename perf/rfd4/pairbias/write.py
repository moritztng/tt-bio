"""RFD4's pair write into the float32 pair stream: the trunk's two ops (Hadamard broadcast multiply, then
`residual_dual`) against one `outer_residual` program. ms per call and error vs float64.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/pairbias/write.py --out <json> [--n 128 256 512]

Inputs: random float32 z [1, n, n, 384] and bf16 factors pi, pj [1, n, 384] at the trunk's scales. Error is relative
RMS of the float32 result's update (z' - z) against float64 pj[i] * pi[j], on a 16384-row sample; the bf16 copy is
checked against the float32 result rounded to even (`bf16_copy_mismatch`, a count). Time is the median over `--reps`
of `--inner` back-to-back calls between syncs; AICLK and host load sampled during the run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
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

from tt_bio import outer_residual as O  # noqa: E402
from tt_bio import rne_add  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

ROWS = 16384


def chain(z, pi, pj):
    b, n, c = (int(d) for d in pi.shape)
    u = ttnn.multiply(ttnn.reshape(pj, (b, n, 1, c)), ttnn.reshape(pi, (b, 1, n, c)))
    out = rne_add.residual_dual(z, u)
    ttnn.deallocate(u)
    return list(out)


def timed(fn, dev, reps, inner):
    free = lambda o: [ttnn.deallocate(t) for t in o]  # noqa: E731
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


def rne(x):
    v = x.float().contiguous().view(torch.int32)
    v = (v + 0x7FFF + ((v >> 16) & 1)) & -65536
    return v.view(torch.float32)


def main():
    dev = get_device()
    rows = []
    up = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
    with clocksample.during() as clk:
        for n in a.n:
            gen = torch.Generator().manual_seed(n)
            z = torch.randn(1, n, n, 384, generator=gen) * 2.0
            pi = (torch.randn(1, n, 384, generator=gen) * 0.7).to(torch.bfloat16)
            pj = (torch.randn(1, n, 384, generator=gen) * 0.7).to(torch.bfloat16)
            upd = (pj.double()[:, :, None, :] * pi.double()[:, None, :, :]).reshape(-1, 384)[:ROWS]
            z_t, pi_t, pj_t = up(z, ttnn.float32), up(pi, ttnn.bfloat16), up(pj, ttnn.bfloat16)
            pj4_t = ttnn.reshape(pj_t, (1, n, 1, 384))
            arms = {"chain": lambda: chain(z_t, pi_t, pj_t),
                    "fused": lambda: list(O.outer_residual(z_t, pi_t, pj4_t))}
            for arm, fn in arms.items():
                out, ms, ms_min = timed(fn, dev, a.reps, a.inner)
                z32, zb = (ttnn.to_torch(t) for t in out)
                for t in out:
                    ttnn.deallocate(t)
                got = (z32.double() - z.double()).reshape(-1, 384)[:ROWS]
                err = float(((got - upd).pow(2).mean() / upd.pow(2).mean()).sqrt())
                mism = int((rne(z32).reshape(-1)[:ROWS * 384] != zb.float().reshape(-1)[:ROWS * 384]).sum())
                r = dict(n=n, arm=arm, ms=round(ms, 4), ms_min=round(ms_min, 4), update_rel_rms_vs_f64=err,
                         bf16_copy_mismatch=mism, t=time.strftime("%H:%M:%S"))
                rows.append(r)
                print(json.dumps(r), flush=True)
            for t in (z_t, pi_t, pj_t):
                ttnn.deallocate(t)
    out = dict(rows=rows, aiclk=clk.summary(), load=clk.load and max(clk.load), script="perf/rfd4/pairbias/write.py")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("aiclk", clk.summary(), "load", out["load"])


if __name__ == "__main__":
    main()
