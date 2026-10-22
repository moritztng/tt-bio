"""`outer_residual` alone: ms per call, a digest of both outputs, and the checks against float64, for an A/B of two
trees run as separate processes (the digests must match for an exact lever).

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/ptrans/orz.py --out <json> [--shapes 1x512x384 1x256x384 ...]

Inputs per shape BxNxC: seeded random float32 z [B, N, N, C] and bf16 pi, pj [B, N, C] at the trunk's scales.
`update_rel_rms_vs_f64` is the float32 result's update (z' - z) against float64 pj[i] * pi[j] on the first 16384
rows; `bf16_copy_mismatch` counts bf16 values that differ from the float32 result rounded to even. Time is the median
over `--reps` of `--inner` back-to-back calls between syncs; AICLK and host load sampled during the run.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--shapes", nargs="+", default=["1x512x384", "1x256x384", "1x128x384", "2x96x384", "1x64x64"])
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
from tt_bio.tenstorrent import get_device  # noqa: E402

ROWS = 16384


def rne(x):
    v = x.float().contiguous().view(torch.int32)
    v = (v + 0x7FFF + ((v >> 16) & 1)) & -65536
    return v.view(torch.float32)


def main():
    dev = get_device()
    up = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rows = []
    with clocksample.during() as clk:
        for shape in a.shapes:
            b, n, c = (int(v) for v in shape.split("x"))
            gen = torch.Generator().manual_seed(1000 * b + n + c)
            z = torch.randn(b, n, n, c, generator=gen) * 2.0
            pi = (torch.randn(b, n, c, generator=gen) * 0.7).to(torch.bfloat16)
            pj = (torch.randn(b, n, c, generator=gen) * 0.7).to(torch.bfloat16)
            z_t, pi_t = up(z, ttnn.float32), up(pi, ttnn.bfloat16)
            pj4_t = ttnn.reshape(up(pj, ttnn.bfloat16), (b, n, 1, c))
            call = lambda: O.outer_residual(z_t, pi_t, pj4_t)  # noqa: E731
            out = call()
            ttnn.synchronize_device(dev)
            z32, zb = (ttnn.to_torch(t) for t in out)
            for t in out:
                ttnn.deallocate(t)
            ts = []
            for _ in range(a.reps):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(a.inner):
                    for t in call():
                        ttnn.deallocate(t)
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) / a.inner)
            ts.sort()
            upd = (pj.double()[:, :, None, :] * pi.double()[:, None, :, :]).reshape(-1, c)[:ROWS]
            got = (z32.double() - z.double()).reshape(-1, c)[:ROWS]
            err = float(((got - upd).pow(2).mean() / upd.pow(2).mean()).sqrt())
            mism = int((rne(z32) != zb.float()).sum())
            digest = hashlib.sha256(z32.contiguous().numpy().tobytes() + zb.view(torch.int16).numpy().tobytes())
            r = dict(shape=shape, ms=round(1e3 * ts[len(ts) // 2], 4), ms_min=round(1e3 * ts[0], 4),
                     gb=round(b * n * n * c * 10 / 1e9, 4), update_rel_rms_vs_f64=err, bf16_copy_mismatch=mism,
                     digest=digest.hexdigest()[:16], t=time.strftime("%H:%M:%S"))
            rows.append(r)
            print(json.dumps(r), flush=True)
            for t in (z_t, pi_t, pj4_t):
                ttnn.deallocate(t)
    out = dict(rows=rows, aiclk=clk.summary(), load=clk.load and max(clk.load), script="perf/rfd4/ptrans/orz.py")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("aiclk", clk.summary(), "load", out["load"])


if __name__ == "__main__":
    main()
