"""outer_residual and rms_pair_bias at RFD4 trunk shapes (bfp8 z, `pair_bfp8z`): ms a call per leg, a digest of the
output so knob legs that keep every byte can be checked bit-identical, AICLK sampled during.

    TT_VISIBLE_DEVICES=<chip> python perf/rfd4/b512/dm.py --ns 512,384 --legs "or:base;or:gran=8;or:dm=1;pb:base" --out <jsonl>

A leg is `<prog>:<k=v,...>`: or = outer_residual (gran=GRAN, dm=DM bits 1 z reads, 2 pj4 rows, 4 writes), pb =
pjt=1 pj as [B, n, C] tiles, pjt=0 the same input reshaped to pj4 inside), pb =
rms_pair_bias (dm=RMS_DM bits 1 z reads, 2 writes). DM legs skip work, so their digests differ by design. Legs run
in the given order, the whole list `--rounds` times alternated; the time is the median of `--reps` x `--inner` calls.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="512,384")
ap.add_argument("--legs", required=True)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--inner", type=int, default=10)
ap.add_argument("--c", type=int, default=384)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # perf/: clocksample

import torch  # noqa: E402

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
import ttnn  # noqa: E402

from tt_bio import dit_pair_bias as P  # noqa: E402
from tt_bio import outer_residual as O  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402


def digest(t):
    return hashlib.sha256(ttnn.to_torch(t).contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def main():
    dev = get_device()
    up = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
    c = a.c
    inputs = {}
    for n in (int(v) for v in a.ns.split(",")):
        gen = torch.Generator().manual_seed(n + c)
        z = up(torch.randn(1, n, n, c, generator=gen), ttnn.bfloat8_b)
        pi = up((torch.randn(1, n, c, generator=gen) * 0.7).to(torch.bfloat16), ttnn.bfloat16)
        pj = up((torch.randn(1, n, c, generator=gen) * 0.7).to(torch.bfloat16), ttnn.bfloat16)
        pj4 = ttnn.reshape(pj, (1, n, 1, c))
        wt = P.rms_weight([torch.randn(c, 12) * 0.05, torch.randn(c, 12) * 0.05], dev)
        inputs[n] = (z, pi, pj4, wt, pj)
    legs = [l.strip() for l in a.legs.split(";") if l.strip()]
    with open(a.out, "a") as f:
        for rnd in range(a.rounds):
            for n, (z, pi, pj4, wt, pj) in inputs.items():
                for leg in legs:
                    prog, _, kv = leg.partition(":")
                    knobs = dict(x.split("=") for x in kv.split(",") if "=" in x)
                    O.GRAN, O.DM, P.RMS_DM = int(knobs.get("gran", 4)), 0, 0
                    if prog == "or":
                        O.DM = int(knobs.get("dm", 0))
                        O.PJ_TILES = knobs.get("pjt", "1") == "1"
                        pjx = pj if "pjt" in knobs else pj4   # pjt=0: [B, n, C] reshaped inside, as the trunk on main
                        call = lambda: O.outer_residual(z, pi, pjx)[:1]  # noqa: E731
                    else:
                        P.RMS_DM = int(knobs.get("dm", 0))
                        call = lambda: P.rms_pair_bias(z, wt, 1e-5, 12, n_out=2)  # noqa: E731
                    with clocksample.during() as clk:
                        out = call()
                        ttnn.synchronize_device(dev)
                        dg = digest(out[0])
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
                            ts.append((time.perf_counter() - t0) / a.inner * 1e3)
                    rec = {"round": rnd, "n": n, "leg": leg, "ms": statistics.median(ts), "ms_min": min(ts),
                           "digest": dg, "aiclk": clk.summary()}
                    print(json.dumps(rec), flush=True)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()


main()
