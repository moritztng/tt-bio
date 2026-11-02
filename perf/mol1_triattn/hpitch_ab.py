"""triatt_ln with a head pitch of MT + 1 (triatt_ln.HPITCH) against pitch MT: q, k, v, g sliced to S torch.equal, ms/call.

    TT_VISIBLE_DEVICES=<card> python perf/mol1_triattn/hpitch_ab.py [--n 1024,992,512] [--c 320] [--reps 7]

The plain-read kernel at Mol-1's widths (HiFi2, bf16 outputs). On/off interleaved per rep, median. One JSON line a cell.
"""
import argparse, json, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", default="1024,992,512")
ap.add_argument("--c", default="320")
ap.add_argument("--reps", type=int, default=7)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import triatt_ln as TL  # noqa: E402

dev = T.get_device()
grid = tuple(T.COMPUTE_GRID_MAIN)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731


def flat(r):
    (q, k, v), g, b = r
    return [q, k, v, g, b]


for c in (int(x) for x in a.c.split(",")):
    for n in (int(x) for x in a.n.split(",")):
        gen = torch.Generator().manual_seed(0)
        x = up(torch.randn(n, n, c, generator=gen))
        w = up(torch.randn(c, 4 * c + 32, generator=gen) / c ** 0.5)
        lw, lb = up(torch.ones(c)), up(torch.zeros(c))

        def run(wp):
            TL.HPITCH = wp
            with T.lever_scope(["triatt_ln"]):
                r = TL.ln_qkvgb_heads(x, lw, lb, w, (ttnn.MathFidelity.HiFi2, False, True, False), c // 32, 32, 8,
                                      grid)
            assert r is not None, TL.REJECTS
            return flat(r)

        outs = {}
        for wp in (False, True):
            o = run(wp)
            outs[wp] = [ttnn.to_torch(t)[..., :n, :] if i < 4 else ttnn.to_torch(t) for i, t in enumerate(o)]
            for t in o:
                ttnn.deallocate(t)
        equal = all(torch.equal(p, q) for p, q in zip(outs[False], outs[True]))
        ms = {False: [], True: []}
        for _ in range(3 + a.reps):
            for wp in (False, True):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                o = run(wp)
                ttnn.synchronize_device(dev)
                ms[wp].append((time.perf_counter() - t0) * 1e3)
                for t in o:
                    ttnn.deallocate(t)
        off, on = (statistics.median(ms[k][3:]) for k in (False, True))
        print(json.dumps(dict(n=n, c=c, MT=-(-n // 32), off_ms=round(off, 3), on_ms=round(on, 3),
                              speedup=round(off / on, 3), equal=equal)), flush=True)
        for t in (x, w, lw, lb):
            ttnn.deallocate(t)
