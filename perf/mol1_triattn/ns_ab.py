# triatt_ln at Mol-1's served shape (HiFi2, head pitch, H bias channels): NOCs as served vs triatt_ln.NOCSWAP.
# Plain and transposed read, whole call and writes only (ABL 11). Outputs at ABL 0 must be torch.equal across arms.
# ns_ab.py --cells 1024:320,512:320,1024:448 --reps 7 --rounds 2
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "perf")]
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import triatt_ln as TL
import clocksample
ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="1024:320,512:320,1024:448")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--abl", type=lambda v: [int(x) for x in v.split(",")], default=[0, 11],
                help="ABL masks to time (16: writes to sequential q pages, triatt_ln ABL_WSEQ)")
ap.add_argument("--arms", default="0,1", help="NOCSWAP arms")
a = ap.parse_args()
dev = T.get_device()
grid = tuple(T.COMPUTE_GRID_MAIN)
for cell in a.cells.split(","):
    n, c = map(int, cell.split(":")); H = c // 32
    g = torch.Generator().manual_seed(0)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    x = up(torch.randn(n, n, c, generator=g))
    w = up(torch.randn(c, 4 * c + 32, generator=g) / c ** 0.5)
    lw, lb = up(torch.ones(c)), up(torch.zeros(c))
    fid = ttnn.MathFidelity.HiFi2

    def run(tr):
        with T.lever_scope(["triatt_ln"]):
            r = TL.ln_qkvgb_heads(x, lw, lb, w, (fid, False, True, False), H, 32, H, grid, transposed=tr, pitch=True)
        assert r is not None, TL.REJECTS
        return r

    def flat(r):
        out = []
        for t in r:
            for u in (t if isinstance(t, (list, tuple)) else [t]):
                out.append(ttnn.to_torch(u)); u.deallocate()
        return out

    for tr in (False, True):
        TL.ABL = 0
        got = {}
        for arm in (False, True):
            TL.NOCSWAP = arm
            got[arm] = flat(run(tr))
        rec = {"n": n, "c": c, "tr": tr, "equal": all(torch.equal(p, q) for p, q in zip(got[False], got[True]))}
        print(json.dumps(rec), flush=True)
        for abl in a.abl:
            TL.ABL = abl
            for arm in [x == "1" for x in a.arms.split(",")] * a.rounds:
                TL.NOCSWAP = arm
                flat(run(tr)); ttnn.synchronize_device(dev)
                ts = []
                with clocksample.during(period=0.5) as clk:
                    for _ in range(a.reps):
                        t0 = time.perf_counter(); r = run(tr); ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t0)
                        for t in r:
                            for u in (t if isinstance(t, (list, tuple)) else [t]):
                                u.deallocate()
                print(json.dumps({"n": n, "c": c, "tr": tr, "abl": abl, "nocswap": arm,
                                  "ms": round(sorted(ts)[len(ts) // 2] * 1e3, 3), "aiclk": clk.summary()}), flush=True)
        TL.ABL = 0
    for t in (x, w, lw, lb):
        t.deallocate()
