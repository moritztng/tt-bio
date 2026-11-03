# triatt_ln at Mol-1's served shape (HiFi2, head pitch, H bias channels), plain read: served vs WROT (weight reads
# from a per-core offset, exact) vs ABL 64 (weight reads skipped, timing only). Outputs of WROT must be torch.equal.
# wr_ab.py --cells 256:320,512:320,1024:320,256:448 --reps 9 --rounds 3
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "perf")]
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import triatt_ln as TL
import clocksample
ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="256:320,512:320,1024:320,256:448")
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--rounds", type=int, default=3)
a = ap.parse_args()
dev = T.get_device()
grid = tuple(T.COMPUTE_GRID_MAIN)


def free(r):
    for t in r:
        for u in (t if isinstance(t, (list, tuple)) else [t]):
            u.deallocate()


for cell in a.cells.split(","):
    n, c = map(int, cell.split(":")); H = c // 32
    g = torch.Generator().manual_seed(0)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    x = up(torch.randn(n, n, c, generator=g))
    w = up(torch.randn(c, 4 * c + 32, generator=g) / c ** 0.5)
    lw, lb = up(torch.ones(c)), up(torch.zeros(c))

    def run():
        with T.lever_scope(["triatt_ln"]):
            r = TL.ln_qkvgb_heads(x, lw, lb, w, (ttnn.MathFidelity.HiFi2, False, True, False), H, 32, H, grid, pitch=True)
        assert r is not None, TL.REJECTS
        return r

    def grab():
        r = run(); out = []
        for t in r:
            for u in (t if isinstance(t, (list, tuple)) else [t]):
                out.append(ttnn.to_torch(u)); u.deallocate()
        return out

    TL.ABL, TL.WROT = 0, False
    ref = grab()
    TL.WROT = True
    rot = grab()
    print(json.dumps({"n": n, "c": c, "wrot_equal": all(torch.equal(p, q) for p, q in zip(ref, rot))}), flush=True)
    arms = {"served": (0, False), "wrot": (0, True), "noweight": (64, False)}
    for name in list(arms) * a.rounds:
        TL.ABL, TL.WROT = arms[name]
        free(run()); ttnn.synchronize_device(dev)
        ts = []
        with clocksample.during(period=0.5) as clk:
            for _ in range(a.reps):
                t0 = time.perf_counter(); r = run(); ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t0)
                free(r)
        print(json.dumps({"n": n, "c": c, "arm": name, "ms": round(sorted(ts)[len(ts) // 2] * 1e3, 3),
                          "aiclk": clk.summary()}), flush=True)
    TL.ABL, TL.WROT = 0, False
    for t in (x, w, lw, lb):
        t.deallocate()
