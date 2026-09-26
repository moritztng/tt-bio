"""Does AdamW's host arithmetic scale with threads? Card-free, on the real shape census.

The A/B says the arena bought 0.081 s of 2.369 s, so the arithmetic is not paying for its
temporaries -- it is paying for the passes over memory. About seventeen of them over 1.525 GB
per step is ~26 GB of host traffic in 2.3 s, i.e. 11 GB/s, well under what a modern desktop
memory controller does. That gap is what a thread pool would collect, IF numpy releases the GIL
on these shapes, and the parameters are independent once the global clip coefficient is known.

Measured here before anything in `optim.py` is restructured, because a threaded update loop has
to hoist the device writes out of the loop and that is not a change worth making on a guess.

    arith_threads.py --shapes <json> --out <json>       # --shapes is written if absent
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

LR, B1, B2, EPS, WD = 3e-4, 0.9, 0.999, 1e-8, 0.01


def capture_shapes(tokens, path):
    """The shipped parameter census, once, so this bench never needs a card again."""
    from perf.of3t_perf import step as S
    from perf.of3t_stepfloor.fullstep import declare_all
    from tt_bio.train.tensors import to_host
    out = {}
    held, _ = S.capture(tokens, out)
    params = declare_all(held["trunk"][0], held["sampler"][0], out)
    shapes = {n: list(to_host(t.value).shape) for n, t in params.items()}
    path.write_text(json.dumps({"tokens": tokens, "shapes": shapes}))
    return shapes


class Arena:
    """One float32 buffer per slot, per THREAD. Threads cannot share scratch."""

    def __init__(self):
        self._b = {}

    def f32(self, slot, shape):
        n = int(np.prod(shape))
        b = self._b.get(slot)
        if b is None or b.size < n:
            b = self._b[slot] = np.empty(n, np.float32)
        return b[:n].reshape(shape)


def update(name, master, m, v, grads, bc1, bc2, arena):
    """`AdamW.step`'s per-parameter arithmetic, exactly, minus the device halves."""
    theta = master[name]
    shape = theta.shape
    g = grads[name]
    b1 = arena.f32("b1", shape)
    b2 = arena.f32("b2", shape)
    mm, vv = m[name], v[name]
    np.multiply(mm, B1, out=mm)
    np.multiply(g, 1.0 - B1, out=b1)
    np.add(mm, b1, out=mm)
    np.multiply(vv, B2, out=vv)
    np.multiply(g, g, out=b1)
    np.multiply(b1, 1.0 - B2, out=b1)
    np.add(vv, b1, out=vv)
    np.divide(vv, bc2, out=b1)
    np.sqrt(b1, out=b1)
    np.add(b1, EPS, out=b1)
    np.divide(mm, bc1, out=b2)
    np.divide(b2, b1, out=b2)
    np.multiply(theta, WD, out=b1)
    np.add(b2, b1, out=b2)
    upd = np.multiply(b2, LR, out=b2)
    want = float(np.linalg.norm(upd))
    np.subtract(theta, upd, out=theta)
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--shapes", type=Path,
                    default=REPO / "perf/of3t_p10optim/out/shapes_384.json")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", default="1,2,4,8,16")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if a.shapes.is_file():
        shapes = {n: tuple(s) for n, s in json.loads(a.shapes.read_text())["shapes"].items()}
    else:
        shapes = {n: tuple(s) for n, s in capture_shapes(a.tokens, a.shapes).items()}

    names = list(shapes)
    rng = np.random.default_rng(20260926)
    master = {n: rng.standard_normal(s).astype(np.float32) for n, s in shapes.items()}
    m = {n: np.zeros(s, np.float32) for n, s in shapes.items()}
    v = {n: np.zeros(s, np.float32) for n, s in shapes.items()}
    grads = {n: (rng.standard_normal(s) * 1e-3).astype(np.float32) for n, s in shapes.items()}

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(), "ncpu": os.cpu_count(),
        "loadavg_start": [round(x, 2) for x in os.getloadavg()],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "census": {"params": len(names),
                   "elements": int(sum(int(np.prod(s)) for s in shapes.values()))},
        "arms": {}}

    bc1, bc2 = 1.0 - B1, 1.0 - B2
    for w in [int(x) for x in a.workers.split(",")]:
        arenas = [Arena() for _ in range(max(w, 1))]
        times, loads = [], []
        pool = None if w == 1 else ThreadPoolExecutor(max_workers=w)
        try:
            for rep in range(a.reps):
                loads.append(round(os.getloadavg()[0], 2))
                t0 = time.perf_counter()
                if pool is None:
                    for n in names:
                        update(n, master, m, v, grads, bc1, bc2, arenas[0])
                else:
                    chunks = [names[i::w] for i in range(w)]

                    def run(idx):
                        ar = arenas[idx]
                        for n in chunks[idx]:
                            update(n, master, m, v, grads, bc1, bc2, ar)

                    list(pool.map(run, range(w)))
                times.append(round(time.perf_counter() - t0, 4))
        finally:
            if pool is not None:
                pool.shutdown()
        out["arms"][str(w)] = {"times_s": times, "median_s": round(statistics.median(times), 4),
                               "loadavg": loads}
        print(f"workers {w:3d}  median {out['arms'][str(w)]['median_s']:6.3f}s  "
              f"{times}  load {loads[0]}/{os.cpu_count()}", flush=True)
        a.out.write_text(json.dumps(out, indent=1))

    base = out["arms"]["1"]["median_s"]
    out["speedup_vs_serial"] = {k: round(base / vv["median_s"], 3)
                                for k, vv in out["arms"].items()}
    out["env"]["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["speedup_vs_serial"], indent=1))
    print(f"WROTE {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
