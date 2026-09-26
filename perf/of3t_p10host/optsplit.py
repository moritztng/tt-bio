#!/usr/bin/env python3
"""What the 5.055 s of `AdamW.step()` is made of, at OF3T's own census. One card visit.

`of3t-PLAN.md` prices the optimizer at 5.055 s of a 39.886 s step over 3,152 parameters and
381,302,188 elements, and calls it "host". Host is not a mechanism. The step makes FIVE full
traversals of that census and only two of them are the update:

    grad_norm()        to_host(t.grad)   PCIe read  1   the global clip norm
    the loop           to_host(t.grad)   PCIe read  2   the SAME gradient, read again
    the loop           to_host(t.value)  PCIe read  3   `dev_before`, for the per-step report
    the loop           to_device(theta)  PCIe write 1   the update
    the loop           to_host(t.value)  PCIe read  4   `dev_after`, for the per-step report

This times a crossing of each direction at the census, and the numpy update beside it, so the
split is measured instead of assumed. Shapes come from the live model when `--shapes` names a
census file, otherwise from a log-normal draw pinned to the same count and total.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

N_PARAMS, N_ELEM = 3152, 381_302_188


def census(n_params=N_PARAMS, n_elem=N_ELEM, seed=0):
    """A shape list with OF3T's count and total. Heavy-tailed, as a transformer's is."""
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0.0, 1.6, n_params)
    w = np.maximum(np.round(w / w.sum() * n_elem).astype(np.int64), 32)
    w[-1] += n_elem - int(w.sum())
    return [_shape(int(x)) for x in w]


def _shape(n):
    """A 2D shape for `n` elements, near-square and tile-friendly.

    A (1, n) row is NOT the shape a weight has, and under TILE_LAYOUT it is not even the
    same amount of data: a 32x32 tile pads a single row to 32, so a row-vector census
    measures 32x the bytes the real census crosses. Measured before the fix: a 381.3 M
    element census read in 6.3 s and wrote in 27.1 s, against a real optimizer step of
    5.055 s that crosses it five times.
    """
    r = max(32, int(round(n ** 0.5 / 32)) * 32)
    c = max(32, int(round(n / r / 32)) * 32)
    return (r, c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="0")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO / "perf/of3t_p10host/out/optsplit.json")
    a = ap.parse_args()

    import ttnn, torch
    from tt_bio.train.tensors import to_host, to_device

    sizes = census()
    nel = [r * c for r, c in sizes]
    print(f"census: {len(sizes)} tensors, {sum(nel):,} elements "
          f"(max {max(nel):,}, median {int(np.median(nel)):,})", flush=True)

    dev = ttnn.open_device(device_id=0)
    rows = []
    try:
        host = [np.ascontiguousarray(np.zeros(s, np.float32)) for s in sizes]
        t0 = time.perf_counter()
        devt = [to_device(h, dev) for h in host]
        ttnn.synchronize_device(dev)
        upload_cold = time.perf_counter() - t0
        print(f"upload (cold, {len(devt)} tensors): {upload_cold:.3f}s", flush=True)

        for rep in range(a.reps):
            r = {"rep": rep}
            t0 = time.perf_counter()
            got = [to_host(t) for t in devt]
            r["read_all_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            for i, h in enumerate(host):
                devt[i] = to_device(h, dev)
            ttnn.synchronize_device(dev)
            r["write_all_s"] = time.perf_counter() - t0

            # The numpy update, on arrays of the same census, without any crossing.
            m = [np.zeros(n, np.float32) for n in nel]
            v = [np.zeros(n, np.float32) for n in nel]
            theta = [np.zeros(n, np.float32) for n in nel]
            g = [x.ravel() for x in got]
            t0 = time.perf_counter()
            for i in range(len(sizes)):
                mi, vi, gi, th = m[i], v[i], g[i], theta[i]
                mi *= 0.9;  mi += 0.1 * gi
                vi *= 0.999; vi += 0.001 * (gi * gi)
                th -= 3e-4 * ((mi / 0.1) / (np.sqrt(vi / 0.001) + 1e-8) + 0.01 * th)
            r["update_numpy_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            tot = 0.0
            for gi in g:
                tot += float(gi @ gi)
            r["gradnorm_numpy_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            for i in range(len(sizes)):
                b = theta[i].copy()
                float(np.linalg.norm(theta[i] - b))
            r["report_numpy_s"] = time.perf_counter() - t0
            del m, v, theta, got, g

            r["five_traversals_s"] = 2 * r["read_all_s"] + r["write_all_s"] + r["read_all_s"] * 2
            rows.append(r)
            print(f"rep {rep}: read_all {r['read_all_s']:.3f}s  write_all {r['write_all_s']:.3f}s  "
                  f"update_numpy {r['update_numpy_s']:.3f}s  gradnorm_numpy "
                  f"{r['gradnorm_numpy_s']:.3f}s  report_numpy {r['report_numpy_s']:.3f}s",
                  flush=True)
    finally:
        ttnn.close_device(dev)

    steady = rows[-1]
    # What `step()` pays today against what it needs: 4 reads + 1 write against 1 read + 1
    # write, with the numpy update and the two norms in both.
    today = 4 * steady["read_all_s"] + steady["write_all_s"] + steady["update_numpy_s"] \
        + steady["gradnorm_numpy_s"] + steady["report_numpy_s"]
    floor = steady["read_all_s"] + steady["write_all_s"] + steady["update_numpy_s"] \
        + steady["gradnorm_numpy_s"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"census": {"tensors": len(sizes), "elements": sum(sizes)},
         "upload_cold_s": upload_cold, "reps": rows, "steady": steady,
         "modelled_today_s": today, "modelled_floor_s": floor,
         "modelled_ratio": today / floor}, indent=2))
    print(f"\nmodelled step today {today:.3f}s -> floor {floor:.3f}s  = {today/floor:.3f}x")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
