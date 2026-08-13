"""L2b's screen: where does `_calibrate_linear`'s 1.15-7.90 s per shape actually go?

`_TUNE_MATMUL` is opt-in only because calibration costs a fixed one-time price that a
single-batch invocation does not earn back. model.py names the suspected dominant cost:
compiling candidates that then fail L1 validation and get swallowed by
`except Exception: continue`. If that is right, `_mm_candidates` can reject them
arithmetically and the one-time cost collapses. If it is wrong -- if the time is in the
candidates that SUCCEED, i.e. in the timing reps and the two bitwise checks -- then a
cheaper enumeration buys nothing and L2b is a different fix.

So: enumerate the real per-step shapes, run the same loop, and attribute every second to
{failed compile, exactness check, timing reps, setup}. Measures, does not assume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio.rfd3 import model as m  # noqa: E402

# The four real per-step [D,I,I,C] @ [C,N] pair linears (p15 table), same as p33.
CASES = [
    ("z_transition fc1/fc2", 128, 512),
    ("z_transition fc3", 512, 128),
    ("transition_2 fc1/fc2", 128, 256),
    ("pair_bias to_b", 128, 32),
]


def ckc():
    dev = get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def instrumented_calibrate(x, w, kw, core_grid):
    """`_calibrate_linear`'s loop with a stopwatch on each phase."""
    acc = {"setup_s": 0.0, "fail_s": 0.0, "exact_s": 0.0, "time_s": 0.0,
           "n_cand": 0, "n_fail": 0, "n_inexact": 0, "n_ok": 0, "errors": {}}
    t0 = time.perf_counter()
    rx, rw = m._mm_random_like(x, 0), m._mm_random_like(w, 1)
    rref = ttnn.linear(rx, rw, core_grid=core_grid, **kw)
    ref = ttnn.linear(x, w, core_grid=core_grid, **kw)
    default_t = m._mm_time(lambda: ttnn.linear(x, w, core_grid=core_grid, **kw))
    acc["setup_s"] = time.perf_counter() - t0
    acc["default_ms"] = default_t * 1e3
    budget = default_t / m._TUNE_MIN_GAIN
    best = None
    for pc in m._mm_candidates(x, w, get_device().compute_with_storage_grid_size()):
        acc["n_cand"] += 1
        t1 = time.perf_counter()
        try:
            bad = m._mm_maxabs(ttnn.linear(rx, rw, program_config=pc, **kw), rref) != 0.0
            if not bad:
                bad = m._mm_maxabs(ttnn.linear(x, w, program_config=pc, **kw), ref) != 0.0
            acc["exact_s"] += time.perf_counter() - t1
            if bad:
                acc["n_inexact"] += 1
                continue
            t2 = time.perf_counter()
            t = m._mm_time(lambda: ttnn.linear(x, w, program_config=pc, **kw))
            acc["time_s"] += time.perf_counter() - t2
        except Exception as e:                                       # noqa: BLE001
            acc["fail_s"] += time.perf_counter() - t1
            acc["n_fail"] += 1
            k = type(e).__name__
            acc["errors"][k] = acc["errors"].get(k, 0) + 1
            continue
        acc["n_ok"] += 1
        if t < budget:
            best, budget = pc, t
    for t in (rx, rw, rref, ref):
        ttnn.deallocate(t)
    acc["gain"] = default_t / budget if best is not None else 1.0
    return best, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[250])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    kw_base = dict(compute_kernel_config=ckc(), dtype=ttnn.bfloat16)
    rows = []
    for D in a.batches:
        for I in a.tokens:
            for name, K, N in CASES:
                g = torch.Generator().manual_seed(0)
                x = m.ttnn.from_torch(torch.randn(D, I, I, K, generator=g),
                                      layout=ttnn.TILE_LAYOUT, device=get_device(),
                                      dtype=ttnn.bfloat16)
                w = m.ttnn.from_torch(torch.randn(K, N, generator=g), layout=ttnn.TILE_LAYOUT,
                                      device=get_device(), dtype=ttnn.bfloat16)
                t0 = time.perf_counter()
                best, acc = instrumented_calibrate(x, w, dict(kw_base), None)
                acc["total_s"] = time.perf_counter() - t0
                acc.update(case=name, K=K, N=N, D=D, I=I,
                           tunable=m._tunable(x, w), chose=best is not None)
                rows.append(acc)
                print(f"D={D} I={I} {name:22s} tunable={acc['tunable']!s:5s} "
                      f"total={acc['total_s']:6.2f}s  setup={acc['setup_s']:5.2f} "
                      f"fail={acc['fail_s']:5.2f}({acc['n_fail']}/{acc['n_cand']}) "
                      f"exact={acc['exact_s']:5.2f} time={acc['time_s']:5.2f} "
                      f"gain={acc['gain']:.2f}x default={acc['default_ms']:.3f}ms "
                      f"errs={acc['errors']}", flush=True)
                for t in (x, w):
                    ttnn.deallocate(t)

    tot = sum(r["total_s"] for r in rows)
    print(f"\n[calib] {len(rows)} shapes, {tot:.2f} s total")
    for k in ("setup_s", "fail_s", "exact_s", "time_s"):
        s = sum(r[k] for r in rows)
        print(f"[calib]   {k:9s} {s:6.2f} s = {s / tot * 100:5.1f} %")
    print(f"[calib]   candidates {sum(r['n_cand'] for r in rows)}, "
          f"failed {sum(r['n_fail'] for r in rows)}, "
          f"inexact {sum(r['n_inexact'] for r in rows)}, ok {sum(r['n_ok'] for r in rows)}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
