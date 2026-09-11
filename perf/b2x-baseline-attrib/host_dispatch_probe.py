#!/usr/bin/env python3
"""Is the Boltz-2 512 aa fold waiting on the device, or on Python?

Two numbers in this campaign disagree: the published cell splits 0.382 s of host against 23.12 s
of device, while `b2x-fusion-boundary` derived 2.74 s/fold of non-device time from the op count.
They measure different boundaries. The published split is host time OUTSIDE
`model.predict_step` -- featurisation and the CIF write -- and says nothing about what the host
does during the fold. The derived figure is host issue cost INSIDE it.

So measure the inside directly. `time.thread_time()` on the calling thread counts only the CPU
that thread burns, which for a ttnn model is exactly the Python-side op issue: building configs,
crossing the pybind boundary, and enqueueing. Against the fold wall it separates the two cases:

    main-thread CPU ~= wall   -> the host is issuing flat out and the device waits on Python.
                                 Deleting bytes cannot help; deleting OPS is the lever.
    main-thread CPU << wall   -> the host is ahead of the device and the fold is device-bound.
                                 Issue cost is real but overlapped, and it is not fold time.

`process_time()` is reported too, but it is NOT the discriminator: tt-metal's completion queue
thread busy-waits, so process CPU is close to the wall whatever the answer is.

The last fold wraps every ttnn entry point and counts calls plus the wall spent inside them. That
arm is instrumented and its fold time is not a perf number; its op count is.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg           # injects conf_kwargs, as the cell did
    sys.path[:] = snap
    patch_boltz2_cfg()
    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / "cdk2x2_512.yaml", fix / "cdk2x2_512.a3m"
    msa_dir = Path(__file__).resolve().parent / ".msa_512"

    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "loadavg_at_start": open("/proc/loadavg").read().split()[:3]},
           "folds": []}
    one_fold, meta, state = B.build_fold("boltz2", msa_dir, tgt, a3m)
    out["env"].update({k: meta[k] for k in ("hardware", "grid", "load_s", "n_msa", "card_type")
                       if k in meta})
    a.out.write_text(json.dumps(out, indent=1))

    def timed(tag):
        t0, c0, p0 = time.perf_counter(), time.thread_time(), time.process_time()
        _s, m = one_fold()
        rec = {"arm": tag, "wall_s": round(time.perf_counter() - t0, 3),
               "main_thread_cpu_s": round(time.thread_time() - c0, 3),
               "process_cpu_s": round(time.process_time() - p0, 3),
               "plddt": m.get("plddt"), "loadavg": open("/proc/loadavg").read().split()[:3]}
        rec["main_thread_cpu_pct"] = round(100 * rec["main_thread_cpu_s"] / rec["wall_s"], 1)
        out["folds"].append(rec)
        a.out.write_text(json.dumps(out, indent=1))
        print("  %-10s wall %7.3f s  main-thread CPU %7.3f s (%4.1f %%)  process CPU %7.3f s"
              % (tag, rec["wall_s"], rec["main_thread_cpu_s"], rec["main_thread_cpu_pct"],
                 rec["process_cpu_s"]), flush=True)
        return rec

    timed("cold")
    for i in range(a.reps):
        timed(f"warm{i}")

    # ---- op census: how many ttnn calls, and how much wall is spent inside them ----------
    import ttnn
    n_calls = defaultdict(int)
    in_ttnn = defaultdict(float)
    depth = {"d": 0}
    originals = []

    def wrap(mod, name, fn):
        def w(*args, **kw):
            if depth["d"]:
                return fn(*args, **kw)
            depth["d"] = 1
            t0 = time.perf_counter()
            try:
                return fn(*args, **kw)
            finally:
                in_ttnn[name] += time.perf_counter() - t0
                n_calls[name] += 1
                depth["d"] = 0
        return w

    for mod, prefix in ((ttnn, "ttnn."), (ttnn.experimental, "ttnn.experimental."),
                        (ttnn.transformer, "ttnn.transformer.")):
        for name in dir(mod):
            if name.startswith("_"):
                continue
            fn = getattr(mod, name, None)
            if callable(fn) and type(fn).__name__ in ("FastOperation", "Operation",
                                                      "builtin_function_or_method"):
                originals.append((mod, name, fn))
                setattr(mod, name, wrap(mod, prefix + name, fn))

    rec = timed("census")
    for mod, name, fn in originals:
        setattr(mod, name, fn)
    tot_calls = sum(n_calls.values())
    tot_in = sum(in_ttnn.values())
    out["op_census"] = {
        "n_ttnn_calls": tot_calls,
        "s_inside_ttnn_calls": round(tot_in, 3),
        "s_outside_ttnn_calls": round(rec["wall_s"] - tot_in, 3),
        "us_per_call_wall": round(1e6 * tot_in / max(tot_calls, 1), 2),
        "main_thread_cpu_us_per_call": round(1e6 * rec["main_thread_cpu_s"] / max(tot_calls, 1), 2),
        "top": sorted(((n, name, round(in_ttnn[name], 3)) for name, n in n_calls.items()),
                      reverse=True)[:25],
    }
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["op_census"], indent=1)[:1200], flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
