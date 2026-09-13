#!/usr/bin/env python3
"""The diffusion step's wall inside a real fold, eager and traced, on one card.

The census bounded the step at "51.6 % eager / <=32 % traced" of the fold and measured its
device kernel time at 22.0 ms (BH) / 40.4 ms (WH) per step. Nobody put a wall next to that
kernel time in fold context, so the step's gap fraction has never been a number.

This measures the wall at the one boundary where it is unambiguous: `DiffusionModule.forward`
(eager) and `DiffusionModule.forward_traced` (traced). Both take torch tensors, stage them onto
the device, run the step and come back through `ttnn.to_torch`, which blocks. So a call's host
wall IS that step's end-to-end wall, with no extra synchronisation added and nothing to perturb.

The two arms alternate inside one process, one fold each, round after round. A sequential
all-eager-then-all-traced schedule would hand the second arm a warmer host and a warmer kernel
cache; the arms also share one device open, one weight load and one MSA cache, so the only
difference between them is the flag.

Change no model code: the arm is flipped by setting `_diffusion_trace` on the live AtomDiffusion
objects, which is what `Boltz.__init__(diffusion_trace=)` sets, and the trace region is reserved
by opening the device before `build_fold` does (512 MiB -- 1 GiB hangs at open on this mesh).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {}
OUT_PATH: Path | None = None
CALLS: list = []
ARM = {"name": "?", "round": 0}


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def loadavg():
    return open("/proc/loadavg").read().split()[:3]


def set_arm(traced: bool) -> int:
    """Flip every live AtomDiffusion onto the traced or untraced score-model path."""
    n = 0
    for obj in gc.get_objects():
        try:
            if hasattr(obj, "_diffusion_trace") and hasattr(obj, "score_model"):
                obj._diffusion_trace = traced
                n += 1
        except ReferenceError:
            continue
    return n


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2, help="eager+traced folds per round")
    ap.add_argument("--settle", type=int, default=20,
                    help="sampling steps discarded at the head of each fold")
    ap.add_argument("--workers", type=int, default=8,
                    help="host_thread_cap_env worker count; 8 is the whglx K10 standard")
    ap.add_argument("--host-threads", type=int, default=None)
    ap.add_argument("--trace-region-mib", type=int, default=512)
    ap.add_argument("--open-lock", type=Path, default=None,
                    help="flock this path across the device open. tt_bio's own lock lives in "
                         "/tmp under another account here and _device_init_lock() swallows the "
                         "PermissionError in a bare except, so it serializes nothing.")
    a = ap.parse_args()
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # The cap has to be in the environment before torch builds its pools.
    from tt_bio import runtime as RT
    cap_env = RT.host_thread_cap_env(a.workers, a.host_threads)
    os.environ.update(cap_env)

    import torch
    torch.set_grad_enabled(False)
    RT.bind_host_threads()
    import ttnn  # noqa: F401
    import tt_bio.tenstorrent as T

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "host": os.uname().nodename, "size": a.size,
                  "host_thread_cap_env": cap_env,
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
                  "ttnn": getattr(ttnn, "__file__", "?"),
                  "loadavg_start": loadavg()}
    dump()

    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps
    B.SAMPLING_STEPS = 200
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    # Open the chip ourselves with a region build_fold would size at 1 GiB, which hangs here.
    lk = None
    if a.open_lock:
        import fcntl
        a.open_lock.parent.mkdir(parents=True, exist_ok=True)
        lk = open(a.open_lock, "a+")
        fcntl.flock(lk, fcntl.LOCK_EX)
    try:
        T.get_device(trace_region_size=a.trace_region_mib << 20)
    finally:
        if lk:
            import fcntl
            fcntl.flock(lk, fcntl.LOCK_UN)
            lk.close()
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold(
        "boltz2", HERE / f".msa_{a.size}", fix / f"cdk2x2_{a.size}.yaml",
        fix / f"cdk2x2_{a.size}.a3m")
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    OUT["env"]["trace_region_bytes"] = T.trace_region_size()
    dump()

    DM = T.DiffusionModule
    orig_fwd, orig_tr = DM.forward, DM.forward_traced

    def timed(fn, tag):
        def wrapper(self_obj, *args, **kw):
            t0 = time.perf_counter()
            out = fn(self_obj, *args, **kw)
            CALLS.append((ARM["name"], ARM["round"], tag, time.perf_counter() - t0))
            return out
        return wrapper

    DM.forward, DM.forward_traced = timed(orig_fwd, "eager"), timed(orig_tr, "traced")

    def run(name, traced, rnd):
        ARM["name"], ARM["round"] = name, rnd
        n = set_arm(traced)
        before = len(CALLS)
        t0 = time.perf_counter()
        one_fold()
        wall = time.perf_counter() - t0
        calls = CALLS[before:]
        per = [c[3] for c in calls]
        tags = sorted({c[2] for c in calls})
        settled = per[a.settle:]
        rec = {"arm": name, "round": rnd, "atomdiffusions_flipped": n,
               "fold_wall_s": round(wall, 3), "diffusion_calls": len(per),
               "call_paths": tags,
               "step_ms_median": round(1e3 * st.median(settled), 4) if settled else None,
               "step_ms_mean": round(1e3 * st.fmean(settled), 4) if settled else None,
               "step_ms_min": round(1e3 * min(settled), 4) if settled else None,
               "step_ms_p90": round(1e3 * sorted(settled)[int(0.9 * len(settled))], 4)
               if settled else None,
               "step_ms_total_settled": round(1e3 * sum(settled), 3),
               "settled_calls": len(settled),
               "step_ms_first5": [round(1e3 * x, 3) for x in per[:5]],
               "loadavg": loadavg()}
        OUT.setdefault("folds", []).append(rec)
        dump()
        print(f"  [{name} r{rnd}] fold {wall:.2f} s, {len(per)} diffusion calls via {tags}, "
              f"settled step median {rec['step_ms_median']} ms  load {rec['loadavg'][0]}",
              flush=True)
        return rec

    print("=== fold 0: cold, discarded (kernel warmup, MSA validation) ===", flush=True)
    run("warmup", False, 0)
    for rnd in range(1, a.rounds + 1):
        print(f"=== round {rnd} ===", flush=True)
        run("eager", False, rnd)
        run("traced", True, rnd)

    DM.forward, DM.forward_traced = orig_fwd, orig_tr

    def med(arm):
        v = [f["step_ms_median"] for f in OUT["folds"] if f["arm"] == arm]
        return round(st.median(v), 4) if v else None

    e, t = med("eager"), med("traced")
    OUT["summary"] = {
        "eager_step_ms": e, "traced_step_ms": t,
        "trace_speedup": round(e / t, 4) if e and t else None,
        "eager_spread": round(max(f["step_ms_median"] for f in OUT["folds"] if f["arm"] == "eager")
                              / min(f["step_ms_median"] for f in OUT["folds"]
                                    if f["arm"] == "eager"), 4),
        "traced_spread": round(max(f["step_ms_median"] for f in OUT["folds"]
                                   if f["arm"] == "traced")
                               / min(f["step_ms_median"] for f in OUT["folds"]
                                     if f["arm"] == "traced"), 4),
        "loadavg_end": loadavg()}
    dump()
    print(f"\nEAGER {e} ms/step   TRACED {t} ms/step   "
          f"trace {OUT['summary']['trace_speedup']}x", flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
