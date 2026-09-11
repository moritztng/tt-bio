#!/usr/bin/env python3
"""The device floor of every phase of the Boltz-2 512 aa fold, and why main-thread CPU lied.

`host_phase_cost.py` produced two results that cannot both mean what they look like:

  * TrunkModule reads 12.653 s of main-thread CPU in a 12.882 s wall, 98.2 %, which reads as
    "the trunk is dispatch-bound";
  * one shipped pairformer block replayed from a ttnn trace, with no host work between replays,
    takes 41.401 ms against the same block's 41.455 ms eager synced wall, which reads as "the
    block is 99.9 % device and has no dispatch headroom at all";
  * and injecting 10 us of real host CPU into every ttnn call moved the trunk's wall by -0.014 s
    and its CPU by -0.016 s, i.e. the trunk absorbed the injection without noticing.

The reading that fits all three is that the calling thread BURNS CPU while it waits for room in
tt-metal's dispatch queue, so `time.thread_time()` counts device wait as host work whenever the
queue is full. The first file's control missed it: 400 back-to-back 4096-cube matmuls fit in the
queue, so the main thread never waited at all (issue wall 0.003 s, drain 0.375 s) and the control
proved only that a push is cheap.

So this file does two things:

  queue    the control that was missing. Sweep the number of back-to-back device-bound matmuls
           from below the queue's depth to far above it and watch main-thread CPU. If CPU/wall
           climbs to 1 as the queue fills, `thread_time` is not a host/device discriminator on
           this stack and every "% main-thread CPU" figure in this campaign is a mix of issue
           cost and device wait.
  block    the same demonstration on the code in question: one real pairformer block issued with
           a device sync after every call (queue empty, host cost visible) against the same block
           issued back-to-back without syncs (queue full). Same ops, same device work; if the
           apparent host cost per block moves by ~10x, the artifact is proved where it matters.
  floor    the device floor per phase, by ttnn trace capture of one settled call of each of
           PairformerLayer, MSALayer and the diffusion step, replayed back-to-back with no host
           in between. Times the phase's call count this gives the seconds of the fold a perfect
           dispatch lever cannot reach.

No model code changes: every captured region is unmodified shipped code called with its own
arguments.
"""
from __future__ import annotations

import argparse
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


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def loadavg():
    return open("/proc/loadavg").read().split()[:3]


def queue_control(ttnn, dev):
    """Does main-thread CPU rise to the wall once the dispatch queue is full?"""
    import torch
    n = 2048
    a = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                        device=dev)
    b = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                        device=dev)
    for _ in range(5):
        ttnn.matmul(a, b)
    ttnn.synchronize_device(dev)
    rows = []
    for reps in (100, 400, 1000, 2000, 5000, 12000):
        ttnn.synchronize_device(dev)
        t0, c0 = time.perf_counter(), time.thread_time()
        for _ in range(reps):
            ttnn.matmul(a, b)
        t_issue, c_issue = time.perf_counter() - t0, time.thread_time() - c0
        ttnn.synchronize_device(dev)
        t_total = time.perf_counter() - t0
        rows.append({
            "reps": reps,
            "issue_wall_s": round(t_issue, 4), "issue_cpu_s": round(c_issue, 4),
            "total_wall_s": round(t_total, 4), "drain_s": round(t_total - t_issue, 4),
            "cpu_over_issue_wall": round(c_issue / max(t_issue, 1e-9), 4),
            "cpu_over_total_wall": round(c_issue / max(t_total, 1e-9), 4),
            "apparent_host_us_per_call": round(1e6 * c_issue / reps, 2),
            "device_us_per_call": round(1e6 * t_total / reps, 2),
        })
        r = rows[-1]
        print(f"  reps {reps:6d}  apparent host {r['apparent_host_us_per_call']:8.2f} us/call  "
              f"device {r['device_us_per_call']:7.2f} us/call  CPU/issue-wall "
              f"{r['cpu_over_issue_wall']:.3f}  drain {r['drain_s']:.3f} s", flush=True)
        OUT["queue"] = rows
        dump()
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return rows


def block_two_ways(ttnn, dev, layer, args, kwargs, reps=16):
    """Same block, queue empty vs queue full. The apparent host cost is the whole point."""
    for _ in range(2):
        layer(*args, **kwargs)
    ttnn.synchronize_device(dev)
    # (a) a sync after every call: the queue is empty when the next call starts
    cpus, walls = [], []
    for _ in range(5):
        t0, c0 = time.perf_counter(), time.thread_time()
        layer(*args, **kwargs)
        cpus.append(time.thread_time() - c0)
        ttnn.synchronize_device(dev)
        walls.append(time.perf_counter() - t0)
    # (b) back to back, no syncs: the queue fills and the calling thread waits inside ttnn
    ttnn.synchronize_device(dev)
    t0, c0 = time.perf_counter(), time.thread_time()
    for _ in range(reps):
        layer(*args, **kwargs)
    c_b = time.thread_time() - c0
    t_issue_b = time.perf_counter() - t0
    ttnn.synchronize_device(dev)
    t_b = time.perf_counter() - t0
    return {
        "synced_each_call": {"host_cpu_ms": round(1e3 * st.median(cpus), 4),
                             "wall_ms": round(1e3 * st.median(walls), 4)},
        "back_to_back": {"reps": reps,
                         "apparent_host_cpu_ms_per_call": round(1e3 * c_b / reps, 4),
                         "issue_wall_ms_per_call": round(1e3 * t_issue_b / reps, 4),
                         "wall_ms_per_call": round(1e3 * t_b / reps, 4)},
    }


def trace_floor(ttnn, dev, fn, args, kwargs, reps=16, n_med=5):
    """Device ms per call: capture once, replay back-to-back, no host work between replays."""
    for _ in range(2):
        fn(*args, **kwargs)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    fn(*args, **kwargs)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    meds, issue_cpu = [], []
    for _ in range(n_med):
        t0, c0 = time.perf_counter(), time.thread_time()
        for _ in range(reps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        issue_cpu.append((time.thread_time() - c0) / reps)
        ttnn.synchronize_device(dev)
        meds.append((time.perf_counter() - t0) / reps)
    ttnn.release_trace(dev, tid)
    return {"device_ms_per_call": round(1e3 * st.median(meds), 4),
            "all_ms": [round(1e3 * m, 4) for m in meds],
            "replay_issue_cpu_us_per_call": round(1e6 * st.median(issue_cpu), 3)}


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--phases", default="queue,grab,block,floor")
    ap.add_argument("--grab", default="PairformerLayer,MSALayer,Diffusion",
                    help="classes to grab one settled call of and trace-replay")
    a = ap.parse_args()
    OUT_PATH = a.out
    phases = [p for p in a.phases.split(",") if p]

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    T.get_device(trace_region_size=1 << 30)
    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"
    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "loadavg": loadavg(), "size": a.size}
    dump()
    one_fold, meta, state = B.build_fold("boltz2", HERE / f".msa_{a.size}", tgt, a3m)
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    dev = T.get_device()
    dump()

    # ---- grab one settled call of each class we want a device floor for ---------------------
    grabs: dict[str, dict] = {}
    counts: dict[str, int] = {}
    originals = []

    def _vol(sh):
        v = 1
        for d in sh:
            v *= int(d)
        return v

    def clone(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

    def arm(cname, want):
        cls = getattr(T, cname)
        orig = cls.__dict__["__call__"]
        originals.append((cls, orig))

        def w(self_obj, *args, **kw):
            counts[cname] = counts.get(cname, 0) + 1
            if cname not in grabs and counts[cname] >= 3 and want(self_obj, args):
                grabs[cname] = {"obj": self_obj,
                                "args": tuple(clone(x) for x in args),
                                "kwargs": {k: clone(v) for k, v in kw.items()}}
                print(f"  grabbed {cname} on call {counts[cname]}", flush=True)
            return orig(self_obj, *args, **kw)
        cls.__call__ = w

    print("=== cold fold (warms kernels; grabs happen on the second) ===", flush=True)
    t, m = one_fold()
    OUT["cold_s"] = round(t, 3)
    dump()
    targets = [c for c in a.grab.split(",") if c]
    if "grab" in phases:
        for cname in targets:
            if cname == "PairformerLayer":
                arm(cname, lambda o, ar: getattr(o, "transform_s", False))
            elif cname == "Transition":
                # the pair-track transition, not the single-track or diffusion one: pick it by
                # the volume of its first tensor argument (z is 512x512x128 at 512 aa)
                arm(cname, lambda o, ar: bool(ar) and hasattr(ar[0], "shape")
                    and _vol(ar[0].shape) > 4_000_000)
            else:
                arm(cname, lambda o, ar: True)
    print("=== fold 2 (grabbing) ===", flush=True)
    t, m = one_fold()
    for cls, orig in originals:
        cls.__call__ = orig
    OUT["fold2_s"] = round(t, 3)
    OUT["class_call_counts_per_fold"] = counts
    OUT["grabbed"] = sorted(grabs)
    print("  call counts/fold: " + json.dumps(counts), flush=True)
    dump()

    if "queue" in phases:
        print("=== queue control: does main-thread CPU track the wall once the queue fills? ===",
              flush=True)
        try:
            queue_control(ttnn, dev)
        except Exception as e:                                              # noqa: BLE001
            OUT["queue_error"] = f"{type(e).__name__}: {e}"
            print("  FAILED " + OUT["queue_error"], flush=True)
            dump()

    if "block" in phases and "PairformerLayer" in grabs:
        print("=== one pairformer block, queue empty vs queue full ===", flush=True)
        try:
            g = grabs["PairformerLayer"]
            OUT["block_two_ways"] = block_two_ways(ttnn, dev, g["obj"], g["args"], g["kwargs"])
            print("  " + json.dumps(OUT["block_two_ways"]), flush=True)
        except Exception as e:                                              # noqa: BLE001
            OUT["block_two_ways"] = {"error": f"{type(e).__name__}: {e}"}
            print("  FAILED " + str(OUT["block_two_ways"]), flush=True)
        dump()

    if "floor" in phases:
        print("=== device floor per phase by trace replay ===", flush=True)
        OUT["floor"] = {}
        for cname in targets:
            if cname not in grabs:
                OUT["floor"][cname] = {"error": "not grabbed"}
                continue
            g = grabs[cname]
            try:
                r = trace_floor(ttnn, dev, g["obj"], g["args"], g["kwargs"])
                r["calls_per_fold"] = counts.get(cname)
                r["device_s_per_fold"] = round(
                    1e-3 * r["device_ms_per_call"] * counts.get(cname, 0), 4)
                OUT["floor"][cname] = r
                print(f"  {cname:18s} {r['device_ms_per_call']:9.4f} ms/call x "
                      f"{r['calls_per_fold']} = {r['device_s_per_fold']:7.3f} s/fold "
                      f"(replay issue {r['replay_issue_cpu_us_per_call']:.2f} us/call)",
                      flush=True)
            except Exception as e:                                          # noqa: BLE001
                OUT["floor"][cname] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  {cname} FAILED {OUT['floor'][cname]['error']}", flush=True)
            dump()

    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
