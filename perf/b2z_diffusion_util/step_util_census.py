#!/usr/bin/env python3
"""Core-utilization census of one Boltz-2 512 aa diffusion step.

`b2z-grid-utilization` censused the PairformerLayer (97.28 % duration-weighted core
utilization, hypothesis refuted) but could not reach the diffusion step: tracy's *host*
post-process was OOM-killed on pc's 30 GB reconstructing a device log for ~25k programs.
This harness reads the same two fields per dispatched program straight out of the device
profiler with `ttnn.ReadDeviceProfiler()` + `ttnn.profiler.get_all_programs_perf_data()`,
which never builds the host log, and it runs on qb2 where the tracy path also fits, so the
two instruments can be crossed against each other in one process.

One device open for the whole pass: on qb2 a chip wedges on its 4th `ttnn.open_device` since
its last reset, and `tt-smi -r` would take the sibling card of the board pair down with it.

The precursor is the sibling's: recycling 1, the pairformer/MSA stacks truncated to a couple
of blocks, the fold aborted with a sentinel the moment a settled `Diffusion` call has been
captured. Shapes are the shipped shapes; only the values flowing in are junk, and a bf16
matmul's core count and cycle count do not depend on its values.

    python -m tracy -r --no-op-info-cache -o OUT --op-support-count 40000 -- step_util_census.py --out J
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

FENCE_N = 3
FENCE_DIM = 32


class Grabbed(Exception):
    pass


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def make_fence(ttnn, dev):
    import torch
    t = ttnn.from_torch(torch.ones(1, 1, FENCE_DIM, FENCE_DIM), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev)

    def fence():
        for _ in range(FENCE_N):
            ttnn.exp(t)
        ttnn.synchronize_device(dev)
    return fence


def truncate_stacks(T, keep):
    import gc
    saved = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, (T.Pairformer, T.MSA)) and isinstance(
                    getattr(obj, "blocks", None), list) and len(obj.blocks) > keep:
                saved.append((obj, obj.blocks))
                obj.blocks = obj.blocks[:keep]
        except ReferenceError:
            continue

    def restore():
        for obj, blocks in saved:
            obj.blocks = blocks
    return restore, len(saved)


def snap_programs(ttnn, dev):
    """Per-program core_count / num_available_cores / durations, straight from the device.

    Returns {uid_key: record}. Shape-agnostic about `program_analyses_results`: the binding
    exposes it as a container of named `ProgramSingleAnalysisResult`, and which names a build
    populates is a property of the build, so every name is carried through.
    """
    ttnn.ReadDeviceProfiler(dev)
    data = ttnn.profiler.get_all_programs_perf_data()
    out = {}
    for chip, programs in data.items():
        for p in programs:
            uid = p.program_execution_uid
            key = f"{chip}:{uid.runtime_id}:{uid.trace_id}:{uid.trace_id_counter}"
            res = p.program_analyses_results
            if isinstance(res, dict):
                an = {str(k): {"start": v.start_timestamp, "end": v.end_timestamp,
                               "duration": v.duration} for k, v in res.items()}
            else:
                an = {str(i): {"start": v.start_timestamp, "end": v.end_timestamp,
                               "duration": v.duration} for i, v in enumerate(res)}
            out[key] = {"chip": int(chip), "runtime_id": uid.runtime_id,
                        "trace_id": uid.trace_id, "trace_id_counter": uid.trace_id_counter,
                        "core_count": p.core_count,
                        "num_available_cores": p.num_available_cores,
                        "analyses": an}
    return out


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--keep-blocks", type=int, default=2)
    a = ap.parse_args()
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    OUT["env"] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "reps": a.reps, "size": a.size,
        "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
        "ttnn": getattr(ttnn, "__file__", "?"),
        "profiler_env": {k: os.environ.get(k) for k in (
            "TT_METAL_DEVICE_PROFILER", "TT_METAL_PROFILER_MID_RUN_DUMP",
            "TT_METAL_PROFILER_CPP_POST_PROCESS")},
        "loadavg": open("/proc/loadavg").read().split()[:3],
    }
    dump()

    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps  # noqa: F401

    B.RECYCLING_STEPS = 1
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    T.get_device(trace_region_size=1 << 30)
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold(
        "boltz2", HERE / f".msa_{a.size}", fix / f"cdk2x2_{a.size}.yaml",
        fix / f"cdk2x2_{a.size}.a3m")
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    dev = T.get_device()
    dump()

    grabs: dict = {}
    counts = {"n": 0}
    cls = T.Diffusion
    orig = cls.__dict__["__call__"]

    def clone(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

    def wrapper(self_obj, *args, **kw):
        counts["n"] += 1
        out = orig(self_obj, *args, **kw)
        if "g" not in grabs and counts["n"] >= 2:
            grabs["g"] = {"obj": self_obj,
                          "args": tuple(clone(x) for x in args),
                          "kwargs": {k: clone(v) for k, v in kw.items()}}
            print(f"  grabbed Diffusion on call {counts['n']}", flush=True)
            raise Grabbed
        return out
    cls.__call__ = wrapper

    restore, n_trunc = truncate_stacks(T, a.keep_blocks)
    OUT["env"]["stacks_truncated"] = n_trunc
    OUT["env"]["keep_blocks"] = a.keep_blocks
    print("=== precursor fold (aborted at the grab) ===", flush=True)
    t0 = time.perf_counter()
    try:
        one_fold()
    except Grabbed:
        pass
    finally:
        cls.__call__ = orig
        restore()
    OUT["precursor_s"] = round(time.perf_counter() - t0, 3)
    OUT["diffusion_calls_in_precursor"] = counts["n"]
    dump()
    if "g" not in grabs:
        OUT["error"] = "Diffusion was never grabbed"
        dump()
        print("FAILED " + OUT["error"], flush=True)
        return 1

    g = grabs["g"]
    OUT["arg_shapes"] = [list(x.shape) if hasattr(x, "shape") else type(x).__name__
                         for x in g["args"]]
    fence = make_fence(ttnn, dev)

    for _ in range(3):
        g["obj"](*g["args"], **g["kwargs"])
    ttnn.synchronize_device(dev)
    fence()

    try:
        before = snap_programs(ttnn, dev)
        OUT["api_available"] = True
    except Exception as e:                                    # noqa: BLE001
        before = {}
        OUT["api_available"] = False
        OUT["api_error"] = repr(e)
    OUT["api_programs_before"] = len(before)
    dump()

    print(f"=== profiled region: {a.reps} x Diffusion ===", flush=True)
    walls = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        g["obj"](*g["args"], **g["kwargs"])
        ttnn.synchronize_device(dev)
        walls.append(time.perf_counter() - t0)
    OUT["solo_synced_wall_ms_per_call"] = round(1e3 * st.median(walls), 4)
    OUT["solo_synced_wall_ms_all"] = [round(1e3 * w, 4) for w in walls]

    t0 = time.perf_counter()
    for _ in range(a.reps):
        g["obj"](*g["args"], **g["kwargs"])
    ttnn.synchronize_device(dev)
    OUT["back_to_back_ms_per_call"] = round(1e3 * (time.perf_counter() - t0) / a.reps, 4)
    fence()
    print(f"  Diffusion solo {OUT['solo_synced_wall_ms_per_call']:.4f} ms, "
          f"back-to-back {OUT['back_to_back_ms_per_call']:.4f} ms", flush=True)
    dump()

    if OUT.get("api_available"):
        try:
            after = snap_programs(ttnn, dev)
            new = {k: v for k, v in after.items() if k not in before}
            OUT["api_programs_after"] = len(after)
            OUT["api_programs_new"] = len(new)
            (a.out.parent / (a.out.stem + "_api.json")).write_text(
                json.dumps({"env": OUT["env"], "reps": 2 * a.reps,
                            "programs": list(new.values())}, indent=1))
            print(f"  API: {len(new)} new programs over {2 * a.reps} calls", flush=True)
        except Exception as e:                                # noqa: BLE001
            OUT["api_snapshot_error"] = repr(e)
    OUT["fence"] = {"op": "ttnn.exp", "n": FENCE_N, "dim": FENCE_DIM}
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
