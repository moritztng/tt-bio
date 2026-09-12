#!/usr/bin/env python3
"""Is the math thread's 18.3366 ms/block input-tile wait schedulable, or physical?

Wave 1 measured that TRISC1 spends 57.0 % of its resident time in `cb_wait_front`, blocked on
input tiles that have not arrived, and read that as "movement is the binding term". That is a
statement about how much data moves. It is not a statement about *when* it moves, and nobody
had varied the things that decide how far ahead the reader can run: circular-buffer depth,
the K-block split, and the subblock geometry that sets tiles-per-wait.

This grabs one settled `PairformerLayer` call at the shipped 512 aa shapes and replays it in
paired interleaved arms, one arm per CB topology, in a single process. Every arm is compared
against a `base` arm run immediately before it, so host drift and box contention cancel; the
`base`-vs-`base` ratio is the A/A floor the win has to clear.
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


class Grabbed(Exception):
    """Unwinds out of the precursor fold the moment the call has been captured."""


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def clear_generic_caches(mods):
    """Every generic-op program descriptor is cached by shape, not by CB topology."""
    for m in mods:
        for name in ("_CACHE", "_CACHE_BACK", "_CACHE_GATED", "_SPLIT_CACHE"):
            d = getattr(m, name, None)
            if isinstance(d, dict):
                d.clear()


def arm_apply(name, G, T):
    """Set the CB topology for one arm. Returns a dict describing what was set."""
    if name == "base":
        G.CB_DEPTH = 2
        return {"cb_depth": 2}
    if name.startswith("depth"):
        d = int(name[len("depth"):])
        G.CB_DEPTH = d
        return {"cb_depth": d}
    raise SystemExit(f"unknown arm {name}")


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3, help="block calls timed per arm visit")
    ap.add_argument("--visits", type=int, default=5, help="interleaved visits per arm")
    ap.add_argument("--arms", default="depth3,depth4")
    a = ap.parse_args()
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.mm_generic as G

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "size": a.size, "reps": a.reps, "visits": a.visits,
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "loadavg": open("/proc/loadavg").read().split()[:3]}
    dump()

    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    T.get_device(trace_region_size=int(os.environ.get("TT_BIO_TRACE_REGION_SIZE", 1 << 28)))
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold(
        "boltz2", HERE / f".msa_{a.size}", fix / f"cdk2x2_{a.size}.yaml",
        fix / f"cdk2x2_{a.size}.a3m")
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    dev = T.get_device()
    dump()

    # --- grab a settled PairformerLayer with the shipped shapes -------------------------------
    grabs: dict = {}
    counts = {"n": 0}
    cls = T.PairformerLayer
    orig = cls.__dict__["__call__"]

    def clone(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

    def wrapper(self_obj, *args, **kw):
        counts["n"] += 1
        out = orig(self_obj, *args, **kw)
        if not grabs and counts["n"] >= 2 and getattr(self_obj, "transform_s", False):
            grabs["g"] = {"obj": self_obj,
                          "args": tuple(clone(x) for x in args),
                          "kwargs": {k: clone(v) for k, v in kw.items()}}
            raise Grabbed
        return out
    cls.__call__ = wrapper
    t0 = time.perf_counter()
    try:
        one_fold()
    except Grabbed:
        pass
    finally:
        cls.__call__ = orig
    OUT["precursor_s"] = round(time.perf_counter() - t0, 2)
    OUT["blocks_in_precursor"] = counts["n"]
    dump()
    if not grabs:
        OUT["error"] = "PairformerLayer never grabbed"
        dump()
        return 1
    g = grabs["g"]
    OUT["arg_shapes"] = [list(x.shape) if hasattr(x, "shape") else type(x).__name__
                         for x in g["args"]]

    # --- which modules dispatch the block's 16 generic ops, and how often ---------------------
    import tt_bio.sdpa_generic as SG
    import tt_bio.reblock_permute as RP
    import tt_bio.triatt_qkv as TQ
    import tt_bio.triatt_sdpa as TS
    import tt_bio.trimul_tail as TT
    mods = [G, SG, RP, TQ, TS, TT]

    census = {}
    real_generic = ttnn.generic_op
    frames = {}

    def counting_generic(*args, **kw):
        import traceback
        st_ = traceback.extract_stack(limit=6)
        who = "?"
        for fr in reversed(st_[:-1]):
            nm = Path(fr.filename).stem
            if nm in ("mm_generic", "sdpa_generic", "reblock_permute", "triatt_qkv",
                      "triatt_sdpa", "trimul_tail", "softmax_generic", "mm_dualnoc"):
                who = f"{nm}.{fr.name}"
                break
        census[who] = census.get(who, 0) + 1
        frames.setdefault(who, fr.lineno if who != "?" else 0)
        return real_generic(*args, **kw)

    ttnn.generic_op = counting_generic
    g["obj"](*g["args"], **g["kwargs"])
    ttnn.synchronize_device(dev)
    ttnn.generic_op = real_generic
    OUT["generic_op_census_per_block"] = census
    dump()
    print("  generic-op census " + json.dumps(census), flush=True)

    # --- reference output, bit-exactness bar --------------------------------------------------
    def run_once():
        return g["obj"](*g["args"], **g["kwargs"])

    def to_host(o):
        if isinstance(o, ttnn.Tensor):
            return [ttnn.to_torch(o)]
        if isinstance(o, (list, tuple)):
            return [t for x in o for t in to_host(x)]
        return []

    ref = to_host(run_once())
    ttnn.synchronize_device(dev)

    def timed(reps):
        for _ in range(2):
            run_once()
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        for _ in range(reps):
            run_once()
        ttnn.synchronize_device(dev)
        return 1e3 * (time.perf_counter() - t) / reps

    arms = ["base"] + [x for x in a.arms.split(",") if x]
    samples = {x: [] for x in arms}
    setup = {}
    order = []
    for v in range(a.visits):
        for name in arms:
            order.append(name)
    OUT["arms"] = arms
    for i, name in enumerate(order):
        T._L1_OUT_RUNG.clear()
        clear_generic_caches(mods)
        setup[name] = arm_apply(name, G, T)
        ms = timed(a.reps)
        samples[name].append(round(ms, 4))
        print(f"  [{i+1}/{len(order)}] {name:8s} {ms:8.4f} ms/block", flush=True)
        OUT["samples"] = samples
        dump()
        if name != "base":
            got = to_host(run_once())
            ttnn.synchronize_device(dev)
            eq = all(bool(torch.equal(x, y)) for x, y in zip(ref, got)) and len(got) == len(ref)
            OUT.setdefault("bit_exact", {})[name] = eq
    T._L1_OUT_RUNG.clear()
    clear_generic_caches(mods)
    arm_apply("base", G, T)

    med = {k: round(st.median(v), 4) for k, v in samples.items()}
    OUT["setup"] = setup
    OUT["median_ms"] = med
    b = samples["base"]
    half = len(b) // 2
    OUT["aa_floor_pct"] = round(
        100 * abs(st.median(b[:half]) / st.median(b[half:]) - 1), 3) if half else None
    OUT["ratio_vs_base"] = {k: round(med["base"] / v, 4) for k, v in med.items()}
    dump()
    print(json.dumps({"median_ms": med, "ratio": OUT["ratio_vs_base"],
                      "aa_floor_pct": OUT["aa_floor_pct"],
                      "bit_exact": OUT.get("bit_exact")}, indent=1), flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
