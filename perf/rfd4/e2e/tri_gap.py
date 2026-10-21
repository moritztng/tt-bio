"""Where the triangle multiplication's time goes inside a real 128-token RFD4 design, against the same
modules called standalone, in one process on one card.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/e2e/tri_gap.py --spec <spec.yaml> --weights <bundle> \
        --out <file.json> [--arms default,cap128,cap64,lever:trimul_bres,set:reblock_permute.L1_N_MIN=32,lever:x;set:m.Y=1]

1. Cold design: counts every L1/circular-buffer clash the trimul catches (`_record_trimul_clash`) and
   times the first trimul calls, which pay the clash, the refused program and the narrower compile.
2. Warm design with only the two trimul families synced on both sides: the in-design per-call
   distribution, and the call's operands as the trunk passes them (shape, dtype, layout, memory).
3. Standalone: the same module objects on a bf16 z of that shape with the trunk's real pair mask,
   per call, median and max over --repeats.
4. Arms, standalone: each named chunk cap, lever set, or `set:` override of tt_bio module globals
   (`set:mod.ATTR=literal[+mod.ATTR=literal]`, restored after the arm), per call, and the update's bits against the
   default arm (`torch.equal`), so a width or placement arm that moves a bit is caught here.

AICLK is sampled on a thread throughout and printed per leg.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402

import clocksample  # noqa: E402

N_FLOP = lambda n: 2 * n * n * (4 * 384 * 256 + 256 * 384 + 384 * 384) + 2 * 256 * n ** 3  # noqa: E731


def stats(xs):
    s = sorted(xs)
    return {"n": len(s), "median_ms": round(statistics.median(s) * 1e3, 4), "p10_ms": round(s[len(s) // 10] * 1e3, 4),
            "max_ms": round(s[-1] * 1e3, 4), "sum_s": round(sum(s), 4)}


def parse_arm(arm, T):
    """`(chunk cap, lever set, saved globals)` for an arm; parts combine with ';' (lever:a+b;set:m.X=1).

    `set:` assigns tt_bio module globals in place and returns what it replaced so the caller restores them.
    """
    cap, lev, saved = 0, frozenset(), []
    for part in arm.split(";"):
        if part.startswith("cap"):
            cap = int(part[3:])
        elif part.startswith("lever:"):
            lev = T.parse_levers(part[6:].replace("+", ","))
        elif part.startswith("set:"):
            for kv in part[4:].split("+"):
                k, v = kv.split("=", 1)
                modname, attr = k.rsplit(".", 1)
                tgt = importlib.import_module(f"tt_bio.{modname}")
                saved.append((tgt, attr, getattr(tgt, attr)))
                setattr(tgt, attr, ast.literal_eval(v))
    return cap, lev, saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/home/ttuser/rfd4-tri-runs/work")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=60)
    ap.add_argument("--arms", default="default")
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio import tenstorrent as T
    from tt_bio.rfd4 import engine as E

    clashes = []
    rec = T._record_trimul_clash

    def logged(seq_len, hidden, batch, width):
        clashes.append({"t": time.time(), "seq": seq_len, "hidden": hidden, "batch": batch, "width": width})
        return rec(seq_len, hidden, batch, width)

    T._record_trimul_clash = logged

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    designs = rfd4.prepare_examples(a.spec, work / "inputs")
    features, _ = rfd4.featurize(designs[0].cif, seed=a.seed)
    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    dev = engine._device
    pus = list(engine.trunk.pair_updates.values())
    out = {"spec": a.spec, "n": int(features["padding_mask"].shape[-1]),
           "card_env": os.environ.get("TT_VISIBLE_DEVICES"), "trimul_chunk_default": None}

    # Per-call timers on the trimul instances only; `sync` False leaves them untimed.
    calls = {"tri_mult_out": [], "tri_mult_in": []}
    seen = {}
    state = {"sync": False}

    for pu in pus:
        for attr, fam in (("tri_out", "tri_mult_out"), ("tri_in", "tri_mult_in")):
            mod = getattr(pu, attr)

            def timed(x, mask=None, *args, _m=mod, _f=fam, **kw):
                if _f not in seen:
                    seen[_f] = {"shape": [int(d) for d in x.shape], "dtype": str(x.dtype), "layout": str(x.layout),
                                "memory": str(x.memory_config().buffer_type),
                                "mask_shape": None if mask is None else [int(d) for d in mask.shape],
                                "mask_dtype": None if mask is None else str(mask.dtype)}
                if not state["sync"]:
                    return _m(x, mask, *args, **kw)
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                r = _m(x, mask, *args, **kw)
                ttnn.synchronize_device(dev)
                calls[_f].append(time.perf_counter() - t0)
                return r

            setattr(pu, attr, timed)

    trunk_s = []
    inner = engine.trunk.run_bucketed

    def timed_trunk(*args, **kw):
        t = time.perf_counter()
        r = inner(*args, **kw)
        trunk_s.append(time.perf_counter() - t)
        return r

    engine.trunk.run_bucketed = timed_trunk
    run = lambda: engine.design(features, num_designs=1, nsteps=engine.spec.nsteps, seed=a.seed, rng="upstream")  # noqa

    # Leg 1: cold. The first calls carry compile and any clash retry; time them per call.
    state["sync"] = True
    with clocksample.during(period=1.0) as clk:
        t0 = time.perf_counter()
        run()
        cold_s = time.perf_counter() - t0
    cold_calls = {f: list(v) for f, v in calls.items()}
    out["cold"] = {"design_s": round(cold_s, 3), "clashes": clashes[:], "aiclk": clk.summary(),
                   "first_calls_ms": {f: [round(x * 1e3, 2) for x in v[:4]] for f, v in cold_calls.items()},
                   "steady_median_ms": {f: round(statistics.median(v[20:]) * 1e3, 4) for f, v in cold_calls.items()}}
    out["clash_table"] = {str(k): v for k, v in T._TRIMUL_CHUNK_CLASH.items()}
    print(json.dumps({"cold": out["cold"]}), flush=True)

    # Leg 2: warm, uninstrumented (the trunk timer only).
    state["sync"] = False
    trunk_s.clear()
    with clocksample.during(period=1.0) as clk:
        t0 = time.perf_counter()
        run()
        warm_s = time.perf_counter() - t0
    out["warm_plain"] = {"design_s": round(warm_s, 3), "trunk_s": round(sum(trunk_s), 3), "aiclk": clk.summary(),
                         "load": os.getloadavg()[0]}
    print(json.dumps({"warm_plain": out["warm_plain"]}), flush=True)

    # Leg 3: warm, trimul calls synced.
    for v in calls.values():
        v.clear()
    state["sync"] = True
    trunk_s.clear()
    with clocksample.during(period=1.0) as clk:
        t0 = time.perf_counter()
        run()
        inst_s = time.perf_counter() - t0
    state["sync"] = False
    n = out["n"]
    out["in_design"] = {"design_s": round(inst_s, 3), "trunk_s": round(sum(trunk_s), 3), "aiclk": clk.summary(),
                        "load": os.getloadavg()[0], "operands": seen,
                        "per_call": {f: stats(v) for f, v in calls.items()},
                        "tflops_median": {f: round(N_FLOP(n) / statistics.median(v) / 1e12, 2) for f, v in calls.items()}}
    print(json.dumps({"in_design": out["in_design"]}), flush=True)
    out["trimul_chunk_default"] = T._trimul_chunk_size(n, 256, 1)

    # Legs 4+: standalone on the same module objects (unwrapped), each arm.
    from tt_bio.rfd4 import trunk as TR  # noqa: F401
    n = seen["tri_mult_out"]["shape"][1]  # the trunk's bucketed width, which the timings above ran at
    out["n_trunk"] = n
    torch.manual_seed(0)
    zt = torch.randn(1, n, n, 384)
    z = ttnn.from_torch(zt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    m = torch.nn.functional.pad(features["padding_mask"].float().reshape(1, -1), (0, n - out["n"]))
    pm = ttnn.from_torch((m[:, :, None] * m[:, None, :]).contiguous(), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mods = {}  # the wrappers hold the module objects as keyword defaults
    for attr, fam in (("tri_out", "tri_mult_out"), ("tri_in", "tri_mult_in")):
        fn = getattr(pus[0], attr)
        mods[fam] = fn.__kwdefaults__["_m"]

    ref = {}
    arms = {}
    for arm in a.arms.split(","):
        cap, lev, saved = parse_arm(arm, T)
        T._TRIMUL_CHUNK_CAP = cap
        res = {}
        with clocksample.during(period=0.5) as clk, T.levers(lev):
            for fam, mod in mods.items():
                try:
                    u = mod(z, pm)
                    ttnn.synchronize_device(dev)
                    bits = ttnn.to_torch(u)
                    ttnn.deallocate(u)
                    ts = []
                    for _ in range(a.repeats):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        u = mod(z, pm)
                        ttnn.synchronize_device(dev)
                        ts.append(time.perf_counter() - t0)
                        ttnn.deallocate(u)
                except Exception as exc:  # an arm that does not run at this shape is a result
                    res[fam] = {"error": str(exc)[:400]}
                    continue
                if arm == "default":
                    ref[fam] = bits
                eq = bool(torch.equal(bits, ref[fam])) if fam in ref else None
                d = None if fam not in ref else float((bits.float() - ref[fam].float()).abs().max())
                res[fam] = {**stats(ts), "tflops_median": round(N_FLOP(n) / statistics.median(ts) / 1e12, 2),
                            "bit_identical_to_default": eq, "max_abs_diff": d}
        res["aiclk"] = clk.summary()
        res["clash_table"] = {str(k): v for k, v in T._TRIMUL_CHUNK_CLASH.items()}
        from tt_bio import reblock_permute as RP
        res["reblock_rejects"] = sorted({k[0] for k in RP.REJECTS})
        RP.REJECTS.clear()
        from tt_bio import trimul_tail as TT
        res["gout_rejects"], res["t1_why"] = dict(T.TRIMUL_GOUT_REJECTS), dict(TT.T1_WHY)
        T.TRIMUL_GOUT_REJECTS.clear()
        TT.T1_WHY.clear()
        res["load"] = os.getloadavg()[0]
        arms[arm] = res
        print(json.dumps({arm: res}), flush=True)
        T._TRIMUL_CHUNK_CAP = 0
        for tgt, attr, v in saved:
            setattr(tgt, attr, v)
    out["standalone"] = arms
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
