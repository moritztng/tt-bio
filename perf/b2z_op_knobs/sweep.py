#!/usr/bin/env python3
"""Sweep the ttnn knob space of one Boltz-2 op instance and report paired ratios.

One process, one chip, one device open, program cache hot. For every instance it runs the
INCUMBENT config (what the fold dispatches today) and every arm interleaved:

    incumbent, arm1, incumbent, arm2, incumbent, ... , incumbent

and reports each arm's ratio against the median of the incumbent runs that bracket it, plus the
A/A floor (the spread of the incumbent runs themselves). Absolute microseconds on a shared box
are worthless; the ratio and its floor are not.

Parity: every arm's output is pulled to host and compared with the incumbent's. `max_abs` and
`bit_exact` go in the table. An arm that is faster but not bit-exact is reported, not hidden.

Knob grids are per op kind and are pruned to what applies: there is no point sweeping matmul
subblocks on a permute.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import op_replay as R  # noqa: E402


def grid_for(kind: str, rec: dict, level: str) -> list[str]:
    """The knob combinations worth trying for this op kind, as --knob strings."""
    arms: list[str] = []
    has_ckc = kind in ("matmul", "layernorm", "softmax", "sdpa")
    if has_ckc:
        fids = ["LoFi", "HiFi2", "HiFi3"]
        for f in fids:
            arms.append(f"fidelity={f}")
        arms.append("fp32acc=0")
        arms.append("packerl1=0")
        arms.append("dstfull=1")
        for f in fids:
            arms.append(f"fidelity={f},fp32acc=0")
        arms.append("fidelity=LoFi,fp32acc=0,packerl1=0")
        arms.append("fidelity=HiFi2,fp32acc=0,dstfull=1")
    if kind == "matmul":
        for g in ("8x8", "8x7", "7x7", "8x4", "4x8", "8x2"):
            arms.append(f"grid={g}")
            arms.append(f"grid={g},fidelity=HiFi2,fp32acc=0")
    if kind in ("matmul", "binary", "layernorm", "softmax", "unary"):
        arms.append("outbuf=L1")
    if kind in ("binary", "unary", "copy", "transpose", "permute", "slice", "concat"):
        arms.append("outbuf=L1")
    if level == "wide" and kind == "matmul":
        for g in ("8x8", "8x4"):
            for sh in ("width", "height", "block"):
                arms.append(f"shard={sh},grid={g}")
        arms.append("outdtype=bfloat8_b")
        arms.append("fidelity=LoFi,fp32acc=0,outdtype=bfloat8_b")
    if level == "wide" and kind in ("binary", "layernorm"):
        for g in ("8x8",):
            for sh in ("width", "height", "block"):
                arms.append(f"shard={sh},grid={g}")
    # de-dup, keep order
    seen, out = set(), []
    for a in arms:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def run_one(ttnn, torch, rec, device, knob, reps, bursts, want_out, seed=0):
    # Same seed for every arm of an instance, or the parity column compares an arm's output
    # against the incumbent's output computed from DIFFERENT random operands, which reads as a
    # numerics change on arms that cannot possibly have one (a core-grid change, say).
    torch.manual_seed(seed)
    knobs = R.parse_knobs(knob)
    fn, ins = R.build_call(ttnn, torch, rec, device, knobs)
    if fn is None:
        raise RuntimeError("no builder")
    try:
        us, all_us = R.time_call(ttnn, device, fn, ins, reps, bursts)
        host = None
        if want_out:
            o = fn()
            host = ttnn.to_torch(o).float()
            if R._addr(o) not in {R._addr(t) for t in ins}:
                try:
                    ttnn.deallocate(o)
                except Exception:                                        # noqa: BLE001
                    pass
        return us, all_us, host
    finally:
        for t in ins:
            try:
                ttnn.deallocate(t)
            except Exception:                                            # noqa: BLE001
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "bench_manifest.json"))
    ap.add_argument("--baseline", required=True, help="op_replay.py output, for ranking")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--shard", default="0/1", help="i/n: take every n-th instance starting at i")
    ap.add_argument("--top", type=int, default=24, help="rank instances by ms/fold, keep top N")
    ap.add_argument("--min-share", type=float, default=0.5,
                    help="percent of total replayed device time an instance must carry")
    ap.add_argument("--level", default="base", choices=["base", "wide"])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--bursts", type=int, default=5)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--arms-from", default="",
                    help="a prior sweep json: re-run only arms that beat --arms-min there")
    ap.add_argument("--arms-min", type=float, default=1.05)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    R._init_tables(ttnn)

    recs = {r["id"]: r for r in json.load(open(a.manifest))}
    base = json.load(open(a.baseline))
    rows = [r for r in base["rows"] if "ms_per_fold" in r]
    total = sum(r["ms_per_fold"] for r in rows)
    rows.sort(key=lambda r: -r["ms_per_fold"])
    if a.ids:
        keep = [r for r in rows if r["id"] in set(a.ids.split(","))]
    else:
        keep = [r for r in rows[: a.top] if 100.0 * r["ms_per_fold"] / total >= a.min_share]
    i, n = (int(x) for x in a.shard.split("/"))
    keep = keep[i::n]

    device = R.open_device_retry(ttnn)
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "shard": a.shard, "level": a.level, "reps": a.reps, "bursts": a.bursts,
                   "grid": str(device.compute_with_storage_grid_size()),
                   "baseline_total_ms": round(total, 2)},
           "instances": []}
    print(f"# {len(keep)} instances, shard {a.shard}, level {a.level}", flush=True)

    prior = {}
    if a.arms_from:
        for e in json.load(open(a.arms_from))["instances"]:
            prior[e["id"]] = [r["knob"] for r in e.get("arms", [])
                              if r.get("ratio", 0) >= a.arms_min]

    for br in keep:
        rec = recs[br["id"]]
        arms = prior.get(br["id"], []) if a.arms_from else grid_for(rec["kind"], rec, a.level)
        if a.arms_from and not arms:
            continue
        entry = {"id": br["id"], "unit": rec["unit"], "kind": rec["kind"], "api": rec["api"],
                 "shapes": br["shapes"], "calls_per_fold": rec["calls_per_fold"],
                 "baseline_ms_per_fold": br["ms_per_fold"], "arms": []}
        print(f"== {br['id']} {rec['kind']} {'|'.join(br['shapes'])[:60]} "
              f"({br['ms_per_fold']:.1f} ms/fold, {len(arms)} arms)", flush=True)
        try:
            inc_us, _, inc_out = run_one(ttnn, torch, rec, device, "", a.reps, a.bursts, a.parity)
        except Exception as e:                                           # noqa: BLE001
            entry["error"] = f"incumbent failed: {type(e).__name__}: {str(e)[:160]}"
            out["instances"].append(entry)
            json.dump(out, open(a.out, "w"), indent=1)
            print("   " + entry["error"], flush=True)
            continue
        incs = [inc_us]
        for knob in arms:
            row = {"knob": knob}
            try:
                us, all_us, o = run_one(ttnn, torch, rec, device, knob, a.reps, a.bursts,
                                        a.parity)
                inc2, _, _ = run_one(ttnn, torch, rec, device, "", a.reps, a.bursts, False)
                incs.append(inc2)
                ref = st.median([incs[-2], inc2])
                row.update({"us": round(us, 2), "ratio": round(ref / us, 4),
                            "incumbent_us": round(ref, 2)})
                if a.parity and o is not None and inc_out is not None and o.shape == inc_out.shape:
                    d = (o - inc_out).abs()
                    row["max_abs"] = round(float(d.max()), 6)
                    row["bit_exact"] = bool(float(d.max()) == 0.0)
            except Exception as e:                                       # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
            entry["arms"].append(row)
            msg = row.get("error") or (
                f"{row['ratio']:6.3f}x  ({row['us']:9.2f} us vs {row['incumbent_us']:9.2f})"
                + (f"  maxabs {row['max_abs']:.2e}" if "max_abs" in row else ""))
            print(f"   {knob:42s} {msg}", flush=True)
            json.dump(out, open(a.out, "w"), indent=1)
        entry["incumbent_us_runs"] = [round(x, 2) for x in incs]
        entry["aa_floor"] = round(max(incs) / min(incs), 4) if len(incs) > 1 else None
        best = [r for r in entry["arms"] if "ratio" in r]
        if best:
            b = max(best, key=lambda r: r["ratio"])
            entry["best"] = b
            print(f"   BEST {b['knob']} {b['ratio']:.3f}x   A/A floor {entry['aa_floor']}",
                  flush=True)
        out["instances"].append(entry)
        json.dump(out, open(a.out, "w"), indent=1)

    ttnn.close_device(device)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
