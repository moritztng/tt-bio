#!/usr/bin/env python3
"""CPU-only reducer: clock qualification, the unit spine, and per-op in-fold device seconds.

Re-runnable on an archived run. Refuses to print a table whose clock did not qualify, because a
512 aa fold's seconds are set by AICLK (800 MHz reads 21.90 s where 1350 MHz reads 14.69 s on this
cell) and a header reading taken before the fold is not a measurement.

Three things it does, in order of how much they can be trusted:

1. `clock`   per timed interval, from the 1 kHz sysfs samples `clk.py` wrote next to the run.
             An interval qualifies only if every sample reads the target, there is no read error,
             and no sample gap exceeds 10 ms.
2. `spine`   the unit call census and the unsynced inclusive/exclusive host wall tree, from the
             `counts` phase. These are WEIGHTS and coverage, not device time.
3. `device`  per-op device kernel seconds inside each profiled unit, from tt-metal's ops report,
             windowed on the fence and split back into the census's python-level classes using the
             call sequence the harness recorded in the same region.

The census classes and the device op codes are not the same partition, and that is the whole
difficulty: `linear` and `matmul` are both MatmulDeviceOperation, and `multiply_`, `add_`,
`multiply` and `add` are all BinaryNgDeviceOperation. The ops report carries no python name, so
the split comes from an order-preserving alignment of the recorded python call sequence against
the report's dispatch order, checked on shapes. `align_quality` is reported with every table: a
class split taken from a poor alignment is not quotable.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

GAP_MS = 10.0

# The c10-fold-census published replay prices this row is asked to check, in seconds per fold, and
# which replay arm each one came from. Source: workstreams/c12-profiled-fold.txt drift table of
# 2026-09-17, itself re-derived from perf/c12_orchestrator/replay_bias/replay_bias.json.
CENSUS = {
    "linear":       {"calls": 108608, "published_s": 4.6604, "arm": "grid110",
                     "bare_s": 10.7316, "grid110_s": 4.6613, "byte_x": 1.83},
    "matmul":       {"calls": 1432,   "published_s": 1.479,  "arm": "grid110",
                     "bare_s": 2.1174, "grid110_s": 1.4814, "byte_x": 1.47},
    "multiply_":    {"calls": 42720,  "published_s": 1.751,  "arm": "bare",
                     "bare_s": 1.7511, "grid110_s": None, "byte_x": 2.23},
    "layer_norm":   {"calls": 35304,  "published_s": 1.533,  "arm": "bare",
                     "bare_s": 1.5326, "grid110_s": None, "byte_x": 1.22,
                     "note": "published as layer_norm 1.357 + layer_norm_w 0.176"},
    "add_":         {"calls": 14616,  "published_s": 0.8473, "arm": "bare",
                     "bare_s": 0.8473, "grid110_s": None, "byte_x": 1.32},
    "add":          {"calls": 13728,  "published_s": 0.1403, "arm": "bare",
                     "bare_s": 0.1403, "grid110_s": None, "byte_x": 1.03},
    "multiply":     {"calls": 12464,  "published_s": 0.1256, "arm": "bare",
                     "bare_s": 0.1256, "grid110_s": None, "byte_x": 1.00},
}
CENSUS_PRICED_S = 10.5368          # the 8 priced classes, c10-fold-census
FOLD_S_OF_RECORD = 14.881          # unprofiled median at a pinned during-sampled 1350 MHz
F_FIXED_S = 3.9830                 # c10-fixed-cost, clock-immune term
GENERIC_FLOOR_S = 4.0841           # ttnn.generic_op traffic floor, priced by nobody


def load_clock(path: Path):
    if not path.is_file():
        return None
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    return {"hdr": lines[0], "samples": lines[1:]}


def qualify(clock, t0, t1, target, node):
    """Did the clock hold `target` on `node` for the whole of [t0, t1]?"""
    if clock is None or t0 is None or t1 is None:
        return {"ok": False, "why": "no clock samples or no interval stamps"}
    key = str(node)
    s = [r for r in clock["samples"] if t0 <= r["t"] <= t1]
    if len(s) < 10:
        return {"ok": False, "why": "only %d samples in the interval" % len(s)}
    v = [r.get(key) for r in s]
    errs = [x for x in v if not isinstance(x, int)]
    ints = [x for x in v if isinstance(x, int)]
    ts = [r["t"] for r in s]
    gaps = [1e3 * (b - a) for a, b in zip(ts, ts[1:])]
    out = {"n": len(s), "min": min(ints) if ints else None, "max": max(ints) if ints else None,
           "read_errors": len(errs), "max_gap_ms": round(max(gaps), 2) if gaps else None,
           "target": target, "node": node}
    out["ok"] = bool(ints) and not errs and out["min"] == out["max"] == target \
        and out["max_gap_ms"] is not None and out["max_gap_ms"] <= GAP_MS
    if not out["ok"]:
        out["why"] = ("clock did not hold %d (min %s max %s, %d read errors, max gap %s ms)"
                      % (target, out["min"], out["max"], len(errs), out["max_gap_ms"]))
    return out


def spine(run: Path, target: int, node: int):
    f = run / "counts.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    clock = load_clock(run / "clock.jsonl")
    legs = []
    for l in d.get("legs", []):
        q = qualify(clock, l.get("t0"), l.get("t1"), target, node)
        legs.append({"arm": l["arm"], "fold_s": l["fold_s"], "clock": q})
    plain = [l["fold_s"] for l in legs if l["arm"] == "plain" and l["clock"]["ok"]]
    brack = [l for l in d.get("legs", []) if l["arm"] == "bracket"]
    qual_brack = [l for l, q in zip(brack, [x for x in legs if x["arm"] == "bracket"])
                  if q["clock"]["ok"]]
    rows = []
    if qual_brack:
        # one qualified bracket fold carries the tree; more than one, take the median leg by wall
        pick = sorted(qual_brack, key=lambda l: l["fold_s"])[len(qual_brack) // 2]
        rows = pick["tree"]["rows"]
        notes = pick["tree"]["notes"]
    else:
        notes = ["no bracket leg qualified on the clock"]
    return {"legs": legs,
            "fold_s_plain_qualified": [round(x, 4) for x in plain],
            "fold_s_plain_median": round(st.median(plain), 4) if plain else None,
            "aa_floor_s": round(max(plain) - min(plain), 4) if len(plain) > 1 else None,
            "bracket_cost_ratio": d.get("bracket_cost_ratio"),
            "warm_fold_s": d.get("warm_fold_s"),
            "env": d.get("env"), "tree": rows, "notes": notes}


def ops_report(path: Path):
    op = gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")
    with op as f:
        return list(csv.DictReader(f))


def fnum(r, k):
    try:
        return float(r.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def dim(r, slot, ax):
    v = str(r.get("%s_%s_PAD[LOGICAL]" % (slot, ax), "") or "")
    n = ""
    for c in v.strip():
        if c.isdigit():
            n += c
        else:
            break
    return int(n) if n else 0


def rshape(r, slot):
    t = tuple(dim(r, slot, a) for a in "WZYX")
    return t if (t[2] and t[3]) else None


def is_fence(r, dim_=32):
    """The harness's fence: 3 x ttnn.exp on a 32x32 bf16 tile, a shape nothing in the fold uses."""
    s = rshape(r, "INPUT_0")
    return s is not None and s[2] == dim_ and s[3] == dim_ and s[0] <= 1 and s[1] <= 1


def window(rows, fence_n=3):
    """Rows strictly between the first and the second run of `fence_n` consecutive fence ops."""
    runs, i = [], 0
    while i < len(rows):
        if is_fence(rows[i]):
            j = i
            while j < len(rows) and is_fence(rows[j]):
                j += 1
            if j - i >= fence_n:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if len(runs) < 2:
        return None, {"fence_runs": len(runs), "why": "need two fence runs to window"}
    a, b = runs[0][1], runs[1][0]
    return rows[a:b], {"fence_runs": len(runs), "window": [a, b], "n_rows": b - a}


def align(opseq, rows):
    """Order-preserving map from ops-report rows to the python op name that dispatched them.

    A python ttnn call dispatches 0 or 1 programs in almost every case, so the alignment walks both
    sequences and consumes a python call per report row, skipping python calls that dispatch
    nothing. A row is only credited to a name when the name's recorded operand shapes contain the
    row's INPUT_0 shape; otherwise the python cursor advances. The fraction of rows credited is
    `align_quality`, and it is reported rather than assumed.
    """
    def norm(s):
        return tuple(int(x) for x in s) if s else None

    names, credited, k = [], 0, 0
    for r in rows:
        want = rshape(r, "INPUT_0")
        hit = None
        j = k
        while j < len(opseq) and j < k + 64:
            nm, ins, kwins = opseq[j]
            shapes = [norm(x) for x in (ins or []) if x] + [norm(x) for x in (kwins or []) if x]
            if want is not None and any(
                    s is not None and tuple(s[-2:]) == tuple(want[-2:]) for s in shapes):
                hit = nm
                k = j + 1
                break
            j += 1
        if hit is None:
            if k < len(opseq):
                hit = opseq[k][0]
                k += 1
        else:
            credited += 1
        names.append(hit or "?")
    return names, round(credited / max(len(rows), 1), 4)


def unit_device(run: Path, unit: str, target: int, node: int):
    """Per-op device kernel seconds for one profiled unit, per call of that unit."""
    j = run / "unit.json"
    d = json.loads(j.read_text())
    reps = d["env"]["reps"]
    csvs = sorted((run / "tracy").rglob("ops_perf_results*.csv"))
    if not csvs:
        return {"unit": unit, "error": "no ops_perf_results csv under %s" % (run / "tracy")}
    rows = ops_report(csvs[-1])
    win, wmeta = window(rows)
    if win is None:
        return {"unit": unit, "error": wmeta["why"], "meta": wmeta}
    clock = load_clock(run / "clock.jsonl")
    q = qualify(clock, d.get("t_region0"), d.get("t_region1"), target, node)
    by_code: dict = defaultdict(lambda: {"ns": 0.0, "n": 0})
    for r in win:
        e = by_code[r["OP CODE"]]
        e["ns"] += fnum(r, "DEVICE KERNEL DURATION [ns]")
        e["n"] += 1
    by_class: dict = defaultdict(lambda: {"ns": 0.0, "n": 0})
    quality = None
    seqp = run / "unit.opseq.json"
    if seqp.is_file():
        opseq = json.loads(seqp.read_text())
        # window the python sequence on its own fence marks, the same 32x32 exp triple
        marks = [i for i, (nm, ins, _k) in enumerate(opseq)
                 if nm == "exp" and ins and ins[0] and tuple(ins[0][-2:]) == (32, 32)]
        if len(marks) >= 6:
            lo, hi = marks[2] + 1, marks[3]
            opseq = opseq[lo:hi]
        names, quality = align(opseq, win)
        for nm, r in zip(names, win):
            e = by_class[nm]
            e["ns"] += fnum(r, "DEVICE KERNEL DURATION [ns]")
            e["n"] += 1
    tot_ns = sum(e["ns"] for e in by_code.values())
    return {"unit": unit, "reps": reps, "clock": q, "window": wmeta,
            "synced_wall_ms_per_call": d.get("synced_wall_ms_per_call"),
            "device_ms_per_call": round(tot_ns / 1e6 / reps, 5),
            "programs_per_call": round(len(win) / reps, 2),
            "align_quality": quality,
            "by_op_code": {k: {"ms_per_call": round(v["ns"] / 1e6 / reps, 5),
                               "n_per_call": round(v["n"] / reps, 2)}
                           for k, v in sorted(by_code.items(), key=lambda kv: -kv[1]["ns"])},
            "by_class": {k: {"ms_per_call": round(v["ns"] / 1e6 / reps, 5),
                             "n_per_call": round(v["n"] / reps, 2)}
                         for k, v in sorted(by_class.items(), key=lambda kv: -kv[1]["ns"])}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="runs/<name> of the counts session")
    ap.add_argument("--unit-run", type=Path, action="append", default=[],
                    help="runs/<name> of a profiled unit session; repeatable")
    ap.add_argument("--target", type=int, default=1350)
    ap.add_argument("--node", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    out = {"target_MHz": a.target, "node": a.node,
           "census": CENSUS, "fold_s_of_record": FOLD_S_OF_RECORD}
    out["spine"] = spine(a.run, a.target, a.node)
    out["units"] = [unit_device(p, p.name, a.target, a.node) for p in a.unit_run]

    sp = out["spine"]
    if sp:
        print("FOLD: plain qualified %s, median %s s, A/A floor %s s, bracket cost %sx"
              % (sp["fold_s_plain_qualified"], sp["fold_s_plain_median"], sp["aa_floor_s"],
                 sp["bracket_cost_ratio"]))
        print("%-58s %7s %10s %10s %10s" % ("path", "calls", "incl_s", "excl_s", "median_ms"))
        for r in sp["tree"][:30]:
            print("%-58s %7d %10.4f %10.4f %10.4f"
                  % (r["path"][-58:], r["calls"], r["incl_s"], r["excl_s"], r["median_ms"]))
        for n in sp["notes"]:
            print("  note:", n)
    for u in out["units"]:
        print("\nUNIT %s" % u["unit"])
        if "error" in u:
            print("  ERROR", u["error"])
            continue
        print("  synced wall %.4f ms/call, device %.4f ms/call, %s programs/call, clock %s"
              % (u["synced_wall_ms_per_call"], u["device_ms_per_call"],
                 u["programs_per_call"], "OK" if u["clock"]["ok"] else u["clock"].get("why")))
        print("  align_quality", u["align_quality"])
        for k, v in list(u["by_op_code"].items())[:14]:
            print("    %-36s %9.4f ms %8.1f n" % (k, v["ms_per_call"], v["n_per_call"]))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
