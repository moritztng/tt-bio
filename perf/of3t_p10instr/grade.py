#!/usr/bin/env python3
"""The pre-registered grade (PREREGISTRATION.md), computed from evalnoise.py artifacts.

    grade.py noise out/N3a.json out/N3b.json
    grade.py ab --history out/N3a.json,out/N3b.json \
        --design two=out/AB2.json:TF7:B2trunk:I_TFs1 \
        --design twelve=out/AB12.json:I_D12:X12b:I_D12s1 --out out/VERDICT.json

`--draws N` grades eval seeds 0..N-1 and `--bar design=T` holds a design to a fixed bar instead
of F/3 (PREREG-128.md). H is always taken on the seeds every run of a weights shares.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

GRADED = list(range(32))          # eval seeds 0..31; 20260926 is reported, never graded
BAR_RATIO = 1 / 3                 # 0.60 A structure bar over its 1.84 A seed floor


def _calls(path):
    """`path` is one artifact, or several joined by `+` (one checkpoint per process)."""
    if "+" in path:
        parts = [_calls(p) for p in path.split("+")]
        by = {}
        for _, b in parts:
            for lab, rows in b.items():
                if lab in by:
                    raise SystemExit(f"{lab} appears twice in {path}")
                by[lab] = rows
        return {"head": ",".join(sorted({r.get("head") for r, _ in parts})),
                "dirty": any(r.get("dirty") for r, _ in parts),
                "aiclk_line": " | ".join(r.get("aiclk_line", "") for r, _ in parts)}, by
    r = json.loads(Path(path).read_text())
    if r.get("trimul", {}).get("host_in_proj"):
        raise SystemExit(f"{path}: {r['trimul']} -- a trimul cut its in-projection from the "
                         "start checkpoint's host copy, so this artifact did not score its weights")
    if not r.get("ok"):
        raise SystemExit(f"{path}: run did not finish ok ({r.get('error')})")
    by = {}
    for c in r["calls"]:
        if c["rep"] == 0 and c["context"] == "shipped":
            by.setdefault(c["label"], {})[c["seed"]] = {t: v["loss"]
                                                        for t, v in c["targets"].items()}
    return r, by


def _score(seeds_rows, seeds=None):
    """M(X): mean over the four targets of the mean over graded seeds."""
    seeds = GRADED if seeds is None else seeds
    targets = sorted(next(iter(seeds_rows.values())))
    return st.mean(st.mean(seeds_rows[s][t] for s in seeds) for t in targets)


def _modes(xs, gap=1.0):
    """Clusters of sorted values separated by more than `gap`."""
    xs = sorted(xs)
    out, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] > gap:
            out.append(cur)
            cur = []
        cur.append(x)
    out.append(cur)
    return [{"n": len(c), "mean": st.mean(c), "min": c[0], "max": c[-1]} for c in out]


def history(paths):
    """H: the largest |M| difference between two processes scoring the same weights."""
    runs = [_calls(p)[1] for p in paths]
    diffs = {}
    for lab in sorted({lab for r in runs for lab in r}):
        have = [r[lab] for r in runs if lab in r]
        seeds = [s for s in GRADED if all(s in h for h in have)]
        ms = [_score(h, seeds) for h in have]
        if len(ms) > 1:
            diffs[lab] = max(ms) - min(ms)
    return max(diffs.values()), diffs


def noise(a):
    out = {}
    for p in a.paths:
        rec, by = _calls(p)
        for lab, rows in by.items():
            tgt = {}
            for t in sorted(next(iter(rows.values()))):
                xs = [rows[s][t] for s in GRADED]
                tgt[t] = {"mean": st.mean(xs), "median": st.median(xs), "sd": st.stdev(xs),
                          "se": st.stdev(xs) / math.sqrt(len(xs)), "min": min(xs),
                          "max": max(xs), "modes": _modes(xs),
                          "old_seed_20260926": rows.get(20260926, {}).get(t)}
            out.setdefault(Path(p).stem, {})[lab] = {"M": _score(rows), "targets": tgt,
                                                     "aiclk": rec.get("aiclk_line")}
    H, hd = history(a.paths)
    out["history"] = {"H": H, "per_weights": hd}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


def _delta(by, a, b):
    targets = sorted(next(iter(by[a].values())))
    d = [st.mean(by[a][s][t] - by[b][s][t] for t in targets) for s in GRADED]
    return st.mean(d), 1.96 * st.stdev(d) / math.sqrt(len(d)), d


def ab(a):
    global GRADED
    GRADED = list(range(a.draws))
    bars = dict((k, float(v)) for k, v in (b.split("=") for b in a.bar))
    H, hd = history(a.history.split(","))
    res = {"H": H, "H_per_weights": hd, "draws": a.draws, "bar_ratio": BAR_RATIO,
           "fixed_bars": bars, "designs": {}}
    for spec in a.design:
        name, rest = spec.split("=", 1)
        path, arm_a, arm_b, floor_b = rest.split(":")
        rec, by = _calls(path)
        delta, se_h, d = _delta(by, arm_a, arm_b)
        fdelta, fse_h, _ = _delta(by, arm_a, floor_b)
        h, hf = se_h + H, fse_h + H
        F = abs(fdelta)
        T = bars.get(name, BAR_RATIO * F)
        if F <= hf:
            grade = "UNRESOLVED"
            why = f"floor F {F:.4f} is inside its own half-width {hf:.4f}"
        elif abs(delta) + h <= T:
            grade, why = "PASS", f"|Delta| + h = {abs(delta) + h:.4f} <= T {T:.4f}"
        elif abs(delta) - h > T:
            grade, why = "FAIL", f"|Delta| - h = {abs(delta) - h:.4f} > T {T:.4f}"
        else:
            grade, why = "UNRESOLVED", f"|Delta| {abs(delta):.4f} +- h {h:.4f} straddles T {T:.4f}"
        # Draws at which this spread would resolve, if Delta and sd(d_s) hold.
        room = (T - abs(delta) - H) if abs(delta) < T else (abs(delta) - T - H)
        need = math.ceil((1.96 * st.stdev(d) / room) ** 2) if room > 0 else None
        res["designs"][name] = {
            "artifact": path, "arm_a": arm_a, "arm_b": arm_b, "floor_b": floor_b,
            "M": {lab: _score(by[lab]) for lab in by},
            "delta": delta, "h": h, "se_half": se_h, "F": F, "F_signed": fdelta, "h_F": hf,
            "T": T, "grade": grade, "why": why, "sd_d": st.stdev(d), "draws_to_resolve": need, "aiclk": rec.get("aiclk_line"),
            "head": rec.get("head"), "dirty": rec.get("dirty")}
    g = {k: v["grade"] for k, v in res["designs"].items()}
    if all(x == "PASS" for x in g.values()):
        res["verdict"] = "GO"
    elif g.get("twelve") == "FAIL":
        res["verdict"] = "NO-GO"
    else:
        res["verdict"] = "PARTIAL"
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"VERDICT: {res['verdict']}  " + "  ".join(f"{k} {v}" for k, v in g.items()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("noise")
    n.add_argument("paths", nargs="+")
    n.add_argument("--out")
    b = sub.add_parser("ab")
    b.add_argument("--history", required=True, help="comma-separated noise artifacts")
    b.add_argument("--design", action="append", required=True,
                   help="name=artifact.json:ARM_A:ARM_B:FLOOR_B")
    b.add_argument("--draws", type=int, default=32, help="grade eval seeds 0..N-1")
    b.add_argument("--bar", action="append", default=[], help="design=T, a fixed bar")
    b.add_argument("--out")
    a = ap.parse_args()
    return noise(a) if a.cmd == "noise" else ab(a)


if __name__ == "__main__":
    raise SystemExit(main())
