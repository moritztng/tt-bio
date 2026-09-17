#!/usr/bin/env python3
"""Turn one perf/hall800/runs/<name> directory into summary.json.

Pairs each fold's own runtime_s (results.json, which starts after model load) with the AICLK
actually observed while that fold ran. A fold time without its clock is not a measurement on
this card, so the summary refuses to report a mean if any sample inside the folding window
missed the 1350 MHz target.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CYCLE_HZ = 1e6  # MHz -> Hz


def rows(d: Path):
    out = []
    for p in sorted(d.glob("out/**/results.json")) + sorted(d.glob("out/results.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        for r in data if isinstance(data, list) else [data]:
            if isinstance(r, dict) and r.get("id"):
                out.append(r)
    seen, uniq = set(), []
    for r in out:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        uniq.append(r)
    return uniq


def clock(d: Path, node: str):
    p = d / ("clock_n%s.jsonl" % node)
    if not p.exists():
        return None
    mhz, errs, head = [], 0, {}
    for i, line in enumerate(p.read_text().splitlines()):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if i == 0 and "target" in o:
            head = o
            continue
        v = o.get("mhz")
        if isinstance(v, int):
            mhz.append(v)
        else:
            errs += 1
    if not mhz:
        return {"node": node, "samples": 0, "errors": errs, **head}
    tgt = head.get("target", 1350)
    return {"node": node, "force_status": head.get("force_status"), "target": tgt,
            "samples": len(mhz), "errors": errs, "min": min(mhz), "max": max(mhz),
            "mean": round(statistics.fmean(mhz), 1),
            "frac_at_target": round(sum(1 for v in mhz if v >= tgt) / len(mhz), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--cards", default="0")
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--wall", type=float, required=True)
    ap.add_argument("--rc", type=int, required=True)
    a = ap.parse_args()

    rs = rows(a.dir)
    ok = [r for r in rs if r.get("status") == "ok" and r.get("runtime_s") is not None]
    ts = [float(r["runtime_s"]) for r in ok]
    warm = ts[1:] if len(ts) > 1 else []
    clocks = {n: clock(a.dir, n) for n in a.cards.split(",")}
    mean_mhz = [c["mean"] for c in clocks.values() if c and c.get("mean")]
    s = {
        "run": a.dir.name, "cards": a.cards, "n_cards": len(a.cards.split(",")),
        "reps_requested": a.reps, "rc": a.rc, "wall_s": a.wall,
        "folds_ok": len(ok), "folds_seen": len(rs),
        "runtime_s_per_fold": ts,
        "cold_fold_s": ts[0] if ts else None,
        "warm_folds": warm,
        "warm_mean_s": round(statistics.fmean(warm), 2) if warm else None,
        "warm_min_s": min(warm) if warm else None,
        "warm_max_s": max(warm) if warm else None,
        "warm_sigma_pct": (round(100 * statistics.stdev(warm) / statistics.fmean(warm), 2)
                            if len(warm) > 1 else None),
        "clock": clocks,
        "plddt": [r.get("plddt") for r in ok],
        "ptm": [r.get("ptm") for r in ok],
        "iptm": [r.get("iptm") for r in ok],
        "errors": [r.get("error") for r in rs if r.get("status") != "ok"],
    }
    if warm and mean_mhz:
        mhz = statistics.fmean(mean_mhz)
        s["warm_mean_cycles"] = int(round(s["warm_mean_s"] * mhz * CYCLE_HZ))
        s["clock_mean_mhz"] = round(mhz, 1)
    if warm:
        # throughput per card, and the chip count the capacity question asks for
        per_card = s["warm_mean_s"] / s["n_cards"]
        s["fold_s_per_card_slot"] = round(per_card, 2)
        s["folds_per_day_per_chip"] = round(86400.0 / s["warm_mean_s"], 1)
        s["chips_for_20k_per_day"] = round(20000 * s["warm_mean_s"] / 86400.0, 2)
        s["chips_for_5k_per_day"] = round(5000 * s["warm_mean_s"] / 86400.0, 2)
    (a.dir / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
