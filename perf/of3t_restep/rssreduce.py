#!/usr/bin/env python3
"""Reduce an rssprofile JSONL to one phase table, and date it.

Written as its own file because the profile's process may not survive to write a summary:
the whole point of the JSONL is that it is flushed per sample and outlives an `os._exit`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GIB = 1024.0 ** 3


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def spans(samples):
    out, cur = [], None
    for r in samples:
        if cur is None or r["phase"] != cur["phase"]:
            if cur:
                out.append(cur)
            cur = {"phase": r["phase"], "t_in": r["t"], "rss_in_gib": r["rss_gib"],
                   "rss_max_gib": r["rss_gib"], "rss_min_gib": r["rss_gib"], "samples": 0}
        cur["t_out"] = r["t"]
        cur["rss_out_gib"] = r["rss_gib"]
        cur["rss_max_gib"] = max(cur["rss_max_gib"], r["rss_gib"])
        cur["rss_min_gib"] = min(cur["rss_min_gib"], r["rss_gib"])
        cur["sm_calls_at_exit"] = r["sm_calls"]
        cur["samples"] += 1
    if cur:
        out.append(cur)
    for s in out:
        s["dwell_s"] = round(s["t_out"] - s["t_in"], 2)
        s["delta_gib"] = round(s["rss_out_gib"] - s["rss_in_gib"], 3)
        s["sawtooth_gib"] = round(s["rss_max_gib"] - s["rss_min_gib"], 3)
    return out


def main():
    src = Path(sys.argv[1])
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    hdr = rows[0]
    foot = next((r for r in rows if r.get("footer")), None)
    samples = [r for r in rows if "t" in r]
    breach = next((r for r in samples if "BREACH" in r), None)
    ph = spans(samples)
    peak = max(s["rss_max_gib"] for s in ph)

    d = {
        "what": "of3t-restep: where an exactness-ON OF3T training step's host memory goes, "
                "phase by phase. R216 asked whether the exact softmax's float64 temporary is "
                "the whole 19.0 GiB. It is not, and it is not even the largest term.",
        "row": "of3t-restep",
        "axis": "phase-to-phase inside ONE process on one board; host RSS, not device DRAM. "
                "Memory is load-insensitive, so a contended box does not invalidate it -- but "
                "no second in this file is quotable as a timing, because host_quiet was RED.",
        "host": "pc",
        "card": 0,
        "board_class": "p150a, custom 130-core firmware",
        "env": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "mem_total_gib": hdr["mem_total_gib"],
            "swap": "none",
            "avail_at_start_gib": hdr["avail_at_start_gib"],
            "started_utc": hdr["started_utc"],
            "sample_period_s": hdr["period_s"],
            "self_kill_floor_gib": hdr["floor_gib"],
        },
        "config": {
            "crop": None, "cycles": None, "diffusion_samples": None, "reps": None,
            "exact_training": None, "argv": hdr["argv"],
        },
        "completed": foot is not None,
        "breached": breach is not None,
        "breach": None if breach is None else {
            "t_s": breach["t"], "phase": breach["phase"], "rss_gib": breach["rss_gib"],
            "avail_gib": breach["avail_gib"], "note": breach["BREACH"]},
        "peak_rss_gib": peak,
        "peak_is_a_lower_bound": breach is not None,
        "phases": ph,
    }
    a = hdr["argv"]
    for k, flag in (("crop", "--tokens"), ("cycles", "--cycles"),
                    ("diffusion_samples", "--samples"), ("reps", "--reps")):
        if flag in a:
            d["config"][k] = int(a[a.index(flag) + 1])
    if "--exact" in a:
        d["config"]["exact_training"] = a[a.index("--exact") + 1]
    if foot:
        d["footer"] = {k: foot[k] for k in ("rc", "wall_s", "vmhwm_gib", "softmax",
                                            "exact_training_ops") if k in foot}
    out = src.with_suffix(".summary.json")
    out.write_text(json.dumps(d, indent=1))
    print(f"-> {out}")
    print(f"{'phase':20s} {'dwell_s':>8s} {'rss_in':>8s} {'rss_out':>8s} {'delta':>8s} {'max':>8s}")
    for s in ph:
        print(f"{s['phase']:20s} {s['dwell_s']:8.2f} {s['rss_in_gib']:8.3f} "
              f"{s['rss_out_gib']:8.3f} {s['delta_gib']:+8.3f} {s['rss_max_gib']:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
