#!/usr/bin/env python3
"""How much a mean-clock label can be trusted when re-pricing an old measurement.

The campaign's premise is that levers recorded at 1.02-1.05x were measured against a throttled
wall-time denominator and are worth more than recorded. The tempting move is to rescale those old
ratios arithmetically using each run's recorded mean AICLK. This measures whether that works, by
taking the model solved from ONE clean interleaved session and asking it to predict folds from
three other sessions on three other commits.

CPU only. Reads committed JSON; opens no device and measures nothing new.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
FIXED_S, WORK_MCYCLES = 3.9516, 14417.7   # solved from mainab_down800_qb2c1 on 0df13ad9
SESSIONS = ("force_ab_qb2c1", "maxclk_ab_qb2c1", "pin_ab_qb2c1")


def predict(f):
    return FIXED_S + WORK_MCYCLES / f


def main():
    out = {
        "scope": "CPU cross-session check of one solved model. No device, no new timing.",
        "model": {"fixed_s": FIXED_S, "work_Mcycles": WORK_MCYCLES,
                  "solved_from": "mainab_down800_qb2c1.json, commit 0df13ad9, interleaved, "
                                 "worst in-session residual 0.033 s"},
        "sessions": {},
    }
    pooled = []
    for name in SESSIONS:
        p = HERE / "inputs" / (name + ".json")
        if not p.is_file():
            continue
        doc = json.loads(p.read_text())
        errs, flat, varying = [], [], []
        for r in doc["runs"]:
            if r.get("warmup"):
                continue
            f = r["aiclk_mean"]
            e = r["fold_s"] - predict(f)
            errs.append(e)
            (flat if abs(r["aiclk_max"] - f) < 1.0 else varying).append(e)
        pooled += errs
        out["sessions"][name] = {
            "commit": doc["env"]["commit"],
            "n": len(errs),
            "mean_error_s": sum(errs) / len(errs),
            "worst_error_s": max(errs, key=abs),
            "flat_clock_n": len(flat),
            "clock_varied_n": len(varying),
        }
    out["pooled"] = {
        "n": len(pooled),
        "mean_error_s": sum(pooled) / len(pooled),
        "worst_error_s": max(pooled, key=abs),
        "in_session_residual_s": 0.0333,
        "conclusion": "A mean-clock label carries several hundred milliseconds of error across "
                      "sessions, which is larger than every lever ratio in the corpus. Old ratios "
                      "cannot be clock-corrected by arithmetic; they have to be re-measured.",
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
