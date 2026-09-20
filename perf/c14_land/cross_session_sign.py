#!/usr/bin/env python3
"""Direction-only sign test across every clean-provenance APB session.

WHY A SECOND STATISTIC, AND WHY IT IS NOT A SECOND BITE. The registered rule is a conjunction --
mean above the session's own A/A 2-SEM floor, AND paired permutation p < 0.05 -- and it returned
NULL in all three sessions. That verdict stands and this does not replace it. What it cannot
answer is the question the landing decision actually turns on, because the two sessions failed it
on DIFFERENT halves: session 2 cleared the permutation and not the floor, session 3 cleared the
floor and not the permutation, and session 3 failed only on n, at the arithmetic minimum p a
4-block design admits. Neither is evidence of no effect.

The magnitudes cannot be pooled: session 2's A/A floor is 0.3087 s against session 3's 0.0314 s,
a 10x difference in noise, and session 2 carries a 0.9247 s block whose own A/A arm moved
1.4875 s. Averaging those with session 3's would let the contaminated session dominate. Block
SIGN is the statistic that survives that, because it does not weigh a loud block more than a
quiet one.

STATED PLAINLY: this test was chosen after the block signs were visible, so it is EXPLORATORY and
its p-value is not a pre-registered one. It is reported beside the three NULLs, never instead of
them.

THE CONTROL IS THE LOAD-BEARING PART. Each block also yields an A/A difference -- its two base
folds against each other -- from the same folds, the same harness and the same ordering rule. A
sign test on those must NOT come back significant: under the null the base-to-base sign is
arbitrary. If it did, the bracket has a systematic within-block drift and the on-arm's sign
majority would be that drift rather than the lever.

  python3 perf/c14_land/cross_session_sign.py --out perf/c14_land/apb_cross_session.json
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Provenance verdicts recorded when each session was scored, not chosen here. Session 1 ran on a
# tree 21 files of tt_bio/ away from the arm it claimed, so it measured a different model and is
# excluded for that reason alone.
SESSIONS = [
    ("session2", "perf/c14_land/apb2_ab_scored.json", "CLEAR"),
    ("session3", "perf/c14_land/apb3_ab_scored_final.json", "CLEAR"),
]


def sign_test(xs):
    """Two-sided exact binomial on the count of positives, zeros dropped."""
    nz = [x for x in xs if x != 0]
    n, k = len(nz), sum(1 for x in nz if x > 0)
    if n == 0:
        return {"n": 0, "k_positive": 0, "p_two_sided": None}
    lo, hi = min(k, n - k), max(k, n - k)
    tail = sum(comb(n, i) for i in range(0, lo + 1)) + sum(comb(n, i) for i in range(hi, n + 1))
    return {"n": n, "k_positive": k, "p_two_sided": round(tail / 2 ** n, 6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = {"sessions": [], "note": __doc__.strip().splitlines()[0]}
    deltas, aa = [], []
    for name, path, prov in SESSIONS:
        d = json.loads((REPO / path).read_text())
        pb = d["per_block"]
        out["sessions"].append({
            "session": name, "provenance": prov, "blocks": len(pb),
            "delta_s": [round(b["delta_s"], 4) for b in pb],
            "aa_s": [round(b["aa_s"], 4) for b in pb],
            "registered_verdict": d["verdict"],
            "mean_delta_s": d.get("mean_delta_s"),
            "aa_floor_2sem_s": d.get("aa_floor_2sem_s"),
            "perm_p_two_sided": d.get("perm_p_two_sided"),
        })
        deltas += [b["delta_s"] for b in pb]
        aa += [b["aa_s"] for b in pb]

    out["on_arm_sign_test"] = sign_test(deltas)
    out["aa_control_sign_test"] = sign_test(aa)
    out["all_block_deltas_s"] = [round(x, 4) for x in deltas]
    out["median_block_delta_s"] = round(sorted(deltas)[len(deltas) // 2], 4)

    ctrl = out["aa_control_sign_test"]["p_two_sided"]
    on = out["on_arm_sign_test"]["p_two_sided"]
    out["control_clean"] = ctrl is not None and ctrl >= 0.05
    out["verdict"] = (
        "DIRECTION CONFIRMED (exploratory): block sign majority is not explained by "
        "within-block drift, because the A/A arm from the same folds is not significant"
        if out["control_clean"] and on is not None and on < 0.05 else
        "NO DIRECTION CLAIM" if out["control_clean"] else
        "INSTRUMENT SUSPECT: the A/A control is itself one-sided, so a sign majority on the "
        "on arm cannot be attributed to the lever")
    out["magnitude_claim"] = (
        "NONE. A sign test says the effect is positive; it says nothing about how large. The "
        "magnitude of record stays session 3's +0.0551 s, taken on the only board pair that was "
        "idle for every leg.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in (
        "on_arm_sign_test", "aa_control_sign_test", "median_block_delta_s",
        "control_clean", "verdict", "magnitude_claim")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
