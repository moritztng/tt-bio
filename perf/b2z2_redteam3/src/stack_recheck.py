#!/usr/bin/env python3
"""Re-derive the campaign's headline stack ratio from the raw folds, not from the row's summary.

Host only. Reads `perf/b2z2_union/out/timing512_step_qb2c1.json` as committed on
`wk/b2z2-bh-union-step` and recomputes 1.12862x from the 65 warm folds behind it.

Why this exists: this campaign has had three headline figures fail re-derivation after the fact (a
denominator retired twice, a shard ratio that divided by the cell instead of its own benchlocked
base, and a step ratio quoted as a fold ratio). The parent three-lever union was re-derived from its
own folds by `union_recheck.py` and held. The five-lever stack never was, and it is the number every
summary of wave 2 now leads with.

    python3 perf/b2z2_orch/stack_recheck.py [--json out.json] [--runs path]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = "perf/b2z2_union/out/timing512_step_qb2c1.json"
REF = "origin/wk/b2z2-bh-union-step"
CLAIMED_PAIRED = 1.12862
CLAIMED_GLOBAL = 1.12290
CLAIMED_AA = 1.01162
PUBLISHED_DIGEST = "a91aa44441f0d9c5"


def load(path: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text())
    out = subprocess.run(["git", "show", f"{REF}:{SRC}"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"could not read {REF}:{SRC} -- {out.stderr.strip()}")
    return json.loads(out.stdout)


def ratios(runs: list[dict], arm: str, base_of) -> list[float]:
    return [base_of(r) / r["fold_s"] for r in runs if r["arm"] == arm if base_of(r)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--runs", help="read the timing json from a file instead of git")
    a = ap.parse_args()

    d = load(a.runs)
    warm = [r for r in d["runs"] if not r.get("cold")]
    base = [r for r in warm if r["arm"] == "base"]
    stack = [r for r in warm if r["arm"] == "STACK"]

    # Per rep, the base folds by position -- the arms sit between them.
    bypos = collections.defaultdict(dict)
    for r in base:
        bypos[r["rep"]][r["pos"]] = r["fold_s"]

    def adjacent(r):
        """The base fold immediately before this arm in its own rep -- the row's own pairing."""
        cands = [p for p in bypos[r["rep"]] if p < r["pos"]]
        return bypos[r["rep"]][max(cands)] if cands else None

    def rep_median(r):
        return st.median(list(bypos[r["rep"]].values()))

    def global_median(_r):
        return st.median([x["fold_s"] for x in base])

    schemes = {
        "adjacent base in own rep (the row's method)": adjacent,
        "median base of own rep": rep_median,
        "global median base": global_median,
    }
    rows = []
    for name, f in schemes.items():
        rs = ratios(warm, "STACK", f)
        rows.append({"pairing": name, "n": len(rs), "median": st.median(rs),
                     "mean": st.fmean(rs), "min": min(rs), "max": max(rs)})

    # A/A floor, ESTIMATOR-MATCHED. The headline is a median of 10 paired ratios, so its floor has
    # to be the spread of that same estimator under the null, not the spread of one fold against
    # one fold. Computing the worst single base-vs-base pair instead gives ~1.10x on this box and
    # would "refute" a claim it does not address -- a 12.4 % base spread at loadavg 3.95-17.5 makes
    # single pairs wild while their median is stable. Take every base-vs-its-own-predecessor ratio,
    # then look at the distribution of the median of 10 of them.
    import itertools
    import random

    aa_pairs = []
    for r in base:
        b = adjacent(r)
        if b:
            aa_pairs.append(b / r["fold_s"])
    rng = random.Random(0)
    meds = sorted(st.median(rng.sample(aa_pairs, 10)) for _ in range(20000))
    aa_floor = max(meds[int(0.975 * len(meds))], 1.0 / meds[int(0.025 * len(meds))])
    aa_worst_single = max(max(x, 1.0 / x) for x in aa_pairs)

    sep = min(x["fold_s"] for x in base) > max(x["fold_s"] for x in stack)

    print(f"source            {REF}:{SRC}")
    print(f"warm folds        {len(warm)}  (base {len(base)}, STACK {len(stack)})")
    print(f"base digest       {sorted({r['sha256'] for r in base})}  "
          f"{'== published' if {r['sha256'] for r in base} == {PUBLISHED_DIGEST} else '!! NOT the published digest'}")
    print(f"STACK flags       {stack[0]['flags']}")
    print()
    print(f"  {'pairing':<44} {'n':>3} {'median':>9} {'mean':>9} {'min':>8} {'max':>8}")
    for r in rows:
        print(f"  {r['pairing']:<44} {r['n']:>3} {r['median']:>8.5f}x {r['mean']:>8.5f}x "
              f"{r['min']:>7.4f}x {r['max']:>7.4f}x")
    print()
    print(f"A/A floor, ESTIMATOR-MATCHED (95 % band of the median of 10 base-vs-base pairs): "
          f"{aa_floor:.5f}x")
    print(f"  row claimed {CLAIMED_AA:.5f}x. For contrast the worst SINGLE base-vs-base pair is "
          f"{aa_worst_single:.5f}x,")
    print(f"  which is the wrong statistic for a median-of-10 headline and would 'refute' a claim "
          f"it does not address.")
    print(f"distributions separate (slowest STACK beats fastest base)? {sep}")
    print()
    # Every arm in the session against the corrected floor. The point is not the headline, which
    # clears it comfortably -- it is which of the session's SMALLER claims survive a floor that is
    # 2.5x the one quoted beside them.
    print()
    print(f"  {'arm':<8} {'n':>3} {'paired median':>14}  readable against {aa_floor:.5f}x ?")
    arm_rows = []
    for arm in sorted({r["arm"] for r in warm} - {"base"}):
        rs = ratios(warm, arm, adjacent)
        m = st.median(rs)
        ok = m > aa_floor
        arm_rows.append({"arm": arm, "n": len(rs), "paired_median": m, "readable": ok})
        print(f"  {arm:<8} {len(rs):>3} {m:>13.5f}x  {'yes' if ok else 'NO -- inside the floor'}")

    head = rows[0]["median"]
    print(f"CLAIMED paired  {CLAIMED_PAIRED:.5f}x")
    print(f"RE-DERIVED      {head:.5f}x   -> {(head/CLAIMED_PAIRED-1)*100:+.2f} %")
    print(f"CLAIMED global  {CLAIMED_GLOBAL:.5f}x")
    gm = st.median([x["fold_s"] for x in base]) / st.median([x["fold_s"] for x in stack])
    print(f"RE-DERIVED      {gm:.5f}x   -> {(gm/CLAIMED_GLOBAL-1)*100:+.2f} %")
    spread = max(r["median"] for r in rows) / min(r["median"] for r in rows)
    print()
    print(f"SENSITIVITY to the pairing choice: {spread:.5f}x across the three schemes "
          f"({(spread-1)*100:.2f} %), against an A/A floor of {aa_floor:.5f}x.")

    out = {"claimed_paired": CLAIMED_PAIRED, "claimed_global": CLAIMED_GLOBAL,
           "rederived_global": gm, "aa_floor_estimator_matched": aa_floor,
           "aa_worst_single_pair": aa_worst_single,
           "distributions_separate": sep, "pairing_sensitivity": spread, "schemes": rows, "arms": arm_rows,
           "base_digest_is_published": sorted({r["sha256"] for r in base}) == [PUBLISHED_DIGEST]}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
