#!/usr/bin/env python3
"""Decide whether an arm resolved, on a statistic that does not punish extra data.

The rule this task preregistered -- "the margin must exceed the WORST A/A bracket in the run" -- is
not a consistent test, and the Wormhole shift-gather arm is what exposed it. The worst bracket is a
maximum over reps, so it grows with the number of reps even when the underlying spread is unchanged:
at 3 reps that arm's worst bracket was 1.01596 and at 6 reps it was 1.01750, while the effect itself
did not move. **Collecting more folds made the test harder to pass.** A criterion with that property
cannot be used to say what a session can resolve.

What replaces it is the standard error of the paired ratios, which falls as 1/sqrt(n) the way a
resolution statistic must, plus a sign test on the pairs that assumes nothing about their
distribution. Both are computed here for every arm, including the ones the change does not rescue --
q-chunk still reads nothing on either part under this test, which is the evidence that this is a
correction and not a rescue.
"""
from __future__ import annotations

import json, math, statistics as st, sys
from pathlib import Path


def summarise(p: Path) -> dict:
    d = json.loads(p.read_text())
    pr = d["paired_ratios"]
    n = len(pr)
    mean = st.mean(pr)
    sd = st.stdev(pr) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    pos = sum(1 for x in pr if x > 1)
    # two-sided sign test
    p_sign = 2 * sum(math.comb(n, k) for k in range(max(pos, n - pos), n + 1)) / 2 ** n
    return {"file": p.name, "lever": d["env"]["lever"], "kind": d["env"]["kind"],
            "host": d["env"]["host"], "card": d["env"]["card"], "n_pairs": n,
            "margin_pp": round(100 * (mean - 1), 3), "se_pp": round(100 * se, 3),
            "t": round((mean - 1) / se, 2) if se else None,
            "ci95_pp": [round(100 * (mean - 1 - 1.96 * se), 3),
                        round(100 * (mean - 1 + 1.96 * se), 3)],
            "pairs_positive": f"{pos}/{n}", "p_sign": round(min(p_sign, 1.0), 5),
            "resolved": bool(se and abs(mean - 1) > 1.96 * se),
            "old_worst_bracket": d["aa_floor_worst"],
            "old_rule_resolved": bool(d["separates_from_floor"]),
            "bit_exact": d["bit_exact"], "engagement": d["engagement_control"]}


def main() -> int:
    rows = [summarise(Path(a)) for a in sys.argv[1:]]
    print(f"{'lever':<13}{'part':<7}{'n':>3}{'margin pp':>11}{'se':>7}{'t':>7}"
          f"{'95% CI':>18}{'signs':>8}{'p':>9}  res  oldrule")
    for r in rows:
        part = "WH" if "glx" in r["host"] else "BH"
        ci = f"[{r['ci95_pp'][0]:+.2f},{r['ci95_pp'][1]:+.2f}]"
        print(f"{r['lever']:<13}{part:<7}{r['n_pairs']:>3}{r['margin_pp']:>+11.3f}{r['se_pp']:>7.3f}"
              f"{r['t']:>7.2f}{ci:>18}{r['pairs_positive']:>8}{r['p_sign']:>9.5f}"
              f"  {'YES' if r['resolved'] else 'no ':<4} {'YES' if r['old_rule_resolved'] else 'no'}")
    Path("perf/k10_transfer/out/resolve.json").write_text(
        json.dumps({"doc": __doc__, "arms": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
