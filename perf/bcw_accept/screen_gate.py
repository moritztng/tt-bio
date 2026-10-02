"""Where the two arms diverge is the SCREEN gate, not the accept gate.

Arm B reads 0 accepted of 11, but it also reads 0 candidates SCORED: 9 of its 11 trajectories
never reached MPNN at all. The RULE's 17:16Z addendum pre-committed to separating those two
zeros, and this is the measurement that does it.

The screen gate is one threshold: `min_plddt_screen = 0.60` on `plddt_metric` bound to the
BINDER chain in the complex (bindcraft/filters.py:826-830, `rank.py` calls it "binder confidence
in the complex"). There is no `min_iptm_screen` in DEFAULT_SETTINGS, so i_pTM does not gate at
screen at all -- arm A passed a trajectory at i_pTM 0.20 and rejected one at i_pTM 0.87.

That matters for reading this: the metric averages over the SAME 150 binder residues in both
arms. It is not a complex-average that a 614-residue receptor could drag down by arithmetic, so
the arms are comparable on it directly.

Two denominators, deliberately kept apart:
  * passed-screen is taken from the trajectory CSVs, the same authority report.py denominates on,
    and is COMPLETE for both arms (n = 10 and n = 11).
  * the pLDDT readings are parsed from the arm logs, and arm A is MISSING 3 of its 10: campaign.sh
    truncates the log on relaunch, so the first sitting's lines went when armA1 resumed. Quoted
    with the n actually in hand, never backfilled.
"""
import argparse
import csv
import re
import sys
from itertools import combinations
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out"
MIN_PLDDT_SCREEN = 0.60

SCREEN = re.compile(
    r"(passed|rejected at) screen design stage\s+i_pTM=([0-9.]+)\s+pLDDT=([0-9.]+)")


def terminated(tag):
    """Every resolved trajectory's terminating stage, from the table report.py denominates on."""
    table = OUT / tag / "1_Trajectories" / "!_Trajectories.csv"
    if not table.exists():
        return []
    with open(table, newline="") as f:
        return [row.get("terminated") or "completed" for row in csv.DictReader(f)]


def screen_readings(tag):
    """(passed, i_pTM, pLDDT) per screen-gate evaluation found in the arm log."""
    log = OUT / f"{tag}.log"
    if not log.exists():
        return []
    return [(verb == "passed", float(iptm), float(plddt))
            for verb, iptm, plddt in SCREEN.findall(log.read_text(errors="replace"))]


def fisher_one_sided(a, b, c, d):
    """P(arm A reaches a or more | the arms share a rate): the upper hypergeometric tail.

    Same body as power.py's, which `power.py --selftest` pins against scipy's
    alternative="greater". Imported rather than copied would be better; it is 4 lines and
    power.py owns the pinned one, so this calls it.
    """
    from power import fisher_one_sided as pinned
    return pinned(a, b, c, d)


def permutation_p(xs, ys, trials_cap=200000):
    """Exact one-sided permutation test on the mean gap, xs > ys.

    Exhaustive over C(n, len(xs)) relabellings when that is small enough, which it is here
    (C(18, 7) = 31824). No scipy, no normal approximation, no tie correction to get wrong.
    """
    pool = list(xs) + list(ys)
    n, k = len(pool), len(xs)
    total = comb(n, k)
    if total > trials_cap:
        return None, total
    observed = sum(xs) / len(xs) - sum(ys) / len(ys)
    hits = 0
    for pick in combinations(range(n), k):
        left = sum(pool[i] for i in pick)
        gap = left / k - (sum(pool) - left) / (n - k)
        if gap >= observed - 1e-12:
            hits += 1
    return hits / total, total


def summarise(name, tags):
    stages = [s for t in tags for s in terminated(t)]
    reads = [r for t in tags for r in screen_readings(t)]
    n = len(stages)
    at_screen = sum(1 for s in stages if s == "screen")
    passed = n - at_screen
    plddt = sorted(p for _, _, p in reads)
    print(f"=== {name}: pooled over {', '.join(tags)}")
    print(f"    n = {n} RESOLVED, passed screen {passed}, terminated at screen {at_screen}")
    print(f"    terminating stages: {dict(sorted((s, stages.count(s)) for s in set(stages)))}")
    if reads:
        mid = plddt[len(plddt) // 2] if len(plddt) % 2 else (plddt[len(plddt) // 2 - 1] + plddt[len(plddt) // 2]) / 2
        print(f"    screen-gate binder pLDDT, {len(reads)} of {n} readings in the log: {plddt}")
        print(f"      median {mid:.3f}, min {plddt[0]:.2f}, max {plddt[-1]:.2f}, gate {MIN_PLDDT_SCREEN}")
        if len(reads) < n:
            print(f"      {n - len(reads)} reading(s) MISSING: log truncated on relaunch, not backfilled")
    return n, passed, [p for _, _, p in reads], [i for _, i, _ in reads]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arma", default="armA1")
    ap.add_argument("--armb", default="armB1,armB2,armB3,armB4")
    args = ap.parse_args()

    na, pa, plddt_a, iptm_a = summarise("ARM A (~352 tokens, domain III crop)",
                                        args.arma.split(","))
    print()
    nb, pb, plddt_b, iptm_b = summarise("ARM B (800 tokens, full ectodomain)",
                                        args.armb.split(","))

    print()
    print("--- passed the screen gate, complete n from the trajectory tables")
    print(f"    arm A {pa}/{na} = {pa / na:.3f}   arm B {pb}/{nb} = {pb / nb:.3f}")
    p = fisher_one_sided(pa, na - pa, pb, nb - pb)
    print(f"    Fisher one-sided, arm A passes screen more often than arm B: p = {p:.5f}")

    if plddt_a and plddt_b:
        print()
        print("--- screen-gate binder pLDDT, the one number the gate reads")
        perm, total = permutation_p(plddt_a, plddt_b)
        ga = sum(plddt_a) / len(plddt_a)
        gb = sum(plddt_b) / len(plddt_b)
        print(f"    mean arm A {ga:.3f} (n={len(plddt_a)})  arm B {gb:.3f} (n={len(plddt_b)})"
              f"  gap {ga - gb:+.3f}")
        if perm is not None:
            print(f"    exact permutation, one-sided, arm A higher: p = {perm:.5f}"
                  f"  over all {total} relabellings")
        print(f"    arm A above the {MIN_PLDDT_SCREEN} gate: "
              f"{sum(1 for x in plddt_a if x >= MIN_PLDDT_SCREEN)}/{len(plddt_a)}   "
              f"arm B: {sum(1 for x in plddt_b if x >= MIN_PLDDT_SCREEN)}/{len(plddt_b)}")

    if iptm_a and iptm_b:
        print()
        print("--- screen-stage i_pTM, which does NOT gate at screen (no min_iptm_screen)")
        ia = sum(iptm_a) / len(iptm_a)
        ib = sum(iptm_b) / len(iptm_b)
        perm, _ = permutation_p(iptm_a, iptm_b)
        print(f"    mean arm A {ia:.3f}  arm B {ib:.3f}  gap {ia - ib:+.3f}"
              + (f"   permutation p = {perm:.5f}" if perm is not None else ""))
        print("    quoted because it separates 'the interface is worse' from 'the binder is")
        print("    placed less confidently'. Only the second one is what the gate cuts on.")


if __name__ == "__main__":
    main()
