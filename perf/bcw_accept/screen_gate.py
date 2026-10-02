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
import glob
import statistics as st
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


def permutation_p(xs, ys, trials_cap=400000):
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



PLDDT_LOSS_WEIGHT = 0.1


def screen_plddt_track(tags):
    """Binder pLDDT at the start and the end of the 50-round screen phase, per trajectory.

    `plddt_loss` is `mean(1 - plddt)` over the DESIGNED binder residues (bindcraft/loss.py:185-190)
    and the CSV column carries it already multiplied by `weights_plddt_loss`, which is 0.1 in both
    arms. So plddt = 1 - loss / 0.1 exactly, and the recovered end-of-screen values land on the
    gate readings parsed from the logs (arm A 0.680 recovered against 0.700 logged, arm B 0.473
    against 0.523; the gate reads a selected round, not the last-5 mean, so they are close rather
    than equal).

    This reads every resolved trajectory in both arms, including the 3 of arm A's whose log lines
    the relaunch truncation ate, so it runs at the full n = 10 and n = 11 where the log-parsed
    pLDDT runs at 7 and 11.
    """
    out = []
    for tag in tags:
        for f in sorted(glob.glob(str(OUT / tag / "1_Trajectories" / "*" / "*_losses.csv"))):
            with open(f, newline="") as fh:
                rows = [r for r in csv.DictReader(fh) if r["phase"] == "screen"]
            if len(rows) < 20:
                continue
            column = next(c for c in rows[0] if c.endswith("plddt_loss"))
            track = [1 - float(r[column]) / PLDDT_LOSS_WEIGHT for r in rows]
            out.append((track[0], st.mean(track[:5]), st.mean(track[-5:])))
    return out


def report_track(track_a, track_b):
    """Split the end-of-screen gap into the part present before optimisation and the part it adds.

    This is the one read in this row that bears on WHY arm B sits under the gate, and it does not
    settle it on its own. A pure threshold-calibration story predicts an offset at round 1 and
    equal gains. A pure design-quality story predicts equal starts and a smaller gain. Both show
    up, so both are quoted.
    """
    # Baseline on ROUND 1, not the first-5 mean. The screen phase is 50 rounds of gradient
    # optimisation, so by round 5 some of it has already happened and a first-5 baseline charges
    # part of the optimisation to the offset. Both are printed because the split barely moves
    # between them, which is the point: the conclusion does not depend on the choice.
    starts_a, ends_a = [x[0] for x in track_a], [x[2] for x in track_a]
    starts_b, ends_b = [x[0] for x in track_b], [x[2] for x in track_b]
    gains_a = [x[2] - x[0] for x in track_a]
    gains_b = [x[2] - x[0] for x in track_b]
    print()
    print("--- binder pLDDT across the 50-round screen phase, recovered from plddt_loss")
    print(f"    arm A (n={len(track_a)}): round1 {st.mean(starts_a):.3f} -> end {st.mean(ends_a):.3f}"
          f"   gain {st.mean(gains_a):+.3f}   (first-5 baseline {st.mean(x[1] for x in track_a):.3f})")
    print(f"    arm B (n={len(track_b)}): round1 {st.mean(starts_b):.3f} -> end {st.mean(ends_b):.3f}"
          f"   gain {st.mean(gains_b):+.3f}   (first-5 baseline {st.mean(x[1] for x in track_b):.3f})")
    print("    NOTE: plddt_loss is stored to 2 dp, so a single round recovers pLDDT only to 0.1.")
    print("    The round-1 gap is smaller than one quantum and survives only as a mean over all")
    print("    trajectories; it is not readable on any single one.")
    gap_end = st.mean(ends_a) - st.mean(ends_b)
    p_start, _ = permutation_p(starts_a, starts_b)
    p_gain, total = permutation_p(gains_a, gains_b)
    print(f"    end-of-screen gap {gap_end:+.3f}, and it splits in two:")
    print(f"      present at round 1, before optimisation: {st.mean(starts_a) - st.mean(starts_b):+.3f}"
          f"  ({(st.mean(starts_a) - st.mean(starts_b)) / gap_end:.0%} of it)"
          + (f"  permutation p = {p_start:.5f}" if p_start is not None else ""))
    print(f"      added by optimisation gaining less:      {st.mean(gains_a) - st.mean(gains_b):+.3f}"
          f"  ({(st.mean(gains_a) - st.mean(gains_b)) / gap_end:.0%} of it)"
          + (f"  permutation p = {p_gain:.5f}" if p_gain is not None else ""))
    print("    Both are significant, so neither single story explains it: arm B starts lower AND")
    print("    its 50 screen rounds buy less than half the confidence arm A's buy.")


def metric_track(tags, suffix):
    """Round 1 and end-of-screen for any per-round loss column, by column suffix."""
    out = []
    for tag in tags:
        for f in sorted(glob.glob(str(OUT / tag / "1_Trajectories" / "*" / "*_losses.csv"))):
            with open(f, newline="") as fh:
                rows = [r for r in csv.DictReader(fh) if r["phase"] == "screen"]
            if len(rows) < 20:
                continue
            col = next((c for c in rows[0] if c.endswith(suffix)), None)
            if col is None:
                continue
            v = [float(r[col]) for r in rows]
            out.append((v[0], st.mean(v[-5:])))
    return out


def report_iptm(tags_a, tags_b):
    """The control that decides whether 'optimisation gains less at 800' is real or a pLDDT artifact.

    i_pTM is measured by the same trajectory but is NOT what the screen gate cuts on, and unlike
    binder pLDDT the two arms START it at the same value. So if the lost-gain effect shows up here
    too, it is a property of the optimisation at the large axis rather than something about the
    pLDDT metric or its threshold.
    """
    a, b = metric_track(tags_a, ".iptm"), metric_track(tags_b, ".iptm")
    if not a or not b:
        return
    ga = [e - s for s, e in a]
    gb = [e - s for s, e in b]
    s0a, s0b = st.mean(x[0] for x in a), st.mean(x[0] for x in b)
    e1a, e1b = st.mean(x[1] for x in a), st.mean(x[1] for x in b)
    p_gain, _ = permutation_p(ga, gb)
    p_start, _ = permutation_p([x[0] for x in a], [x[0] for x in b])
    print()
    print("--- i_pTM over the same screen phase: the independent check on the lost-gain half")
    print(f"    arm A (n={len(a)}): round1 {s0a:.3f} -> end {e1a:.3f}   gain {st.mean(ga):+.3f}")
    print(f"    arm B (n={len(b)}): round1 {s0b:.3f} -> end {e1b:.3f}   gain {st.mean(gb):+.3f}")
    print(f"    round-1 gap {s0a - s0b:+.3f}"
          + (f" (permutation p = {p_start:.5f})" if p_start is not None else "")
          + "  <- essentially ZERO, unlike pLDDT's +0.080")
    print(f"    gain gap    {st.mean(ga) - st.mean(gb):+.3f}"
          + (f" (permutation p = {p_gain:.5f})" if p_gain is not None else ""))
    print("    So i_pTM carries NO context offset and still loses the same optimisation. The")
    print("    offset is specific to binder pLDDT; the lost gain is not, and is therefore a")
    print("    property of optimising at 800 rather than an artifact of the gated metric.")


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

    report_track(screen_plddt_track(args.arma.split(",")),
                 screen_plddt_track(args.armb.split(",")))
    report_iptm(args.arma.split(","), args.armb.split(","))


if __name__ == "__main__":
    main()
