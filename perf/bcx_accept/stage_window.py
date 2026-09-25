#!/usr/bin/env python3
"""The stage number BindCraft 2 prints is a max over the stage's rounds, so it is not
comparable between stages of different length.

This matters because it manufactured a finding. Read off the run logs, the device arms
look like they fall apart at `harden`: 4 of 8 completed trajectories terminate there and
the anneal->harden i_pTM step has a device median of -0.35 against the reference's +0.00.
`harden` is the discretization stage, so the story writes itself.

It is an artifact. `harden` runs 5 rounds and `anneal` runs 45 (settings.py:604), and the
log prints each stage's BEST round. A max over 45 draws from a noisy series is higher than
a max over 5 from the same series whether or not anything changed. On the device's l135 the
step is -0.35 against anneal's 45-round max and **+0.02** against its last five rounds.

The quantity that is matched across arms is the PEAK-TO-END GAP inside one stage: the
stage's max minus the max of its final five rounds. Both arms run the same 45 anneal
rounds, so the gap is a property of the trajectory, not of the budget. The reference's
anneal peaks and ends at the same place (gap 0.01). The device's l135 peaks at 0.84 and
ends at 0.47 (gap 0.37), which says the degradation happened DURING anneal and `harden`
is only the first stage whose short budget cannot hide it.

Reads `1_Trajectories/<traj>/<traj>_losses.csv`, which carries every round.

Usage:  stage_window.py --device LABEL=CSV [...] --reference LABEL=CSV [...]
        stage_window.py --window 5 ...
"""
import csv
import re
import statistics as st
import sys

STAGES = ("screen", "refine", "anneal", "harden", "mutate")
METRIC = "hPDL1.iptm"
RE_DRAW = re.compile(r"_l(\d+)_([0-9a-f]+)_losses\.csv$")


def series(path):
    """stage -> list of the metric, in round order."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            v = (row.get(METRIC) or "").strip()
            if not v:
                continue
            out.setdefault(row["phase"], []).append(float(v))
    return out


def stat(vals, window):
    tail = vals[-window:]
    return max(vals), max(tail), st.median(tail), len(vals)


def main(argv):
    side, specs, window = None, [], 5
    args = argv[1:]
    while args:
        a = args.pop(0)
        if a in ("--device", "--reference"):
            side = a[2:]
        elif a == "--window":
            window = int(args.pop(0))
        else:
            label, _, path = a.partition("=")
            specs.append((label, side, path))

    rows = []
    for label, s, path in specs:
        m = RE_DRAW.search(path)
        rows.append((label, s, f"l{m.group(1)}" if m else "?", series(path)))

    print(f"\n# Per-stage i_pTM: the printed max against the last {window} rounds")
    print(f"# gap = stage max - max of its final {window} rounds. A stage that ends where it")
    print("# peaked has gap ~0. The gap is round-count matched, the printed max is not.\n")
    head = f"{'arm':<13}{'side':<5}{'draw':<7}{'stage':<8}{'n':>4}{'printed max':>13}{'last' + str(window) + ' max':>11}{'gap':>8}{'last' + str(window) + ' median':>13}"
    print(head)
    print("-" * len(head))
    gaps = {"device": {}, "reference": {}}
    for label, s, draw, ser in rows:
        for stage in STAGES:
            vals = ser.get(stage)
            if not vals:
                continue
            mx, tmx, tmed, n = stat(vals, window)
            gap = mx - tmx
            gaps[s].setdefault(stage, []).append(gap)
            mark = "  <--" if gap >= 0.15 else ""
            print(f"{label:<13}{s[:3]:<5}{draw:<7}{stage:<8}{n:>4}{mx:>13.2f}{tmx:>11.2f}"
                  f"{gap:>8.2f}{tmed:>13.2f}{mark}")
        print()

    print(f"## peak-to-end gap by stage, the round-count-matched statistic")
    print(f"{'stage':<8}{'device median':>15}{'range':>14}{'n':>5}"
          f"{'reference median':>20}{'range':>14}{'n':>5}")
    for stage in STAGES:
        row = f"{stage:<8}"
        for s in ("device", "reference"):
            g = gaps[s].get(stage, [])
            w = 15 if s == "device" else 20
            if g:
                row += f"{st.median(g):>{w}.2f}{f'{min(g):.2f}-{max(g):.2f}':>14}{'n=' + str(len(g)):>5}"
            else:
                row += f"{'.':>{w}}{'.':>14}{'n=0':>5}"
        print(row)

    print(f"\n## anneal->harden step, printed against matched")
    print(f"{'arm':<13}{'side':<11}{'draw':<7}{'printed step':>14}{'matched step':>15}")
    for label, s, draw, ser in rows:
        a, h = ser.get("anneal"), ser.get("harden")
        if not a or not h:
            continue
        printed = max(h) - max(a)
        matched = max(h) - max(a[-window:])
        print(f"{label:<13}{s:<11}{draw:<7}{printed:>+14.2f}{matched:>+15.2f}")


    print(f"\n## round-to-round jitter: median |i_pTM(r) - i_pTM(r-1)| inside a stage")
    print("Immune to the stage's round count -- it is a median over consecutive differences.")
    print("BindCraft 2 advances a stage on its BEST round, so a jittery arm is graded on an")
    print("outlier and a stage with a small budget gets few chances to draw one.\n")
    print(f"{'stage':<8}{'device median':>15}{'range':>14}{'n':>5}"
          f"{'reference median':>20}{'range':>14}{'n':>5}")
    jit = {"device": {}, "reference": {}}
    for label, s_, draw, ser in rows:
        for stage in STAGES:
            vals = ser.get(stage)
            if not vals or len(vals) < 3:
                continue
            d = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
            jit[s_].setdefault(stage, []).append(st.median(d))
    for stage in STAGES:
        row = f"{stage:<8}"
        for s_ in ("device", "reference"):
            g = jit[s_].get(stage, [])
            w = 15 if s_ == "device" else 20
            if g:
                row += f"{st.median(g):>{w}.3f}{f'{min(g):.3f}-{max(g):.3f}':>14}{'n=' + str(len(g)):>5}"
            else:
                row += f"{'.':>{w}}{'.':>14}{'n=0':>5}"
        print(row)

    print(f"\n## per-trajectory jitter, all gradient stages pooled")
    print(f"{'arm':<13}{'side':<11}{'draw':<7}{'median |d|':>12}{'rounds':>8}")
    for label, s_, draw, ser in rows:
        d = []
        for stage in STAGES:
            vals = ser.get(stage) or []
            d += [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
        if d:
            print(f"{label:<13}{s_:<11}{draw:<7}{st.median(d):>12.3f}{len(d) + 1:>8}")


if __name__ == "__main__":
    main(sys.argv)
