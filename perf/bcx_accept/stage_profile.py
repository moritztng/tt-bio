#!/usr/bin/env python3
"""One row per trajectory, every stage reading it produced, both arms.

The acceptance rate cannot decide this campaign. Separating a 0.167 device rate from a
0.33 reference rate at 80 % power needs ~109 trajectories an arm, 39 days of reference
CPU at the measured 8.65 h a trajectory (state/bcx/DECISION-RULE.md). At the counts BCX
can afford the two Clopper-Pearson intervals overlap whatever the loop does, so that
comparison returns "consistent" either way and decides nothing.

A trajectory already prints more than the accept/reject bit at its end. It prints a
metric pair at every design stage it clears and at the one that rejects it, so the same
compute buys up to five readings instead of one. This reads those out of both arms' run
logs and puts them in one table.

Three things it is careful about:

1. THE LOG NUMBER IS THE STAGE BEST, not the stage mean (perf/bcx_accept/stage_instrument.py).
   Both arms print the same quantity so the comparison is sound, but do not mix a row here
   with a summary.csv mean.
2. A SATURATED i_pTM CARRIES NO INFORMATION. predicted_tm_score is a .max() over alignments,
   so one collapsed PAE row pins it at the length ceiling. DECISION-RULE.md grades a mutate
   verdict on pLDDT only; this drops any i_pTM >= SATURATED from the i_pTM statistics on both
   arms and marks the cell with a star.
3. A REPEATED DRAW IS ONE SAMPLE. Two arms that drew the same binder are a reproducibility
   control, not two trajectories, and pooling both doubles it into the statistics.

Usage:
  stage_profile.py --device LABEL=LOG [...] --reference LABEL=LOG [...]
"""
import re
import statistics as st
import sys

STAGES = ("screen", "refine", "anneal", "harden", "mutate")
SATURATED = 0.995

RE_TRAJ = re.compile(r"=== trajectory (\d+) \| (\S+) \| accepted (\d+)/(\d+) ===")
RE_STAGE = re.compile(
    r"(passed|rejected at) (\w+) design stage\s+i_pTM=([\d.]+)\s+pLDDT=([\d.]+)"
    r"(?:\s+due to \[([^\]]*)\])?")
RE_KEPT = re.compile(r"(\d+) of (\d+) redesigns passed")
# BC2 applies the filter set to the finished trajectory BEFORE it spends MPNN redesigns on it
# (.campaign_state.json terminated."final"). A trajectory that dies there cleared all five design
# stages, so it is COMPLETED and belongs in the denominator -- it is not in flight, and it is not
# an MPNN-candidate rejection either. accept_s4's l152 is the campaign's first of these.
RE_FINAL = re.compile(
    r"trajectory rejected\s+i_pTM=([\d.]+)\s+pLDDT=([\d.]+)(?:\s+due to \[([^\]]*)\])?")
RE_DRAW = re.compile(r"_l(\d+)_([0-9a-f]+)$")
RE_STAMP = re.compile(r"^\d{4}-\d\d-\d\dT[\d:]+Z ")

# Draws DECISION-RULE.md rules out, with the reason it gives. Keyed on the draw hash so a
# reading is excluded wherever it appears, not only in the arm it was first noticed in.
INVALID = {
    "3c946b4d257ac696": "stale mask, served l142's",
    "922fc15bad085ec1": "stale mask, served l142's",
    "3150367da865d53b": "i_pTM 1.0 for all 15 mutate rounds, pre-mask-fix tree",
}


class Traj:
    def __init__(self, arm, side, index, name):
        self.arm, self.side, self.index, self.name = arm, side, index, name
        m = RE_DRAW.search(name)
        self.length = int(m.group(1)) if m else 0
        self.draw = m.group(2) if m else name
        self.stage = {}          # stage -> (i_pTM, pLDDT)
        self.terminal = None
        self.verdict = "in flight"
        self.failed = ""
        self.invalid = INVALID.get(self.draw, "")
        self.duplicate_of = None
        self.kept = None

    @property
    def usable(self):
        return not self.invalid and self.duplicate_of is None

    def iptm(self, stage):
        v = self.stage.get(stage)
        return None if v is None or v[0] >= SATURATED else v[0]

    def plddt(self, stage):
        v = self.stage.get(stage)
        return None if v is None else v[1]

    def term_label(self):
        if self.verdict == "rejected":
            return f"{self.terminal}[{self.failed}]"
        return {"ACCEPTED": "ACCEPTED", "crashed": "crash", "in flight": "in flight"}[self.verdict]


def parse(arm, side, path):
    out, cur = [], None
    for raw in open(path, errors="replace"):
        line = RE_STAMP.sub("", raw).rstrip()
        m = RE_TRAJ.search(line)
        if m:
            cur = Traj(arm, side, int(m.group(1)), m.group(2))
            out.append(cur)
            continue
        if cur is None:
            continue
        m = RE_STAGE.search(line)
        if m:
            verb, stage, iptm, plddt, failed = m.groups()
            cur.stage[stage] = (float(iptm), float(plddt))
            if verb == "rejected at":
                cur.terminal, cur.verdict = stage, "rejected"
                cur.failed = failed or ""
            continue
        m = RE_FINAL.search(line)
        if m:
            cur.terminal, cur.verdict = "final", "rejected"
            cur.failed = m.group(3) or ""
            continue
        m = RE_KEPT.search(line)
        if m:
            # BC2 prints this line whether the refold ensemble kept a candidate or not,
            # and it prints "0 of 10 redesigns passed" for a trajectory it REJECTED there.
            # Matching the line instead of reading the leading count turns every refold
            # rejection into an acceptance: traj_off_s3 read 4 accepted against the 1 its
            # own .campaign_state.json and 2_Refolded/!_Refolded.csv record.
            kept = int(m.group(1))
            cur.terminal = "validation"
            cur.verdict = "ACCEPTED" if kept else "rejected"
            cur.failed = "" if kept else f"0 of {m.group(2)} refolds"
            cur.kept = kept
        elif line.startswith("Traceback") and cur.verdict == "in flight":
            cur.terminal, cur.verdict = "crash", "crashed"
    return out


def dedupe(trajs):
    """The same draw hash is the same trajectory. Keep the one that got furthest."""
    best = {}
    for t in trajs:
        prev = best.get(t.draw)
        if prev is None or (len(t.stage), t.verdict != "crashed") > (len(prev.stage),
                                                                    prev.verdict != "crashed"):
            if prev is not None:
                prev.duplicate_of = t.arm
            best[t.draw] = t
        else:
            t.duplicate_of = prev.arm


def cell(t, stage):
    v = t.stage.get(stage)
    if v is None:
        return "    .    "
    return f"{v[0]:.2f}/{v[1]:.2f}" + ("*" if v[0] >= SATURATED else " ")


def band(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "  .  ", "  .  ", 0
    return f"{st.median(vals):.2f}", f"{min(vals):.2f}-{max(vals):.2f}", len(vals)


def main(argv):
    side, specs = None, []
    for a in argv[1:]:
        if a in ("--device", "--reference"):
            side = a[2:]
        else:
            label, _, path = a.partition("=")
            specs.append((label, side, path))
    trajs = []
    for label, s, path in specs:
        trajs += parse(label, s, path)
    dedupe(trajs)

    head = ("arm            traj side draw    terminal          "
            + "".join(f"{s:>11}" for s in STAGES))
    print("\n# Stage profile -- i_pTM/pLDDT, the stage BEST as the run log prints it")
    print("# * = i_pTM saturated at the length ceiling, no information (DECISION-RULE.md)\n")
    print(head)
    print("-" * len(head))
    for t in trajs:
        flag = f"  INVALID: {t.invalid}" if t.invalid else (
            f"  repeat draw, see {t.duplicate_of}" if t.duplicate_of else "")
        print(f"{t.arm:<14} {t.index:<4} {t.side[:3]:<4} l{t.length:<6} {t.term_label():<17} "
              + "".join(f"{cell(t, s):>11}" for s in STAGES) + flag)

    for metric, get in (("i_pTM", Traj.iptm), ("pLDDT", Traj.plddt)):
        print(f"\n## {metric} by stage, usable trajectories only")
        print(f"{'stage':<8}{'device median':>14}{'range':>13}{'n':>5}"
              f"{'reference median':>19}{'range':>13}{'n':>5}")
        for s in STAGES:
            row = f"{s:<8}"
            for side_ in ("device", "reference"):
                vals = [get(t, s) for t in trajs if t.side == side_ and t.usable]
                med, rng, n = band(vals)
                width = 14 if side_ == "device" else 19
                row += f"{med:>{width}}{rng:>13}{('n=' + str(n)):>5}"
            print(row)

    print("\n## anneal -> harden, the transition the two arms do not share")
    print("harden is the only stage with dropout off and the only one whose optimizer is")
    print("OneHotSequenceOptimizer: it discretizes the sequence in 5 rounds")
    print("(bindcraft/trajectory.py:219 and :233).\n")
    print(f"{'arm':<14} {'side':<10} {'draw':<7} {'i_pTM anneal->harden':<28} pLDDT anneal->harden")
    deltas = {"device": [], "reference": []}
    for t in trajs:
        if not t.usable:
            continue
        a, h = t.iptm("anneal"), t.iptm("harden")
        pa, ph = t.plddt("anneal"), t.plddt("harden")
        if a is None or h is None:
            continue
        deltas[t.side].append(h - a)
        print(f"{t.arm:<14} {t.side:<10} l{t.length:<6} "
              f"{a:.2f} -> {h:.2f}  ({h - a:+.2f})           {pa:.2f} -> {ph:.2f}  ({ph - pa:+.2f})")
    for side_ in ("device", "reference"):
        d = deltas[side_]
        if d:
            print(f"{side_:>14}: n={len(d)}  median {st.median(d):+.2f}  "
                  f"range {min(d):+.2f} to {max(d):+.2f}")

    print("\n## terminal stage, completed trajectories")
    for side_ in ("device", "reference"):
        hist = {}
        for t in trajs:
            if t.side == side_ and t.usable and t.verdict in ("rejected", "ACCEPTED"):
                hist[t.terminal] = hist.get(t.terminal, 0) + 1
        total = sum(hist.values())
        parts = ", ".join(f"{k} {v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1]))
        print(f"{side_:>10}: {total} completed -- {parts or 'none'}")

    print("\n## readings, which is what this instrument buys")
    for side_ in ("device", "reference"):
        n_t = sum(1 for t in trajs if t.side == side_ and t.usable)
        n_r = sum(len(t.stage) for t in trajs if t.side == side_ and t.usable)
        done = sum(1 for t in trajs if t.side == side_ and t.usable
                   and t.verdict in ("rejected", "ACCEPTED"))
        print(f"{side_:>10}: {n_r} stage readings from {n_t} usable trajectories "
              f"({done} of them with a terminal verdict)")


if __name__ == "__main__":
    main(sys.argv)
