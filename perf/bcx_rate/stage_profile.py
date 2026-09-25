#!/usr/bin/env python3
"""BindCraft 2 stage profile: where trajectories die, and how confident they were at each stage.

Why this exists rather than an acceptance count. The campaign's GO condition was the shipped
PD-L1 example's accepted-binder count reproduced within seed variance. Two measurements killed
that as a test: `examples/pdl1.json` requests ten accepted designs with no attempt limit, so the
count is ten on any run that terminates; and read as a rate instead, separating a 0.167 device
rate from a 0.33 reference at 80% power needs 109 trajectories per arm -- 39 days of reference
CPU against a competition window that opens 28 September.

A trajectory carries about five stage verdicts before it carries one acceptance bit, and each
verdict comes with an i_pTM and a pLDDT. So this instrument gets roughly five readings per
trajectory where the rate gets one, from exactly the same compute. It answers the question the
rate was standing in for: does the device loop optimise the way BindCraft 2's own JAX does.

    python3 perf/bcx_rate/stage_profile.py device:accept_s4.log reference:ref_s1.stamped.log

Prefix each log with the arm class it belongs to. A log may be stamped or raw; the timestamps are
ignored here.

Read a mutate-stage verdict on pLDDT alone. `predicted_tm_score` is a `.max()` over alignments,
so one collapsed PAE row pins i_pTM at the length's ceiling -- on either arm. The columns are
printed because their absence would be its own signal, not because a saturated one means anything.
"""
import argparse
import re
import sys
from collections import defaultdict

STAGES = ["screen", "refine", "anneal", "harden", "mutate"]
TRAJECTORY = re.compile(r"^=== trajectory (\d+) \| (\S+)")
VERDICT = re.compile(
    r"^\s*(passed|rejected at) (\w+) design stage\s+i_pTM=([\d.]+)\s+pLDDT=([\d.]+)"
    r"(?:\s+due to \[([^\]]*)\])?")


def parse(path):
    """One record per trajectory: its draw, its stage readings, where it stopped and why."""
    trajectories = []
    current = None
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        # A stamped log prefixes an ISO timestamp; drop anything before the first '=' or space run.
        line = re.sub(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ", "", line)
        match = TRAJECTORY.match(line)
        if match:
            current = {"index": int(match.group(1)), "design": match.group(2),
                       "stages": {}, "terminated": None, "filters": None}
            trajectories.append(current)
            continue
        match = VERDICT.match(line)
        if match and current is not None:
            outcome, stage, iptm, plddt, filters = match.groups()
            current["stages"][stage] = (float(iptm), float(plddt))
            if outcome == "rejected at":
                current["terminated"] = stage
                current["filters"] = filters or ""
    return trajectories


def length_of(design):
    match = re.search(r"_l(\d+)_", design)
    return int(match.group(1)) if match else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", metavar="CLASS:LOG",
                        help="arm class ('device' or 'reference') and the run log")
    args = parser.parse_args(argv)

    by_class = defaultdict(list)
    for spec in args.logs:
        if ":" not in spec:
            sys.exit(f"{spec}: prefix the log with its arm class, e.g. device:{spec}")
        arm_class, path = spec.split(":", 1)
        for trajectory in parse(path):
            trajectory["class"] = arm_class
            trajectory["log"] = path.rsplit("/", 1)[-1]
            by_class[arm_class].append(trajectory)

    header = f"{'class':<10}{'log':<24}{'draw':>5}  " + "".join(f"{s:<14}" for s in STAGES)
    print(header)
    print(f"{'':<10}{'':<24}{'':>5}  " + "".join(f"{'i_pTM/pLDDT':<14}" for _ in STAGES))
    for arm_class in sorted(by_class):
        for t in by_class[arm_class]:
            cells = ""
            for stage in STAGES:
                if stage in t["stages"]:
                    iptm, plddt = t["stages"][stage]
                    mark = "x" if t["terminated"] == stage else " "
                    cells += f"{iptm:.2f}/{plddt:.2f}{mark}     "
                else:
                    cells += f"{'-':<14}"
            t["end"] = t["terminated"] or (
                "running" if len(t["stages"]) < len(STAGES) else "completed")
            end = t["end"]
            print(f"{arm_class:<10}{t['log']:<24}{length_of(t['design']):>5}  {cells}{end}"
                  + (f" [{t['filters']}]" if t["filters"] else ""))

    print("\nPer-stage range by arm class. 'x' above marks the stage a trajectory died at.")
    for stage in STAGES:
        line = f"  {stage:<8}"
        for arm_class in sorted(by_class):
            values = [t["stages"][stage] for t in by_class[arm_class] if stage in t["stages"]]
            if not values:
                line += f"{arm_class}: -   "
                continue
            iptms = sorted(v[0] for v in values)
            plddts = sorted(v[1] for v in values)
            line += (f"{arm_class}: n={len(values)} "
                     f"i_pTM {iptms[0]:.2f}-{iptms[-1]:.2f} "
                     f"pLDDT {plddts[0]:.2f}-{plddts[-1]:.2f}   ")
        print(line)

    print("\nTerminal stage:")
    for arm_class in sorted(by_class):
        counts = defaultdict(int)
        for t in by_class[arm_class]:
            counts[t["end"]] += 1  # a trajectory still running has not reached anything
        print(f"  {arm_class:<10}" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    print("\nGO on this instrument needs the device's per-stage medians inside the reference's "
          "per-stage range,\nand no stage at which the device terminates and the reference never "
          "does. Neither is decided by\na range alone while the reference carries one trajectory.")


if __name__ == "__main__":
    main()
