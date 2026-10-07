"""The #17 paired comparison as a table: per-stage confidences, both arms, same seeds.

    python3 perf/bci_accept/paired_table.py host=.bci/pc_arm_true.log card=.bci/card_campaign_chip1.log

Reads BindCraft 2's own printed stage lines rather than reconstructing them from `losses.csv`.
That is not a shortcut, it is the only correct source. `trajectory.py:313` prints what
`judge_stage` measured, and `judge_stage` (`:252-256`) evaluates the stage filters against
`stage_filter_predictions(...)`, a separate fold from the per-round predictions the loss recorder
writes. The two disagree: a smoke trajectory whose `losses.csv` ends `screen` on i_pTM 0.19 printed
`passed screen design stage i_pTM=0.24`. The printed line is also the quantity the reporter's own
table is made of, so a table built from the CSV would silently be comparing something else.

Trajectories are paired by NUMBER, not by name. `campaign.py:162` draws trajectory N's key as
`fold_in(PRNGKey(campaign_seed), N)` and neither term reads the arm, so trajectory N starts from a
bit-identical sequence in both arms. The directory name comes from `design_hash(tuned_settings...)`
(`:170`) and `tuned_settings` carries the arm's own settings, so the names differ while the seeds
do not. Pairing by name would pair nothing.
"""
import argparse
import collections
import re
import sys

#: `=== trajectory 3 | design_l60_07a15d3555be7f73 | accepted 0/3 ===`, campaign_log.py:28.
HEADER = re.compile(r"^=== trajectory (\d+) \| (\S+) \| accepted (\d+)/(\d+)")
#: `  passed harden design stage  i_pTM=0.83  pLDDT=0.91`, and the rejected forms from
#: `stage_outcome` (campaign_log.py:41-49): `rejected at <stage> design stage ... due to [...]`
#: and the final-stage `trajectory rejected ...`.
PASSED = re.compile(r"^\s+passed (\w+) design stage(.*)$")
REJECTED = re.compile(r"^\s+rejected at (\w+) design stage(.*?)(?: due to \[(.*)\])?$")
FINAL_REJECT = re.compile(r"^\s+trajectory rejected(.*?)(?: due to \[(.*)\])?$")
#: `  i_pTM=0.83  pLDDT=0.91`, from `metric_field`.
METRIC = re.compile(r"(\w[\w.]*)=([-\d.]+)")
#: `campaign done: 0 accepted design(s) after 1 trajectory, ranked by i_pDAE`
DONE = re.compile(r"^campaign done: (\d+) accepted design\(s\) after (\d+) trajector")

STAGES = ("screen", "refine", "anneal", "harden", "mutate")


def read_arm(path):
    """One arm's trajectories: {number: {"design": str, "stages": {stage: (verdict, metrics)}}}."""
    trajectories = collections.OrderedDict()
    current = None
    accepted = requested = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            header = HEADER.match(line)
            if header:
                number = int(header.group(1))
                current = trajectories.setdefault(
                    number, {"design": header.group(2), "stages": collections.OrderedDict()})
                continue
            done = DONE.match(line)
            if done:
                accepted, requested = int(done.group(1)), int(done.group(2))
                continue
            if current is None:
                continue
            passed = PASSED.match(line)
            if passed:
                current["stages"][passed.group(1)] = ("passed", dict(METRIC.findall(passed.group(2))))
                continue
            rejected = REJECTED.match(line)
            if rejected:
                current["stages"][rejected.group(1)] = (
                    "REJECTED", dict(METRIC.findall(rejected.group(2))))
                continue
            final = FINAL_REJECT.match(line)
            if final:
                current["stages"]["final"] = ("REJECTED", dict(METRIC.findall(final.group(1))))
    return {"trajectories": trajectories, "accepted": accepted, "requested": requested}


def cell(arm, number, stage, metric):
    entry = arm["trajectories"].get(number)
    if entry is None:
        return "-"
    outcome = entry["stages"].get(stage)
    if outcome is None:
        return ""                                  # the stage has not been reached yet
    verdict, metrics = outcome
    value = metrics.get(metric)
    if value is None:
        return "x" if verdict == "REJECTED" else "?"
    return f"{value}{'*' if verdict == 'REJECTED' else ''}"


def main():
    parser = argparse.ArgumentParser(
        description="the #17 paired comparison, one arm per name=log argument")
    parser.add_argument("arms", nargs="+", metavar="name=log",
                        help="e.g. host=.bci/pc_arm_true.log card=.bci/card_campaign_chip1.log")
    parser.add_argument("--metric", default="i_pTM",
                        help="confidence to tabulate (default i_pTM; pLDDT if the filters report it)")
    args = parser.parse_args()

    arms = collections.OrderedDict()
    for spec in args.arms:
        if "=" not in spec:
            parser.error(f"{spec!r} is not name=log")
        name, _, path = spec.partition("=")
        arms[name] = read_arm(path)

    numbers = sorted({n for arm in arms.values() for n in arm["trajectories"]})
    if not numbers:
        print("no trajectories in any arm yet", file=sys.stderr)
        return 1

    width = max(10, max(len(n) for n in arms) + 2)
    print(f"per-stage {args.metric}, paired by trajectory number. "
          "* = the stage rejected the trajectory, x = rejected with no reported value, "
          "blank = not reached yet.")
    header = f"{'traj':>5} {'stage':>8}" + "".join(f"{n:>{width}}" for n in arms)
    print(header)
    print("-" * len(header))
    for number in numbers:
        for stage in STAGES:
            if not any(stage in arm["trajectories"].get(number, {"stages": {}})["stages"]
                       for arm in arms.values()):
                continue
            row = f"{number:>5} {stage:>8}"
            for arm in arms.values():
                row += f"{cell(arm, number, stage, args.metric):>{width}}"
            print(row)

    print()
    for name, arm in arms.items():
        if arm["accepted"] is None:
            reached = len(arm["trajectories"])
            print(f"{name}: campaign still running, {reached} trajector"
                  f"{'y' if reached == 1 else 'ies'} started, no accepted count yet")
        else:
            print(f"{name}: accepted {arm['accepted']} of {arm['requested']}")
    print("\nAcceptance at this budget resolves nothing on its own: Fisher two-sided on the "
          "reporter's own 0 of 8 against 3 of 8 is p=0.200. Read the harden rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
