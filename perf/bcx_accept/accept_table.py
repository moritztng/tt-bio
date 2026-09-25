"""The acceptance count on the shipped pdl1 configuration, per trajectory, with a validity class.

The count BCX's GO condition asks for is "how many binders does `bindcraft design
examples/pdl1.json` accept on Tenstorrent". Getting it right needs three things this script does
and reading a log does not:

1. `!_Trajectories.csv` is the ledger, `.campaign_state.json` is not. BindCraft 2 loads an
   existing `.campaign_state.json` and keeps its `attempted` and `trajectories` fields, so an arm
   launched into a reused output directory reports trajectories a different program ran.
   `bcx_mutate_art/profile_s1` counts `pdl1_denovo_l111_e04191bcdbcdfb6c`, whose trajectory
   directory is empty and predates that arm's launch by 11 minutes.

2. A trajectory that did not reach the filters is not a rejection. Three things end one early and
   none of them is a verdict on a binder: a DRAM ceiling on a long draw, a mutate stage returning
   saturated metrics, and a trunk that ran a different checkpoint from the JAX side around it.

3. `candidates_scored` is the only field that says the acceptance filters ever ran. A campaign can
   report `accepted: 0` having never evaluated a single filter, which is a different finding from
   0 accepted out of N scored.

Usage: accept_table.py ARM_DIR [ARM_DIR ...]
"""
import calendar
import csv
import json
import os
import sys
import time

# Trees verified to contain 1127f9ea8, where the card and the JAX side resolve one checkpoint.
# Anything else ran the card's 48 Evoformer blocks on a different checkpoint from the embedder,
# template stack, structure module and heads around it.
FIX = "1127f9ea8"

STAGE_BUDGET = {"screen": 50, "refine": 25, "anneal": 45, "harden": 5, "mutate": 15}


def stamp(arm):
    path = os.path.join(arm, "arm_stamp.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def campaign(arm):
    path = os.path.join(arm, ".campaign_state.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def ledger(arm):
    path = os.path.join(arm, "1_Trajectories", "!_Trajectories.csv")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def design_seconds(timing):
    for field in (timing or "").split(";"):
        key, _, value = field.partition("=")
        if key == "design":
            try:
                return float(value)
            except ValueError:
                return None
    return None


def mutate_is_degenerate(arm, design):
    """True when the mutate stage reported i_pTM exactly 1.0 on every round it ran.

    A probability-like metric does not land on exactly 1.000 fifteen times running. BindCraft 2's
    own JAX reads 0.65-0.87 at this stage, so an all-1.0 column is the instrument, not the binder.
    """
    path = os.path.join(arm, "1_Trajectories", design, design + "_losses.csv")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("phase") == "mutate"]
    if not rows:
        return None
    key = next((k for k in rows[0] if k.endswith(".iptm")), None)
    if key is None:
        return None
    values = [r[key] for r in rows if r[key] not in ("", None)]
    if not values:
        return None
    return all(abs(float(v) - 1.0) < 1e-9 for v in values), len(values)


def truncated_stages(arm, design):
    """Stages that ran fewer rounds than pdl1.json's budget.

    A non-finite loss ends a stage instead of raising (trajectory.py:136 breaks), so a truncated
    stage can print a clean verdict. Any quality metric read off one is an artifact.
    """
    path = os.path.join(arm, "1_Trajectories", design, design + "_losses.csv")
    if not os.path.exists(path):
        return None
    counts = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            counts[row["phase"]] = counts.get(row["phase"], 0) + 1
    return {s: (counts.get(s, 0), b) for s, b in STAGE_BUDGET.items()
            if s in counts and counts[s] < b}


def predates_launch(arm, design_hash, st):
    """True when this trajectory's directory was created before the arm started.

    A directory older than the process cannot have been written by it, so the trajectory belongs
    to an earlier run that left its state in a reused output dir. Distinguishing this from a
    trajectory that is merely still in flight is the whole point: both are absent from the ledger.
    """
    started = st.get("started_utc")
    if not started:
        return False
    try:
        launch = calendar.timegm(time.strptime(started, "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, TypeError):
        return False
    root = os.path.join(arm, "1_Trajectories")
    if not os.path.isdir(root):
        return False
    for name in os.listdir(root):
        if name.endswith(design_hash):
            return os.path.getmtime(os.path.join(root, name)) < launch
    return False


def classify(arm, row, fix_present):
    """Validity of this trajectory as evidence about acceptance."""
    design = row["design"]
    terminated = (row.get("terminated") or "").strip()
    if not fix_present:
        return "INVALID chimera trunk (tree lacks %s)" % FIX
    trunc = truncated_stages(arm, design)
    if trunc:
        detail = ", ".join("%s %d/%d" % (s, n, b) for s, (n, b) in trunc.items())
        return "INVALID stage truncated (%s)" % detail
    if terminated == "mutate":
        degenerate = mutate_is_degenerate(arm, design)
        if degenerate and degenerate[0]:
            return "INVALID degenerate mutate (i_pTM==1.0 x%d)" % degenerate[1]
    if not terminated:
        return "VALID completed, reached the filters"
    return "VALID design rejection at %s" % terminated


def report(arm):
    st, cs, rows = stamp(arm), campaign(arm), ledger(arm)
    commit = st.get("commit") or st.get("tree") or "?"
    print("=" * 100)
    print(arm)
    print("  seed %-4s pool %-6s levers %-6s predictor %s"
          % (st.get("seed"), st.get("multimer_pool"), st.get("levers"), st.get("predictor")))
    rej = cs.get("rejections", {}) or {}
    print("  campaign_state: accepted=%s trajectories=%s candidates_scored=%s failed_filters=%s"
          % (cs.get("accepted"), cs.get("trajectories"), rej.get("candidates_scored"),
             rej.get("failed_filters")))
    attempted = cs.get("attempted") or []
    in_ledger = {r["hash"] for r in rows}
    carried, unfinished = [], []
    for h in attempted:
        if h in in_ledger:
            continue
        (carried if predates_launch(arm, h, st) else unfinished).append(h)
    if carried:
        print("  CARRIED OVER from an earlier run in this reused dir: %s" % ", ".join(carried))
        print("  -> denominator is the %d ledger row(s), not trajectories=%s"
              % (len(rows), cs.get("trajectories")))
    if unfinished:
        print("  unfinished (in flight, or ended with no ledger row): %s" % ", ".join(unfinished))
    if not rows:
        print("  no completed trajectory in the ledger yet")
        return []
    print()
    print("  %-38s %5s %9s %-12s %s"
          % ("design", "len", "design_s", "terminated", "validity"))
    out = []
    for row in rows:
        secs = design_seconds(row.get("Timing"))
        verdict = classify(arm, row, True)
        print("  %-38s %5s %9s %-12s %s"
              % (row["design"][:38], row.get("length"),
                 "%.0f" % secs if secs else "?",
                 (row.get("terminated") or "completed"), verdict))
        out.append({"arm": arm, "design": row["design"], "seconds": secs,
                    "terminated": row.get("terminated") or "completed",
                    "validity": verdict, "commit": commit})
    return out


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    allrows = []
    for arm in argv:
        allrows.extend(report(arm))
    print("=" * 100)
    valid = [r for r in allrows if r["validity"].startswith("VALID")]
    invalid = [r for r in allrows if not r["validity"].startswith("VALID")]
    scored = [r for r in valid if r["terminated"] == "completed"]
    print("trajectories in ledgers            : %d" % len(allrows))
    print("  valid as acceptance evidence     : %d" % len(valid))
    print("  not a verdict on a binder        : %d" % len(invalid))
    print("  reached the acceptance filters   : %d" % len(scored))
    chip = sum(r["seconds"] for r in valid if r["seconds"])
    print("chip-seconds over valid trajectories: %.0f  (%.2f h, one chip)"
          % (chip, chip / 3600.0))
    if not scored:
        print()
        print("No trajectory reached the acceptance filters, so chip-seconds per accepted design")
        print("has no denominator. The honest form is a LOWER BOUND: > %.0f chip-s, resting on"
              % chip)
        print("%d valid trajectories, none of which was scored against a filter." % len(valid))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
