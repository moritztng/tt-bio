"""The acceptance count on the shipped pdl1 configuration, per trajectory, with a validity class.

The count BCX's GO condition asks for is "how many binders does `bindcraft design
examples/pdl1.json` accept on Tenstorrent". Getting it right needs three things this script does
and reading a log does not:

1. `!_Trajectories.csv` is the ledger, `.campaign_state.json` is not. BindCraft 2 loads an
   existing `.campaign_state.json` and keeps its `attempted` and `trajectories` fields, so an arm
   launched into a reused output directory reports trajectories a different program ran.
   `bcx_mutate_art/profile_s1` counts `pdl1_denovo_l111_e04191bcdbcdfb6c`, whose trajectory
   directory is empty and predates that arm's launch by 11 minutes.

2. A trajectory that did not reach the filters is not a rejection. Four things end one early and
   none of them is a verdict on a binder: a DRAM ceiling on a long draw, a stage returning
   saturated metrics, a trunk that ran a different checkpoint from the JAX side around it, and a
   crash in binder optimization after hallucination succeeded. The last one is the dangerous one,
   because it leaves a complete-looking ledger row with an empty `terminated` field and an arm
   stamp reading `exit 0`.

3. `candidates_scored` is the only field that says the acceptance filters ever ran. A campaign can
   report `accepted: 0` having never evaluated a single filter, which is a different finding from
   0 accepted out of N scored.

Usage: accept_table.py ARM_DIR[=COMMIT] [ARM_DIR[=COMMIT] ...]

Run it from inside the tt-bio worktree so the COMMIT ancestry check can reach git. An arm whose
commit is neither given nor stamped is reported UNVERIFIED rather than counted, because a chimera
trunk and a fixed one leave identical project folders.
"""
import calendar
import csv
import json
import os
import subprocess
import sys
import time

# Trees verified to contain 1127f9ea8, where the card and the JAX side resolve one checkpoint.
# Anything else ran the card's 48 Evoformer blocks on a different checkpoint from the embedder,
# template stack, structure module and heads around it.
FIX = "1127f9ea8"

# The validation route, `ttbio_predictor._route`. Without it a trajectory that passes every design
# stage dies in `predict_validation_ensemble` asking the resident pool for `model_1_ptm`, so a
# no-route arm KEEPS its design rejections and DISCARDS its successes. Its rows are a rejection
# profile, not an acceptance ratio: pooling them into accepted/attempted biases the rate toward
# zero by construction. Checked by CONTENT, not ancestry -- the same change landed twice, as
# c014a2a7d on wk/bcx-armtree and 2bf8a136b on wk/bcx-mutate, and an ancestry test against either
# sha reports the other lineage as unfixed.
#
# What the route does is also what the accepted count MEANS. BindCraft 2 holds validation out of
# design, and on the shipped pdl1.json all five multimer checkpoints are design models, so
# validation runs on the monomer pair. Those are not card-resident, so the validation folds run on
# BindCraft 2's own JAX on the host. A device arm's accepted count is therefore a device design
# loop judged by the reference's own validation folds -- which makes the filter identical on both
# arms and the comparison clean, and means the count does not certify a device validation fold.
ROUTE_FILE = "perf/bcx_predictor/ttbio_predictor.py"
ROUTE_MARK = "def _route"

# The MSA-mask cache key. Before the fix `EvoformerOnDevice._mask` keyed on `tuple(mask.shape)`,
# and that shape is the token axis AFTER `_pad_inputs` rounds it up to 32, so every binder length
# in a bucket shares it. One `EvoformerOnDevice` is built per campaign, so the cache outlives every
# trajectory: the first draw in a bucket populates the entry and every later draw in that bucket is
# served ITS mask, running the difference as padding. Against the 115-residue PD-L1 target, binders
# 142, 151 and 144 all land on (1, 288) carrying 257, 266 and 259 real residues. The fixed key is
# (shape, blake2b(content)). The pair-mask cache was never affected: it already carried the mask's
# sum. `perf/bcx_mutate/mask_cache_collision.py` is the card-free demonstration.
MASK_FILE = "perf/bcx_predictor/splice.py"
MASK_MARK = "hashlib.blake2b"

# hPDL1, the target every arm in this campaign designs against. Used only when an arm's own
# `sequences.jsonl` is missing, which is the case for arms from trees that predate it. Verifiable
# from any arm that does have one: `target_hPDL1` is 115 characters, and that is the length of the
# sequence in BindCraft 2's `settings/target/hPDL1.json`. The table says which source it used.
TARGET_FALLBACK = 115

# The stage floors BindCraft 2 rejects on, from `settings/core/default.json:25-29,113-117` (BC2 at
# 7a2dfdb). `examples/pdl1.json` overrides none of them and the `binder` modality does not either,
# so these are the bars every arm in this campaign was judged against. `screen` and `refine` have
# no i_pTM floor at all, which is why passing `screen` says nothing about interface confidence.
STAGE_FLOOR = {"screen": (None, 0.60), "refine": (None, 0.60), "anneal": (0.50, 0.65),
               "harden": (0.50, 0.65), "mutate": (0.50, 0.60), "final": (0.70, 0.70)}

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


def last_recorded_stage(arm, design):
    """The last stage this trajectory actually recorded rounds for.

    `terminated` can name a stage that runs no gradient rounds. `pool_full`'s l146 reads
    `terminated=final`, the gate that sits after the design stages and judges the trajectory on
    the metrics the LAST design stage left behind. Testing the literal `terminated` value for
    saturation finds no rounds under that name, returns nothing, and lets a verdict read off a
    saturated mutate stage into the denominator as an ordinary design rejection. That is exactly
    what happened: mutate read i_pTM 1.0 on all 15 rounds and `final` rejected it on pLDDT 0.6.
    """
    path = os.path.join(arm, "1_Trajectories", design, design + "_losses.csv")
    if not os.path.exists(path):
        return None
    seen = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if row["phase"] not in seen:
                seen.append(row["phase"])
    return seen[-1] if seen else None


def stage_is_degenerate(arm, design, stage):
    """True when `stage` reported i_pTM pinned at exactly 1.0 on every round it ran.

    A probability-like metric does not land on exactly 1.000 round after round. BindCraft 2's own
    JAX reads 0.65-0.87 where we read 1.0, so an all-1.0 column is the instrument, not the binder.

    Not mutate-specific, though mutate is where it was first seen: profile_s1's
    l151_3c946b4d257ac696 was rejected at SCREEN on i_pTM 1.0 with pLDDT 0.33. Keying this on the
    stage name would have classified that trajectory as a design rejection and quietly put a
    broken reading into the acceptance denominator.
    """
    path = os.path.join(arm, "1_Trajectories", design, design + "_losses.csv")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("phase") == stage]
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


def arm_commit(arm, explicit):
    """The commit the arm's tree was at, from the command line or the launcher's own stamp.

    Nothing inside a BindCraft 2 project folder records which tree produced it, so an arm that ran
    a chimera trunk and an arm that ran the checkpoint fix write byte-identical `arm_stamp.json`
    shapes. Provenance therefore has to come from outside the project dir: either written on the
    command line as ARM_DIR=COMMIT, or from a sibling `<name>_stamp.txt` the launcher wrote.
    """
    if explicit:
        return explicit
    for key in ("commit", "tree_commit"):
        value = stamp(arm).get(key)
        if value:
            return value
    sibling = os.path.join(os.path.dirname(arm.rstrip("/")),
                           os.path.basename(arm.rstrip("/")) + "_stamp.txt")
    if os.path.exists(sibling):
        with open(sibling) as handle:
            for line in handle:
                field, _, value = line.strip().partition(" ")
                if field == "commit" and value.strip():
                    return value.strip()
    return None


def carries_fix(commit):
    """True/False when git can answer, None when the commit is unknown or unreachable here."""
    if not commit:
        return None
    try:
        done = subprocess.run(["git", "merge-base", "--is-ancestor", FIX, commit],
                              capture_output=True)
    except OSError:
        return None
    return True if done.returncode == 0 else (False if done.returncode == 1 else None)


def carries_route(commit):
    """True/False when git can read the tree, None when the commit is unknown or unreachable.

    Reads the file out of the commit instead of testing ancestry, because the route landed twice
    as two cherry-picks of one change and neither sha is an ancestor of the other.
    """
    if not commit:
        return None
    try:
        done = subprocess.run(["git", "show", "%s:%s" % (commit, ROUTE_FILE)],
                              capture_output=True)
    except OSError:
        return None
    if done.returncode != 0:
        return None
    return ROUTE_MARK in done.stdout.decode("utf-8", "replace")


def carries_mask_fix(commit):
    """True/False when git can read the tree, None when it cannot."""
    if not commit:
        return None
    try:
        done = subprocess.run(["git", "show", "%s:%s" % (commit, MASK_FILE)], capture_output=True)
    except OSError:
        return None
    if done.returncode != 0:
        return None
    return MASK_MARK in done.stdout.decode("utf-8", "replace")


def target_length(arm):
    """Residues in the target, off the arm's own `sequences.jsonl` rather than a constant.

    Every chain in the first recorded state except the binder. The padded bucket is a function of
    the TOTAL token count, so the target length is half the arithmetic and hardcoding it silently
    moves every bucket edge if the campaign is ever pointed at another target.
    """
    path = os.path.join(arm, "sequences.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        line = handle.readline()
    if not line.strip():
        return None
    try:
        states = json.loads(line)["states"]
    except (ValueError, KeyError):
        return None
    state = next(iter(states.values()))
    return sum(len(v) for k, v in state.items() if k != "binder") or None


def target_or_fallback(arm):
    """(residues, source). An arm with no `sequences.jsonl` still gets a bucket arithmetic."""
    own = target_length(arm)
    return (own, "this arm's sequences.jsonl") if own else (TARGET_FALLBACK, "hPDL1 constant")


def draw_order(arm):
    """Binder lengths in the order THIS PROCESS drew them, from the sibling run log.

    The ledger is not enough. A trajectory that died on the DRAM ceiling leaves no ledger row and
    still populated the mask cache, so reading order off `!_Trajectories.csv` can call a stale draw
    fresh. The log records every draw, including the ones that left nothing behind.
    """
    sibling = os.path.join(os.path.dirname(arm.rstrip("/")),
                           os.path.basename(arm.rstrip("/")) + ".log")
    if not os.path.exists(sibling):
        return None
    order = []
    with open(sibling, "rb") as handle:
        for raw in handle:
            line = raw.decode("utf-8", "replace")
            if "=== trajectory" not in line or "pdl1_denovo_" not in line:
                continue
            name = line.split("|")[1].strip()
            hash_ = name.rsplit("_", 1)[-1]
            length = name.split("_l", 1)[1].split("_", 1)[0]
            try:
                order.append((hash_, int(length)))
            except ValueError:
                continue
    return order or None


def rejections(arm):
    """{design hash: (stage, i_pTM, pLDDT, filter)} from the run log.

    BindCraft 2 prints the rejection without naming the design, so the line is attached to the
    trajectory banner above it. The margin against `STAGE_FLOOR` is the difference between a
    collapse and a near miss, and a count of 0 means something different in each case.
    """
    sibling = os.path.join(os.path.dirname(arm.rstrip("/")),
                           os.path.basename(arm.rstrip("/")) + ".log")
    if not os.path.exists(sibling):
        return {}
    out, current = {}, None
    with open(sibling, "rb") as handle:
        for raw in handle:
            line = raw.decode("utf-8", "replace")
            if "=== trajectory" in line and "pdl1_denovo_" in line:
                current = line.split("|")[1].strip().rsplit("_", 1)[-1]
            elif current and "rejected at" in line and "due to" in line:
                words = line.split()
                stage = words[words.index("at") + 1]
                got = {}
                for word in words:
                    key, _, value = word.partition("=")
                    if key in ("i_pTM", "pLDDT"):
                        try:
                            got[key] = float(value)
                        except ValueError:
                            pass
                which = line.split("due to")[1].strip().strip("[]")
                out[current] = (stage, got.get("i_pTM"), got.get("pLDDT"), which)
                current = None
    return out


def margin(rejection):
    """How far under its floor the rejecting metric was, or None when it cannot be computed."""
    stage, iptm, plddt, which = rejection
    floors = STAGE_FLOOR.get(stage)
    if not floors:
        return None
    floor = floors[0] if which == "i_pTM" else floors[1]
    got = iptm if which == "i_pTM" else plddt
    if floor is None or got is None:
        return None
    return got - floor


def stale_mask(arm, design):
    """The earlier draw whose MSA mask this one was served, or None when its mask was its own.

    Returns (bucket, first_draw_length) so the classification can say which draw poisoned it.
    """
    target, _ = target_or_fallback(arm)
    order = draw_order(arm)
    if not target or not order:
        return None
    hash_ = design.rsplit("_", 1)[-1]
    seen = {}
    for h, length in order:
        bucket = -(-(target + length) // 32) * 32
        if h == hash_:
            return (bucket, seen[bucket]) if bucket in seen else None
        seen.setdefault(bucket, length)
    return None


def candidates(arm, design):
    """The MPNN candidate rows this trajectory was scored on, from `2_Refolded/!_Refolded.csv`.

    A trajectory that finished hallucination has NOT reached the acceptance filters. Between the
    two sits binder optimization: ten MPNN redesigns, each refolded on the VALIDATION models and
    tested against the filter set. Only those rows carry `Binder_RMSD` and `Unbound_Binder_pLDDT`,
    and only those rows carry an `outcome`.

    The distinction is not academic. `accept_s3` passed all five design stages at i_pTM 0.85 and
    then died in `MPNN_stage.predict_validation_ensemble` because the validation models
    (`model_1_ptm`, `model_2_ptm`) are monomer checkpoints and the resident pool holds the five
    multimer ones. The ledger row it left behind has an empty `terminated` field, which reads
    identically to a trajectory that was scored and accepted nothing.
    """
    path = os.path.join(arm, "2_Refolded", "!_Refolded.csv")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [r for r in csv.DictReader(handle) if r.get("design", "").startswith(design)]


def classify(arm, row, fix_present, mask_fix):
    """Validity of this trajectory as evidence about acceptance."""
    design = row["design"]
    terminated = (row.get("terminated") or "").strip()
    if fix_present is None:
        return "UNVERIFIED tree, no commit recorded for this arm"
    if not fix_present:
        return "INVALID chimera trunk (tree lacks %s)" % FIX
    if mask_fix is False:
        served = stale_mask(arm, design)
        if served:
            return ("INVALID stale MSA mask (bucket %d, served draw l%d's mask)" % served)
    trunc = truncated_stages(arm, design)
    if trunc:
        detail = ", ".join("%s %d/%d" % (s, n, b) for s, (n, b) in trunc.items())
        return "INVALID stage truncated (%s)" % detail
    if terminated:
        judged = terminated if stage_is_degenerate(arm, design, terminated) else last_recorded_stage(arm, design)
        degenerate = stage_is_degenerate(arm, design, judged)
        if degenerate and degenerate[0]:
            return "INVALID saturated %s (i_pTM==1.0 x%d)%s" % (
                judged, degenerate[1], "" if judged == terminated else ", rejected at " + terminated)
    if not terminated:
        scored = candidates(arm, design)
        if not scored:
            return "INVALID hallucination completed, validation never ran (0 candidates scored)"
        accepted = [c for c in scored if c.get("outcome") == "passed"]
        return "VALID scored, %d of %d candidates passed" % (len(accepted), len(scored))
    return "VALID design rejection at %s" % terminated


def report(argument):
    arm, _, explicit = argument.partition("=")
    st, cs, rows = stamp(arm), campaign(arm), ledger(arm)
    commit = arm_commit(arm, explicit)
    fix_present = carries_fix(commit)
    route_present = carries_route(commit)
    mask_fix = carries_mask_fix(commit)
    print("=" * 100)
    print(arm)
    print("  seed %-4s pool %-6s levers %-6s predictor %s"
          % (st.get("seed"), st.get("multimer_pool"), st.get("levers"), st.get("predictor")))
    print("  tree %s, carries %s: %s"
          % (commit or "UNRECORDED", FIX,
             {True: "yes", False: "NO, chimera trunk", None: "unknown"}[fix_present]))
    print("  validation route: %s"
          % {True: "present, so this arm could reach the filters",
             False: "ABSENT, rejection profile only -- its successes crash unrecorded",
             None: "unknown"}[route_present])
    print("  MSA-mask cache: %s"
          % {True: "keyed on content",
             False: "keyed on SHAPE -- a draw after the first in its 32-bucket ran a stale mask",
             None: "unknown"}[mask_fix])
    if mask_fix is False:
        target, source = target_or_fallback(arm)
        order = draw_order(arm)
        print("  bucket arithmetic: target %d residues (%s), %s"
              % (target, source,
                 "draw order from the run log" if order else "NO run log, mask test not run"))
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
    print("  %-38s %5s %9s %-12s %-16s %s"
          % ("design", "len", "design_s", "terminated", "margin", "validity"))
    out = []
    rejected = rejections(arm)
    for row in rows:
        secs = design_seconds(row.get("Timing"))
        verdict = classify(arm, row, fix_present, mask_fix)
        hit = rejected.get(row["hash"])
        if hit:
            delta = margin(hit)
            note = "[%s] %s" % (hit[3], "%+.2f" % delta if delta is not None else "no floor")
        else:
            note = ""
        print("  %-38s %5s %9s %-12s %-16s %s"
              % (row["design"][:38], row.get("length"),
                 "%.0f" % secs if secs else "?",
                 (row.get("terminated") or "completed"), note, verdict))
        out.append({"arm": arm, "design": row["design"], "seconds": secs,
                    "terminated": row.get("terminated") or "completed",
                    "validity": verdict, "commit": commit or "UNRECORDED",
                    "route": route_present, "rejection": hit,
                    "margin": margin(hit) if hit else None})
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
    unverified = [r for r in allrows if r["validity"].startswith("UNVERIFIED")]
    invalid = [r for r in allrows if not r["validity"].startswith(("VALID", "UNVERIFIED"))]
    scored = [r for r in valid if r["validity"].startswith("VALID scored")]
    eligible = [r for r in valid if r["route"] is True]
    profile_only = [r for r in valid if r["route"] is not True]
    print("trajectories in ledgers            : %d" % len(allrows))
    print("  valid as acceptance evidence     : %d" % len(valid))
    print("  not a verdict on a binder        : %d" % len(invalid))
    print("  tree unverified, provenance missing: %d" % len(unverified))
    print("  reached the acceptance filters   : %d" % len(scored))
    print("  of the valid, ACCEPTANCE-ELIGIBLE : %d  (tree carries the validation route)"
          % len(eligible))
    print("  of the valid, rejection profile   : %d  (no route: keeps its failures, drops its"
          % len(profile_only))
    print("                                        successes, so it cannot carry a ratio)")
    profile = [r for r in valid if r["rejection"]]
    if profile:
        print()
        print("rejection profile over the valid trajectories:")
        for r in sorted(profile, key=lambda r: (r["rejection"][0], r["design"])):
            stage, iptm, plddt, which = r["rejection"]
            print("  %-34s %-7s [%s] i_pTM %.2f pLDDT %.2f, %s its floor"
                  % (r["design"][:34], stage, which, iptm, plddt,
                     "%+.2f under" % r["margin"] if r["margin"] is not None
                     else "no floor at"))
    chip = sum(r["seconds"] for r in valid if r["seconds"])
    print("chip-seconds over valid trajectories: %.0f  (%.2f h, one chip)"
          % (chip, chip / 3600.0))
    accepted = 0
    for row in scored:
        head, _, _ = row["validity"].partition(" of ")
        accepted += int(head.rsplit(" ", 1)[-1])
    if scored:
        print()
        print("ACCEPTANCE: %d accepted over %d acceptance-eligible trajectories."
              % (accepted, len(eligible)))
        print("DECISION-RULE B reads a 0 only at 6 or more completed eligible trajectories;")
        print("this stands at %d." % len(eligible))
        chip_eligible = sum(r["seconds"] for r in eligible if r["seconds"])
        if accepted:
            print("chip-seconds per accepted design: %.0f over %d eligible trajectories"
                  % (chip_eligible / accepted, len(eligible)))
            print("  design-loop chip time only. The validation ensemble runs on the host, so it")
            print("  is wall clock and not chip time; see ROUTE_FILE above.")
    else:
        print()
        print("No trajectory reached the acceptance filters, so chip-seconds per accepted design")
        print("has no denominator. The honest form is a LOWER BOUND: > %.0f chip-s, resting on"
              % chip)
        print("%d valid trajectories, none of which was scored against a filter." % len(valid))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
