#!/usr/bin/env python3
"""Read the acceptance pair off disk and apply the arithmetic the verdict needs.

One place that turns the two campaigns into numbers, so no pass retypes a count or a p-value by
hand. Per arm: the trajectory budget spent, accepted designs, candidates scored, which filter each
rejection failed, which stage each trajectory terminated at, the round time, and the axis the
ENGINE priced -- which is not the axis arithmetic gives. 614 + 150 = 764 buckets to 768, and
BindCraft 2 runs it at 800.

It deliberately does NOT print a verdict. The rule lives in state/bcw-accept.md, written before the
counts existed, and a script that also decided would be the place to quietly change it.
"""
import argparse
import csv
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from power import P0_DEN, P0_NUM, cp_upper, fisher_one_sided, power   # noqa: E402

OUT = HERE.parents[1] / "perf/bcw_accept/out"


def resolved_trajectories(d):
    """Trajectories that actually REACHED a verdict, which is not the number the budget charged.

    BindCraft 2 charges the budget when a trajectory is CLAIMED and writes its row when it
    FINISHES (`tt_bio.bindcraft2.print_resumption`). A campaign killed mid-flight -- qb2 hard-hung
    twice in six hours and took every arm with it both times -- therefore leaves `trajectories` in
    `.campaign_state.json` counting claims that never resolved: armA1 read 6 charged against 3
    rows in the table, armB1 read 2 against 1.

    Those in-flight trajectories are CENSORED, not rejected. They neither accepted nor failed, so
    putting them in the denominator of an accept rate silently understates the uncertainty: it was
    reporting `0 of 6` for an arm that had actually resolved 3. The denominator is the table.
    """
    table = d / "1_Trajectories" / "!_Trajectories.csv"
    if not table.exists():
        return 0
    with open(table, newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def stamp(tag):
    """The rung's own JSON header, printed before the campaign starts."""
    log = OUT / f"{tag}.log"
    if not log.exists():
        return {}
    raw = log.read_text(errors="replace")
    i = raw.find('{\n "host"')
    return {} if i < 0 else json.JSONDecoder().raw_decode(raw[i:])[0]


def arm(tag):
    d = OUT / tag
    state = d / ".campaign_state.json"
    st = json.loads(state.read_text()) if state.exists() else {}
    rj = d / "rounds.json"
    rounds = json.loads(rj.read_text()) if rj.exists() else []
    gaps = [round(b["t"] - a["t"], 2) for a, b in zip(rounds, rounds[1:])
            if b["slot"] == a["slot"]]
    s = stamp(tag)
    rej = st.get("rejections", {})
    return {
        "tag": tag,
        "target": s.get("target"), "binder": s.get("binder"), "card": s.get("card"),
        "seed": s.get("seed"), "commit": (s.get("commit") or "")[:9],
        "axis_arithmetic": s.get("tokens"), "axis_engine": s.get("design_tokens"),
        "axis_seam": s.get("evoformer_axis"), "auto": s.get("auto_would_choose"),
        "budget": s.get("max_trajectories"),
        # charged = claims against the budget; resolved = verdicts reached. They differ by
        # whatever was in flight when the box went down, and only `resolved` is a denominator.
        "charged": st.get("trajectories", 0),
        "resolved": resolved_trajectories(d),
        "in_flight": max(0, st.get("trajectories", 0) - resolved_trajectories(d)),
        "accepted": st.get("accepted", 0),
        "candidates_scored": rej.get("candidates_scored", 0),
        "candidates_rejected": rej.get("candidates_rejected", 0),
        "failed_filters": rej.get("failed_filters", {}),
        "terminated": rej.get("terminated", {}),
        "rounds_marked": len(rounds),
        # The first gap carries compile and is not a round time.
        "round_median_s": round(statistics.median(gaps[1:]), 2) if len(gaps) > 1 else None,
        "aiclk": s.get("aiclk_run"),
        "wall_s": round(rounds[-1]["t"] - rounds[0]["t"], 1) if len(rounds) > 1 else None,
        "triatt": rounds[-1]["triatt"] if rounds else None,
    }


def show(a):
    p0 = P0_NUM / P0_DEN
    print("--- {}: {} + {} aa, card {}, seed {}, tree {}".format(
        a["tag"], a["target"], a["binder"], a["card"], a["seed"], a["commit"]))
    print("    axis: arithmetic {}, engine priced {}, seam {}".format(
        a["axis_arithmetic"], a["axis_engine"], a["axis_seam"]))
    print("    auto chose: {}".format(a["auto"]))
    print("    n = {} RESOLVED of {} charged against a budget of {}, ACCEPTED {}, "
          "candidates scored {}, rejected {}".format(
              a["resolved"], a["charged"], a["budget"], a["accepted"],
              a["candidates_scored"], a["candidates_rejected"]))
    if a["in_flight"]:
        print("    {} charged but not resolved: claimed and either still running or killed "
              "in flight. Charged to the budget, censored, NOT counted in n.".format(
                  a["in_flight"]))
    print("    rejected on: {}".format(a["failed_filters"] or "nothing scored yet"))
    print("    terminated at: {}".format(a["terminated"] or "no trajectory finished"))
    print("    {} gradient rounds marked, median {} s (first gap dropped, it carries "
          "compile), campaign wall {} s".format(
              a["rounds_marked"], a["round_median_s"], a["wall_s"]))
    print("    AICLK over the run: {}".format(a["aiclk"] or "written at close, run still open"))
    print("    triatt cumulative: {}".format(a["triatt"]))
    n = a["resolved"]
    if n:
        print("    at this n: CP95 upper bound for {} of {} is {:.4f}; "
              "P(0 of {} | p0={}/{}) = {:.4f}".format(
                  a["accepted"], n, cp_upper(n), n, P0_NUM, P0_DEN, (1 - p0) ** n))


def merge(counts):
    """Add the counts of one arm across its campaigns.

    An arm accumulates n two ways, and both are pooled here. A dead campaign is RESUMED into its
    own folder, which continues its own seed and builds trajectory recipes it has not attempted
    (`resume` and `max_trajectories` are excluded from BindCraft 2's design identity). A second
    card gets a campaign under a DIFFERENT seed: same target, same binder, same tree, same board.
    Pooling those is the design, not a convenience. A second campaign at the SAME seed would
    reproduce the first trajectory for trajectory and add no n at all, which is why campaign.sh
    takes the seed as an argument. The labels and their seeds are printed so a pool can be checked
    for a repeated seed, and a repeat is called out rather than silently counted twice.
    """
    pooled = {"tags": [c["tag"] for c in counts], "seeds": [c["seed"] for c in counts],
              "cards": [c["card"] for c in counts],
              "commits": sorted({c["commit"] for c in counts}),
              "charged": 0, "resolved": 0, "in_flight": 0, "accepted": 0,
              "candidates_scored": 0, "candidates_rejected": 0,
              "failed_filters": {}, "terminated": {}}
    for c in counts:
        for k in ("charged", "resolved", "in_flight", "accepted", "candidates_scored",
                  "candidates_rejected"):
            pooled[k] += c[k]
        for k in ("failed_filters", "terminated"):
            for name, v in (c[k] or {}).items():
                pooled[k][name] = pooled[k].get(name, 0) + v
    return pooled


def show_pool(name, pooled):
    p0 = P0_NUM / P0_DEN
    n, acc = pooled["resolved"], pooled["accepted"]
    print("=== {} POOLED over {}, seeds {}, cards {}, tree {}".format(
        name, ", ".join(pooled["tags"]), pooled["seeds"], pooled["cards"],
        ", ".join(pooled["commits"])))
    if len(set(pooled["seeds"])) != len(pooled["seeds"]):
        print("    !! a seed is REPEATED in this pool: those campaigns run the same "
              "trajectories and must not both be counted")
    print("    n = {} RESOLVED ({} charged, {} in flight or lost), ACCEPTED {}, "
          "candidates scored {}, rejected {}".format(
              n, pooled["charged"], pooled["in_flight"], acc,
              pooled["candidates_scored"], pooled["candidates_rejected"]))
    print("    rejected on: {}".format(pooled["failed_filters"] or "nothing scored yet"))
    print("    terminated at: {}".format(pooled["terminated"] or "no trajectory finished"))
    if n:
        print("    CP95 upper bound for {} of {} is {:.4f}; P(0 of {} | p0={}/{}) = {:.4f}".format(
            acc, n, cp_upper(n), n, P0_NUM, P0_DEN, (1 - p0) ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arma", default="armA1",
                    help="comma-separated labels of arm A's campaigns, pooled")
    ap.add_argument("--armb", default="armB1,armB2,armB3,armB4",
                    help="comma-separated labels of arm B's campaigns, pooled")
    args = ap.parse_args()
    pools = []
    for name, tags in (("ARM A", args.arma), ("ARM B", args.armb)):
        counts = []
        for tag in [t for t in tags.split(",") if t]:
            if (OUT / tag / ".campaign_state.json").exists():
                c = arm(tag)
                counts.append(c)
                show(c)
            else:
                print("--- {}: NOT STARTED (no campaign state on disk)".format(tag))
        if counts:
            pooled = merge(counts)
            show_pool(name, pooled)
            pools.append(pooled)
        print()
    if len(pools) == 2:
        A, B = pools
        na, nb = A["resolved"], B["resolved"]
        if na and nb:
            p = fisher_one_sided(A["accepted"], na - A["accepted"],
                                 B["accepted"], nb - B["accepted"])
            print("Fisher one-sided, arm A {}/{} against arm B {}/{}: p = {:.4f}".format(
                A["accepted"], na, B["accepted"], nb, p))
            print("Power this pair had at n = {} an arm, if arm B were truly 0: {:.3f}".format(
                min(na, nb), power(min(na, nb), P0_NUM / P0_DEN)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
