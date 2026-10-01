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
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from power import P0_DEN, P0_NUM, cp_upper, fisher_one_sided, power   # noqa: E402

OUT = HERE.parents[1] / "perf/bcw_accept/out"


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
        "trajectories_done": st.get("trajectories", 0),
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
    print("    n = {} of {} trajectories, ACCEPTED {}, candidates scored {}, rejected {}".format(
        a["trajectories_done"], a["budget"], a["accepted"],
        a["candidates_scored"], a["candidates_rejected"]))
    print("    rejected on: {}".format(a["failed_filters"] or "nothing scored yet"))
    print("    terminated at: {}".format(a["terminated"] or "no trajectory finished"))
    print("    {} gradient rounds marked, median {} s (first gap dropped, it carries "
          "compile), campaign wall {} s".format(
              a["rounds_marked"], a["round_median_s"], a["wall_s"]))
    print("    AICLK over the run: {}".format(a["aiclk"] or "written at close, run still open"))
    print("    triatt cumulative: {}".format(a["triatt"]))
    n = a["trajectories_done"]
    if n:
        print("    at this n: CP95 upper bound for {} of {} is {:.4f}; "
              "P(0 of {} | p0={}/{}) = {:.4f}".format(
                  a["accepted"], n, cp_upper(n), n, P0_NUM, P0_DEN, (1 - p0) ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arma", default="armA1")
    ap.add_argument("--armb", default="armB1")
    args = ap.parse_args()
    arms = []
    for tag in (args.arma, args.armb):
        if (OUT / tag / ".campaign_state.json").exists():
            a = arm(tag)
            arms.append(a)
            show(a)
        else:
            print("--- {}: NOT STARTED (no campaign state on disk)".format(tag))
    if len(arms) == 2:
        A, B = arms
        na, nb = A["trajectories_done"], B["trajectories_done"]
        if na and nb:
            p = fisher_one_sided(A["accepted"], na - A["accepted"],
                                 B["accepted"], nb - B["accepted"])
            print("\nFisher one-sided, arm A {}/{} against arm B {}/{}: p = {:.4f}".format(
                A["accepted"], na, B["accepted"], nb, p))
            print("Power this pair had at n = {} an arm, if arm B were truly 0: {:.3f}".format(
                min(na, nb), power(min(na, nb), P0_NUM / P0_DEN)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
