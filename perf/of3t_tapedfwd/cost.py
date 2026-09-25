#!/usr/bin/env python3
"""Price the tape-gated routes against the training step they sit in.

No card. The inputs are this row's own `out/fires_*.json` and the step decomposition
`of3t-stepfloor` banked, fetched from git so the arithmetic is re-checkable anywhere:

    git show origin/wk/of3t-stepfloor:perf/of3t_stepfloor/out/step_rekey_b_384.json

Why the step number is quoted from another row's artifact rather than re-measured here: a
full step is 466.70 s of which 456.67 s is the backward, and nothing this row changes touches
it. What this row owes is the FORWARD's share and what the levers are worth inside it, and
that is measured on this row's own card.

    python3 perf/of3t_tapedfwd/cost.py --fires perf/of3t_tapedfwd/out/fires_384.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

STEPFLOOR = "origin/wk/of3t-stepfloor"
STEP_RUN = "perf/of3t_stepfloor/out/step_rekey_b_384.json"
#: The rep `of3t-gpugap` headlines. Named so the choice is visible rather than implied: it is
#: the median-ish steady rep of arm B, not the fastest and not the cold one.
HEADLINE_REP = 2


def banked_step():
    blob = subprocess.run(["git", "show", f"{STEPFLOOR}:{STEP_RUN}"],
                          capture_output=True, text=True, check=True).stdout
    return json.loads(blob)["reps"][HEADLINE_REP]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fires", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    f = json.loads(a.fires.read_text())
    step = banked_step()

    arms = {}
    for arm in ("A", "B", "C"):
        xs = [r["s"] for r in f["reps"] if r["arm"] == arm]
        if xs:
            arms[arm] = {"median_s": round(statistics.median(xs), 4), "n": len(xs),
                         "min_s": round(min(xs), 4), "max_s": round(max(xs), 4),
                         "spread_pct": round(100 * (max(xs) - min(xs)) / statistics.median(xs), 2)}

    out = {"doc": "of3t-tapedfwd: the tape-gated routes, priced against the banked 384 step",
           "fires_source": str(a.fires), "fires_host": f["env"]["host"],
           "fires_clock": f.get("clock_line"), "arms": arms,
           "step_source": f"{STEPFLOOR}:{STEP_RUN} rep {HEADLINE_REP}",
           "step": {k: step[k] for k in
                    ("trunk_taped_cycle_s", "diffusion_s", "losses_s", "seed_upload_s",
                     "backward_s", "optimizer_s", "step_s") if k in step}}

    fwd = step["trunk_taped_cycle_s"]
    tot = step["step_s"]
    out["forward_share_pct"] = round(100 * fwd / tot, 3)
    out["backward_share_pct"] = round(100 * step["backward_s"] / tot, 3)
    # The ceiling first, because it bounds every answer below it and needs nothing measured on
    # this row's card: delete the taped forward ENTIRELY and the step still pays everything else.
    out["ceiling_if_forward_were_free_x"] = round(tot / (tot - fwd), 4)

    if "A" in arms and "B" in arms:
        a_s, b_s = arms["A"]["median_s"], arms["B"]["median_s"]
        out["lever_cost_s_on_this_card"] = round(b_s - a_s, 4)
        out["lever_cost_ratio"] = round(b_s / a_s, 4)
        # The ratio, not the seconds, is what crosses boxes: this row measures on a pc p150a
        # and the step was measured on a qb2 p300c. Applying the RATIO to the banked forward
        # assumes the two cards' route mix is the same, which the counter deltas check.
        recovered = fwd * (1 - a_s / b_s)
        out["forward_seconds_recoverable_at_384"] = round(recovered, 4)
        out["step_speedup_if_every_lever_came_back_x"] = round(tot / (tot - recovered), 5)
    if "B" in arms and "C" in arms:
        out["tape_recording_overhead_s"] = round(arms["C"]["median_s"] - arms["B"]["median_s"], 4)
        out["tape_recording_share_of_taped_forward_pct"] = round(
            100 * (arms["C"]["median_s"] - arms["B"]["median_s"]) / arms["C"]["median_s"], 2)

    # Which routes the tape gives up, from the counters rather than from the source.
    def merged(arm):
        m = {}
        for r in f["reps"]:
            if r["arm"] != arm:
                continue
            for k, v in r["counters"].items():
                if isinstance(v, list):
                    m.setdefault(k, [0, 0])
                    for i, x in enumerate(v[:2]):
                        m[k][i] += x
                elif isinstance(v, dict):
                    d = m.setdefault(k, {})
                    for kk, vv in v.items():
                        d[kk] = d.get(kk, 0) + vv
        return m
    ca, cb = merged("A"), merged("B")
    lost = {}
    for k, v in ca.items():
        served_a = v[0] if isinstance(v, list) else v.get("served", v.get("l1", 0))
        vb = cb.get(k)
        served_b = 0 if vb is None else (vb[0] if isinstance(vb, list)
                                         else vb.get("served", vb.get("l1", 0)))
        if served_a != served_b:
            lost[k] = {"untaped_served": served_a, "taped_served": served_b}
    out["routes_lost_under_tape"] = lost
    out["routes_served_untaped_but_zero"] = sorted(
        k for k, v in ca.items()
        if (v[0] if isinstance(v, list) else v.get("served", 1)) == 0)

    # The taping gate's own call sites, which cover the routes that keep no counter at all.
    sites = {}
    for r in f["reps"]:
        if r["arm"] == "A":
            for k, v in r["taping_sites"].items():
                sites[k] = sites.get(k, 0) + v
    n = max(1, sum(1 for r in f["reps"] if r["arm"] == "A"))
    out["taping_sites_per_forward"] = {k: v // n for k, v in
                                       sorted(sites.items(), key=lambda kv: -kv[1])}
    out["taping_gates_per_forward_total"] = sum(sites.values()) // n

    # Did any refusal latch grow while a taped arm ran? This is the recorded suspicion, tested
    # at runtime instead of read: arm A runs again after arm C in every rep, so a poisoned
    # latch would show as arm A serving less in rep 1 than in rep 0.
    latches = {}
    for r in f["reps"]:
        for k, v in r["counters"].items():
            if "OVER_L1" in k or "REFUSED" in k:
                latches.setdefault(k, []).append((r["rep"], r["arm"], v))
    out["latch_growth"] = latches
    per_rep_A = [sum((v[0] if isinstance(v, list) else v.get("served", 0))
                     for v in ([r["counters"][k] for k in r["counters"]]))
                 for r in f["reps"] if r["arm"] == "A"]
    out["arm_A_total_served_per_rep"] = per_rep_A
    out["arm_A_served_is_constant_across_reps"] = len(set(per_rep_A)) == 1

    print(json.dumps(out, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
