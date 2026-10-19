"""Order the quality-reference campaigns so a session that runs out of money still leaves a usable reference.

A rented box is a budget, not a promise: the 2026-10-19 session capped mid-campaign and left 14 of 90
trajectories. What it left was usable only by luck. This decides the order deliberately, so that the FIRST
campaigns to run are the ones `_bc4_quality_check.py` cannot pass without, and the last are the ones that only
add power.

The gate wants two targets at n >= 30 each and an A/A control that reads PASS. Against the 14 banked pdl1
trajectories (arm A1, seeds 101-103) that makes the priority:

  1. the A/A control's MISSING arm (pdl1 A2). Without it there is no A/A at all and the gate fails outright.
  2. the second target (il2r). The gate needs two targets and il2r has nothing banked, so all of it is required.
  3. pdl1 A1 top-ups. These only rebalance the A/A and thicken an arm that is already past 30 when pooled.

So the gate is satisfied after SIX of the nine campaigns, not nine. The last three are power, not pass/fail.

    python bc4_session_plan.py --request state/bc4/QUALITY-REF-REQUEST.json --budget-min 150
"""
from __future__ import annotations

import json

# Measured on the 2026-10-19 B200, three campaigns concurrent on one card, over the 14 trajectories it delivered
# (artifacts/bc4-quality-ref/pdl1/*/runs.jsonl): 1192.5 s of wall clock for 14 trajectories.
DELIVERED_S_PER_TRAJECTORY = 85.2
B200_DPH = 6.251
# What a box costs before it computes anything: clone, pip install, fetch-weights, first JAX import.
SETUP_S = 900
# One campaign's own span is the delivered rate times however many campaigns share the card.
DEFAULT_CONCURRENCY = 3


def campaign_span_s(trajectories: int, concurrency: int) -> float:
    """Wall clock for ONE campaign while `concurrency` of them share the GPU."""
    return trajectories * DELIVERED_S_PER_TRAJECTORY * concurrency


def _priority(run: dict, banked_arms: set[str]) -> tuple[int, str]:
    target, arm = run["target"], run["arm"]
    if target == "pdl1" and arm not in banked_arms:
        return 0, run["id"]          # the A/A control's missing arm
    if target != "pdl1":
        return 1, run["id"]          # the second target, nothing banked
    return 2, run["id"]              # top-ups to an arm that already has data


def plan(request: dict, budget_s: float, concurrency: int = DEFAULT_CONCURRENCY,
         setup_s: float = SETUP_S) -> dict:
    """Order the runs and say how many of them the budget actually buys."""
    banked_arms = {b["arm"] for b in request.get("banked_runs", []) if b["target"] == "pdl1"}
    runs = sorted(request.get("runs", []), key=lambda r: _priority(r, banked_arms))

    waves, wave = [], []
    for run in runs:
        wave.append(run)
        if len(wave) == concurrency:
            waves.append(wave)
            wave = []
    if wave:
        waves.append(wave)

    spent, scheduled, dropped = setup_s, [], []
    for w in waves:
        span = max(campaign_span_s(r["trajectories"], len(w)) for r in w)
        if spent + span > budget_s:
            dropped += w
            continue
        spent += span
        scheduled.append({"runs": [r["id"] for r in w], "span_s": round(span),
                          "done_by_s": round(spent)})
    # A wave that does not fit is not a reason to skip a later, cheaper one, but the ordering means a dropped
    # wave is always lower priority than every scheduled one, so report it plainly rather than backfilling.
    gate_needs = [r["id"] for r in runs if _priority(r, banked_arms)[0] < 2]
    scheduled_ids = [i for w in scheduled for i in w["runs"]]

    # The number the buyer actually needs: the wall clock at which the last gate-critical campaign finishes, walked
    # over the SAME waves so it cannot drift from the schedule above.
    need_s, seen = setup_s, set()
    for w in waves:
        if seen >= set(gate_needs):
            break
        need_s += max(campaign_span_s(r["trajectories"], len(w)) for r in w)
        seen |= {r["id"] for r in w}
    return {
        "order": [r["id"] for r in runs],
        "waves": scheduled,
        "dropped": [r["id"] for r in dropped],
        "concurrency": concurrency,
        "budget_s": budget_s,
        "projected_spend_s": round(spent),
        "projected_cost_usd": round(spent / 3600 * B200_DPH, 2),
        "gate_satisfiable": all(i in scheduled_ids for i in gate_needs),
        "gate_needs": gate_needs,
        "min_budget_min_for_gate": round(need_s / 60, 1),
        "min_cost_usd_for_gate": round(need_s / 3600 * B200_DPH, 2),
        "rate_note": (
            f"{DELIVERED_S_PER_TRAJECTORY} s per trajectory delivered at {concurrency}-way concurrency, measured "
            "on the 2026-10-19 B200 over the 14 trajectories it left (1192.5 s wall for 14). Setup is "
            f"{setup_s} s. Both are measurements, not estimates; re-derive them if the box or the pin changes."),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", required=True)
    ap.add_argument("--budget-min", type=float, required=True, help="hard wall-clock budget for the whole session")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    with open(a.request) as fh:
        request = json.load(fh)
    p = plan(request, a.budget_min * 60, a.concurrency)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(p, fh, indent=1)
            fh.write("\n")
    print(f"budget {a.budget_min:g} min at concurrency {p['concurrency']}: "
          f"{len([i for w in p['waves'] for i in w['runs']])} of {len(p['order'])} campaigns, "
          f"${p['projected_cost_usd']}, gate_satisfiable={p['gate_satisfiable']}")
    for w in p["waves"]:
        print(f"  wave {', '.join(w['runs'])}  span {w['span_s']}s  done by {w['done_by_s']}s")
    if p["dropped"]:
        print(f"  DROPPED (budget): {', '.join(p['dropped'])}")
    return 0 if p["gate_satisfiable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
