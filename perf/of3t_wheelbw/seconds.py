#!/usr/bin/env python3
"""Seconds off the crop-384 taped training step, from this row's node census and a LIVE price.

A saving of this shape is a product of two numbers with different owners:

  VERBS DELETED  is this row's, and it is a COUNT. `census.py` reads it off the tape at crop
                 384 rather than modelling it, because it is the half that is cheap to get
                 wrong. It does not depend on a board or a clock.

  PER-VERB COST  is not this row's. It is a property of the step, it moves when the step
                 moves, and LEDGER R209 is what happens when a row forgets that: the campaign
                 priced four days of work at 2.107 ms per backward verb against a 466.702 s
                 step, after `502ed112e` had put a host float64 softmax and layer norm inside
                 every `tape()` and changed the taped forward by ~65x. Both constants are void
                 until re-taken, so neither is written down here.

So this script REFUSES to price against an artifact that has expired, rather than multiplying
by whatever is on disk. The check is R209's own cheapest form and the canonical copy is
`perf/of3t_orchestrator/bwd/baseline_expiry.py` on `wk/of3t-orchestrator`; it is inlined here
because this row's instrument has to run without that branch checked out.

Two readings, kept apart because their axes differ:

  A  VERBS DELETED x the step's own per-verb backward rate, both from the same run, quoted as
     a fraction of THAT run's step. Never as a fraction of 466.702 s, which R209 voided.

  B  PER-NODE delta measured by `speed.py` here. An independent number, and on its own board.
     It is a check on the shape of A: if the closures do not get faster, A is arithmetic over
     a verb count and nothing else.

Neither is a step-level A/B, and that is deliberate: a step-level A/B would spend two full
steps to read a difference the closures own entirely, and the step's own warm spread is wider
than the whole saving.

Every arm this prices runs `exact_training(False)` (the sprint's grading convention, pass 450)
and the price artifact must say so, because an arm carrying the host float64 term in its
denominator cannot see a device kernel at all.
"""
import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"

# The paths a taped training step's timing depends on. Same list as the canonical script.
MEASURED_PATHS = ["tt_bio/autograd.py", "tt_bio/taped_ttnn.py", "tt_bio/tenstorrent.py",
                  "tt_bio/train/", "tt_bio/kernels/"]

# The verbs each ACCEPTED substitution deletes per taped node, after the two withdrawals.
# `narrow` and `concat` are absent on purpose: ttnn.pad cannot front-pad a tile-layout tensor
# on device and ttnn.concat_bw throws on anything that is not rank 4, so both composed forms
# still ship. The arithmetic lives here rather than in census.py so the census stays a raw
# count and a withdrawal changes one table, not a measurement.
ACCEPTED = {
    "autograd:mul|both":            (2, 1),
    "autograd:relu|one":            (2, 1),
    "autograd:sigmoid|one":         (3, 1),
    "autograd:silu|one":            (6, 1),
    "taped_ttnn:relu":              (2, 1),
    "taped_ttnn:sigmoid":           (3, 1),
    "taped_ttnn:silu":              (6, 1),
    "taped_ttnn:multiply|fastpath": (2, 1),
}


def expiry(path, head="HEAD"):
    """LIVE / STALE / UNDATABLE for a banked perf artifact. R209's check, inlined."""
    d = json.loads(pathlib.Path(path).read_text())
    commit = (d.get("env") or {}).get("commit") or d.get("commit")
    if not commit:
        return "UNDATABLE", f"{path} records no env.commit, so it cannot be dated", d
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", commit, head],
                         cwd=REPO, capture_output=True)
    if anc.returncode != 0:
        return "STALE", f"{commit[:9]} is not an ancestor of {head}", d
    log = subprocess.run(["git", "log", "--oneline", f"{commit}..{head}", "--", *MEASURED_PATHS],
                         cwd=REPO, capture_output=True, text=True).stdout.split("\n")
    since = [l for l in log if l]
    if since:
        return "STALE", (f"{len(since)} commits touched the measured paths since "
                         f"{commit[:9]}, most recently {since[0]}"), d
    return "LIVE", f"{commit[:9]} is an ancestor of {head}, measured paths untouched since", d


def deleted(cen):
    """Verbs deleted, from the census's RAW counters and the table above."""
    branch, taped, calls = cen["branch"], cen["taped_calls"], cen["calls"]
    per = {}
    for key, (before, after) in ACCEPTED.items():
        if key.startswith("autograd:"):
            n = sum(v for k, v in branch.items() if k.startswith(key + "|"))
        elif key.endswith("|fastpath"):
            n = calls.get(key, 0)
        else:
            n = taped.get(key, 0)
        per[key] = {"nodes": n, "verbs_deleted": n * (before - after)}
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=str(OUT / "census_384.json"))
    ap.add_argument("--price", default=None,
                    help="artifact carrying bwd_ms_per_call and the step it measured, under "
                         "exact_training(False). Must pass the R209 expiry check.")
    ap.add_argument("--out", default=str(OUT / "seconds.json"))
    a = ap.parse_args()

    res = {"doc": __doc__.split("\n")[0], "reading_A": None, "reading_B": None, "owed": []}

    cen = json.loads(pathlib.Path(a.census).read_text()) if pathlib.Path(a.census).is_file() else None
    if cen is None or "branch" not in cen:
        res["owed"].append(f"{a.census} carries no node census -- reading A needs the count")
    else:
        per = deleted(cen)
        n = sum(v["verbs_deleted"] for v in per.values())
        res["verbs_deleted"] = {"total": n, "by_op": per, "tape_nodes": cen.get("tape_nodes"),
                                "census_commit": (cen.get("env") or {}).get("commit"),
                                "axis": "count, crop 384, board-insensitive"}

    if a.price is None:
        res["owed"].append("no --price artifact given. The per-verb backward cost and the step "
                           "it belongs to are VOID until re-taken (R209); this row does not "
                           "carry a constant for them.")
    else:
        state, why, pd = expiry(a.price)
        res["price"] = {"artifact": a.price, "expiry": state, "why": why}
        if state != "LIVE":
            res["owed"].append(f"{a.price} is {state}: {why}. Refusing to price against it.")
        elif "verbs_deleted" in res:
            ms = pd["bwd_ms_per_call"]
            step = pd["step_s"]
            s = res["verbs_deleted"]["total"] * ms / 1000.0
            res["reading_A"] = {
                "seconds_off_step": s,
                "bwd_ms_per_call": ms,
                "step_s": step,
                "pct_of_measured_step": 100.0 * s / step,
                "exact_training": pd.get("exact_training"),
                "board": pd.get("board"), "aiclk": pd.get("aiclk_during"),
                "axis": ("seconds off the step THIS price artifact measured, under "
                         "exact_training(False). Not a fraction of 466.702 s (R209)."),
            }

    b = {}
    for f in sorted(OUT.glob("speed_*.json")):
        d = json.loads(f.read_text())
        b[f.name] = {"board": d.get("board"), "dtype": d.get("dtype"),
                     "shape": d.get("shape"), "aiclk": d.get("aiclk_during"),
                     "exact_training": d.get("exact_training"),
                     "per_op_delta_ms": {k: (v.get("delta_s_per_node") or 0) * 1e3
                                         for k, v in d.get("ops", {}).items()},
                     "aa_floor_rel_pct": {k: (v.get("aa_floor_rel") or 0) * 100
                                          for k, v in d.get("ops", {}).items()}}
    res["reading_B"] = b or None
    if not b:
        res["owed"].append("no speed_*.json -- reading B needs the card")

    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 1 if res["owed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
