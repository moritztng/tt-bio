#!/usr/bin/env python3
"""Seconds off the 466.70 s crop-384 step, from the census and the two per-verb instruments.

Two readings, kept apart because their axes differ and neither is the other's check by
accident:

  A  VERBS DELETED x the step's OWN per-verb rate. `of3t-gpugap` measured 168,922 backward
     verb calls costing 356.00 s on the very step being priced -- tt-quietbox2 dev3, p300c,
     1350 MHz -- so 2.107 ms is what one backward verb costs THERE. This is the reading that
     answers "seconds off 466.70 s", because both halves come from the same run.

  B  PER-NODE delta measured by `speed.py` here, on pc's p150a. An independent number on a
     different board class, so it cannot be quoted as seconds off a p300c step. It is a check
     on the shape of A: if the closures do not actually get faster, A is arithmetic over a
     verb count and nothing else.

Neither is a step-level A/B. A step-level A/B would spend 466 s per arm to read a difference
the closures own entirely, and the step's own warm spread -- 459.32, 456.67 and 540.87 s
across three warm reps of `step_rekey_b_384.json` -- is wider than the whole saving.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"


# The verbs each ACCEPTED substitution deletes per taped node, after the two withdrawals.
# `narrow` and `concat` are absent on purpose: ttnn.pad cannot front-pad a tile-layout tensor
# on device and ttnn.concat_bw throws on anything that is not rank 4, so both composed forms
# still ship. The arithmetic lives here rather than in census.py so the census stays a raw
# count and a withdrawal changes one table, not a measurement.
ACCEPTED = {
    "autograd:mul|both":          (2, 1),
    "autograd:relu|one":          (2, 1),
    "autograd:sigmoid|one":       (3, 1),
    "autograd:silu|one":          (6, 1),
    "taped_ttnn:relu":            (2, 1),
    "taped_ttnn:sigmoid":         (3, 1),
    "taped_ttnn:silu":            (6, 1),
    "taped_ttnn:multiply|fastpath": (2, 1),
}


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


def load(p):
    q = pathlib.Path(p)
    return json.loads(q.read_text()) if q.is_file() else None


def main() -> int:
    gap = load(REPO / "perf/of3t_gpugap/PARTITION.json")
    step = load(REPO / "perf/of3t_stepfloor/out/step_rekey_b_384.json")
    cen = load(OUT / "census_384.json")
    if gap is None or step is None:
        print("missing the banked step artifacts", file=sys.stderr)
        return 2
    rate = gap["backward_verb_rate"]
    ms = rate["bwd_ms_per_call"]
    rep = next(r for r in step["reps"] if r["step_s"] == 466.702)
    warm = [r["step_s"] for r in step["reps"] if not r["cold"]]

    res = {
        "axis": ("seconds off the crop-384 taped training step recorded on tt-quietbox2 dev3, "
                 "a p300c, AICLK median 1350 MHz over 578 DURING samples"),
        "step_s": rep["step_s"], "backward_s": rep["backward_s"],
        "warm_step_spread_s": [min(warm), max(warm)],
        "bwd_calls": rate["backward_calls"], "bwd_ms_per_call": ms,
        "reading_A": None, "reading_B": None,
    }
    if cen is None:
        print("census_384.json not written yet -- reading A needs the node count")
    else:
        per = deleted(cen)
        n = sum(v["verbs_deleted"] for v in per.values())
        s = n * ms / 1000.0
        res["reading_A"] = {
            "verbs_deleted": n, "by_op": per, "tape_nodes": cen.get("tape_nodes"),
            "seconds_off_step": s,
            "pct_of_step": 100.0 * s / rep["step_s"],
            "pct_of_backward": 100.0 * s / rep["backward_s"],
            "bwd_calls_after": rate["backward_calls"] - n,
            "note": ("the census counts THIS row's crop-384 forward; the 168,922 call total "
                     "is of3t-gpugap's, on the same configuration but a separate run"),
        }
    b = {}
    for f in sorted(OUT.glob("speed_*.json")):
        d = load(f)
        b[f.name] = {"board": d.get("board"), "dtype": d.get("dtype"),
                     "shape": d.get("shape"), "aiclk": d.get("aiclk_during"),
                     "per_op_delta_ms": {k: (v.get("delta_s_per_node") or 0) * 1e3
                                         for k, v in d.get("ops", {}).items()},
                     "aa_floor_rel_pct": {k: (v.get("aa_floor_rel") or 0) * 100
                                          for k, v in d.get("ops", {}).items()}}
    res["reading_B"] = b or None
    (OUT / "seconds.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
