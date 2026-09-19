#!/usr/bin/env python3
"""Score a bracketed base/on/base fold A/B session.

PRE-REGISTERED 2026-09-19 09:0xZ, with blocks 0-3 visible and blocks 4-11 not yet run.
Written before the data it judges exists, so the rule cannot be fitted to the answer.

Block layout, as the driver emits it: leg 0 = base, leg 1 = on, leg 2 = base. Bracketing the
on-arm between two base arms is what makes a drifting neighbour separable from a lever: a
monotone drift over a block lands in the base-to-base difference, and the lever lands in the
on-arm's departure from the bracket's midpoint.

  delta_i  = mean(base_i0, base_i2) - on_i1      seconds SAVED by the lever in block i
  aa_i     = base_i0 - base_i2                   the block's own A/A draw, expectation 0

GO requires BOTH, and they are deliberately different in kind:

  1. FLOOR RULE: |mean(delta)| must exceed 2 * SEM of the session's own A/A draws, where
     SEM(aa) = stdev(aa_i)/sqrt(n). The A/A draw carries 2*var(base) against the delta's
     var(on) + var(base)/2, so it is the wider of the two and the floor is conservative.
  2. PAIRED RULE (uses the bracket): a two-sided paired permutation test over the per-block
     deltas, sign flips, exact when 2^n is small enough to enumerate. p < 0.05.

AMENDED 09:1xZ, same pass, still before blocks 4-11 ran, and the reason is stated because an
amendment after the fact would be worthless. Rule 1 was first registered as the harness's own
"session ratio must beat aa_floor_ratio_max" (apb_fold_ab.py:415), and that statistic is a
MAX over blocks: it grows with n while the mean it judges shrinks its error as 1/sqrt(n), so a
longer, better session scores itself harder. At 4 blocks it already reads 1.02473 against a
1.00472 session ratio, and 12 blocks can only widen it. It is the wrong shape for judging a
mean. Both are computed and both are reported; the GO condition uses the SEM form. This is a
correction to the harness's rule, and it applies to any bracketed session this campaign scores,
not just this one.

Rule 2 has a floor on what it can return: with n blocks the smallest two-sided p is 2/2^n, so
n=4 cannot go below 0.125 and n>=6 is needed to reach 0.05 at all. Stated here so a NULL at
small n is read as "not yet decidable" rather than as evidence of no effect.

NULL if the mean delta is positive but fails either rule: record and drop, per the row's kill
criterion. NEGATIVE if the mean delta is below zero and rule 2 rejects: that is the
pre-registered failure mechanism arriving, namely nlp_concat_heads keeping the padded head
lanes so proj_g/proj_o run at 16x64=1024 instead of 768, and the two wider matmuls eating the
epilogue saving. Either outcome is a result, not a measurement failure.

Pre-registered expected value, from the orchestrator's op-ladder derivation: +0.0915-0.1173 s.
Scatter for this screen class is 2.7x with a ~12 % chance of the wrong sign.
"""
import argparse
import itertools
import json
import statistics
from pathlib import Path


def median(xs):
    return statistics.median(xs)


def load_blocks(d: Path, size: str):
    blocks = {}
    for f in sorted(d.glob(f"{size}_*.json")):
        parts = f.stem.split("_")          # <size>_<arm>_<block>_<leg>
        arm, blk, leg = parts[1], int(parts[2]), int(parts[3])
        rec = json.loads(f.read_text())
        folds = [x["fold_s"] for x in rec["folds"]]
        clocks = [x["clock"] for x in rec["folds"]]
        digests = {x["cif_sha256"] for x in rec["folds"]}
        served = [tuple(x["apb_served_declined"]) for x in rec["folds"]]
        blocks.setdefault(blk, {})[leg] = {
            "arm": arm, "folds": folds, "median": median(folds),
            "clk_min": min(c["aiclk_min"] for c in clocks),
            "clk_max": max(c["aiclk_max"] for c in clocks),
            "clk_mean": round(statistics.mean(c["aiclk_mean"] for c in clocks), 1),
            "digests": digests, "served": served,
            "load": [x.get("loadavg1") for x in rec["folds"]],
        }
    return blocks


def perm_p(deltas):
    """Two-sided paired permutation test by sign flips. Exact up to 2^20."""
    n = len(deltas)
    obs = abs(statistics.mean(deltas))
    if n == 0:
        return None, 0
    if n <= 20:
        hits = total = 0
        for signs in itertools.product((1, -1), repeat=n):
            total += 1
            if abs(statistics.mean(s * d for s, d in zip(signs, deltas))) >= obs - 1e-12:
                hits += 1
        return hits / total, total
    return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--size", default="512")
    ap.add_argument("--out")
    a = ap.parse_args()

    blocks = load_blocks(Path(a.dir), a.size)
    complete = {k: v for k, v in blocks.items() if {0, 1, 2} <= set(v)}
    partial = sorted(set(blocks) - set(complete))

    rows, deltas, aas, base_folds, on_folds, aa_ratios = [], [], [], [], [], []
    base_digests, on_digests = set(), set()
    served_on, served_base = set(), set()
    for b in sorted(complete):
        legs = complete[b]
        b0, on, b2 = legs[0]["median"], legs[1]["median"], legs[2]["median"]
        delta = (b0 + b2) / 2 - on
        aa = b0 - b2
        deltas.append(delta)
        aas.append(aa)
        aa_ratios.append(max(b0 / b2, b2 / b0))
        base_folds += legs[0]["folds"] + legs[2]["folds"]
        on_folds += legs[1]["folds"]
        base_digests |= legs[0]["digests"] | legs[2]["digests"]
        on_digests |= legs[1]["digests"]
        served_base |= set(legs[0]["served"]) | set(legs[2]["served"])
        served_on |= set(legs[1]["served"])
        rows.append({"block": b, "base0": b0, "on": on, "base2": b2,
                     "delta_s": round(delta, 4), "aa_s": round(aa, 4),
                     "clk": [min(legs[l]["clk_min"] for l in (0, 1, 2)),
                             max(legs[l]["clk_max"] for l in (0, 1, 2))],
                     "load": round(statistics.mean(
                         x for l in (0, 1, 2) for x in legs[l]["load"] if x is not None), 2)})

    n = len(deltas)
    summary = {
        "size_aa": a.size, "blocks_complete": n, "blocks_partial": partial,
        "per_block": rows,
        "mean_delta_s": round(statistics.mean(deltas), 4) if n else None,
        "median_delta_s": round(median(deltas), 4) if n else None,
        "stdev_delta_s": round(statistics.stdev(deltas), 4) if n > 1 else None,
        "sem_delta_s": round(statistics.stdev(deltas) / n ** 0.5, 4) if n > 1 else None,
        "blocks_positive": sum(1 for d in deltas if d > 0),
        "mean_aa_s": round(statistics.mean(aas), 4) if n else None,
        "max_abs_aa_s": round(max(abs(x) for x in aas), 4) if n else None,
        "session_ratio_base_over_on": round(
            statistics.mean(base_folds) / statistics.mean(on_folds), 5) if n else None,
        "aa_floor_ratio_max": round(max(aa_ratios), 5) if n else None,
        "clock_min": min(r["clk"][0] for r in rows) if n else None,
        "clock_max": max(r["clk"][1] for r in rows) if n else None,
        "loadavg1_mean": round(statistics.mean(r["load"] for r in rows), 2) if n else None,
        "firing_on": sorted(served_on), "firing_base": sorted(served_base),
        "digests_base": len(base_digests), "digests_on": len(on_digests),
        "digest_moves_with_flag": bool(base_digests & on_digests) is False,
    }
    p, perms = perm_p(deltas)
    summary["perm_p_two_sided"] = p
    summary["perm_space"] = perms

    aa_sem = (statistics.stdev(aas) / n ** 0.5) if n > 1 else None
    summary["aa_sem_s"] = round(aa_sem, 4) if aa_sem else None
    summary["aa_floor_2sem_s"] = round(2 * aa_sem, 4) if aa_sem else None
    rule1 = aa_sem is not None and abs(summary["mean_delta_s"]) > 2 * aa_sem
    rule1_harness = (summary["session_ratio_base_over_on"] or 0) > (
        summary["aa_floor_ratio_max"] or 9)
    summary["rule1_harness_max_ratio_form"] = rule1_harness
    summary["perm_p_floor"] = round(2 / 2 ** n, 4) if n else None
    rule2 = p is not None and p < 0.05
    if n < 2:
        verdict = "INSUFFICIENT"
    elif rule1 and rule2 and summary["mean_delta_s"] > 0:
        verdict = "GO"
    elif summary["mean_delta_s"] is not None and summary["mean_delta_s"] < 0 and rule2:
        verdict = "NEGATIVE"
    else:
        verdict = "NULL"
    summary["rule1_mean_beats_2sem_aa"] = rule1
    summary["rule2_paired_perm_p_lt_0.05"] = rule2
    summary["verdict"] = verdict
    summary["prereg"] = {"expected_s": [0.0915, 0.1173],
                         "rule": "both rule1 and rule2, pre-registered before blocks 4-11 ran"}

    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
