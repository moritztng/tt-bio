#!/usr/bin/env python3
"""D53's three questions, answered from the per-tensor array that already existed.

ORCHESTRATOR AMENDMENT 1 asks for a per-tensor dump on `device_gradient.py` and a re-run at
`--structs all` against `diffcap043`, because `device_gradient_043all.json` keeps only
aggregates plus worst10/best10 and the mass-weighted headline A23 requires is therefore
underivable. The dump flag already existed (`--dump-per-tensor`) and the run was already
taken: `perf/of3t_rebase/device_gradient_043pt.json` carries all 547 tensors with `rel_l2`,
`ref_norm`, `device_norm`, `norm_ratio` and `cos`.

It is the SAME measurement as the quoted one, checked rather than assumed: every one of the 26
aggregate keys is equal, all 48 `forward_rel` entries are bit-identical doubles, the 48-entry
accumulation probe is bit-identical, and `worst10` matches. Two runs against different
captured boundaries do not agree to the last bit of a double 97 times. So this is a host-only
re-analysis; no card was opened and no number was re-measured.

The three answers:

  1. the MASS-WEIGHTED rel for the diffusion arm, with the A20 denominator -- the full
     reference mass our module is asked to cover, not the mass it happened to reach;
  2. the per-block DiT error profile against the per-block MASS profile
     (`perf/of3t_orchestrator/BLOCK_MASS_PROFILE.json`);
  3. the test of D53's open hypothesis: inside the
     `attention_pair_bias.layer_norm_a.layer_norm_s.weight` family, does rel rise with
     ||g_ref||? The six points on the record are the six WORST of 24, a selected tail, and a
     selected tail manufactures that trend whether or not it exists. This runs the whole
     family and reports it refuted if the family shows no trend.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

MODEL_SQ = 10.279642678524985          # 0.4.3, exhaustive over 4,170 tensors
COND_SQ = 3.7979352325432836           # diffusion_conditioning, measured by this row
FAMILY = "attention_pair_bias.layer_norm_a.layer_norm_s.weight"
BLOCK = re.compile(r"^diffusion_transformer\.blocks\.(\d+)\.")


def spearman(xs, ys):
    """Rank correlation without scipy. Ties take their average rank."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx if sxx else 0.0
    r2 = (sxy * sxy) / (sxx * syy) if sxx and syy else 0.0
    return slope, my - slope * mx, r2


def mw(rows):
    """A23 rule 2: rel_l2 over the concatenated set, mass-weighted by construction."""
    den = sum(r["ref_norm"] ** 2 for r in rows)
    if not den:
        return None
    return (sum((r["rel_l2"] * r["ref_norm"]) ** 2 for r in rows) / den) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tensor", type=Path,
                    default=Path("perf/of3t_rebase/device_gradient_043pt.json"))
    ap.add_argument("--aggregate", type=Path,
                    default=Path("perf/of3t_rebase/device_gradient_043all.json"))
    ap.add_argument("--block-mass", type=Path,
                    default=Path("perf/of3t_conditioning/BLOCK_MASS_PROFILE_043.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("perf/of3t_conditioning/MASS_ON_THE_DIFFUSION_ARM.json"))
    a = ap.parse_args()

    P = json.loads(a.per_tensor.read_text())
    A = json.loads(a.aggregate.read_text())
    rows = P["per_tensor"]

    # ---- 0. is the array the quoted run? ----------------------------------------------------
    keys = [k for k in A if k not in ("per_tensor", "per_tensor_dumped")]
    ident = {"aggregate_keys_equal": sum(1 for k in keys if A[k] == P.get(k)),
             "aggregate_keys": len(keys),
             "forward_rel_bit_identical": A["forward_rel"] == P["forward_rel"],
             "forward_rel_n": len(A["forward_rel"]),
             "accumulation_probe_bit_identical":
                 A["accumulation_probe"] == P["accumulation_probe"],
             "worst10_equal": A["worst10"] == P["worst10"]}

    # ---- 1. the mass-weighted headline, with the A20 denominator ----------------------------
    cmp_sq = sum(r["ref_norm"] ** 2 for r in rows)
    # `share_of_diffusion_sq_norm` is cmp_sq over the WHOLE captured reference, which includes
    # the 26 conditioning tensors our module does not contain. Back the total out of it, then
    # remove the conditioning to get the mass this module is actually asked to cover.
    total_capture_sq = cmp_sq / A["share_of_diffusion_sq_norm"]
    in_scope_sq = total_capture_sq - COND_SQ
    head = {
        "mass_weighted_rel_l2": mw(rows),
        "median_over_tensors": A["median_rel"],
        "worst_rel": A["worst_rel"], "worst_tensor": A["worst_tensor"],
        "compared": len(rows), "over_5e-2": A["over_5e-2"],
        "compared_pct_of_model": 100.0 * cmp_sq / MODEL_SQ,
        "in_scope_pct_of_model": 100.0 * in_scope_sq / MODEL_SQ,
        "reach_pct_of_scope": 100.0 * cmp_sq / in_scope_sq,
        "unreached_pct_of_model": 100.0 * (in_scope_sq - cmp_sq) / MODEL_SQ,
        "device_weights_named": A["device_weights_named"],
        "device_weights_reachable": A["device_weights_reachable"],
        "zero_model_mass_weighted": 1.0,
    }

    # ---- 2. the per-block DiT profile against the per-block mass profile --------------------
    prof, blocks = {}, {}
    for r in rows:
        m = BLOCK.match(r["tensor"])
        if m:
            blocks.setdefault(m.group(1), []).append(r)
    mass = {}
    if a.block_mass.is_file():
        mass = json.loads(a.block_mass.read_text()).get("dit_24_blocks_pct_of_model", {})
    for k in sorted(blocks, key=int):
        b = blocks[k]
        sq = sum(r["ref_norm"] ** 2 for r in b)
        prof[k] = {"tensors": len(b), "mass_weighted_rel": mw(b),
                   "median_rel": sorted(r["rel_l2"] for r in b)[len(b) // 2],
                   "worst_rel": max(r["rel_l2"] for r in b),
                   "compared_pct_of_model": 100.0 * sq / MODEL_SQ,
                   "block_pct_of_model_reference": mass.get(k)}

    # ---- 3. D53's hypothesis, on the WHOLE family instead of its worst six ------------------
    fam = sorted((r for r in rows if r["tensor"].endswith(FAMILY) and BLOCK.match(r["tensor"])),
                 key=lambda r: int(BLOCK.match(r["tensor"]).group(1)))
    xs = [math.log10(r["ref_norm"]) for r in fam]
    ys = [math.log10(r["rel_l2"]) for r in fam]
    rho = spearman([r["ref_norm"] for r in fam], [r["rel_l2"] for r in fam])
    slope, intercept, r2 = ols(xs, ys)
    top6 = sorted(fam, key=lambda r: -r["rel_l2"])[:6]
    rho6 = spearman([r["ref_norm"] for r in top6], [r["rel_l2"] for r in top6])
    family = {
        "leaf": FAMILY, "present": len(fam), "of_blocks": 24,
        "spearman_rel_vs_ref_norm_full_family": rho,
        "spearman_on_the_worst_six_only": rho6,
        "loglog_slope": slope, "loglog_intercept": intercept, "loglog_r2": r2,
        "points": [{"block": int(BLOCK.match(r["tensor"]).group(1)), "rel_l2": r["rel_l2"],
                    "ref_norm": r["ref_norm"], "norm_ratio": r["norm_ratio"], "cos": r["cos"],
                    "pct_of_model": 100.0 * r["ref_norm"] ** 2 / MODEL_SQ} for r in fam],
    }

    # ---- 4. what the magnitude hypothesis missed: DIRECTION --------------------------------
    # The family table answers D53's question and then asks a better one. Four of the 24 have a
    # NEGATIVE cosine against the reference -- our gradient points the other way on those
    # tensors -- and block 8's norm ratio is 19.24. Neither is a precision effect, and neither
    # is visible in `rel` alone: rel^2 = 1 + r^2 - 2rc, so rel bounds r to [1-rel, 1+rel] and
    # says nothing about c. This is the cheap statistic the run already carried.
    def band(sel):
        b = [r for r in rows if sel(r)]
        sq = sum(r["ref_norm"] ** 2 for r in b)
        return {"tensors": len(b), "pct_of_model": 100.0 * sq / MODEL_SQ,
                "pct_of_compared_mass": 100.0 * sq / cmp_sq}

    dit = [r for r in rows if BLOCK.match(r["tensor"])]
    dit_sq = sum(r["ref_norm"] ** 2 for r in dit)
    direction = {
        "cos_below_0": band(lambda r: r["cos"] is not None and r["cos"] < 0.0),
        "cos_below_0.5": band(lambda r: r["cos"] is not None and r["cos"] < 0.5),
        "cos_below_0.9": band(lambda r: r["cos"] is not None and r["cos"] < 0.9),
        "cos_at_or_above_0.99": band(lambda r: r["cos"] is not None and r["cos"] >= 0.99),
        "norm_ratio_over_2": band(lambda r: r["norm_ratio"] is not None
                                  and r["norm_ratio"] > 2.0),
        "norm_ratio_under_0.5": band(lambda r: r["norm_ratio"] is not None
                                     and r["norm_ratio"] < 0.5),
        "what": "rel cannot separate these. A tensor with cos = -0.8 and r = 1.87 (DiT block 0's "
                "layer_norm_s.weight) and a tensor with cos = 0.999 and r = 0.5 can read the "
                "same rel, and only one of them is a precision result.",
        "dit_pct_of_compared_mass": 100.0 * dit_sq / cmp_sq,
        "mass_weighted_rel_dit_only": mw(dit),
        "mass_weighted_rel_outside_dit": mw([r for r in rows if not BLOCK.match(r["tensor"])]),
    }
    # The heaviest disagreements, by the mass they actually move rather than by rel.
    worst_by_mass = sorted(rows, key=lambda r: -(r["rel_l2"] * r["ref_norm"]) ** 2)[:12]
    direction["worst_12_by_mass_moved"] = [
        {"tensor": r["tensor"], "rel_l2": r["rel_l2"], "norm_ratio": r["norm_ratio"],
         "cos": r["cos"], "pct_of_model": 100.0 * r["ref_norm"] ** 2 / MODEL_SQ,
         "pct_of_squared_error": 100.0 * (r["rel_l2"] * r["ref_norm"]) ** 2
                                 / sum((q["rel_l2"] * q["ref_norm"]) ** 2 for q in rows)}
        for r in worst_by_mass]

    # ---- 5. by LEAF, because a worst tensor names the tail and not the locus ---------------
    # One tensor is 94 % of the squared error, which is exactly the shape that tempts a reader
    # into calling block 8 the location. Group the whole compared set by leaf name first: if
    # the leaf misbehaves across blocks, the locus is the op, and block 8 is only where the
    # mass happens to sit.
    leaves = {}
    for r in rows:
        leaf = BLOCK.sub("", r["tensor"])
        leaves.setdefault(leaf, []).append(r)
    err_tot = sum((r["rel_l2"] * r["ref_norm"]) ** 2 for r in rows)
    by_leaf = sorted(
        ({"leaf": k, "tensors": len(v), "mass_weighted_rel": mw(v),
          "median_rel": sorted(q["rel_l2"] for q in v)[len(v) // 2],
          "worst_rel": max(q["rel_l2"] for q in v),
          "min_cos": min(q["cos"] for q in v if q["cos"] is not None),
          "max_norm_ratio": max(q["norm_ratio"] for q in v if q["norm_ratio"] is not None),
          "pct_of_model": 100.0 * sum(q["ref_norm"] ** 2 for q in v) / MODEL_SQ,
          "pct_of_squared_error":
              100.0 * sum((q["rel_l2"] * q["ref_norm"]) ** 2 for q in v) / err_tot}
         for k, v in leaves.items() if len(v) > 1),
        key=lambda d: -d["pct_of_squared_error"])

    rep = {"what": "D53's three questions, answered from the per-tensor array of the SAME run "
                   "the campaign quotes. Host-only re-analysis; no card opened.",
           "the_rerun_was_not_needed": ident,
           "headline_A23": head, "dit_block_profile": prof, "family_hypothesis": family,
           "direction_not_magnitude": direction, "by_leaf": by_leaf}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"the_rerun_was_not_needed": ident, "headline_A23": head,
                      "family_hypothesis": {k: v for k, v in family.items() if k != "points"}},
                     indent=1, default=str))
    print("\nDiT per-block: blk tensors  mw_rel   median   worst  compared%model  block%model")
    for k, v in prof.items():
        print(f"  {k:>2} {v['tensors']:>5}  {v['mass_weighted_rel']:8.4f} "
              f"{v['median_rel']:8.4f} {v['worst_rel']:8.3f}  "
              f"{v['compared_pct_of_model']:10.5f}  {v['block_pct_of_model_reference']}")
    print("\nfamily: blk  rel_l2      ref_norm    %model     r        cos")
    for p in family["points"]:
        print(f"  {p['block']:>2}  {p['rel_l2']:10.4f}  {p['ref_norm']:10.6f}  "
              f"{p['pct_of_model']:8.5f}  {p['norm_ratio']:7.4f}  {p['cos']:9.6f}")
    print("\ndirection:")
    for k in ("cos_below_0", "cos_below_0.5", "cos_below_0.9", "cos_at_or_above_0.99",
              "norm_ratio_over_2", "norm_ratio_under_0.5"):
        v = direction[k]
        print(f"  {k:<22} {v['tensors']:>4} tensors  {v['pct_of_model']:8.4f} % of model  "
              f"{v['pct_of_compared_mass']:8.4f} % of compared mass")
    print(f"  DiT holds {direction['dit_pct_of_compared_mass']:.4f} % of the compared mass; "
          f"mass-weighted rel inside the DiT {direction['mass_weighted_rel_dit_only']:.4f}, "
          f"outside it {direction['mass_weighted_rel_outside_dit']:.4f}")
    print("\nworst 12 by the mass they move:")
    for r in direction["worst_12_by_mass_moved"]:
        print(f"  {r['pct_of_squared_error']:6.2f} % of sq error | rel {r['rel_l2']:9.4f} "
              f"r {r['norm_ratio']:8.4f} cos {r['cos']:9.6f} | {r['pct_of_model']:7.4f} % model"
              f" | {r['tensor']}")
    print("\nby leaf (multi-block leaves, top 10 by squared error):")
    print("   %sqerr  %model  n   mw_rel   median   worst   min_cos  max_r   leaf")
    for d in by_leaf[:10]:
        print(f"  {d['pct_of_squared_error']:7.3f} {d['pct_of_model']:7.3f} {d['tensors']:>3} "
              f"{d['mass_weighted_rel']:8.4f} {d['median_rel']:8.4f} {d['worst_rel']:7.3f} "
              f"{d['min_cos']:8.4f} {d['max_norm_ratio']:7.3f}  {d['leaf']}")
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
