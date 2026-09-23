#!/usr/bin/env python3
"""D191: where the trunk's backward error GROWS between padded width 64 and 384.

of3t-frame384 measured the two endpoints in one frame on one host: ours reads 0.8354121633 at
384 and 0.3833065668 at 64 against the capture's own float64, while upstream's own bf16 floor is
flat (0.3739383921 / 0.3739375769, both built on qb1). This row decomposes the DIFFERENCE.

Why a difference and not a share. A mass-weighted rel L2 is

    mw^2 = sum_t m_t rel_t^2 = sum_t ||ours_t - ref_t||^2 / ||g_ref||^2

so the per-tensor terms are additive in mw^2 and

    GROWTH_t = m_t rel_t^2 (384) - m_t rel_t^2 (64)

sums EXACTLY to mw(384)^2 - mw(64)^2. That is arithmetic, not a model, and the script checks the
closure. A share cannot be used in its place: a share moves when its denominator collapses, and
the question here is which tensors' absolute error grew.

The denominator happens not to move at all, and that is measured rather than assumed
(REF_WIDTH_INVARIANCE below): upstream's float64 trunk gradient is the same function of width to
3e-15, because the content is the same 56 real tokens at both widths and upstream's pad terms are
exactly zero. So a growth in m_t rel_t^2 IS a growth in ||ours_t - ref_t||^2, and both are
reported.

The scoring math is of3t-trunkg043's `score.py`, imported rather than reimplemented, so this row
shares one definition of rel_l2, of the D35 per-tensor triple and of the leaf aggregation with
every row above it (A23).
"""
import argparse, hashlib, json, math, os, sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "of3t_trunkg043"))
from score import triple                                                    # noqa: E402

PRE = "pairformer_stack.blocks."
# of3t-apbback's BISECT_BLK47 scope, verbatim: `attn_pair_bias.* and single_transition.*` of
# block 47 and nothing else. 16 tensors. The softmax containment question is asked against
# exactly this set, because that is the set the 51.55 % was measured over.
BISECT_BLOCK = 47
BISECT_LEAF_PREFIXES = ("attn_pair_bias.", "single_transition.")


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def trunk(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = d["grads"] if isinstance(d, dict) and "grads" in d else d
    return {k: v for k, v in g.items() if k.startswith(PRE) and v is not None}


def leaf_of(k):
    return k.split(".", 3)[3]


def section_of(k):
    return leaf_of(k).split(".")[0]


def block_of(k):
    return int(k.split(".")[2])


def rows_for(ours, ref, keys):
    """The per-tensor table at one width. err_sq is the ABSOLUTE squared error."""
    out = {}
    for k in keys:
        rel, ratio, cos, nr, nm = triple(ours[k], ref[k])
        out[k] = {"rel_l2": rel, "norm_ratio": ratio, "cos": cos,
                  "ref_norm": nr, "our_norm": nm,
                  "err_sq": float((ours[k].reshape(-1).to(torch.float64)
                                   - ref[k].reshape(-1).to(torch.float64)).pow(2).sum()),
                  "ref_sq": nr * nr}
    return out


def mw(rows, keys):
    tot = sum(rows[k]["ref_sq"] for k in keys)
    return math.sqrt(sum(rows[k]["err_sq"] for k in keys) / tot), tot


def agg(growth, keyfn, keys):
    """Aggregate the growth. Reports the DIFFERENCE and both endpoints, never a share alone."""
    o = defaultdict(lambda: {"n": 0, "at_384": 0.0, "at_64": 0.0, "growth": 0.0,
                             "abs_err_sq_384": 0.0, "abs_err_sq_64": 0.0,
                             "ref_sq_384": 0.0, "ref_sq_64": 0.0})
    for k in keys:
        g = growth[k]
        a = o[keyfn(k)]
        a["n"] += 1
        for f in ("at_384", "at_64", "growth", "abs_err_sq_384", "abs_err_sq_64",
                  "ref_sq_384", "ref_sq_64"):
            a[f] += g[f]
    total = sum(growth[k]["growth"] for k in keys)
    for a in o.values():
        a["share_of_the_growth"] = a["growth"] / total if total else float("nan")
        r = a["abs_err_sq_384"] / a["abs_err_sq_64"] if a["abs_err_sq_64"] else float("inf")
        a["abs_err_sq_ratio"] = r
        # ||e||^2 ratio written as 6^p, 6 = 384/64. p separates a pair-axis leak (p>2) from a
        # single-axis one (0.8<p<2) from a uniform shape-keyed multiplier (p equal everywhere).
        a["width_exponent_p"] = (math.log(r) / math.log(6.0)
                                 if r > 0 and math.isfinite(r) else float("nan"))
        a["rel_l2_384"] = math.sqrt(a["abs_err_sq_384"] / a["ref_sq_384"]) if a["ref_sq_384"] else 0.0
        a["rel_l2_64"] = math.sqrt(a["abs_err_sq_64"] / a["ref_sq_64"]) if a["ref_sq_64"] else 0.0
    return dict(sorted(o.items(), key=lambda kv: -kv[1]["growth"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    for n in ("ref384", "ref64", "floor384", "floor64", "ours384", "ours64"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--boundary384", required=True)
    ap.add_argument("--boundary64", required=True)
    ap.add_argument("--floor-host", required=True, metavar="HOST",
                    help="D189: the floor is host-dependent at the percent level, so the host "
                         "that produced it is part of every ratio quoted from it")
    ap.add_argument("--arm-host", required=True, metavar="HOST")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {"what": __doc__.strip().splitlines()[0],
           "scored_on": os.uname().nodename,
           "device_involved": False,
           "hosts": {"floor_and_local_float64_reference_built_on": a.floor_host,
                     "device_arm_built_on": a.arm_host,
                     "scored_on": os.uname().nodename},
           "widths": [64, 384], "controls": {}, "digests": {}}
    paths = {"ref384": a.ref384, "ref64": a.ref64, "floor384": a.floor384,
             "floor64": a.floor64, "ours384": a.ours384, "ours64": a.ours64,
             "boundary384": a.boundary384, "boundary64": a.boundary64}
    for n, p in paths.items():
        out["digests"][n] = {"path": p, "sha256": sha256_file(p)}

    # ---- P2: is the c64 boundary exactly the n384 boundary cropped? -----------------------
    b3 = torch.load(a.boundary384, map_location="cpu", weights_only=False)
    b6 = torch.load(a.boundary64, map_location="cpu", weights_only=False)
    cid = {}
    for fld, sl in (("s_in", lambda t: t[:, :64]),
                    ("z_in", lambda t: t[:, :64, :64]),
                    ("single_mask", lambda t: t[..., :64]),
                    ("pair_mask", lambda t: t[..., :64, :64])):
        x, y = sl(b3[fld].to(torch.float64)), b6[fld].to(torch.float64)
        cid[fld] = {"shape_384": list(b3[fld].shape), "shape_64": list(b6[fld].shape),
                    "bit_identical": bool(torch.equal(x, y)),
                    "max_absdiff": float((x - y).abs().max())}
    real = int((b3["single_mask"].reshape(-1) > 0).sum())
    pad3, pad6 = 384 - real, 64 - real
    cid["real_tokens"] = real
    cid["pad_rows"] = {"64": pad6, "384": pad3, "ratio": pad3 / pad6}
    cid["pad_cells_z"] = {"64": 64 * 64 - real * real, "384": 384 * 384 - real * real,
                          "ratio": (384 * 384 - real * real) / (64 * 64 - real * real)}
    for w, b in ((384, b3), (64, b6)):
        m = (b["single_mask"].reshape(-1) > 0)
        s = b["s_in"].to(torch.float64)
        z = b["z_in"].to(torch.float64)
        cid[f"w{w}_input_mass"] = {
            "s_norm": float(s.norm()), "s_real_norm": float(s[:, m].norm()),
            "z_norm": float(z.norm()),
            "z_real_norm": float(z[:, m][:, :, m].norm())}
    out["controls"]["BOUNDARY_IS_A_CROP"] = cid
    del b3, b6

    # ---- the six arms --------------------------------------------------------------------
    ref3, ref6 = trunk(a.ref384), trunk(a.ref64)
    keys = sorted(set(ref3) & set(ref6))
    out["scope"] = {"compared": len(keys), "in_ref_384": len(ref3), "in_ref_64": len(ref6),
                    "blocks": len({block_of(k) for k in keys})}

    # ---- P1: is the reference the same function at both widths? --------------------------
    worst, worst_t, n_bit = 0.0, None, 0
    sq3 = sum(float(torch.linalg.vector_norm(ref3[k].to(torch.float64))) ** 2 for k in keys)
    sq6 = sum(float(torch.linalg.vector_norm(ref6[k].to(torch.float64))) ** 2 for k in keys)
    for k in keys:
        x, y = ref3[k].to(torch.float64), ref6[k].to(torch.float64)
        if torch.equal(x, y):
            n_bit += 1
        nr = float(torch.linalg.vector_norm(y))
        r = float(torch.linalg.vector_norm(x - y)) / (nr + 1e-300)
        if r > worst:
            worst, worst_t = r, k
    out["controls"]["REF_WIDTH_INVARIANCE"] = {
        "what": "upstream 0.4.3 float64 at width 384 against the same at width 64, per tensor",
        "compared": len(keys), "bit_identical": n_bit,
        "worst_rel_l2": worst, "worst_tensor": worst_t,
        "squared_gradient_norm_384": sq3, "squared_gradient_norm_64": sq6,
        "denominator_relative_move": abs(sq3 - sq6) / sq6,
        "why": "the denominator of every figure below. If it moved, a difference of numerators "
               "would not be a difference of absolute errors."}

    # ---- A/A and A16, measured here, at both widths --------------------------------------
    aa = rows_for(ref3, ref3, keys)
    out["controls"]["A_over_A"] = {"mass_weighted_rel_l2": mw(aa, keys)[0],
                                   "exactly_zero": mw(aa, keys)[0] == 0.0}
    del aa
    for w, ref in ((384, ref3), (64, ref6)):
        z = rows_for({k: torch.zeros_like(ref[k]) for k in keys}, ref, keys)
        v = mw(z, keys)[0]
        out["controls"][f"A16_zero_gradient_w{w}"] = {"mass_weighted_rel_l2": v,
                                                      "exactly_one": v == 1.0}
        del z

    arms = {}
    for nm, (op, rp, w) in {"ours_384": (a.ours384, None, 384),
                            "ours_64": (a.ours64, None, 64),
                            "floor_384": (a.floor384, None, 384),
                            "floor_64": (a.floor64, None, 64)}.items():
        g = trunk(op)
        miss = [k for k in keys if k not in g]
        if miss:
            raise SystemExit(f"{nm}: {len(miss)} reference tensors have no gradient, e.g. {miss[:3]}")
        arms[nm] = rows_for(g, ref3 if w == 384 else ref6, keys)
        del g
    del ref3, ref6

    # ---- REPRODUCTION: the four published figures, before any growth number ---------------
    pub = {"ours_384": 0.8354121633458239, "floor_384": 0.3739383921130369,
           "ours_64": 0.3833065667807629, "floor_64": 0.3739375768802167}
    rep = {}
    for nm, want in pub.items():
        got, tot = mw(arms[nm], keys)
        rep[nm] = {"published_by_of3t_frame384": want, "recomputed_here": got,
                   "abs_diff": abs(got - want), "within_1e_12": abs(got - want) <= 1e-12,
                   "reference_squared_norm": tot}
    out["controls"]["REPRODUCES_FRAME384"] = rep
    out["MATCHED"] = {
        "ours_384_vs_local_f64": rep["ours_384"]["recomputed_here"],
        "ours_64_vs_local_f64": rep["ours_64"]["recomputed_here"],
        "floor_384_vs_local_f64": rep["floor_384"]["recomputed_here"],
        "floor_64_vs_local_f64": rep["floor_64"]["recomputed_here"],
        "ratio_384": rep["ours_384"]["recomputed_here"] / rep["floor_384"]["recomputed_here"],
        "ratio_64": rep["ours_64"]["recomputed_here"] / rep["floor_64"]["recomputed_here"],
        "ours_384_over_ours_64": rep["ours_384"]["recomputed_here"] / rep["ours_64"]["recomputed_here"],
        "floor_384_over_floor_64": rep["floor_384"]["recomputed_here"] / rep["floor_64"]["recomputed_here"],
        "floor_host": a.floor_host}

    # ---- the growth, per tensor ----------------------------------------------------------
    def growth_table(hi, lo):
        t3, t6 = arms[hi], arms[lo]
        d3 = sum(t3[k]["ref_sq"] for k in keys)
        d6 = sum(t6[k]["ref_sq"] for k in keys)
        g = {}
        for k in keys:
            c3 = t3[k]["err_sq"] / d3
            c6 = t6[k]["err_sq"] / d6
            g[k] = {"at_384": c3, "at_64": c6, "growth": c3 - c6,
                    "abs_err_sq_384": t3[k]["err_sq"], "abs_err_sq_64": t6[k]["err_sq"],
                    "ref_sq_384": t3[k]["ref_sq"], "ref_sq_64": t6[k]["ref_sq"],
                    "rel_l2_384": t3[k]["rel_l2"], "rel_l2_64": t6[k]["rel_l2"],
                    "norm_ratio_384": t3[k]["norm_ratio"], "norm_ratio_64": t6[k]["norm_ratio"],
                    "cos_384": t3[k]["cos"], "cos_64": t6[k]["cos"]}
        return g

    gr = growth_table("ours_384", "ours_64")
    gf = growth_table("floor_384", "floor_64")

    tot_growth = sum(gr[k]["growth"] for k in keys)
    closure = (rep["ours_384"]["recomputed_here"] ** 2 - rep["ours_64"]["recomputed_here"] ** 2)
    out["controls"]["CLOSURE"] = {
        "sum_of_per_tensor_growths": tot_growth,
        "mw384_sq_minus_mw64_sq": closure,
        "abs_diff": abs(tot_growth - closure),
        "within_1e_12": abs(tot_growth - closure) <= 1e-12,
        "why": "the decomposition is exact or it is not a decomposition"}

    tot_floor = sum(gf[k]["growth"] for k in keys)
    out["controls"]["FLOOR_GROWTH"] = {
        "what": "upstream's own bf16 recipe measured the same way. The control for 'is any of "
                "this width response a property of the comparison rather than of our arm'.",
        "sum_of_per_tensor_growths": tot_floor,
        "sum_of_absolute_per_tensor_growths": sum(abs(gf[k]["growth"]) for k in keys),
        "ours_sum_of_absolute_per_tensor_growths": sum(abs(gr[k]["growth"]) for k in keys),
        "floor_over_ours_signed": tot_floor / tot_growth,
        "floor_abs_over_ours_abs": (sum(abs(gf[k]["growth"]) for k in keys)
                                    / sum(abs(gr[k]["growth"]) for k in keys))}

    out["GROWTH"] = {
        "total": tot_growth,
        "units": "mw^2, i.e. sum_t ||ours_t - ref_t||^2 / ||g_ref||^2. Additive by construction.",
        "by_section": agg(gr, section_of, keys),
        "by_leaf_family": dict(list(agg(gr, leaf_of, keys).items())[:24]),
        "by_block": {str(b): v for b, v in agg(gr, block_of, keys).items()},
        "floor_by_section": agg(gf, section_of, keys)}

    rank = sorted(keys, key=lambda k: -gr[k]["growth"])
    out["TENSORS"] = {
        "top_by_growth": [dict(tensor=k, share_of_the_growth=gr[k]["growth"] / tot_growth,
                               **gr[k]) for k in rank[:a.top]],
        "most_negative": [dict(tensor=k, share_of_the_growth=gr[k]["growth"] / tot_growth,
                               **gr[k]) for k in rank[-5:]],
        "concentration": {
            f"top{n}_share": sum(gr[k]["growth"] for k in rank[:n]) / tot_growth
            for n in (1, 5, 10, 20, 50, 100)},
        "n_tensors_that_grew": sum(1 for k in keys if gr[k]["growth"] > 0),
        "n_tensors_that_shrank": sum(1 for k in keys if gr[k]["growth"] < 0)}

    # ---- SOFTMAX containment -------------------------------------------------------------
    bis = [k for k in keys if block_of(k) == BISECT_BLOCK
           and leaf_of(k).startswith(BISECT_LEAF_PREFIXES)]
    bg = sum(gr[k]["growth"] for k in bis)
    fam = [k for k in keys if section_of(k) == "attn_pair_bias"]
    out["SOFTMAX"] = {
        "what": "of3t-apbback's softmax result is scoped to 16 tensors of ONE block. This asks "
                "how much of the trunk's width growth lives inside that scope. No device run: "
                "this row holds no card and the bisect cannot be re-taken at a second width.",
        "bisect_scope": {"tensors": len(bis), "of_the_trunk": len(keys),
                         "names": sorted(bis),
                         "growth": bg, "share_of_the_growth": bg / tot_growth,
                         "at_384": sum(gr[k]["at_384"] for k in bis),
                         "at_64": sum(gr[k]["at_64"] for k in bis),
                         "abs_err_sq_384": sum(gr[k]["abs_err_sq_384"] for k in bis),
                         "abs_err_sq_64": sum(gr[k]["abs_err_sq_64"] for k in bis)},
        "attn_pair_bias_all_48_blocks": {
            "tensors": len(fam), "growth": sum(gr[k]["growth"] for k in fam),
            "share_of_the_growth": sum(gr[k]["growth"] for k in fam) / tot_growth},
        "block47_whole": {
            "growth": sum(gr[k]["growth"] for k in keys if block_of(k) == 47),
            "share_of_the_growth": sum(gr[k]["growth"] for k in keys if block_of(k) == 47) / tot_growth}}

    # ---- our own arm against itself across widths ----------------------------------------
    # The reference does not move, so ours(384) - ours(64) IS the width perturbation of our arm.
    o3, o6 = trunk(a.ours384), trunk(a.ours64)
    per = []
    for k in keys:
        x, y = o3[k].to(torch.float64), o6[k].to(torch.float64)
        per.append((float((x - y).pow(2).sum()), k))
    per.sort(reverse=True)
    dn = sum(p for p, _ in per)
    out["WIDTH_DELTA_OF_OUR_OWN_ARM"] = {
        "what": "ours(384) - ours(64), per tensor. The reference is width-invariant, so this is "
                "the whole width effect with no reference in it.",
        "squared_norm": dn,
        "relative_to_the_reference_squared_norm": dn / sq6,
        "top": [{"tensor": k, "sq": p, "share": p / dn} for p, k in per[:12]]}
    del o3, o6

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({
        "reproduces": {k: v["within_1e_12"] for k, v in rep.items()},
        "ref_width_invariance_worst_rel": worst,
        "closure_ok": out["controls"]["CLOSURE"]["within_1e_12"],
        "total_growth": tot_growth,
        "floor_growth": tot_floor,
        "top_sections": {k: (v["growth"], v["share_of_the_growth"], v["width_exponent_p"])
                         for k, v in list(out["GROWTH"]["by_section"].items())[:6]},
        "softmax_scope_share": out["SOFTMAX"]["bisect_scope"]["share_of_the_growth"],
        "top5": [(r["tensor"], r["share_of_the_growth"]) for r in out["TENSORS"]["top_by_growth"][:5]],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
