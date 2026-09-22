#!/usr/bin/env python3
"""The frame-matched reading at crop 384: our trunk arm and upstream's own bf16, one reference.

of3t-apbback established that `the trunk is 5.41x upstream's own bf16` pairs a CAPTURE-driven
numerator with a FULL-MODEL denominator (REFAUDIT.json, and of3t-orchestrator's FRAME_TABLE.json
which summarises it). Scored in one frame the same device tensors read 0.9565x -- but only at
crop 64, because both local references were c64 dumps. This closes that at 384.

The scoring math is of3t-trunkg043's `score.py`, imported rather than reimplemented, so this row
and every row above it share one definition of mass-weighted rel_l2 (A23), of the three-number
per-tensor triple (D35) and of the error-mass leaf aggregation.

FOUR CONTROLS, all before any headline number:

  BIT        the plain and the `--checkpoint` arm at crop 64 must be bit-identical on all 2,736
             tensors. `--checkpoint` is the only change to the producer and it is what makes the
             384 arm fit; if recompute moved a bit, the 384 references would be a different
             function from the c64 pair the campaign already published.
  CROSS-HOST the same producer, same tree, same boundary, run on qb1 against the artifact qb2
             banked. Both are CPU float64 and no device is involved on either side, so there is
             no hardware to attribute any difference to (D155): whatever this reads is the
             interpreter and the BLAS, and it is quoted rather than assumed away.
  A/A        the float64 reference scored against itself. Must be exactly 0.
  A16        a gradient of exact zeros against the same reference. Must be exactly 1.0, or the
             ceiling is not where the protocol says it is.

The A26-style reachable bar is recomputed for THIS scope from THIS floor -- sqrt(2)*floor/r, the
arithmetic of3t-modelboundary uses (0.10592054683439786 / 1.0165697057473722 * sqrt(2) =
0.14735268326440318) -- and not borrowed from a scope it was not measured at.
"""
import argparse, hashlib, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "of3t_trunkg043"))
from score import score, by_leaf, per_block                                  # noqa: E402

PIN = "1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4"
PRE = "pairformer_stack.blocks."


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def trunk(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = d["grads"] if isinstance(d, dict) and "grads" in d else d
    return ({k: v for k, v in g.items() if k.startswith(PRE) and v is not None},
            d if isinstance(d, dict) else {})


def sq(g):
    return sum(float(torch.linalg.vector_norm(v.to(torch.float64))) ** 2 for v in g.values())


def identity(a, b, keys):
    """Bit identity, reported as a count and a worst case rather than as a boolean."""
    n_eq, worst, worst_t = 0, 0.0, None
    for k in keys:
        x, y = a[k].to(torch.float64), b[k].to(torch.float64)
        if torch.equal(x, y):
            n_eq += 1
        d = float((x - y).abs().max())
        if d > worst:
            worst, worst_t = d, k
    return {"compared": len(keys), "bit_identical": n_eq,
            "max_absdiff": worst, "worst_tensor": worst_t,
            "all_bit_identical": n_eq == len(keys)}


def summarise(s):
    return {"compared": s["tensors"], "mass_weighted_rel_l2": s["mass_weighted_rel_l2"],
            "median_rel_l2_over_tensors": s["median_rel_l2_over_tensors"],
            "mass_weighted_norm_ratio": s["mass_weighted_norm_ratio"],
            "mass_weighted_cos": s["mass_weighted_cos"],
            "reference_squared_norm": s["reference_squared_norm"],
            "over_per_tensor_bar": s["over_per_tensor_bar"],
            "over_per_tensor_bar_mass": s["over_per_tensor_bar_mass"],
            "worst_by_error_mass": s["worst_by_error_mass"],
            "worst_by_rel": s["worst_by_rel"],
            "top8_by_error_mass": s["top8_by_error_mass"]}


SECTION = "pairformer_stack"


def model_projection(mb, path, inframe_vs_f64, inframe_vs_bf16):
    """What the GRADIENTS clause would read if its trunk section were scored IN FRAME.

    NOT a measurement, and the file says so in its own first field. It is arithmetic on two
    measurements: of3t-modelboundary's model figure, and this row's in-frame reading of the
    one section whose device file the two rows share. A mass-weighted rel L2 is
    sqrt(sum_i m_i r_i^2), so swapping one section's r is exact given its mass share -- there
    is no modelling step. What it assumes is the part to argue with: that the other sections'
    own frames are already matched. Each of them is a separate capture-driven arm
    (`arms.renorm.scopes`) and this row measured none of them, so the projection is a FLOOR on
    what an in-frame model reading would be, not an estimate of it.
    """
    bar = mb["bars"]["A26_reachable_bar_vs_their_bf16"]
    o = {"what": "A PROJECTION, not a measurement. See this function's docstring in "
                 "perf/of3t_frame384/frame384.py.",
         "from_artifact": path,
         "A26_reachable_bar_vs_their_bf16": bar,
         "section_substituted": SECTION,
         "same_device_file_both_rows":
             mb["arms"]["renorm"]["scopes"][-1].split("=", 1)[-1],
         "assumption": "the other sections' arms are already frame-matched. None of them was "
                       "measured here, so this is a LOWER BOUND on an in-frame model reading."}
    for stat, inframe in (("renorm_vs_UPSTREAM_BF16", inframe_vs_bf16),
                          ("renorm_vs_FLOAT64", inframe_vs_f64)):
        sec = mb["per_section"][stat][SECTION]
        m = sec["pct_of_model_mass"] / 100.0
        cross = sec["mass_weighted_rel_l2"]
        tot = mb["stats"][stat]["mass_weighted_rel_l2"]
        moved = tot ** 2 - m * (cross ** 2 - inframe ** 2)
        proj = moved ** 0.5 if moved > 0 else float("nan")
        o[stat] = {
            "model_as_measured": tot,
            "trunk_section_mass_share_of_the_model": m,
            "trunk_section_cross_frame": cross,
            "trunk_section_in_frame_this_row": inframe,
            "trunk_share_of_the_models_error_mass": m * cross ** 2 / tot ** 2,
            "model_projected_in_frame": proj,
            "multiple_of_the_A26_bar_as_measured": tot / bar,
            "multiple_of_the_A26_bar_projected": proj / bar,
            "clause_would_pass": proj <= bar}
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-f64-n384", required=True)
    ap.add_argument("--ref-bf16-n384", required=True)
    ap.add_argument("--ours-n384", required=True)
    ap.add_argument("--ref-model-f64", required=True)
    ap.add_argument("--c64-plain", required=True)
    ap.add_argument("--c64-ckpt", required=True)
    ap.add_argument("--c64-bf16-plain", default="")
    ap.add_argument("--c64-bf16-ckpt", default="")
    ap.add_argument("--c64-banked", required=True,
                    help="the artifact of3t-trunkg043 banked on qb2, for the cross-host control")
    ap.add_argument("--ref-f64-report", required=True,
                    help="ref_grad.py's own report for the float64 arm; the probe (real tokens "
                         "against padded tokens, and both input norms) is lifted from it rather "
                         "than re-derived, so the two files cannot disagree")
    ap.add_argument("--refs-built-on", required=True, metavar="HOST",
                    help="the host that produced the LOCAL float64 reference AND the floor. "
                         "D189: upstream's own bf16 autocast arm differs 6.0 %% between qb1 and "
                         "qb2 on this boundary while the float64 arm agrees to 2e-16, so a bar "
                         "is a host-dependent claim and D155's rule applies to it as much as to "
                         "a digest. The campaign's own 0.3147698293887927 floor records no host.")
    ap.add_argument("--arm-built-on", required=True, metavar="HOST",
                    help="the host and card the DEVICE arm was produced on, which is not this one")
    ap.add_argument("--model-ref-built-on", required=True, metavar="HOST")
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--crop", type=int, default=384,
                    help="the crop every figure in the output is labelled with (D180)")
    ap.add_argument("--reproduces", type=float, default=2.159527121735274,
                    help="the campaign's own published cross-frame figure at this crop, which "
                         "the CROSSFRAME block is a reproduction control for")
    ap.add_argument("--reproduces-from", default="perf/of3t_ditmodel/TRUNK_D174.json "
                                                 "stats.MASKON_vs_FLOAT64")
    ap.add_argument("--model-artifact", default="",
                    help="of3t-modelboundary's MODEL_withtrunk_n384.json. The GRADIENTS clause "
                         "reads it, and the pairformer_stack section of its `renorm` arm is the "
                         "SAME device file this row scores in frame, so the two can be composed "
                         "arithmetically. The composition is published as a PROJECTION and "
                         "labelled one -- it is not a re-measurement of the model.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {"what": __doc__.strip().splitlines()[0],
           "host": os.uname().nodename,
           "device_involved": False,
           "note": "every arm in this file is CPU. The device arm was banked by "
                   "of3t-modelboundary on qb2 and is scored, not re-run, here.",
           "crop": a.crop, "controls": {}, "digests": {},
           "hosts": {"scored_on": os.uname().nodename,
                     "local_float64_reference_and_the_floor_built_on": a.refs_built_on,
                     "device_arm_built_on": a.arm_built_on,
                     "model_frame_float64_built_on": a.model_ref_built_on,
                     "why": "D189. The floor is host-dependent at the percent level and the "
                            "float64 reference is not, so the ratio below is only quotable "
                            "with the host that produced its DENOMINATOR beside it."}}

    with open(a.ref_f64_report) as fh:
        f64rep = json.load(fh)
    probe = f64rep.get("probe", {})
    out["float64_reference_run"] = {
        "report": a.ref_f64_report, "policy": f64rep.get("policy"),
        "tree": f64rep.get("tree"), "blocks": f64rep.get("blocks"),
        "crop": f64rep.get("crop"), "checkpointed": f64rep.get("checkpointed"),
        "threads": f64rep.get("threads"),
        "boundary": f64rep.get("boundary"), "boundary_sha256": f64rep.get("boundary_sha256"),
        "cotangent_from": f64rep.get("cotangent_from"),
        "cotangent_sha256": f64rep.get("cotangent_sha256"),
        "seconds_forward_backward": f64rep.get("seconds_forward_backward"),
        "peak_rss_gb": f64rep.get("peak_rss_gb"), "probe": probe}

    # Every input is digested BEFORE it is loaded, not only the pinned one. A reference is
    # part of the measurement's identity (D187), and a path is not a version (A24-AMENDMENT):
    # five of these six files were produced by this row this week and carry no pin anywhere
    # else, so the digest recorded here is what a successor re-checks them against.
    for nm, path in (("REF_MODEL_f64", a.ref_model_f64),
                     ("REF_LOCAL_f64_n384", a.ref_f64_n384),
                     ("REF_LOCAL_bf16_n384", a.ref_bf16_n384),
                     ("OURS_n384", a.ours_n384),
                     ("c64_f64_plain", a.c64_plain), ("c64_f64_ckpt", a.c64_ckpt),
                     ("c64_bf16_plain", a.c64_bf16_plain), ("c64_bf16_ckpt", a.c64_bf16_ckpt),
                     ("c64_banked_on_qb2", a.c64_banked)):
        if not path:
            continue
        d = sha256_file(path)
        e = {"path": path, "bytes": os.path.getsize(path), "sha256": d}
        if nm == "REF_MODEL_f64":
            e["matches_pin"] = d == PIN
            if d != PIN:
                raise SystemExit(f"REF_MODEL_f64 digest {d} does not match the pin {PIN}")
        out["digests"][nm] = e
        print("SHA", nm, d, flush=True)

    # ---- CONTROL: --checkpoint is inert, at crop 64, before any 384 number ------------------
    gp, rp = trunk(a.c64_plain)
    gc, rc = trunk(a.c64_ckpt)
    keys64 = sorted(set(gp) & set(gc))
    out["controls"]["BIT_checkpoint_is_inert_c64"] = {
        **identity(gp, gc, keys64),
        "plain_loss": rp.get("loss"), "checkpointed_loss": rc.get("loss"),
        "what": "the same producer at crop 64 with and without --checkpoint, same host, same "
                "interpreter. This is the claim that lets the 384 arms use the flag."}
    print("BIT", out["controls"]["BIT_checkpoint_is_inert_c64"]["bit_identical"], flush=True)

    if a.c64_bf16_plain and a.c64_bf16_ckpt:
        bp, mbp = trunk(a.c64_bf16_plain)
        bc, mbc = trunk(a.c64_bf16_ckpt)
        kb = sorted(set(bp) & set(bc))
        out["controls"]["BIT_checkpoint_is_inert_c64_bf16auto"] = {
            **identity(bp, bc, kb),
            "plain_loss": mbp.get("loss"), "checkpointed_loss": mbc.get("loss"),
            "what": "the same control on the bf16 autocast arm. torch.utils.checkpoint saves "
                    "and restores the autocast state for the recompute, so this is the claim "
                    "that the recomputed forward is the SAME mixed-precision forward."}
        print("BIT bf16", out["controls"]["BIT_checkpoint_is_inert_c64_bf16auto"]
              ["bit_identical"], flush=True)
        del bp, bc

    # ---- CONTROL: cross-host, against the artifact the campaign already published -----------
    gb, rb = trunk(a.c64_banked)
    keysb = sorted(set(gp) & set(gb))
    s = score(gp, gb, keysb); s.pop("_rows")
    out["controls"]["CROSSHOST_qb1_vs_the_banked_c64_artifact"] = {
        **identity(gp, gb, keysb),
        "mass_weighted_rel_l2": s["mass_weighted_rel_l2"],
        "qb1_loss": rp.get("loss"), "banked_loss": rb.get("loss"),
        "qb1_squared_gradient_norm": sq(gp), "banked_squared_gradient_norm": sq(gb),
        "what": "both sides CPU float64, upstream 0.4.3, same tree, same boundary, same "
                "cotangent, same torch 2.8.0+cpu; python 3.10.12 on qb1 against 3.12.3 on qb2. "
                "No device on either side, so there is no card to attribute a difference to."}
    print("CROSSHOST", out["controls"]["CROSSHOST_qb1_vs_the_banked_c64_artifact"]
          ["mass_weighted_rel_l2"], flush=True)
    del gp, gc, gb

    # ---- the three references at 384 --------------------------------------------------------
    f64, rf = trunk(a.ref_f64_n384)
    bf16, rbf = trunk(a.ref_bf16_n384)
    ours, ro = trunk(a.ours_n384)
    model, _ = trunk(a.ref_model_f64)
    keys = sorted(f64)
    for nm, g, meta in (("REF_LOCAL_f64_n384", f64, rf), ("REF_LOCAL_bf16_n384", bf16, rbf),
                        ("OURS_n384", ours, ro), ("REF_MODEL_f64", model, {})):
        n2 = sq(g)
        out.setdefault("refs", {})[nm] = {
            "trunk_tensors": len(g), "trunk_squared_gradient_norm": n2,
            "trunk_gradient_norm": float(np.sqrt(n2)),
            "policy": meta.get("policy"), "checkpointed": meta.get("checkpointed"),
            "loss": meta.get("loss")}
    missing = [k for k in keys if k not in ours]
    out["scope"] = {"reference_tensors": len(keys), "ours_absent": len(missing),
                    "real_tokens": probe.get("real_tokens"),
                    "padded_tokens": probe.get("tokens"),
                    "what_crop_names_here": "the boundary is the SAME 56-residue target in both "
                                            "the c64 and the n384 dumps; the axis this row calls "
                                            "crop is the PADDED width, 64 against 384. Every "
                                            "figure is labelled with it (D180) and nothing here "
                                            "claims a different target was folded.",
                    "absent": missing[:8],
                    "share_of_the_trunks_squared_gradient_norm": 1.0,
                    "what": "every tensor of the 48-block stack carries a reference gradient and "
                            "every one is placed, so the compared set is the whole trunk scope"}

    # ---- A/A and A16, before any headline --------------------------------------------------
    aa = score(f64, f64, keys); aa.pop("_rows")
    out["controls"]["AA_the_reference_against_itself"] = {
        "mass_weighted_rel_l2": aa["mass_weighted_rel_l2"],
        "exactly_zero": aa["mass_weighted_rel_l2"] == 0.0,
        "mass_weighted_norm_ratio": aa["mass_weighted_norm_ratio"]}
    z = score({k: torch.zeros_like(f64[k]) for k in keys}, f64, keys); z.pop("_rows")
    out["controls"]["A16_zero_gradient_baseline"] = {
        "mass_weighted_rel_l2": z["mass_weighted_rel_l2"],
        "exactly_one": z["mass_weighted_rel_l2"] == 1.0,
        "median_rel_l2_over_tensors": z["median_rel_l2_over_tensors"],
        "what": "measured in this row's own scorer, not inherited"}
    print("AA", aa["mass_weighted_rel_l2"], "A16", z["mass_weighted_rel_l2"], flush=True)

    # ---- FRAME: the local float64 reference against the MODEL float64 ----------------------
    kf = sorted(set(f64) & set(model))
    s = score(f64, model, kf)
    rows = s.pop("_rows")
    out["FRAME_ref_f64_n384__vs__grads_f64_043"] = {
        **summarise(s), "by_leaf_error_mass": by_leaf(rows),
        "what": "the frame itself at crop 384: the capture's own float64 gradient, with no "
                "device op and no bf16 anywhere on its path, against the pinned model-frame "
                "float64. Nothing inside the port can reach this distance."}
    print("FRAME", s["mass_weighted_rel_l2"], flush=True)

    # ---- MATCHED: ours and the floor, both against the SAME local float64 ------------------
    ou = score(ours, f64, keys)
    rows_o = ou.pop("_rows")
    fl = score(bf16, f64, keys)
    rows_f = fl.pop("_rows")
    r = fl["mass_weighted_norm_ratio"]
    floor = fl["mass_weighted_rel_l2"]
    bar = (2 ** 0.5) * floor / r if r else float("nan")
    ob = score(ours, bf16, keys); ob.pop("_rows")
    out["MATCHED"] = {
        "ours_vs_REF_LOCAL_f64_n384": summarise(ou),
        "ours_vs_REF_LOCAL_bf16_n384": {
            **summarise(ob),
            "what": "the exact SHAPE of the GRADIENTS clause -- our arm against upstream's OWN "
                    "bf16 step, which is what `stats.renorm_vs_UPSTREAM_BF16` is at model "
                    "scope -- but with both sides driven from the same capture. The clause "
                    "compares this against the A26 bar below."},
        "ratio_ours_vs_their_bf16_over_bar": (ob["mass_weighted_rel_l2"] /
                                              ((2 ** 0.5) * fl["mass_weighted_rel_l2"]
                                               / fl["mass_weighted_norm_ratio"])),
        "floor_REF_LOCAL_bf16_vs_REF_LOCAL_f64_n384": summarise(fl),
        "ratio_ours_over_floor": ou["mass_weighted_rel_l2"] / floor if floor else None,
        "A26_style_reachable_bar_for_this_scope": bar,
        "A26_bar_arithmetic": "sqrt(2) * floor / mass_weighted_norm_ratio(floor), the same "
                              "arithmetic of3t-modelboundary uses for its own scope",
        "inside_the_A26_style_bar": ou["mass_weighted_rel_l2"] <= bar,
        "ours_by_leaf_error_mass": by_leaf(rows_o),
        "floor_by_leaf_error_mass": by_leaf(rows_f),
        "ours_per_block": per_block(rows_o, a.blocks),
        "floor_per_block": per_block(rows_f, a.blocks)}
    print("MATCHED ours", ou["mass_weighted_rel_l2"], "floor", floor,
          "ratio", out["MATCHED"]["ratio_ours_over_floor"], "bar", bar, flush=True)

    # ---- CROSS-FRAME, for the record and as a reproduction control -------------------------
    kx = sorted(set(ours) & set(model))
    xf = score(ours, model, kx); xf.pop("_rows")
    out["CROSSFRAME_ours_vs_grads_f64_043"] = {
        **summarise(xf),
        "reproduces": a.reproduces,
        "published_in": a.reproduces_from,
        "what": "the campaign's own crop-384 trunk figure, recomputed here through this row's "
                "scorer on the same banked device tensors. It is a reproduction control: if it "
                "does not come back, the arm or the scorer is not the published one."}
    print("CROSSFRAME", xf["mass_weighted_rel_l2"], flush=True)

    # ---- MODEL_PROJECTION: what the charter clause would read with this row's frame --------
    if a.model_artifact:
        with open(a.model_artifact) as fh:
            mb = json.load(fh)
        out["MODEL_PROJECTION"] = model_projection(
            mb, a.model_artifact,
            inframe_vs_f64=ou["mass_weighted_rel_l2"],
            inframe_vs_bf16=ob["mass_weighted_rel_l2"])
        print("PROJECTION", json.dumps(
            {k: v for k, v in out["MODEL_PROJECTION"].items()
             if isinstance(v, (int, float, bool))}, indent=1), flush=True)

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({
        "FRAME": out["FRAME_ref_f64_n384__vs__grads_f64_043"]["mass_weighted_rel_l2"],
        "MATCHED_ours": ou["mass_weighted_rel_l2"], "MATCHED_floor": floor,
        "ratio": out["MATCHED"]["ratio_ours_over_floor"], "bar": bar,
        "inside": out["MATCHED"]["inside_the_A26_style_bar"],
        "CROSSFRAME": xf["mass_weighted_rel_l2"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
