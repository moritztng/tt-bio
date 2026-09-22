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
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {"what": __doc__.strip().splitlines()[0],
           "host": os.uname().nodename,
           "device_involved": False,
           "note": "every arm in this file is CPU. The device arm was banked by "
                   "of3t-modelboundary on qb2 and is scored, not re-run, here.",
           "crop": 384, "controls": {}, "digests": {}}

    d = sha256_file(a.ref_model_f64)
    out["digests"]["REF_MODEL_f64"] = {"path": a.ref_model_f64, "sha256": d,
                                       "matches_pin": d == PIN}
    if d != PIN:
        raise SystemExit(f"REF_MODEL_f64 digest {d} does not match the pin {PIN}")

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
    out["MATCHED"] = {
        "ours_vs_REF_LOCAL_f64_n384": summarise(ou),
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
        "reproduces": 2.159527121735274,
        "published_in": "perf/of3t_ditmodel/TRUNK_D174.json stats.MASKON_vs_FLOAT64",
        "what": "the campaign's own crop-384 trunk figure, recomputed here through this row's "
                "scorer on the same banked device tensors. It is a reproduction control: if it "
                "does not come back, the arm or the scorer is not the published one."}
    print("CROSSFRAME", xf["mass_weighted_rel_l2"], flush=True)

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
