#!/usr/bin/env python3
"""of3t-trajfull deliverables 1 and 2: score the whole reference set, and CHECK `scope.coupled`.

Host-only. The two banks are already on disk: `w/theirs` is upstream 0.4.3's own 20-step
trajectory at 761 tensors and is not re-run, `w/trajfull` is ours with the widened dump.
`trajwide.py`'s `score_step`, `growth` and `scope_share` are imported rather than copied, so
the arithmetic is the same arithmetic `traj_refatom.json` was scored with and the two numbers
are comparable.

THE FAILURE MODE HERE READS AS PROGRESS. A wrong name or a wrong split scores more tensors,
the percentage goes up, and nothing complains. Three controls run BEFORE the scored number,
all pre-registered in the brief, and the artifact carries them:

  1. k=1 bit-identity. lr(1) = 0 on both sides, so each side's k=1 dump is its own w_0. Every
     one of the 180 newly written tensors must reproduce the reference's k=1 value exactly,
     in fp32 or after bf16 rounding -- the same two dtypes the 581 already split between. One
     that does not means the map is wrong, not that our port differs.
  2. The split round-trip, asserted inside `namemap.emit` at every rung of the arm, reported
     here from the arm's own evidence.
  3. The three family subtotals: `not_covered_families` empty, and the added mass equal to
     0.3755971869 % of the model squared gradient norm. Higher means something is scored twice.
"""
import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PERF = os.path.dirname(_HERE)
for p in (os.path.join(_PERF, "of3t_trajwide"), _PERF):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                       # noqa: E402
import trajwide as TW                                                    # noqa: E402
import namemap as NM                                                     # noqa: E402

RUNS = TW.RUNS
BAR = 89.2105                    # the diffusion module's whole share, rounded DOWN (D209)
BEFORE = 88.83498302148425       # `traj_refatom.json`, the previous best coupled trajectory
EXPECTED_ADDED = 0.3755971869    # BAR_exact - BEFORE, from the reference's own gradient mass


# ------------------------------------------------------------------------- control 1, k=1

def k1_bit_identity(dt, do, new_names, all_names):
    """Each side's k=1 dump is its own w_0. Ours must be the reference's, exactly."""
    wt, wo = TW.load_w(dt, 1), TW.load_w(do, 1)
    out = {}
    for label, names in (("new_180", new_names), ("previously_scored", all_names - new_names)):
        fp32 = bf16 = 0
        bad = []
        for n in sorted(names):
            a, r = wo[n].astype(np.float32), wt[n].astype(np.float32)
            if a.shape != r.shape:
                bad.append({"tensor": n, "why": "shape %s vs %s" % (a.shape, r.shape)})
            elif np.array_equal(a, r):
                fp32 += 1
            elif np.array_equal(a, NM._bf16(r)):
                bf16 += 1
            else:
                d = np.abs(a.astype(np.float64) - r.astype(np.float64))
                bad.append({"tensor": n, "max_abs_diff": float(d.max()),
                            "rel": float(d.max() / (np.abs(r).max() + 1e-300))})
        out[label] = {"tensors": len(names), "bit_identical_fp32": fp32,
                      "bit_identical_after_bf16_rounding": bf16,
                      "not_identical": len(bad), "failures": bad[:12]}
    out["pass"] = all(v["not_identical"] == 0 for v in out.values() if isinstance(v, dict))
    return out


# -------------------------------------------------------------------------- scope.coupled

def _series(log, key):
    return [(r["k"], r.get(key)) for r in (log or []) if r.get(key) is not None]


def coupled_check(ours_sl, theirs_sl, scored, dt, do, names, W0ck):
    """`scope.coupled`, computed rather than declared.

    The clause exists because a UNION of per-section trajectories satisfies a scope
    percentage while testing none of the coupling: every boundary this campaign holds is a
    frozen capture of upstream's r=0 step, so a per-section trajectory reads step-0 inputs at
    every k. What has to be true instead is that each side computed its own gradient at its
    own w_{k-1} and stepped its own weights with it.

    Six checks, each falsifiable from the artifacts:

      1  two independent producers -- different `side`, different arm, different bank.
      2  the two gradient-norm series are different at every rung, so neither side consumed
         the other's gradient.
      3  each side's gradient tracks ITS OWN weights. lr(1) = 0, so w_1 = w_0 and the rung-2
         gradient is taken at the same weights as rung 1: the two agree to float noise. From
         rung 3 the weights have moved and the gradient moves with them. A side driven by
         anything other than its own weight path cannot show both halves.
      4  both sides moved at every rung from k=2, on disk: w_k != w_{k-1}.
      5  the two trajectories are different trajectories -- rel_d > 0 at every rung from 2.
      6  one trajectory pair, not a union: 20 rungs from exactly one directory per side, and
         every scored name present in every rung on both sides.
    """
    ck = {}
    ours_log, theirs_log = ours_sl.get("steps") or [], theirs_sl.get("steps") or []
    og = dict(_series(ours_log, "grad_norm"))
    tg = dict(_series(theirs_log, "grad_norm_last_sample"))
    ks = sorted(set(og) & set(tg))

    ck["1_independent_producers"] = {
        "ours_side": ours_sl.get("side"), "theirs_side": theirs_sl.get("side"),
        "ours_arm": ours_sl.get("arm"), "theirs_arm": theirs_sl.get("arm"),
        "ours_bank": do, "theirs_bank": dt,
        "ours_complete": ours_sl.get("complete"), "theirs_complete": theirs_sl.get("complete"),
        "ours_rungs": len(ours_log), "theirs_rungs": len(theirs_log),
        "pass": (ours_sl.get("side") == "ours" and theirs_sl.get("side") == "theirs"
                 and ours_sl.get("arm") != theirs_sl.get("arm")
                 and os.path.realpath(do) != os.path.realpath(dt)
                 and bool(ours_sl.get("complete")) and bool(theirs_sl.get("complete"))
                 and len(ours_log) >= 20 and len(theirs_log) >= 20)}

    same = [k for k in ks if og[k] == tg[k]]
    ck["2_gradient_norms_differ_every_rung"] = {
        "rungs": len(ks), "rungs_where_equal": same,
        "ours_k1": og.get(1), "theirs_k1": tg.get(1),
        "pass": not same}

    # Our side's `grad_norm` is the norm of the ACCUMULATED gradient over all 48 noise
    # levels, so it is a function of our weights alone: lr(1) = 0, w_1 = w_0, and the rung-2
    # gradient is taken at the same weights as rung 1. The reference side logs
    # `grad_norm_last_sample`, the last accumulation sample only, and `partition()` redraws
    # the 48 levels into four blocks at every rung -- so that series moves between rungs
    # because the last block holds different noise levels, not because weights moved, and it
    # cannot carry this witness. The reference side gets the two witnesses it CAN carry: its
    # warm-up rung moved nothing, from the bank rather than the log, and the first-step sign
    # test below, which is the direct form of "neither side advanced from the other's
    # gradient".
    warm = abs(og[2] - og[1]) / (abs(og[1]) + 1e-300) if {1, 2} <= set(og) else None
    move = abs(og[3] - og[2]) / (abs(og[2]) + 1e-300) if {2, 3} <= set(og) else None
    wt1, wo1 = TW.load_w(dt, 1), TW.load_w(do, 1)
    wt2, wo2 = TW.load_w(dt, 2), TW.load_w(do, 2)
    theirs_w1_is_ckpt = sum(1 for n in names if np.array_equal(wt1[n], W0ck[n]))
    agree = tot = 0
    for n in names:
        so = np.sign(wo2[n].astype(np.float64) - wo1[n].astype(np.float64))
        st = np.sign(wt2[n].astype(np.float64) - wt1[n].astype(np.float64))
        live = (so != 0) & (st != 0)
        tot += int(live.sum())
        agree += int(((so == st) & live).sum())
    frac = agree / tot if tot else None
    del wt1, wo1, wt2, wo2
    ck["3_each_side_stepped_from_its_own_gradient"] = {
        "ours_grad_norm_k1": og.get(1), "ours_grad_norm_k2": og.get(2),
        "ours_grad_norm_k3": og.get(3),
        "ours_rel_change_while_weights_frozen_k1_to_k2": warm,
        "ours_rel_change_after_the_first_real_step_k2_to_k3": move,
        "ours_gradient_is_a_function_of_its_own_weights": bool(
            warm is not None and move is not None and warm < 1e-6 < move),
        "theirs_w1_equals_the_fp32_checkpoint": "%d of %d" % (theirs_w1_is_ckpt, len(names)),
        "theirs_warmup_rung_moved_nothing": theirs_w1_is_ckpt == len(names),
        "first_step_sign_agreement": frac, "first_step_elements_compared": tot,
        "why_the_sign_test": "at k=2 each side takes its first non-zero-lr step, and Adam's "
                             "first update is -lr*sign(g) elementwise, so the sign of each "
                             "side's displacement IS the sign of that side's own accumulated "
                             "gradient. A side advanced from the other's gradient would agree "
                             "on every element. Strictly below 1 falsifies that.",
        "theirs_grad_norm_last_sample_is_partition_dependent": True,
        "pass": bool(warm is not None and move is not None and warm < 1e-6 < move
                     and theirs_w1_is_ckpt == len(names)
                     and frac is not None and frac < 1.0)}

    # One pass over both banks: checks 4 and 6 read the same 40 npz files, and one rung is
    # ~813 MB, so they are answered together rather than twice.
    moved, absent = {"ours": [], "theirs": []}, {"ours": {}, "theirs": {}}
    nameset = set(names)
    for side, d in (("ours", do), ("theirs", dt)):
        prev = None
        for k in range(1, 21):
            cur = TW.load_w(d, k)
            absent[side][k] = len(nameset - set(cur))
            if prev is not None:
                moved[side].append(
                    {"k": k, "tensors_unchanged_from_k_minus_1":
                     sum(1 for n in names if np.array_equal(cur[n], prev[n])),
                     "tensors": len(names)})
            prev = cur
        del prev
    ck["4_both_sides_step_their_own_weights_every_rung"] = {
        "ours_rungs_with_no_tensor_moving": [r["k"] for r in moved["ours"]
                                             if r["tensors_unchanged_from_k_minus_1"]
                                             == r["tensors"]],
        "theirs_rungs_with_no_tensor_moving": [r["k"] for r in moved["theirs"]
                                               if r["tensors_unchanged_from_k_minus_1"]
                                               == r["tensors"]],
        "worst_ours_unchanged": max(r["tensors_unchanged_from_k_minus_1"]
                                    for r in moved["ours"]),
        "worst_theirs_unchanged": max(r["tensors_unchanged_from_k_minus_1"]
                                      for r in moved["theirs"]),
        "per_rung": moved}
    c4 = ck["4_both_sides_step_their_own_weights_every_rung"]
    c4["pass"] = not c4["ours_rungs_with_no_tensor_moving"] and \
        not c4["theirs_rungs_with_no_tensor_moving"]

    zero = [r["k"] for r in scored if r["k"] >= 2 and r["rel_d"] == 0.0]
    both = [r["k"] for r in scored if r["k"] >= 2
            and (r["d_ours_norm"] == 0.0 or r["d_theirs_norm"] == 0.0)]
    ck["5_the_two_trajectories_are_different"] = {
        "rungs_with_rel_d_zero": zero, "rungs_where_a_side_did_not_move": both,
        "min_rel_d_from_k2": min((r["rel_d"] for r in scored if r["k"] >= 2), default=None),
        "pass": not zero and not both}

    ck["6_one_trajectory_pair_not_a_union"] = {
        "banks": {"ours": do, "theirs": dt},
        "rungs": sorted(absent["ours"]),
        "rungs_missing_a_scored_name": {s: [k for k, v in absent[s].items() if v]
                                        for s in absent},
        "pass": all(v == 0 for s in absent for v in absent[s].values())
        and sorted(absent["ours"]) == sorted(absent["theirs"]) == list(range(1, 21))}

    ck["coupled"] = all(ck[k]["pass"] for k in ck if k != "coupled")
    return ck


# --------------------------------------------------------------------------------- scoring

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="trajfull")
    ap.add_argument("--theirs-arm", default="theirs", dest="theirs_arm")
    ap.add_argument("--before-arm", default="refatom", dest="before_arm")
    ap.add_argument("--out-dir", default=RUNS, dest="out_dir")
    ap.add_argument("--out", default="perf/of3t_trajfull/traj_trajfull.json")
    a = ap.parse_args()

    import torch
    t0 = TW.time.time()
    TW.refpath.install()

    dt, do = TW.wdir(a.out_dir, a.theirs_arm), TW.wdir(a.out_dir, a.arm)
    db = TW.wdir(a.out_dir, a.before_arm)
    ks = sorted(set(TW.have_steps(dt)) & set(TW.have_steps(do)))
    if ks != list(range(1, 21)):
        raise SystemExit("need 20 paired rungs, have %s" % ks)

    sd = torch.load(TW.CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    w1t, w1o = TW.load_w(dt, 1), TW.load_w(do, 1)
    names = sorted(n for n in w1t
                   if n in w1o and TW.PREFIX + n in sd
                   and tuple(sd[TW.PREFIX + n].shape) == tuple(w1t[n].shape))
    before_names = set(TW.load_w(db, 1)) if os.path.isdir(db) else set()
    new_names = set(names) - before_names
    print("scoring %d tensors, %d of them new" % (len(names), len(new_names)), flush=True)

    # ---- control 1, before a single scored number is read
    c1 = k1_bit_identity(dt, do, new_names, set(names))
    print("k=1 bit-identity: %s" % json.dumps({k: {kk: vv for kk, vv in v.items()
                                                   if kk != "failures"}
                                               for k, v in c1.items()
                                               if isinstance(v, dict)}), flush=True)
    if not c1["pass"]:
        raise SystemExit("k=1 bit-identity FAILED -- the map is wrong; nothing below is "
                         "scorable. %s" % json.dumps(c1)[:2000])

    W0 = {n: sd[TW.PREFIX + n].to(torch.float64).numpy().astype(np.float32) for n in names}
    nw0 = math.sqrt(sum(float(W0[n].astype(np.float64).ravel()
                              @ W0[n].astype(np.float64).ravel()) for n in names))
    W0o = TW.load_w(do, 1)
    scored = []
    for k in ks:
        r = TW.score_step(names, k, TW.load_w(do, k), TW.load_w(dt, k), W0, nw0, W0o)
        scored.append(r)
        print("k=%2d rel_d=%.6e worst=%.3e (%s)" % (r["k"], r["rel_d"],
                                                    r["worst_per_tensor"], r["worst_tensor"]),
              flush=True)

    scope = TW.scope_share(names)
    added = scope["pct_of_model_sq_grad_norm"] - BEFORE
    c3 = {"not_covered_families": scope["not_covered_families"],
          "n_tensors_in_reference_not_scored": scope["n_tensors_in_reference_not_scored"],
          "added_pct_of_model_sq_grad_norm": added,
          "expected_added": EXPECTED_ADDED,
          "added_minus_expected": added - EXPECTED_ADDED,
          "pass": (not scope["not_covered_families"]
                   and scope["n_tensors_in_reference_not_scored"] == 0
                   and abs(added - EXPECTED_ADDED) < 1e-6)}

    def steplog(arm):
        p = os.path.join(a.out_dir, "steplog_%s.json" % arm)
        return json.load(open(p)) if os.path.exists(p) else {}

    ours_sl, theirs_sl = steplog(a.arm), steplog(a.theirs_arm)
    coup = coupled_check(ours_sl, theirs_sl, scored, dt, do, names, W0)
    scope["coupled"] = bool(coup["coupled"])
    scope["coupled_checks"] = coup

    # The reference arm was banked before `of3t-refsweep` repointed `refpath.OF3PKG` from
    # `of3t_refprec/` to `of3t-campaign-refs/`, so the two strings differ and a path
    # comparison refuses a scorable pairing. What the clause is FOR is that 0.4.3 and 0.5.0
    # are a different function at this boundary (D120: 7.66979728e-01 against
    # 1.94959719e-05), and a tree digest answers that question directly where a string
    # cannot. `of3t-refatom` hit the same split and recorded the same block; this is that
    # block, recomputed here rather than quoted, plus the pin `refpath.py` itself carries.
    their_tree = theirs_sl.get("ref_tree")
    PIN = "1b27f5754b32b8e35e98f3f9246ebe09a9dcccc36b6f8e688d2b5e11b1aa57b5"
    sys.path.insert(0, os.path.join(_PERF, "of3t_trunk043ref"))
    import tree_digest as TD
    from pathlib import Path
    digests = [TD.digest(Path(r) / "openfold3") for r in (TW.OF3PKG, their_tree)]
    # D149, added by of3t-orchestrator pass 357 when the ratchet caught this file. The digests
    # above answer "are the two trees the same content", which is a different question from
    # "which tree did this process resolve". `refpath.install()` above puts two package trees on
    # sys.path, and this script then calls `torch.load(..., weights_only=False)` on upstream's
    # checkpoint -- an unpickle that will import whatever `openfold3.*` classes the pickle names,
    # from whichever tree resolves first. A named constant is not a resolution, so read it back.
    #
    # Three outcomes and all three are recorded rather than assumed. Resolves to the tree under
    # test: fine. Resolves elsewhere: STOP, because that is exactly the of3t-trajwide failure
    # this ratchet exists for. Not importable at all: also fine, and recorded as such -- nothing
    # can have come from the wrong tree if nothing came from any tree.
    try:
        import openfold3
        resolved_of3 = os.path.abspath(openfold3.__file__)
    except Exception as e:                                               # noqa: BLE001
        resolved_of3 = "NOT IMPORTABLE: %s" % e
    else:
        if os.path.realpath(os.path.dirname(os.path.dirname(resolved_of3))) != \
                os.path.realpath(TW.OF3PKG):
            raise SystemExit("D149: openfold3 resolved to %s, not the tree under test %s"
                             % (resolved_of3, TW.OF3PKG))
    tree_identity = {
        "reference_side_ran_on": their_tree, "refpath_OF3PKG": TW.OF3PKG,
        "openfold3_resolved_to": resolved_of3,
        "digests": digests, "pinned_expected_of3pkg043": PIN,
        "same_tree": len({d["tree_sha256"] for d in digests}) == 1,
        "matches_pin": all(d["tree_sha256"] == PIN for d in digests),
        "why": "of3t-refsweep repointed refpath.OF3PKG after this reference bank was taken; "
               "the paths differ and the trees do not"}
    if not (tree_identity["same_tree"] and tree_identity["matches_pin"]):
        raise SystemExit("reference tree identity FAILED: %s" % json.dumps(tree_identity)[:1500])

    res = {
        "what": "of3t-trajfull: the coupled 20-step trajectory over the WHOLE reference "
                "parameter set at the diffusion_module boundary, 761 of 761 tensors.",
        "arm": a.arm, "refatom": ours_sl.get("refatom"), "namemap": ours_sl.get("namemap"),
        "w0_baseline": "own", "ref_tree": their_tree,
        "reference_tree_identity": tree_identity,
        "scope": scope,
        "bar_pct_of_model_sq_grad_norm": BAR,
        "previous_best_coupled_pct": BEFORE,
        "steps_scored": ks, "steps_asked": 20, "steps": len(ks),
        "accumulate_grad_batches": 4, "warmup_no_steps": TW.SCHED["warmup_no_steps"],
        "controls": {"k1_bit_identity": c1,
                     "split_roundtrip_every_rung": (ours_sl.get("evidence") or {})
                     .get("namemap_emit"),
                     "family_subtotals": c3},
        "shipped_defaults": {k: v for k, v in TW.shipped_defaults().items()
                             if k in ("betas", "weight_decay", "plateau_until", "lr",
                                      "warmup_steps")},
        "d1": {"ours_norm": scored[0]["d_ours_norm"],
               "theirs_norm": scored[0]["d_theirs_norm"], "rel_d": scored[0]["rel_d"],
               "zero_both_sides": scored[0]["d_ours_norm"] == 0.0 == scored[0]["d_theirs_norm"]},
        "growth_k2_20": TW.growth(scored),
        "per_step": scored,
        "our_step_log": ours_sl.get("steps"), "their_step_log": theirs_sl.get("steps"),
        "our_evidence": ours_sl.get("evidence"), "their_evidence": theirs_sl.get("evidence"),
        "timing_s": {"scoring": TW.time.time() - t0,
                     "our_arm_wall_s": ours_sl.get("wall_s"),
                     "their_arm_wall_s": theirs_sl.get("wall_s")},
        "sha256": {"diffusion_boundary": TW.sha256(TW.DIFFCAP),
                   "cond_boundary": TW.sha256(TW.CAP), "checkpoint": TW.sha256(TW.CKPT)},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_step", "our_step_log", "their_step_log",
                                   "our_evidence", "their_evidence", "controls")},
                     indent=1, default=str)[:3500])
    print("scope.pct_of_model_sq_grad_norm = %.10f   bar %.4f   coupled=%s"
          % (scope["pct_of_model_sq_grad_norm"], BAR, scope["coupled"]))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
