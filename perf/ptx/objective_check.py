#!/usr/bin/env python3
"""The COMPOSED af3 objective against a float64 reference of the same composition.

`perf/ptxft/losscheck.py` checks the eleven quantities one at a time. Every one of them
passes and the composition can still be wrong, in four ways a per-term check cannot see:

  WEIGHTS    a term entering at the wrong weight is a correct loss trained at the wrong
             strength. The weights are not asserted here, they are RECONSTRUCTED from
             upstream's own `configs/configs_base.py` and compared to `LOSS_WEIGHTS`.
  FAN-IN     `smooth_lddt` and `bond` both seed `pred_dist`; `af3_loss` accumulates them
             into one array. A reference built from separate leaves would never exercise
             that sum, so every device output here is ONE shared torch leaf and the
             reference backward is ONE call.
  LEAKAGE    four terms take `pred_xyz` and are supposed to reach only their own logits,
             because upstream builds every bin label under `no_grad`. If any of them
             leaks, the reference's `pred_xyz` gradient carries a contribution ours does
             not and the check fails on `pred_xyz` alone.
  SKIPPING   `af3_loss` drops a zero-weight term instead of computing it and multiplying
             by zero, and drops a term whose inputs are absent. Both must leave the total
             and every seed unchanged against a reference that does compute them.

Then a directional finite difference, because agreeing with upstream's autograd proves the
two agree, not that either differentiates the scalar it returns. The FD is the only arm
here that compares the gradient against the loss VALUE.

Upstream is imported, not copied: `~/ref/Protenix` at 4c355be, same as losscheck.
"""

import argparse
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

REF = os.path.expanduser("~/ref/Protenix")


def upstream():
    """Protenix's loss module, with optree stubbed. Returns the module."""
    if REF not in sys.path:
        sys.path.insert(0, REF)
    if "optree" not in sys.modules:
        m = types.ModuleType("optree")
        m.tree_map = m.tree_flatten = lambda *a, **k: None
        sys.modules["optree"] = m
    import protenix.model.loss as L
    return L


def upstream_weights():
    """`LOSS_WEIGHTS` rebuilt from upstream's config and upstream's own products.

    `configs/configs_base.py:401-407` holds seven alphas; `loss.py:1463-1475` multiplies
    them into the eight weights `ProtenixLoss` applies. Both are read as text rather than
    imported, because importing the config pulls the full training stack. The stage-3
    finetuning values are upstream's own comments on those same lines ("or 1 in finetuning
    stage 3", "or 1 in finetuning stages", "or 0 in finetuning stages").
    """
    src = open(os.path.join(REF, "configs", "configs_base.py")).read()
    import re
    block = src[src.index('"alpha_confidence"'):src.index('"plddt": {')]
    a = {k: float(v) for k, v in re.findall(r'"(alpha_\w+|smooth_lddt)":\s*([0-9.e+-]+)', block)}
    def stage(pae, bond, slddt):
        return {"plddt": a["alpha_confidence"] * a["alpha_except_pae"],
                "pde": a["alpha_confidence"] * a["alpha_except_pae"],
                "resolved": a["alpha_confidence"] * a["alpha_except_pae"],
                "pae": a["alpha_confidence"] * pae,
                "mse": a["alpha_diffusion"],
                "bond": a["alpha_diffusion"] * bond,
                "smooth_lddt": a["alpha_diffusion"] * slddt,
                "distogram": a["alpha_distogram"]}
    return a, {"pretrain": stage(a["alpha_pae"], a["alpha_bond"], a["smooth_lddt"]),
               "finetune": stage(1.0, 1.0, 0.0)}


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n = np.linalg.norm(a - b)
    d = np.linalg.norm(b)
    return float(n / d) if d > 0 else float(n)


def make_batch(a):
    """One synthetic example carrying everything all eight terms need.

    Keys are the ones `objectives._TERMS` and `_NEEDS` name, so this is the interface the
    recipe's `dataset.batch()` has to produce, not a shape invented for the check.
    """
    from tt_bio.train import losses as X
    rng = np.random.default_rng(a.seed)
    N, S, NB = a.n, a.samples, 64
    true = rng.normal(0, 8, (N, 3))
    pred = true + rng.normal(0, 1.2, (S, N, 3))
    cmask = np.ones(N, bool)
    cmask[rng.choice(N, size=max(1, N // 10), replace=False)] = False
    rep = np.ones(N, bool)
    sigma = X.sample_noise_level(rng, (S,))
    tdist = np.linalg.norm(true[:, None, :] - true[None, :, :], axis=-1)
    pdist = np.linalg.norm(pred[:, :, None, :] - pred[:, None, :, :], axis=-1)
    dmask = (cmask[:, None] * cmask[None, :]).astype(np.float64)
    is_nuc = np.zeros(N, bool)
    ld, lw = X.atom_bespoke_lddt(pred[:, cmask], true[cmask], is_nuc[cmask],
                                 np.ones(int(cmask.sum()), bool), rep[cmask])
    batch = {
        "true_xyz": true, "coord_mask": cmask, "true_dist": tdist,
        "lddt_pair_mask": X.lddt_mask(tdist, dmask, is_nuc),
        "bond_mask": (np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) == 1).astype(float),
        "per_atom_lddt": ld, "per_atom_weight": lw,
        "frame_atom_index": np.stack([(np.arange(N) - 1) % N, np.arange(N),
                                      (np.arange(N) + 1) % N], 1),
        "edm_scale": X.edm_scale(sigma), "resolution": 2.0,
    }
    outputs = {
        "pred_xyz": pred, "pred_dist": pdist,
        "distogram_logits": rng.normal(0, 1.5, (N, N, NB)),
        "pde_logits": rng.normal(0, 1.5, (S, N, N, NB)),
        "pae_logits": rng.normal(0, 1.5, (S, N, N, NB)),
        "plddt_logits": rng.normal(0, 1.5, (S, int(cmask.sum()), 50)),
        "resolved_logits": rng.normal(0, 1.5, (S, N, 2)),
    }
    extra = {"rep": rep, "is_nuc": is_nuc, "dmask": dmask, "sigma": sigma, "N": N, "S": S}
    return batch, outputs, extra


def reference(batch, outputs, extra, weights, *, f64_align=True):
    """Upstream's eight `nn.Module`s composed into ONE scalar at `weights`, differentiated once.

    Returns (total, {output name: gradient}). Every output is a single leaf shared by every
    term that reads it, which is what makes this a reference for the COMPOSITION rather than
    for eight terms that happen to be added up afterwards.

    `f64_align` is the one instrument correction and it is load-bearing. `MSELoss` casts its
    Kabsch to float32 at `loss.py:1136-1146` ("Some ops in weighted_rigid_align do not support
    BFloat16 training"), which puts a 1e-8 floor under the largest term in the objective and
    would be read here as our error. Binding `torch.float32` to `torch.float64` for the
    duration of that one call makes those three casts no-ops; nothing else in the call site
    reads the name. Both arms are reported, so the correction is visible rather than assumed.
    """
    import torch
    L = upstream()
    torch.set_default_dtype(torch.float64)
    T = lambda x, g=False: torch.tensor(np.asarray(x), dtype=torch.float64, requires_grad=g)
    N, S = extra["N"], extra["S"]
    cmask = batch["coord_mask"]
    f = lambda x: T(np.asarray(x, np.float64))

    leaves = {k: T(v, True) for k, v in outputs.items()}
    # pLDDT's logits live on the masked subset (that is the shape `losses.plddt` takes), and
    # upstream wants full N. Scatter rather than slice so the leaf keeps its gradient.
    full_plddt = torch.zeros((S, N, 50), dtype=torch.float64)
    full_plddt = full_plddt.index_copy(1, torch.tensor(np.flatnonzero(cmask)),
                                       leaves["plddt_logits"])

    terms = {}
    terms["distogram"] = lambda: L.DistogramLoss()(
        logits=leaves["distogram_logits"], true_coordinate=f(batch["true_xyz"]),
        coordinate_mask=f(cmask), rep_atom_mask=f(extra["rep"]))
    terms["pde"] = lambda: L.PDELoss()(
        logits=leaves["pde_logits"], pred_coordinate=leaves["pred_xyz"],
        true_coordinate=f(batch["true_xyz"]), coordinate_mask=f(cmask),
        rep_atom_mask=f(extra["rep"]))
    terms["pae"] = lambda: L.PAELoss()(
        logits=leaves["pae_logits"], pred_coordinate=leaves["pred_xyz"],
        true_coordinate=f(batch["true_xyz"]), coordinate_mask=f(cmask),
        frame_atom_index=torch.tensor(batch["frame_atom_index"]),
        rep_atom_mask=f(extra["rep"]), has_frame=f(np.ones(N)))
    terms["plddt"] = lambda: L.PLDDTLoss()(
        logits=full_plddt, pred_coordinate=leaves["pred_xyz"],
        true_coordinate=f(batch["true_xyz"]), coordinate_mask=f(cmask),
        is_nucleotide=f(extra["is_nuc"]), is_polymer=f(np.ones(N)),
        rep_atom_mask=f(extra["rep"]))
    # No atom_mask, which is upstream's OWN call at loss.py:1783-1786 and is not the same
    # function: with a mask the denominator is eps + sum(mask) (loss.py:1055), without it a
    # plain mean. Passing an all-ones mask here instead reads 2.08e-08 against the adapter
    # in objectives._TERMS, which is eps/N: the reference being wrong, not the objective.
    terms["resolved"] = lambda: L.ExperimentallyResolvedLoss()(
        logits=leaves["resolved_logits"], coordinate_mask=f(cmask))
    terms["bond"] = lambda: L.BondLoss()(
        pred_distance=leaves["pred_dist"], true_distance=f(batch["true_dist"]),
        distance_mask=f(extra["dmask"]), bond_mask=f(batch["bond_mask"]),
        per_sample_scale=f(batch["edm_scale"]))
    terms["smooth_lddt"] = lambda: L.SmoothLDDTLoss()(
        pred_distance=leaves["pred_dist"], true_distance=f(batch["true_dist"]),
        distance_mask=f(extra["dmask"]), lddt_mask=f(batch["lddt_pair_mask"]))

    def mse():
        keep = torch.float32
        if f64_align:
            torch.float32 = torch.float64     # the three casts at loss.py:1139-1146
        try:
            return L.MSELoss()(pred_coordinate=leaves["pred_xyz"],
                               true_coordinate=f(batch["true_xyz"]), coordinate_mask=f(cmask),
                               is_dna=f(np.zeros(N)), is_rna=f(np.zeros(N)),
                               is_ligand=f(np.zeros(N)), per_sample_scale=f(batch["edm_scale"]))
        finally:
            torch.float32 = keep
    terms["mse"] = mse

    # Upstream's own aggregation, loss.py:1607: cum_loss = cum_loss + weight * loss, over
    # every term in loss_fns INCLUDING the zero-weighted ones. Kept that way deliberately --
    # `af3_loss` skips them, and the point of the reference is to not skip.
    per_term, total = {}, torch.zeros((), dtype=torch.float64)
    for name, fn in terms.items():
        v = fn()
        per_term[name] = float(v.item())
        total = total + weights[name] * v
    total.backward()
    grads = {k: (v.grad.numpy().copy() if v.grad is not None else np.zeros(v.shape))
             for k, v in leaves.items()}
    return float(total.item()), grads, per_term


def ours(batch, outputs, weights):
    from tt_bio.train import objectives
    return objectives.objective("af3")(batch, outputs, weights=weights)


def fd_ours(batch, outputs, weights, seeds, eps, keys=None):
    """Central difference of OUR composition along its own gradient direction.

    The direction is the gradient itself, normalised, so the predicted change is
    `2*eps*||g||` and any missing term shows up as a deficit rather than as noise in an
    unrelated direction.
    """
    keys = list(seeds) if keys is None else [k for k in keys if k in seeds]
    nrm = np.sqrt(sum(float((seeds[k] ** 2).sum()) for k in keys))
    d = {k: seeds[k] / nrm for k in keys}
    def at(sign):
        o = dict(outputs)
        for k in keys:
            o[k] = outputs[k] + sign * eps * d[k]
        return ours(batch, o, weights)[0]
    return (at(+1) - at(-1)) / (2 * eps), nrm


def fd_upstream(batch, outputs, extra, weights, seeds, eps, keys):
    nrm = np.sqrt(sum(float((seeds[k] ** 2).sum()) for k in keys))
    d = {k: seeds[k] / nrm for k in keys}
    def at(sign):
        o = dict(outputs)
        for k in keys:
            o[k] = outputs[k] + sign * eps * d[k]
        return reference(batch, o, extra, weights)[0]
    return (at(+1) - at(-1)) / (2 * eps), nrm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bar", type=float, default=1e-12)
    ap.add_argument("--fd-bar", type=float, default=1e-6)
    a = ap.parse_args()
    from tt_bio.train import losses as X

    fails = []
    print(f"# objective_check: the COMPOSED af3 objective vs a float64 reference of the same "
          f"composition\n# N={a.n} tokens, {a.samples} diffusion samples, seed {a.seed}, "
          f"float64 both sides, bar {a.bar:.0e} relative")

    # ---------------------------------------------------------------- 1. the weight table
    alphas, rebuilt = upstream_weights()
    print(f"\n## weights, rebuilt from {REF}/configs/configs_base.py:401-407 through "
          f"loss.py:1463-1475")
    print("   " + ", ".join(f"{k}={v:g}" for k, v in sorted(alphas.items())))
    print(f"{'stage':<10} {'term':<12} {'ours':>10} {'upstream product':>18}")
    for stage in ("pretrain", "finetune"):
        for term in sorted(rebuilt[stage]):
            o, u = X.LOSS_WEIGHTS[stage][term], rebuilt[stage][term]
            bad = "" if o == u else "   <-- MISMATCH"
            if bad:
                fails.append(f"weight {stage}/{term}")
            print(f"{stage:<10} {term:<12} {o:>10g} {u:>18g}{bad}")

    batch, outputs, extra = make_batch(a)

    # ------------------------------------------ 2. composite value and gradient, both stages
    for stage in ("pretrain", "finetune"):
        w = X.LOSS_WEIGHTS[stage]
        total, breakdown, seeds = ours(batch, outputs, w)
        for align, label in ((True, "float64 Kabsch"), (False, "upstream float32 Kabsch")):
            ref_total, ref_grads, per_term = reference(batch, outputs, extra, w,
                                                       f64_align=align)
            ve = rel(total, ref_total)
            ok = ve <= a.bar or not align
            if not ok:
                fails.append(f"{stage} total")
            print(f"\n## {stage}: composite value, reference with {label}")
            print(f"   ours {total:.15g}   upstream {ref_total:.15g}   rel {ve:.3e}"
                  f"{'' if ok else '   <-- FAIL'}")
            if not align:
                continue
            print(f"   active terms {sum(1 for b in breakdown.values() if b['value'] is not None)}"
                  f" of 8; skipped "
                  f"{[k for k, b in breakdown.items() if b.get('skipped')]}")
            print(f"{'   term':<15} {'value':>14} {'weight':>9} {'contribution':>14} "
                  f"{'upstream value':>16} {'rel':>10}")
            for term in sorted(breakdown):
                b = breakdown[term]
                if b["value"] is None:
                    print(f"   {term:<12} {'--':>14} {b['weight']:>9g} {0.0:>14.6g} "
                          f"{per_term[term]:>16.6g}  (skipped: {b['skipped']})")
                    continue
                r = rel(b["value"], per_term[term])
                if r > a.bar:
                    fails.append(f"{stage} term {term}")
                print(f"   {term:<12} {b['value']:>14.8g} {b['weight']:>9g} "
                      f"{b['contribution']:>14.8g} {per_term[term]:>16.8g} {r:>10.2e}"
                      f"{'' if r <= a.bar else '  <-- FAIL'}")
            print(f"\n## {stage}: composite gradient, one backward over shared leaves")
            print(f"{'   output':<18} {'shape':>18} {'rel err':>12}  seeded by")
            seeded_by = {}
            from tt_bio.train.objectives import _SEED
            for t, k in _SEED.items():
                if w[t] != 0.0:
                    seeded_by.setdefault(k, []).append(t)
            for k in sorted(ref_grads):
                g = seeds.get(k)
                by = ", ".join(seeded_by.get(k, [])) or "nothing at this stage"
                if g is None:
                    r = float(np.linalg.norm(ref_grads[k]))
                    note = "ours: no seed" if r == 0 else "   <-- FAIL ours has no seed"
                    if r != 0:
                        fails.append(f"{stage} missing seed {k}")
                    print(f"   {k:<15} {str(ref_grads[k].shape):>18} "
                          f"{r:>12.3e}  {by}  ({note})")
                    continue
                r = rel(g, ref_grads[k])
                if r > a.bar:
                    fails.append(f"{stage} grad {k}")
                print(f"   {k:<15} {str(np.shape(g)):>18} {r:>12.3e}  {by}"
                      f"{'' if r <= a.bar else '   <-- FAIL'}")

    # ------------------------------------------------- 3. directional finite difference
    w = X.LOSS_WEIGHTS["finetune"]          # all eight terms live
    total, _, seeds = ours(batch, outputs, w)
    print(f"\n## directional finite difference, finetune stage (all 8 terms active)")
    print(f"   direction = the composed gradient itself, normalised; predicted dL/deps = "
          f"||g||")
    logit_only = [k for k in seeds if k != "pred_xyz"]
    for label, keys, fn in (
            ("ours, all 7 outputs", None, lambda e, k: fd_ours(batch, outputs, w, seeds, e, k)),
            ("upstream, logits+pred_dist", logit_only,
             lambda e, k: fd_upstream(batch, outputs, extra, w, seeds, e, k))):
        best = None
        print(f"   {label}:")
        for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            got, nrm = fn(eps, keys)
            r = abs(got - nrm) / nrm
            best = r if best is None else min(best, r)
            print(f"      eps {eps:.0e}   predicted {nrm:.12g}   measured {got:.12g}   "
                  f"rel {r:.3e}")
        ok = best <= a.fd_bar
        if not ok:
            fails.append(f"fd {label}")
        print(f"      best-of-sweep {best:.3e}, bar {a.fd_bar:.0e}"
              f"{'' if ok else '   <-- FAIL'}")

    # --------------------------------------------------------------- 4. the conditionals
    print(f"\n## conditionals")
    w = dict(X.LOSS_WEIGHTS["finetune"])
    t_all, b_all, s_all = ours(batch, outputs, w)
    # (a) a zero weight skips the term and changes nothing.
    w0 = dict(w, pae=0.0)
    t0, b0, s0 = ours(batch, outputs, w0)
    expect = t_all - w["pae"] * b_all["pae"]["value"]
    r = rel(t0, expect)
    if r > a.bar or "pae_logits" in s0:
        fails.append("zero-weight skip")
    print(f"   pae weight -> 0: total {t0:.12g} vs sum of the rest {expect:.12g}, rel {r:.2e}; "
          f"pae seed present: {'pae_logits' in s0} (must be False)")
    # (b) an absent input is recorded as SKIPPED, never as a zero that enters the sum.
    o_less = {k: v for k, v in outputs.items() if k != "pae_logits"}
    t1, b1, s1 = ours(batch, o_less, w)
    r1 = rel(t1, expect)
    if r1 > a.bar or b1["pae"].get("skipped") is None or b1["pae"]["value"] is not None:
        fails.append("absent-input skip")
    print(f"   pae_logits absent: total {t1:.12g}, rel {r1:.2e}, breakdown says "
          f"{b1['pae'].get('skipped')!r}, value {b1['pae']['value']!r} (must be None, not 0.0)")
    # (c) every other seed is untouched by either skip.
    worst = max(rel(s1[k], s_all[k]) for k in s1)
    if worst > a.bar:
        fails.append("skip perturbed another seed")
    print(f"   the other {len(s1)} seeds under (b): worst rel change {worst:.2e}")
    # (d) the per-example resolution gate. Upstream zeroes the four confidence terms when the
    #     example's resolution is outside [0.1, 4.0] (loss.py:1734-1759 and :1595-1601).
    #     `af3_loss` has no such gate; it is expressible only as a per-call weight override.
    conf = ("plddt", "pde", "pae", "resolved")
    w_gated = dict(w, **{k: 0.0 for k in conf})
    t2, b2, s2 = ours(batch, outputs, w_gated)
    drop = sum(w[k] * b_all[k]["value"] for k in conf)
    r2 = rel(t2, t_all - drop)
    if r2 > a.bar:
        fails.append("resolution-gate override")
    print(f"   resolution gate (res outside [0.1, 4.0]): the override "
          f"weights={{{', '.join(k + ': 0.0' for k in conf)}}} reproduces upstream's "
          f"`loss = 0.0 * loss`; total {t2:.12g}, rel {r2:.2e}, confidence seeds present: "
          f"{sorted(k for k in s2 if k.endswith('_logits') and k[:-7] in conf)}")
    print(f"   NOT AUTOMATIC: batch['resolution'] = {batch['resolution']} is carried by the "
          f"example and `af3_loss` never reads it. See the state doc's CONDITIONALS.")

    print()
    if fails:
        print(f"OBJECTIVE FAILS ({len(fails)}): " + ", ".join(fails))
        return 1
    print("OBJECTIVE PASSES: the eight-term composition, its weights, its fan-in, its "
          "skipping and its gradient all agree with a float64 reference of the same "
          "composition, and the gradient predicts the loss change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
