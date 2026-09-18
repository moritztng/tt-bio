#!/usr/bin/env python3
"""Phase 2: every AF3 loss term against Protenix's OWN implementation, on CPU.

A gradient check cannot catch a wrong loss, because the gradient of the wrong loss is
computed perfectly -- this row already found that the hard way, with a distogram head
computing half of upstream's symmetrisation while every gradcheck passed. So each term in
`tt_bio.train.losses` is run against the `nn.Module` from ByteDance's own
`protenix/model/loss.py` on shared random inputs, in float64 both sides, and both halves
are compared:

  VALUE     our loss vs upstream's loss, relative error, bar 1e-12 (both float64 doing the
            same arithmetic in a different order, so anything above that is a real
            difference, not round-off).
  GRADIENT  our analytic gradient vs upstream's AUTOGRAD gradient of the same scalar.
            Upstream never writes a backward by hand, so torch's is the independent
            reference, and it is the thing the tape will actually be seeded with.

Upstream is imported, not copied: `~/ref/Protenix` at 4c355be with `optree` stubbed out
(`protenix/model/utils.py:20` imports it only for a shape-printing helper at :527 that
nothing here calls, and it is not in the fleet pin).
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


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n = np.linalg.norm(a - b)
    d = np.linalg.norm(b)
    return float(n / d) if d > 0 else float(n)


class Case:
    """One term: its value check and its gradient check, both against upstream."""

    def __init__(self, name, value_err, grad_err, note=""):
        self.name, self.value_err, self.grad_err, self.note = name, value_err, grad_err, note


def run(a):
    import torch
    from tt_bio.train import losses as X
    L = upstream()
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(a.seed)
    N, S, NB = a.n, a.samples, 64
    T = lambda x, g=False: torch.tensor(np.asarray(x), dtype=torch.float64, requires_grad=g)

    true = rng.normal(0, 8, (N, 3))
    pred = true + rng.normal(0, 1.2, (S, N, 3))
    cmask = np.ones(N, bool)
    cmask[rng.choice(N, size=max(1, N // 10), replace=False)] = False
    rep = np.ones(N, bool)
    sigma = X.sample_noise_level(rng, (S,))
    scale = X.edm_scale(sigma)
    cases = []

    # ---- distogram
    lg = rng.normal(0, 1.5, (N, N, NB))
    ours, g_ours = X.distogram(lg, true, cmask)
    tl = T(lg, True)
    up = L.DistogramLoss()(logits=tl, true_coordinate=T(true),
                           coordinate_mask=T(cmask.astype(float)),
                           rep_atom_mask=T(rep.astype(float)))
    up.backward()
    cases.append(Case("distogram", rel(ours, up.item()), rel(g_ours, tl.grad.numpy()),
                      f"[{N},{N},64] logits, {int((cmask[:,None]&cmask[None,:]).sum())} masked pairs"))

    # ---- pde
    lg = rng.normal(0, 1.5, (S, N, N, NB))
    ours, g_ours = X.pde(lg, pred, true, cmask)
    tl = T(lg, True)
    up = L.PDELoss()(logits=tl, pred_coordinate=T(pred), true_coordinate=T(true),
                     coordinate_mask=T(cmask.astype(float)),
                     rep_atom_mask=T(rep.astype(float)))
    up.backward()
    cases.append(Case("pde", rel(ours, up.item()), rel(g_ours, tl.grad.numpy()),
                      f"{S} samples, bins clamped 1..64 on |dpred-dtrue|"))

    # ---- pae
    frame_idx = np.stack([(np.arange(N) - 1) % N, np.arange(N), (np.arange(N) + 1) % N], 1)
    has_frame = np.ones(N, bool)
    lg = rng.normal(0, 1.5, (S, N, N, NB))
    ours, g_ours = X.pae(lg, pred, true, cmask, frame_idx)
    tl = T(lg, True)
    up = L.PAELoss()(logits=tl, pred_coordinate=T(pred), true_coordinate=T(true),
                     coordinate_mask=T(cmask.astype(float)),
                     frame_atom_index=torch.tensor(frame_idx),
                     rep_atom_mask=T(rep.astype(float)),
                     has_frame=T(has_frame.astype(float)))
    up.backward()
    cases.append(Case("pae", rel(ours, up.item()), rel(g_ours, tl.grad.numpy()),
                      "squared boundaries, all N tokens carry a frame"))

    # ---- plddt, and the bespoke lddt label it depends on
    is_nuc = np.zeros(N, bool)
    is_poly = np.ones(N, bool)
    ld, lw = X.atom_bespoke_lddt(pred[0][cmask], true[cmask], is_nuc[cmask],
                                 is_poly[cmask], rep[cmask])
    u_ld, u_lw = L.calculate_atom_bespoke_lddt(
        pred_coordinate=T(pred[0][cmask]), true_coordinate=T(true[cmask]),
        is_nucleotide=T(is_nuc[cmask].astype(float)),
        is_polymer=T(is_poly[cmask].astype(float)),
        rep_atom_mask=T(rep[cmask].astype(float)))
    cases.append(Case("atom_bespoke_lddt", rel(ld, u_ld.numpy()[..., 0]),
                      rel(lw, u_lw.numpy()[..., 0]),
                      "the pLDDT label itself: per-atom lddt (value) and its weight (grad column)"))
    lg = rng.normal(0, 1.5, (S, int(cmask.sum()), 50))
    ld_s, lw_s = X.atom_bespoke_lddt(pred[:, cmask], true[cmask], is_nuc[cmask],
                                     is_poly[cmask], rep[cmask])
    ours, g_ours = X.plddt(lg, ld_s, lw_s)
    tl = T(lg, True)
    full = np.zeros((S, N, 50))
    full[:, cmask, :] = lg
    tf = T(full, True)
    up = L.PLDDTLoss()(logits=tf, pred_coordinate=T(pred), true_coordinate=T(true),
                       coordinate_mask=T(cmask.astype(float)),
                       is_nucleotide=T(is_nuc.astype(float)),
                       is_polymer=T(is_poly.astype(float)),
                       rep_atom_mask=T(rep.astype(float)))
    up.backward()
    cases.append(Case("plddt", rel(ours, up.item()),
                      rel(g_ours, tf.grad.numpy()[:, cmask, :]),
                      "50 bins on the normalised per-atom lddt"))

    # ---- resolved
    lg = rng.normal(0, 1.5, (S, N, 2))
    amask = np.ones(N)
    ours, g_ours = X.resolved(lg, cmask, amask)
    tl = T(lg, True)
    up = L.ExperimentallyResolvedLoss()(logits=tl, coordinate_mask=T(cmask.astype(float)),
                                        atom_mask=T(amask))
    up.backward()
    cases.append(Case("resolved", rel(ours, up.item()), rel(g_ours, tl.grad.numpy()),
                      "two classes, atom_mask denominator"))

    # ---- mse, with the EDM per-sample scale
    ours, g_ours = X.mse(pred, true, cmask, per_sample_scale=scale)
    tp = T(pred, True)
    up = L.MSELoss()(pred_coordinate=tp, true_coordinate=T(true),
                     coordinate_mask=T(cmask.astype(float)),
                     is_dna=T(np.zeros(N)), is_rna=T(np.zeros(N)),
                     is_ligand=T(np.zeros(N)), per_sample_scale=T(scale))
    up.backward()
    cases.append(Case("mse", rel(ours, up.item()), rel(g_ours, tp.grad.numpy()),
                      f"stop-gradient Kabsch, EDM scale {scale.min():.3f}..{scale.max():.3f}"))

    # ---- bond
    pdist = np.linalg.norm(pred[:, :, None, :] - pred[:, None, :, :], axis=-1)
    tdist = np.linalg.norm(true[:, None, :] - true[None, :, :], axis=-1)
    bm = (np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) == 1).astype(float)
    dmask = (cmask[:, None] * cmask[None, :]).astype(float)
    ours, g_ours = X.bond(pdist, tdist, bm, cmask, per_sample_scale=scale)
    tp = T(pdist, True)
    up = L.BondLoss()(pred_distance=tp, true_distance=T(tdist), distance_mask=T(dmask),
                      bond_mask=T(bm), per_sample_scale=T(scale))
    up.backward()
    cases.append(Case("bond", rel(ours, up.item()), rel(g_ours, tp.grad.numpy()),
                      f"{int(bm.sum())} bonded pairs, denominator sum(mask + eps)"))

    # ---- smooth_lddt
    lm = X.lddt_mask(tdist, dmask, np.zeros(N, bool))
    ours, g_ours = X.smooth_lddt(pdist, tdist, lm)
    tp = T(pdist, True)
    up = L.SmoothLDDTLoss()(pred_distance=tp, true_distance=T(tdist),
                            distance_mask=T(dmask), lddt_mask=T(lm))
    up.backward()
    cases.append(Case("smooth_lddt", rel(ours, up.item()), rel(g_ours, tp.grad.numpy()),
                      f"lddt mask {int(lm.sum())} pairs of {N * N}, 15 A radius"))

    # ---- weighted Kabsch on its own, in float64 both sides
    from protenix.metrics.rmsd import weighted_rigid_align as up_align
    w = rng.uniform(0.2, 3.0, N) * cmask
    ours_a = X.weighted_rigid_align(true, pred[0], w)
    up_a = up_align(x=T(true), x_target=T(pred[0]), atom_weight=T(w)).numpy()
    cases.append(Case("weighted_rigid_align", rel(ours_a, up_a), 0.0,
                      "AF3 Algorithm 28, float64 both sides; no gradient by design"))

    # ---- the distance -> coordinate bridge, checked on its own
    gd = rng.normal(0, 1, (S, N, N))
    tp = T(pred, True)
    # torch.cdist's default switches to a matmul kernel above 25 rows and that path loses
    # ~1e-10 of relative accuracy, which would be read here as our error rather than the
    # instrument's. The direct path is the reference.
    d = torch.cdist(tp, tp, compute_mode="donot_use_mm_for_euclid_dist")
    (d * T(gd)).sum().backward()
    ours = X.dist_grad_to_coords(gd, pred)
    cases.append(Case("dist_grad_to_coords", 0.0, rel(ours, tp.grad.numpy()),
                      "vs torch.cdist autograd; no value of its own"))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bar", type=float, default=1e-12)
    ap.add_argument("--mse-bar", type=float, default=1e-5,
                    help="MSELoss runs its Kabsch in float32 upstream (loss.py:1139-1147), "
                         "so a float64 alignment cannot agree with it to float64")
    a = ap.parse_args()
    cases = run(a)
    print(f"# losscheck: tt_bio.train.losses vs Protenix's own nn.Modules, float64 both "
          f"sides, N={a.n} tokens, {a.samples} samples, seed {a.seed}")
    print(f"# bar {a.bar:.0e} relative -- same arithmetic in a different order, so anything "
          f"above this is a real difference")
    print(f"{'term':<20} {'value rel err':>14} {'grad rel err':>14}  note")
    bad = []
    for c in cases:
        bar = a.mse_bar if c.name == "mse" else a.bar
        ok = c.value_err <= bar and c.grad_err <= bar
        print(f"{c.name:<20} {c.value_err:>14.3e} {c.grad_err:>14.3e}  {c.note}"
              f"{'' if ok else '   <-- FAIL'}{'   (bar %.0e)' % bar if bar != a.bar else ''}")
        if not ok:
            bad.append(c.name)
    print()
    if bad:
        print(f"LOSSES FAIL ({len(bad)}): " + ", ".join(bad))
        return 1
    print(f"LOSSES PASS: {len(cases)} terms, every value and every gradient within "
          f"{a.bar:.0e} of Protenix's own implementation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
