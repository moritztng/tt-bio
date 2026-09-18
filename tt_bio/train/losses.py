"""The AF3 training loss set, transcribed from Protenix with Protenix's own weights.

Every term here is a transcription of ByteDance's `protenix/model/loss.py` at
`4c355be`, not an independent reading of the AF3 paper, because the paper leaves the
reductions and the masking underdetermined and a loss that is 5 % wrong trains a subtly
wrong model that no gradient check can catch. `perf/ptxft/losscheck.py` runs Protenix's
own `nn.Module` for each term on CPU against these functions on shared random inputs, so
the agreement is measured rather than argued.

Provenance, term by term (file:line in the Protenix tree, and the weight from
`configs/configs_base.py:401-407`):

    term          upstream class / line                weight in ProtenixLoss
    distogram     DistogramLoss, loss.py:515           alpha_distogram = 3e-2
    smooth_lddt   SmoothLDDTLoss, loss.py:63           alpha_diffusion * 1.0
    bond          BondLoss, loss.py:299                alpha_diffusion * alpha_bond
    mse           MSELoss, loss.py:1063                alpha_diffusion = 4.0
    plddt         PLDDTLoss, loss.py:1283              alpha_confidence * alpha_except_pae
    pde           PDELoss, loss.py:638                 alpha_confidence * alpha_except_pae
    pae           PAELoss, loss.py:813                 alpha_confidence * alpha_pae
    resolved      ExperimentallyResolvedLoss, :1010    alpha_confidence * alpha_except_pae

alpha_diffusion 4.0, alpha_distogram 3e-2, alpha_confidence 1e-4, alpha_except_pae 1.0,
alpha_pae 0.0 (1.0 in finetuning stage 3), alpha_bond 0.0 (1.0 in finetuning),
smooth_lddt 1.0 (0.0 in finetuning). So the *pretraining* objective is mse + smooth_lddt +
distogram + the three non-PAE confidence heads, and bond and pae only switch on in
finetuning. `LOSS_WEIGHTS` below carries both stages rather than one of them.

THREE THINGS UPSTREAM DOES THAT ARE EASY TO GET WRONG, all load-bearing:

* **The alignment is a stop-gradient.** `MSELoss.forward` calls `weighted_rigid_align`
  inside `torch.no_grad()` and `.detach()`s the result (loss.py:1180-1189, and
  `metrics/rmsd.py:243-252` runs it under `no_grad` again). So the Kabsch fit of the label
  onto the prediction is never differentiated, and the tape does not need a backward for
  an SVD.
* **Every bin label is detached too.** Distogram, PDE, PAE and pLDDT all build their true
  bins under `no_grad`, including pLDDT's, whose label depends on the prediction. The
  gradient of those four terms therefore reaches ONLY the logits.
* **Only mse and bond take the per-sample noise scale.** `SmoothLDDTLoss.forward` has no
  `per_sample_scale` argument at all (loss.py:104). The scale itself is the EDM weight
  `(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2` (loss.py:1638) with
  sigma_data = 16.0 (`generator.py:40`), which is why `edm_scale` lives here.

Everything is float64 numpy on host. The loss value is a scalar reduction over N^2 terms
and a bf16 reduction of it would swamp exactly the improvement being measured; the
gradient handed back to the tape is the analytic one, so the tape differentiates the loss
rather than an approximation of it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "LOSS_WEIGHTS", "SIGMA_DATA", "edm_scale", "sample_noise_level",
    "distogram", "smooth_lddt", "bond", "mse", "plddt", "pde", "pae", "resolved",
    "atom_bespoke_lddt", "lddt_mask", "weighted_rigid_align", "dist_grad_to_coords",
    "cross_entropy_bins", "bin_index",
]

# configs/configs_base.py:401-407. Stage names are upstream's: pretraining runs
# alpha_pae = 0 and alpha_bond = 0, finetuning stage 3 turns both on and smooth_lddt off.
LOSS_WEIGHTS = {
    "pretrain": {"mse": 4.0, "smooth_lddt": 4.0, "bond": 0.0, "distogram": 3e-2,
                 "plddt": 1e-4, "pde": 1e-4, "resolved": 1e-4, "pae": 0.0},
    "finetune": {"mse": 4.0, "smooth_lddt": 0.0, "bond": 4.0, "distogram": 3e-2,
                 "plddt": 1e-4, "pde": 1e-4, "resolved": 1e-4, "pae": 1e-4},
}
SIGMA_DATA = 16.0                      # generator.py:40
DISTOGRAM_GRID = (2.3125, 21.6875, 64)  # loss.py:534-536
PDE_GRID = (0.0, 32.0, 64)              # loss.py:652-655
PAE_GRID = (0.0, 32.0, 64)              # loss.py:826-829, boundaries squared
PLDDT_GRID = (0.0, 1.0, 50)             # loss.py:1300-1303
LDDT_RADIUS = (30.0, 15.0)              # ProtenixLoss.lddt_radius, loss.py:1465-1468


def sample_noise_level(rng, size=(), *, p_mean=-1.2, p_std=1.5, sigma_data=SIGMA_DATA):
    """`TrainingNoiseSampler.__call__`, generator.py:49-60: exp(N(p_mean, p_std))*sigma_data."""
    return np.exp(rng.standard_normal(size) * p_std + p_mean) * sigma_data


def edm_scale(sigma, sigma_data=SIGMA_DATA):
    """The EDM per-sample diffusion weight, loss.py:1636-1639."""
    sigma = np.asarray(sigma, np.float64)
    return (sigma ** 2 + sigma_data ** 2) / (sigma_data * sigma) ** 2


# ------------------------------------------------------------------ shared machinery

def bin_index(values, lo, hi, n_bins, *, n_edges=None, square=False):
    """`sum(values > boundaries)`, which is how every upstream label is assigned.

    Upstream builds the boundaries two different ways and the difference is not cosmetic:
    the distogram uses `linspace(min, max, no_bins - 1)` and indexes directly
    (loss.py:571-585, giving bins in [0, no_bins-1]), while PDE, PAE and pLDDT use
    `linspace(min, max, no_bins + 1)` and then `clamp(bins, 1, no_bins) - 1`
    (loss.py:652-680). `n_edges` selects which, so neither is re-derived at a call site.
    """
    n_edges = (n_bins - 1) if n_edges is None else n_edges
    edges = np.linspace(lo, hi, n_edges)
    if square:
        edges = edges ** 2
    b = (values[..., None] > edges).sum(-1)
    if n_edges == n_bins + 1:
        b = np.clip(b, 1, n_bins) - 1
    return b.astype(np.int64)


def cross_entropy_bins(logits, bins, mask, *, eps=0.0, reduce_axes=2):
    """Masked softmax cross-entropy plus its exact gradient w.r.t. the logits.

    `reduce_axes` is how many trailing axes (after the bin axis) the mask covers: 2 for a
    pair term, 1 for a per-atom term. Anything left of those is a sample axis and gets a
    plain mean, which is what every upstream term does (`loss.mean(dim=-1)` at
    loss.py:774, :1000, :1057, :1397).

    THE EPS SITS INSIDE THE PER-SAMPLE DENOMINATOR AND THAT PLACEMENT IS MEASURABLE. Every
    upstream CE term divides by `self.eps + sum(mask)` with eps = 1e-6 (loss.py:626, :770,
    :999, :1056) and only then averages over samples. Dropping the eps scales value and
    gradient alike by 1 + eps/M; adding it once to a denominator that has already been
    summed over S samples leaves 2/3 of that error behind at S = 3. Both versions of the
    mistake were live here and both were caught by comparing against upstream rather than
    against a tolerance: 5.2e-10 and then 3.4e-10 at 1936 masked pairs, systematic bias in
    the direction of the loss, not round-off.
    """
    x = np.asarray(logits, np.float64)
    x = x - x.max(-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(-1, keepdims=True)
    # Upstream broadcasts a label with no sample axis across the samples
    # (ExperimentallyResolvedLoss does exactly this at loss.py:1046).
    bins = np.broadcast_to(np.asarray(bins), x.shape[:-1])
    m = np.broadcast_to(np.asarray(mask, np.float64), x.shape[:-1])
    axes = tuple(range(-reduce_axes, 0))
    denom = eps + float(np.asarray(mask, np.float64).sum(axis=axes).max())
    if denom <= 0:
        raise ValueError("empty loss mask")
    idx = np.ogrid[tuple(slice(s) for s in bins.shape)]
    pick = p[tuple(idx) + (bins,)]
    ce = -np.log(np.clip(pick, 1e-300, None))
    n_lead = int(np.prod(x.shape[:-1 - reduce_axes])) or 1
    loss = float(((ce * m).sum(axis=axes) / denom).sum() / n_lead)
    grad = p.copy()
    grad[tuple(idx) + (bins,)] -= 1.0
    grad *= (m / (denom * n_lead))[..., None]
    return loss, grad


def dist_grad_to_coords(grad_dist, coords, *, eps=1e-10):
    """Push a [..., N, N] gradient w.r.t. pairwise distance back onto [..., N, 3] coords.

    dD_ij/dx_i = (x_i - x_j)/D_ij, so entry (i, j) feeds both atoms. Summing the
    transpose in is what a per-pair loss means for an atom, and dropping it is a silent
    factor of two on a symmetric mask.
    """
    x = np.asarray(coords, np.float64)
    d = np.linalg.norm(x[..., :, None, :] - x[..., None, :, :], axis=-1)
    g = np.asarray(grad_dist, np.float64)
    g = g + np.swapaxes(g, -1, -2)
    u = (x[..., :, None, :] - x[..., None, :, :]) / np.maximum(d, eps)[..., None]
    return (g[..., None] * u).sum(-2)


def _pdist(x):
    return np.linalg.norm(x[..., :, None, :] - x[..., None, :, :], axis=-1)


def weighted_rigid_align(x, x_target, atom_weight):
    """Weighted Kabsch: rotate/translate `x` onto `x_target`. AF3 Algorithm 28.

    Upstream is `protenix/metrics/rmsd.py:216` wrapping `align_pred_to_true` with
    `allowing_reflection=False`, and it is called under `no_grad` with `.detach()`, so
    this deliberately returns an array with no gradient path. Written in numpy rather
    than reusing `tt_bio.boltz2.weighted_rigid_align`, which is torch and would pull a
    whole model module into the training package for one 3x3 SVD.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(x_target, np.float64)
    w = np.asarray(atom_weight, np.float64)
    w = np.broadcast_to(w, x.shape[:-1])[..., None]
    sw = w.sum(-2, keepdims=True)
    cx = (x * w).sum(-2, keepdims=True) / sw
    cy = (y * w).sum(-2, keepdims=True) / sw
    xc, yc = x - cx, y - cy
    h = np.einsum("...ni,...nj->...ij", w * xc, yc)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum("...ij,...jk->...ik", u, vt)))
    f = np.zeros(h.shape[:-2] + (3, 3))
    f[..., 0, 0] = f[..., 1, 1] = 1.0
    f[..., 2, 2] = d
    r = np.einsum("...ij,...jk,...kl->...il", u, f, vt)
    return np.einsum("...ni,...ij->...nj", xc, r) + cy


def lddt_mask(true_dist, distance_mask, is_nucleotide,
              radius=LDDT_RADIUS):
    """`compute_lddt_mask`, loss.py:454-492. Not symmetric: the radius is the ROW atom's."""
    nuc_r, other_r = radius
    nuc = np.asarray(is_nucleotide, bool)
    c = ((true_dist < nuc_r) * nuc[..., None]
         + (true_dist < other_r) * (~nuc[..., None])).astype(np.float64)
    c = c * (1.0 - np.eye(c.shape[-1]))
    return c * np.asarray(distance_mask, np.float64)


# ------------------------------------------------------------------ the eight terms

def distogram(logits, true_xyz, coord_mask, *, grid=DISTOGRAM_GRID, eps=1e-6):
    """DistogramLoss, loss.py:515-636. `logits` [N, N, 64] on representative atoms."""
    lo, hi, nb = grid
    d = _pdist(np.asarray(true_xyz, np.float64))
    bins = bin_index(d, lo, hi, nb)
    m = np.asarray(coord_mask, bool)
    pm = (m[:, None] & m[None, :]).astype(np.float64)
    return cross_entropy_bins(logits, bins, pm, eps=eps)


def pde(logits, pred_xyz, true_xyz, coord_mask, *, grid=PDE_GRID, eps=1e-6):
    """PDELoss, loss.py:638-777. Label is |pred - true| distance error, bins clamped to 1..64."""
    lo, hi, nb = grid
    err = np.abs(_pdist(np.asarray(pred_xyz, np.float64))
                 - _pdist(np.asarray(true_xyz, np.float64)))
    bins = bin_index(err, lo, hi, nb, n_edges=nb + 1)
    m = np.asarray(coord_mask, bool)
    pm = (m[:, None] & m[None, :]).astype(np.float64)
    return cross_entropy_bins(logits, bins, pm, eps=eps)


def express_in_frame(coords, frames, *, eps=1e-8):
    """`expressCoordinatesInFrame`, modules/frames.py:21-55 (AF3 Algorithm 29)."""
    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)
    a, b, c = frames[..., 0, :], frames[..., 1, :], frames[..., 2, :]
    w1, w2 = unit(a - b), unit(c - b)
    e1, e2 = unit(w1 + w2), unit(w2 - w1)
    e3 = np.cross(e1, e2)
    d = coords[..., None, :, :] - b[..., None, :]
    return np.stack([(d * e1[..., None, :]).sum(-1),
                     (d * e2[..., None, :]).sum(-1),
                     (d * e3[..., None, :]).sum(-1)], axis=-1)


def pae(logits, pred_xyz, true_xyz, coord_mask, frame_atom_index, *, grid=PAE_GRID,
        eps=1e-6):
    """PAELoss, loss.py:813-1007. `logits` [N_frame, N_token, 64] already frame-selected.

    The label is the squared alignment error in bins whose boundaries are the SQUARE of
    `linspace(0, 32, 65)` (loss.py:833), and it is detached, so the gradient is logits-only.
    """
    lo, hi, nb = grid
    pf = np.asarray(pred_xyz, np.float64)[..., frame_atom_index, :]
    tf = np.asarray(true_xyz, np.float64)[..., frame_atom_index, :]
    sq = ((express_in_frame(np.asarray(pred_xyz, np.float64), pf)
           - express_in_frame(np.asarray(true_xyz, np.float64), tf)) ** 2).sum(-1)
    m = np.asarray(coord_mask, bool)
    fmask = (m[frame_atom_index].sum(-1) >= 3)
    pm = (fmask[:, None] & m[None, :])
    sq = sq * pm
    bins = bin_index(sq, lo, hi, nb, n_edges=nb + 1, square=True)
    bins = np.where(pm, bins, nb - 1)
    return cross_entropy_bins(logits, bins, pm.astype(np.float64), eps=eps)


def atom_bespoke_lddt(pred_xyz, true_xyz, is_nucleotide, is_polymer, rep_atom_mask,
                      *, radius=LDDT_RADIUS):
    """`calculate_atom_bespoke_lddt`, loss.py:1212-1281. Returns (per-atom lddt, weight)."""
    nuc_r, other_r = radius
    p = np.asarray(pred_xyz, np.float64)
    t = np.asarray(true_xyz, np.float64)
    m = (np.asarray(rep_atom_mask, bool) & np.asarray(is_polymer, bool))
    pd = np.linalg.norm(p[..., :, None, :] - p[..., None, m, :], axis=-1)
    td = np.linalg.norm(t[..., :, None, :] - t[..., None, m, :], axis=-1)
    dd = np.abs(pd - td)
    l = 0.25 * ((dd < 0.5).astype(np.float64) + (dd < 1.0) + (dd < 2.0) + (dd < 4.0))
    nuc = np.asarray(is_nucleotide, bool)[m]
    loc = ((td < nuc_r) * nuc[None, :] + (td < other_r) * (~nuc[None, :]))
    diag = (1.0 - np.eye(t.shape[-2]))[:, m].astype(bool)
    pm = (loc * diag).astype(np.float64)
    return (l * pm).sum(-1), pm.sum(-1)


def plddt(logits, per_atom_lddt, per_atom_weight, *, grid=PLDDT_GRID, eps=1e-6):
    """PLDDTLoss, loss.py:1283-1439. Label is the normalised per-atom lddt in 50 bins."""
    lo, hi, nb = grid
    v = np.asarray(per_atom_lddt, np.float64) / (np.asarray(per_atom_weight, np.float64) + eps)
    bins = bin_index(v, lo, hi, nb, n_edges=nb + 1)
    return cross_entropy_bins(logits, bins, np.ones(bins.shape[-1:]), reduce_axes=1)


def resolved(logits, coord_mask, atom_mask=None, *, eps=1e-6):
    """ExperimentallyResolvedLoss, loss.py:1010-1060. Two classes: unresolved, resolved."""
    bins = np.asarray(coord_mask, np.int64)
    m = np.ones(bins.shape) if atom_mask is None else np.asarray(atom_mask, np.float64)
    return cross_entropy_bins(logits, bins, m, eps=0.0 if atom_mask is None else eps,
                              reduce_axes=1)


def mse(pred_xyz, true_xyz, coord_mask, *, is_dna=None, is_rna=None, is_ligand=None,
        per_sample_scale=None, weight_mse=1 / 3, w_dna=5.0, w_rna=5.0, w_ligand=10.0,
        eps=1e-6):
    """MSELoss, loss.py:1063-1207: weighted coordinate MSE after a stop-gradient Kabsch.

    `pred_xyz` is [S, N, 3] (upstream's N_sample axis) or [N, 3]. Returns the loss and its
    gradient w.r.t. `pred_xyz`, which is the seed the diffusion module's tape needs.
    """
    p = np.asarray(pred_xyz, np.float64)
    single = p.ndim == 2
    if single:
        p = p[None]
    t = np.asarray(true_xyz, np.float64)
    cm = np.asarray(coord_mask, np.float64)
    zero = np.zeros(t.shape[-2])
    w = (1.0 + w_dna * (zero if is_dna is None else is_dna)
         + w_rna * (zero if is_rna is None else is_rna)
         + w_ligand * (zero if is_ligand is None else is_ligand)) * cm
    # Both coordinate sets are masked BEFORE the align, exactly as upstream does
    # (loss.py:1122-1125); an unresolved atom must not pull the rotation.
    pm = p * cm[None, :, None]
    tm = np.broadcast_to(t * cm[:, None], p.shape)
    aligned = weighted_rigid_align(tm, pm, np.broadcast_to(w, p.shape[:-1]))
    diff = p - aligned
    denom = float(cm.sum()) + eps
    per_sample = (w * (diff ** 2).sum(-1)).sum(-1) / denom
    scale = np.ones(p.shape[0]) if per_sample_scale is None else np.asarray(
        per_sample_scale, np.float64).reshape(p.shape[0])
    loss = float(weight_mse * (per_sample * scale).mean())
    grad = weight_mse * (scale[:, None, None] / p.shape[0]) * 2.0 * w[None, :, None] * diff / denom
    return loss, (grad[0] if single else grad)


def bond(pred_dist, true_dist, bond_mask, coord_mask=None, *, per_sample_scale=None,
         eps=1e-6):
    """BondLoss, loss.py:299-451: squared error on bonded distances only.

    Takes DISTANCES, like upstream's `forward(pred_distance, true_distance, ...)`, and
    returns the gradient w.r.t. the predicted distance. `dist_grad_to_coords` is the
    bridge onto coordinates, kept separate so each half can be checked against upstream
    on its own terms rather than as one composite.

    Note the denominator: upstream divides by `sum(bond_mask + eps)`, i.e. eps is added
    per element and then summed, not once to the total (loss.py:322-324). On an N x N mask
    that is `sum(bond_mask) + N^2 * eps`, a different number from `sum(bond_mask) + eps`,
    and the kind of detail a re-derivation from the paper loses.
    """
    pd = np.asarray(pred_dist, np.float64)
    single = pd.ndim == 2
    if single:
        pd = pd[None]
    bm = np.asarray(bond_mask, np.float64)
    if coord_mask is not None:
        cm = np.asarray(coord_mask, np.float64)
        bm = bm * (cm[:, None] * cm[None, :])
    err = pd - np.asarray(true_dist, np.float64)[None]
    denom = float((bm + eps).sum())
    per_sample = ((err ** 2) * bm[None]).sum((-1, -2)) / denom
    scale = np.ones(pd.shape[0]) if per_sample_scale is None else np.asarray(
        per_sample_scale, np.float64).reshape(pd.shape[0])
    loss = float((per_sample * scale).mean())
    grad = (scale[:, None, None] / pd.shape[0]) * 2.0 * err * bm[None] / denom
    return loss, (grad[0] if single else grad)


def smooth_lddt(pred_dist, true_dist, lddt_pair_mask, *, eps=1e-10):
    """SmoothLDDTLoss, loss.py:63-183 (AF3 Algorithm 27). No per-sample scale upstream.

    `1 - mean_lddt`, where lddt averages `0.25 * sum_t sigmoid(t - |dpred - dtrue|)` over
    the four thresholds 0.5, 1, 2 and 4 A on the masked pairs. Distances in, gradient
    w.r.t. the predicted distance out, as for `bond`.
    """
    pd = np.asarray(pred_dist, np.float64)
    single = pd.ndim == 2
    if single:
        pd = pd[None]
    c = np.asarray(lddt_pair_mask, np.float64)
    dd = pd - np.asarray(true_dist, np.float64)[None]
    a = np.abs(dd)
    s = np.zeros_like(a)
    ds = np.zeros_like(a)
    for th in (0.5, 1.0, 2.0, 4.0):
        sig = 1.0 / (1.0 + np.exp(-(th - a)))
        s += 0.25 * sig
        ds += 0.25 * sig * (1.0 - sig) * (-np.sign(dd))
    denom = c.sum() + eps
    per_sample = (c[None] * s).sum((-1, -2)) / denom
    loss = float(1.0 - per_sample.mean())
    grad = -(c[None] * ds) / (denom * pd.shape[0])
    return loss, (grad[0] if single else grad)
