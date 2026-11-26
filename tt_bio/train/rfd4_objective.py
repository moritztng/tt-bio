"""RFD4-Proteina's training objective, as upstream defines it, in float64 numpy on host.

Transcribed from NVIDIA/Baker-lab RFD4-Proteina at `2db07bb` and its release pretraining
recipe `configs/releases/flow/2026_10_19_hero_pretrain.yaml` (over `training_proteina.yaml`).
`tests/test_rfd4_objective.py` checks every gradient here against central differences, and
`perf/r4t/objective_parity.py` runs upstream's own `ProductSpaceFlowMatcher` on the same
batches, so agreement with upstream is measured, not argued.

What one training step optimises (`proteina.py:593-700`):

    corrupt   x_t = (1 - t) x_0 + t x_1 per modality (bb_ca [n,3] in nm, local_latents [n,32]),
              x_0 ~ N(0, I) on emitted tokens; bb_ca noise is centred, then shifted by one
              N(0, 0.2^2 I) offset per molecule (stochastic centering, `x_0`, `per_group`).
              t ~ mix_unif_beta, drawn independently per modality, shared across the k
              noise candidates of one example.
    network   predicts the velocity v for both modalities, sequence logits (on a detached
              trunk state) and insertion logits (also detached).
    terms     bb_ca, local_latents: x_1-space MSE on supervised tokens, per-token weight,
              divided by the supervised-token count, times 1 / ((1 - t)^2 + 1e-5)
              sequence_prediction: token CE over 147 classes on unconditioned latent tokens
              insertion_prediction: CE of each unindexed token over the indexed tokens
              signed_volume: squared hinge on chiral-centre handedness, weight 2.5
    select    exploration_k = 4: the candidate with the lowest bb_ca + local_latents loss
              wins; only its terms enter the loss (selection is not differentiated).
    total     sum over the five terms of the mean over examples.

The network output is ``v``; the loss lives in ``x_1`` space through
``x_1_hat = (x_t + (1 - t) v) * emit``. Every seed returned here is d total / d output,
so ``ag.backward(roots, seeds)`` gets the exact gradient of upstream's loss.
"""

from __future__ import annotations

import numpy as np

from .objectives import Objective, register

__all__ = [
    "MODES", "TERMS", "WEIGHTS", "T_DISTRIBUTION", "EXPLORATION_K", "STOCHASTIC_CENTERING_SCALE",
    "SEQ_MASK_TOKEN", "RECYCLE", "masks", "sample_t", "recycle_depth", "corrupt", "fm_loss",
    "sequence_loss", "insertion_loss", "signed_volume_loss", "select_winners", "terms",
    "rfd4_loss",
]

MODES = ("bb_ca", "local_latents")
#: The optimised terms, in `ProductSpaceFlowMatcher.optimization_terms` order.
TERMS = ("bb_ca", "local_latents", "insertion_prediction", "sequence_prediction", "signed_volume")
#: `loss.signed_volume_weight` 2.5 and `loss.sequence_prediction.weight` (default 1.0),
#: `training_proteina.yaml` loss block; the other three are unweighted upstream.
WEIGHTS = {"bb_ca": 1.0, "local_latents": 1.0, "insertion_prediction": 1.0,
           "sequence_prediction": 1.0, "signed_volume": 2.5}
#: `loss.t_distribution`, mix_unif_beta: with prob p3 U(0,1), else Beta(p1, p2).
#: `shared_groups` is null, so the two modalities draw independent times.
T_DISTRIBUTION = {"bb_ca": (1.9, 1.0, 0.02), "local_latents": (1.0, 1.5, 0.2)}
EXPLORATION_K = 4
EXPLORATION_LOSSES = ("bb_ca", "local_latents")
STOCHASTIC_CENTERING_SCALE = 0.2      # product_flowmatcher.bb_ca, nm
SEQ_MASK_TOKEN = 0                    # UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]
#: training.n_recycle_*: Poisson(3) clamped to [1, 9], drawn from global_step so every rank
#: agrees; gradient through the last pass only (`nn.recycle.grad: last`).
RECYCLE = {"lam": 3.0, "max": 9, "seed_base": 0x5EED0000, "grad_depth": 1}
_T_EPS = 1e-5


def masks(batch) -> dict:
    """Upstream's emit and supervision masks (`model_masks.py:27-45`), as float64."""
    pad = np.asarray(batch["padding_mask"], bool)
    emit_bb = pad & ~np.asarray(batch["is_in_context_conditioning"], bool)
    emit = {"bb_ca": emit_bb,
            "local_latents": emit_bb & np.asarray(batch["supports_local_latent"], bool)}
    res = np.asarray(batch["is_resolved_mask"], bool)
    sup = {m: e & res for m, e in emit.items()}
    return {"emit": emit, "sup": sup}


def sample_t(rng: np.random.Generator, shape, mode: str) -> np.ndarray:
    """`_sample_t` mix_unif_beta for one modality (`product_space_flow_matcher.py:1287`)."""
    p1, p2, p3 = T_DISTRIBUTION[mode]
    beta = rng.beta(p1, p2, size=shape)
    unif = rng.random(size=shape)
    u = rng.random(size=shape)
    return np.where(u < p3, unif, beta)


def recycle_depth(global_step: int) -> int:
    """`Proteina.handle_recycling` with n_recycle_sync: bit-identical to upstream's draw."""
    rng = np.random.default_rng(RECYCLE["seed_base"] + int(global_step))
    return int(np.clip(rng.poisson(RECYCLE["lam"]), 1, RECYCLE["max"]))


def _group_ids(batch, shape) -> np.ndarray:
    g = batch.get("molecule_iid")
    if g is None:
        raise KeyError("stochastic centering is per molecule (`molecule_iid`), and the batch has "
                       "none; a single shared offset would be a different corruption")
    return np.asarray(g, np.int64).reshape(shape)


def corrupt(batch, x1: dict, t: dict, eps: dict, offsets: dict) -> dict:
    """The forward process `corrupt_batch`, with every random draw passed in.

    ``eps[m]`` is the standard-normal draw [b, n, d] upstream's `sample_noise` makes.
    ``offsets`` maps each molecule id (in ascending order, upstream's `unique()` order) to its
    standard-normal [b, 3] draw for the bb_ca stochastic offset. Returns x_0, x_1, x_t, all
    zeroed outside each modality's emit mask, as upstream stores them.
    """
    mk = masks(batch)["emit"]
    out = {"x_0": {}, "x_1": {}, "x_t": {}}
    for m in MODES:
        e = mk[m][..., None].astype(np.float64)
        x0 = np.asarray(eps[m], np.float64) * e
        if m == "bb_ca":                                       # zero_com_noise
            cnt = np.maximum(e.sum(-2, keepdims=True), 1e-12)  # mean_w_mask
            x0 = (x0 - (x0 * e).sum(-2, keepdims=True) / cnt) * e
        x1m = np.asarray(x1[m], np.float64) * e
        x0_off = x0
        if m == "bb_ca":
            ccm = batch.get("coord_cond_mask")
            off = mk[m] if ccm is None else mk[m] & ~np.asarray(ccm, bool)
            gid = _group_ids(batch, off.shape)
            for g in np.unique(gid[off]):
                gm = (off & (gid == g))[..., None].astype(np.float64)
                x0_off = x0_off + STOCHASTIC_CENTERING_SCALE * np.asarray(
                    offsets[int(g)], np.float64)[:, None, :] * gm
            x0_off = x0_off * e
        tm = np.asarray(t[m], np.float64)[:, None, None]
        out["x_0"][m] = x0
        out["x_1"][m] = x1m
        out["x_t"][m] = ((1.0 - tm) * x0_off + tm * x1m) * e
    return out


def fm_loss(x1, xt, v, t, emit, sup, weight=None):
    """`RDNFlowMatcher.compute_fm_loss` for a ``v``-parameterised head: per-example loss, dL/dv.

    x1, xt, v [b, n, d]; t [b]; emit, sup [b, n] bool; weight [b, n] or None.
    """
    tt = np.asarray(t, np.float64)[:, None, None]
    e = emit[..., None].astype(np.float64)
    s = sup[..., None].astype(np.float64)
    x1_hat = (xt + (1.0 - tt) * v) * e
    err = (x1 - x1_hat) * s
    w = s if weight is None else (np.asarray(weight, np.float64)[..., None] * s)
    nres = np.maximum(sup.sum(-1), 1).astype(np.float64)
    tw = 1.0 / ((1.0 - np.asarray(t, np.float64)) ** 2 + _T_EPS)
    loss = (err ** 2 * w).sum((-1, -2)) / nres * tw
    # d/dv: err = (x1 - x1_hat) s, x1_hat = (xt + (1-t) v) e
    dv = -2.0 * err * w * s * e * (1.0 - tt) * (tw / nres)[:, None, None]
    return loss, dv


def _log_softmax(z):
    m = z.max(-1, keepdims=True)
    ez = np.exp(z - m)
    se = ez.sum(-1, keepdims=True)
    return z - m - np.log(se), ez / se


def sequence_loss(logits, batch, t_ll, sup_ll, *, t_min=0.0, t_max=1.0):
    """`_compute_sequence_prediction_loss` (unweighted CE); per-example loss, dL/dlogits."""
    z = np.asarray(logits, np.float64)
    b, n, c = z.shape
    target = np.asarray(batch["residue_type"], np.int64)
    seq = batch.get("condition_sequence_1_residue")
    conditioned = np.zeros((b, n), bool) if seq is None else (np.asarray(seq) != SEQ_MASK_TOKEN)
    lm = sup_ll & ~conditioned
    safe = np.where(lm, target, 0)
    logp, p = _log_softmax(z)
    ce_tok = -np.take_along_axis(logp, safe[..., None], -1)[..., 0]
    nv = lm.sum(-1)
    snv = np.where(nv > 0, nv, 1).astype(np.float64)
    t_ll = np.asarray(t_ll, np.float64)
    active = ((t_ll >= t_min) & (t_ll <= t_max) & (nv > 0)).astype(np.float64)
    loss = (ce_tok * lm).sum(-1) / snv * active
    onehot = np.zeros_like(z)
    np.put_along_axis(onehot, safe[..., None], 1.0, -1)
    dz = (p - onehot) * (lm * (active / snv)[:, None])[..., None]
    return loss, dz


def insertion_loss(logits, batch):
    """`_compute_insertion_prediction_loss`; per-example loss, dL/dlogits [b, n, n]."""
    z = np.asarray(logits, np.float64)
    b, n, _ = z.shape
    target = np.asarray(batch["insertion_source_idx"], np.int64)
    pad = np.asarray(batch["padding_mask"], bool)
    key_ok = pad & (target < 0)                                  # indexed tokens: candidates
    zm = np.where(key_ok[:, None, :], z, -1e9)
    row = target >= 0                                            # unindexed tokens: queries
    safe = np.where(row, target, 0)
    logp, p = _log_softmax(zm)
    ce_tok = np.where(row, -np.take_along_axis(logp, safe[..., None], -1)[..., 0], 0.0)
    nv = row.sum(-1).astype(np.float64)
    snv = np.where(nv > 0, nv, 1.0)
    has = row.any(-1).astype(np.float64)
    loss = ce_tok.sum(-1) / snv * has
    onehot = np.zeros_like(z)
    np.put_along_axis(onehot, safe[..., None], 1.0, -1)
    dz = (p - onehot) * (row * (has / snv)[:, None])[..., None] * key_ok[:, None, :]
    return loss, dz


def signed_volume_loss(pred, ref, tet, supervised):
    """`signed_volume_loss.signed_volume_loss`; per-example loss, dL/dpred [b, n, 3].

    Upstream evaluates this in float32 (`.float()`); this is the float64 value of the same
    expression, so the two agree to float32 roundoff.
    """
    pred = np.asarray(pred, np.float64)
    ref = np.asarray(ref, np.float64)
    tet = np.asarray(tet, np.int64)
    sup = np.asarray(supervised, bool)
    b = pred.shape[0]
    bi = np.arange(b)[:, None, None]
    has = tet[..., 0] >= 0
    idx = np.maximum(tet, 0)
    p, r = pred[bi, idx], ref[bi, idx]                           # [b, n, 4, 3]
    elig = has & sup & sup[bi, idx].all(-1) & np.isfinite(r).all((-1, -2))
    p = np.where(elig[..., None, None], p, 0.0)
    r = np.where(elig[..., None, None], r, 0.0)
    pv, rv = p[..., :3, :] - p[..., 3:4, :], r[..., :3, :] - r[..., 3:4, :]
    norm = np.maximum(np.linalg.norm(rv, axis=-1), 0.01).prod(-1)
    ref_vol = (rv[..., 0, :] * np.cross(rv[..., 1, :], rv[..., 2, :])).sum(-1)
    c12, c20, c01 = (np.cross(pv[..., 1, :], pv[..., 2, :]), np.cross(pv[..., 2, :], pv[..., 0, :]),
                     np.cross(pv[..., 0, :], pv[..., 1, :]))
    pred_vol = (pv[..., 0, :] * c12).sum(-1)
    nref = ref_vol / norm
    elig = elig & (np.abs(nref) >= 0.1)
    sgn = np.sign(ref_vol)
    score = sgn * pred_vol / norm
    margin = np.minimum(np.abs(nref) * 0.5, 0.2)
    h = np.maximum(margin - score, 0.0)
    cnt = np.maximum(elig.sum(-1), 1).astype(np.float64)
    loss = np.where(elig, h ** 2, 0.0).sum(-1) / cnt
    # d loss / d pred_vol per centre, then through the triple product to the four vertices
    g = np.where(elig, -2.0 * h * sgn / norm, 0.0) / cnt[:, None]       # [b, n]
    gv = np.stack([c12, c20, c01], -2) * g[..., None, None]             # d/d pv_i, [b,n,3,3]
    gp = np.concatenate([gv, -gv.sum(-2, keepdims=True)], -2)            # vertices 0..3
    dpred = np.zeros_like(pred)
    np.add.at(dpred, (np.broadcast_to(bi, idx.shape), idx), gp)
    return loss, dpred


def terms(batch, outputs, *, sequence_weight=None, signed_volume_weight=None):
    """Every upstream loss term per physical example, and d term / d output.

    ``batch`` carries x_1, x_t (dicts per modality), t (dict) and upstream's masks under
    upstream's names; ``outputs`` carries ``v_bb_ca``, ``v_local_latents`` and, when the
    network has those heads, ``sequence_logits`` and ``insertion_logits``. Returns
    ``{term: (per_example [b], {output_key: grad})}``; a term whose inputs are absent is
    left out, never zero-filled, exactly as upstream leaves it out of its loss dict.
    """
    sw = WEIGHTS["sequence_prediction"] if sequence_weight is None else sequence_weight
    vw = WEIGHTS["signed_volume"] if signed_volume_weight is None else signed_volume_weight
    mk = masks(batch)
    lw = batch.get("loss_weight")
    out = {}
    for m in MODES:
        loss, dv = fm_loss(batch["x_1"][m], batch["x_t"][m], np.asarray(outputs[f"v_{m}"], np.float64),
                           batch["t"][m], mk["emit"][m], mk["sup"][m], lw)
        out[m] = (loss, {f"v_{m}": dv})
    if "insertion_logits" in outputs and "insertion_source_idx" in batch:
        loss, dz = insertion_loss(outputs["insertion_logits"], batch)
        out["insertion_prediction"] = (loss, {"insertion_logits": dz})
    if "sequence_logits" in outputs:
        loss, dz = sequence_loss(outputs["sequence_logits"], batch, batch["t"]["local_latents"],
                                 mk["sup"]["local_latents"])
        out["sequence_prediction"] = (sw * loss, {"sequence_logits": sw * dz})
    if vw > 0:
        sup = mk["sup"]["bb_ca"]
        if lw is not None:
            sup = sup & (np.asarray(lw) > 0)
        t = np.asarray(batch["t"]["bb_ca"], np.float64)[:, None, None]
        e = mk["emit"]["bb_ca"][..., None].astype(np.float64)
        v = np.asarray(outputs["v_bb_ca"], np.float64)
        pred = (batch["x_t"]["bb_ca"] + (1.0 - t) * v) * e
        loss, dp = signed_volume_loss(pred, batch["x_1"]["bb_ca"], batch["tetrahedral_vertices"], sup)
        out["signed_volume"] = (vw * loss, {"v_bb_ca": vw * dp * (1.0 - t) * e})
    return out


def select_winners(per_term: dict, k: int, losses=EXPLORATION_LOSSES) -> np.ndarray:
    """`minimum_loss_attempt_indices`: physical index of each example's lowest-loss candidate.

    Ties go to the first candidate, as torch's argmin does.
    """
    sel = sum(np.asarray(per_term[name], np.float64) for name in losses)
    if sel.shape[0] % k:
        raise ValueError(f"physical batch {sel.shape[0]} is not divisible by exploration_k={k}")
    nb = sel.shape[0] // k
    return sel.reshape(nb, k).argmin(1) + np.arange(nb) * k


def rfd4_loss(batch, outputs, weights) -> tuple:
    """The registered objective: ``(total, breakdown, seeds)``.

    With ``batch["exploration_k"] = k > 1`` the physical batch holds k adjacent candidates per
    example; the winner is chosen on bb_ca + local_latents and the losers receive an exactly
    zero seed (they are in upstream's graph too, and their gradient is zero there as well).
    ``weights`` rescales upstream's own weights per term: pass ``WEIGHTS`` for upstream.
    """
    per = terms(batch, outputs, sequence_weight=1.0, signed_volume_weight=1.0)
    k = int(batch.get("exploration_k", 1))
    b = next(iter(per.values()))[0].shape[0]
    win = select_winners({m: per[m][0] for m in EXPLORATION_LOSSES}, k) if k > 1 else np.arange(b)
    keep = np.zeros(b)
    keep[win] = 1.0 / len(win)                    # mean over the logical batch
    total, breakdown, seeds = 0.0, {}, {}
    for name in TERMS:
        w = float(weights.get(name, 0.0))
        if name not in per:
            breakdown[name] = {"value": None, "weight": w, "contribution": 0.0,
                               "skipped": "inputs absent (upstream omits the term too)"}
            continue
        loss, grads = per[name]
        value = float(loss[win].mean())
        total += w * value
        breakdown[name] = {"value": value, "weight": w, "contribution": w * value}
        for key, g in grads.items():
            sg = w * g * keep.reshape((-1,) + (1,) * (g.ndim - 1))
            seeds[key] = sg if key not in seeds else seeds[key] + sg
    breakdown["winners"] = win.tolist()
    return total, breakdown, seeds


register(Objective(
    name="rfd4", fn=rfd4_loss, weights=dict(WEIGHTS),
    # No rollout: flow matching trains on one corrupted time per example. The step's cost is
    # exploration_k forwards at the drawn recycle depth plus one backward through the
    # winner's last pass; that is the recipe, not a knob.
    rollout=1, terms=TERMS,
    doc="RFD4-Proteina flow matching: bb_ca and local_latents x_1-space MSE with the "
        "1/((1-t)^2+1e-5) weight, sequence and insertion CE, signed-volume hinge x2.5, "
        "lowest-loss of 4 noise candidates per example"))
