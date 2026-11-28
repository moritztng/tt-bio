"""Training-side on-card heads loss and backward for RFD4 (D89).

Lives under ``tt_bio/train/`` so inference module ``tt_bio/rfd4/heads.py`` never
imports ``autograd``.

``device_insertion_loss`` evaluates the ``[b, n, n]`` insertion cross-entropy and
its gradient ``dL/dlogits`` via exact float64 :func:`rfd4_objective.insertion_loss`
only on batches with active query rows (``target >= 0``; zero ops and zero PCIe
bytes when no insertions are active, ~262 KB each way at ``n=256`` when active),
avoiding Tenstorrent SFPU ``exp``/``log`` cancellation that produces negative
cross-entropy and >6% gradient error when the loss is small (~4e-3 or below).

``device_candidate_scores`` runs the grad-off K-candidate exploration scoring
through ``FlowGlue.heads`` on card (only reading back tiny ``[K, n, 3]`` +
``[K, n, 32]`` = ~0.14 MB instead of ``[K, n, 768]`` = 3.15 MB).

``device_winner_loss`` runs taped ``FlowGlue.heads`` on card, computes the loss
terms in exact float64 numpy (only transferring insertion logits on batches with
active insertions), uploads the seeds, and runs a single ``ag.backward`` on card.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import ttnn

from .. import autograd as ag
from . import rfd4_objective as RO

__all__ = [
    "device_insertion_loss",
    "device_candidate_scores",
    "device_winner_loss",
    "extract_cand_score_meta",
    "device_candidate_scores_spmd",
    "device_winner_loss_spmd",
]


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _npy_dict(d):
    if isinstance(d, dict):
        return {k: _npy_dict(v) for k, v in d.items()}
    return d.detach().cpu().numpy() if torch.is_tensor(d) else d


def device_insertion_loss(
    logits_tt,
    batch: dict,
    *,
    win: np.ndarray | None = None,
    weight: float = 1.0,
) -> tuple[np.ndarray, object | None]:
    """Exact float64 insertion cross-entropy loss and ``[b, n_pad, n_pad]`` gradient.

    Delegates to :func:`tt_bio.train.rfd4_objective.insertion_loss` (and applies
    the ``keep[win] * weight`` scaling from :func:`rfd4_loss`) so small and
    near-zero cross-entropy losses never suffer SFPU ``exp``/``log`` bias or
    ``p - onehot`` cancellation. Works on both single-chip devices and multi-chip
    SPMD ``MeshDevice``s.

    Returns ``(per_example_loss_np [b], dL_dlogits_tt [b_local, n_pad, n_pad] or None)``.
    When no query rows are active (``target < 0`` everywhere), returns
    ``(zeros(b), None)`` with zero device ops and zero PCIe bytes.
    """
    from .mesh_dp import mesh_world, put_sharded, read_sharded

    z = getattr(logits_tt, "value", logits_tt)
    b_local, n_pad, _ = (int(x) for x in z.shape)
    dev = z.device()
    world = mesh_world(dev)

    target_raw = _as_np(batch["insertion_source_idx"]).astype(np.int64)
    pad_raw = _as_np(batch["padding_mask"]).astype(bool)
    b, n_tok = (int(x) for x in target_raw.shape)
    if world > 1 and b != world * b_local:
        raise ValueError(
            f"device_insertion_loss: batch leading dim {b} != world ({world}) * per-chip ({b_local})"
        )
    if world == 1 and b != b_local:
        raise ValueError(
            f"device_insertion_loss: batch leading dim {b} != logits leading dim ({b_local})"
        )

    row = target_raw >= 0
    if not row.any():
        return np.zeros(b, dtype=np.float64), None

    if win is None:
        keep = np.ones(b, dtype=np.float64)
    else:
        keep = np.zeros(b, dtype=np.float64)
        keep[np.asarray(win, np.int64)] = 1.0 / len(win)

    with ag.no_grad():
        z_t = read_sharded(z, dev) if world > 1 else ttnn.to_torch(z)
        z_cpu = z_t.detach().cpu()
        loss_np = np.zeros(b, dtype=np.float64)
        dz_buf = np.zeros((b, n_pad, n_pad), dtype=np.float32)
        key_ok = pad_raw & (target_raw < 0)
        scale_b = keep * float(weight)
        for bi in np.flatnonzero(row.any(-1)):
            q = np.flatnonzero(row[bi])
            nv = float(len(q))
            kok = key_ok[bi]
            zb = z_cpu[bi, q, :n_tok].float().numpy().astype(np.float64)
            zm = np.where(kok[None, :], zb, -1e9)
            logp, p = RO._log_softmax(zm)
            tgt_q = target_raw[bi, q]
            idx_q = np.arange(len(q))
            loss_np[bi] = float((-logp[idx_q, tgt_q]).sum()) / nv
            if scale_b[bi] != 0.0:
                p[idx_q, tgt_q] -= 1.0
                dz_q = (p * (scale_b[bi] / nv)) * kok[None, :]
                dz_buf[bi, q, :n_tok] = dz_q.astype(np.float32)
        dz_t = torch.from_numpy(np.ascontiguousarray(dz_buf))
        if world > 1:
            dz = put_sharded(dz_t, dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        else:
            dz = ttnn.from_torch(
                dz_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
            )

    return loss_np, dz


def device_candidate_scores(glue, sk_tt, batch: dict, K: int, weights=None) -> int:
    """Select the exploration winner on card via ``FlowGlue.heads`` (grad-off).

    Only ``bb_ca`` ([K, n, 3]) and ``local_latents`` ([K, n, 32]) are computed and
    downloaded (~0.14 MB total), avoiding the [K, n, 768] (3.15 MB) ``s`` readback
    and host CPU head matmuls.
    """
    w = RO.WEIGHTS if weights is None else weights
    n_tok = int(batch["padding_mask"].shape[1])
    # Exploration winner uses only EXPLORATION_LOSSES = ("bb_ca", "local_latents").
    dh = getattr(glue, "_device_heads", None)
    if dh is None:
        raise RuntimeError("device_candidate_scores requires glue._device_heads")
    raw_out = dh(sk_tt, batch, insertion=False, sequence=False)
    ca_tt = next(iter(raw_out["bb_ca"].values()))
    ll_tt = next(iter(raw_out["local_latents"].values()))
    ca_np = ttnn.to_torch(getattr(ca_tt, "value", ca_tt)).float().numpy()[:, :n_tok, :3].astype(np.float64)
    ll_np = ttnn.to_torch(getattr(ll_tt, "value", ll_tt)).float().numpy()[:, :n_tok, :32].astype(np.float64)
    ttnn.deallocate(getattr(ca_tt, "value", ca_tt))
    ttnn.deallocate(getattr(ll_tt, "value", ll_tt))
    ok = {"v_bb_ca": ca_np, "v_local_latents": ll_np}
    _t, bd, _s = RO.rfd4_loss({**_npy_dict(batch), "exploration_k": K}, ok, w)
    return int(bd["winners"][0])


def device_winner_loss(
    glue,
    s_taped,
    bw: dict,
    *,
    bridge=None,
    weights=None,
    ref_terms: dict | None = None,
    win_idx: int = 0,
):
    """Run taped ``FlowGlue.heads`` on card, evaluate loss, and backprop through heads + trunk.

    Keeps the pair-sized ``[1, n, n]`` insertion logits and gradient strictly on card
    via :func:`device_insertion_loss`. Only token-sized outputs/seeds (~0.19 MB each
    way) touch the host for exact float64 ``rfd4_loss`` scoring.
    """
    w = RO.WEIGHTS if weights is None else weights
    n_tok = int(bw["padding_mask"].shape[1])
    s_raw = getattr(s_taped, "value", s_taped)
    n_pad = int(s_raw.shape[1])
    dev = s_raw.device()

    # Production dispatch through FlowGlue.heads in place (taped automatically by taped_ttnn).
    # Re-entrant `ag.tape()` ensures heads attach onto s_taped even when the caller closed its outer tape.
    with ag.tape():
        o = glue.heads(s_taped, bw, insertion=True)
    ca_t = next(iter(o["bb_ca"].values()))
    ll_t = next(iter(o["local_latents"].values()))
    seq_t = o["sequence"]["logits"] if "sequence" in o else None
    ins_t = o["insertion"]["logits"]

    def read_tok(t, d_out: int) -> np.ndarray:
        arr = ttnn.to_torch(getattr(t, "value", t)).float().numpy()
        return arr[:, :n_tok, :d_out].astype(np.float64)

    ok = {
        "v_bb_ca": read_tok(ca_t, 3),
        "v_local_latents": read_tok(ll_t, 32),
    }
    if seq_t is not None:
        ok["sequence_logits"] = read_tok(seq_t, 147)

    # Evaluate non-insertion terms in exact float64 on host.
    bw_np = {**_npy_dict(bw), "exploration_k": 1}
    total_non_ins, bd, seeds = RO.rfd4_loss(bw_np, ok, w)

    # Evaluate insertion loss and [1, n_pad, n_pad] gradient strictly on card.
    ins_w = float(w.get("insertion_prediction", 1.0))
    ins_loss_np, dz_ins = device_insertion_loss(ins_t, bw, weight=ins_w)
    ins_val = float(ins_loss_np[0])
    total = float(total_non_ins) + ins_w * ins_val
    bd["insertion_prediction"] = {
        "value": ins_val,
        "weight": ins_w,
        "weighted": ins_w * ins_val,
        "per_example": ins_loss_np,
    }

    def up_seed(arr: np.ndarray, d_out: int):
        buf = np.zeros((1, n_pad, d_out), dtype=np.float32)
        buf[:, :n_tok, :d_out] = np.asarray(arr, dtype=np.float32)
        return ttnn.from_torch(
            torch.from_numpy(buf), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev
        )

    outs_bwd = [ca_t, ll_t]
    seeds_bwd = [up_seed(seeds["v_bb_ca"], 3), up_seed(seeds["v_local_latents"], 32)]
    if seq_t is not None and "sequence_logits" in seeds:
        outs_bwd.append(seq_t)
        seeds_bwd.append(up_seed(seeds["sequence_logits"], 147))
    if dz_ins is not None:
        outs_bwd.append(ins_t)
        seeds_bwd.append(dz_ins)

    ag.backward(outs_bwd, seeds_bwd)
    for sd in seeds_bwd:
        ttnn.deallocate(sd)
    del o, ca_t, ll_t, seq_t, ins_t, outs_bwd

    # Propagate tiny non-pair host-glue leaf gradients (s_init, conds, pair_taped leaves) if bridged.
    if bridge is not None:
        host_leaves, host_grads = [], []
        for h_t, d_t in bridge.ups:
            if d_t.grad is None:
                continue
            g_h = ttnn.to_torch(d_t.grad).float().reshape(h_t.shape).to(h_t.dtype)
            ttnn.deallocate(d_t.grad)
            d_t.grad = None
            host_leaves.append(h_t)
            host_grads.append(g_h)
        if host_leaves:
            torch.autograd.backward(host_leaves, host_grads)
        bridge.ups.clear()
        bridge.downs.clear()

    terms = {k: v.get("value") for k, v in bd.items() if isinstance(v, dict)}
    ref = {}
    if ref_terms is not None:
        ref = {k: float(ref_terms[k][win_idx]) for k in terms
               if k in ref_terms and win_idx < len(ref_terms[k])}
    return total, terms, ref


_SPMD_LOSS_KEYS = (
    "padding_mask",
    "is_in_context_conditioning",
    "supports_local_latent",
    "is_resolved_mask",
    "loss_weight",
    "x_1",
    "x_t",
    "t",
    "residue_type",
    "condition_sequence_1_residue",
    "tetrahedral_vertices",
    "insertion_source_idx",
)


def _fm_loss_values_only(x1, xt, v, t, emit, sup, weight=None) -> np.ndarray:
    """Bit-identical float64 per-example ``RO.fm_loss`` values without allocating gradients."""
    tt = np.asarray(t, np.float64)[:, None, None]
    e = emit[..., None].astype(np.float64)
    s = sup[..., None].astype(np.float64)
    x1_hat = (xt + (1.0 - tt) * v) * e
    err = (x1 - x1_hat) * s
    w = s if weight is None else (np.asarray(weight, np.float64)[..., None] * s)
    nres = np.maximum(sup.sum(-1), 1).astype(np.float64)
    tw = 1.0 / ((1.0 - np.asarray(t, np.float64)) ** 2 + RO._T_EPS)
    return (err ** 2 * w).sum((-1, -2)) / nres * tw


def extract_cand_score_meta(rank_batches: list[dict]) -> dict[str, Any]:
    """Extract only the tiny token-level arrays needed for vectorized float64 candidate scoring.

    Pure input-batch transformation (zero trainable weights), safe to run ahead-of-time in
    ``SPMDPrefetcher``.
    """
    pad = np.concatenate([_as_np(rb["padding_mask"]).astype(bool) for rb in rank_batches], axis=0)
    in_ctx = np.concatenate([_as_np(rb["is_in_context_conditioning"]).astype(bool) for rb in rank_batches], axis=0)
    sup_ll_flag = np.concatenate([_as_np(rb["supports_local_latent"]).astype(bool) for rb in rank_batches], axis=0)
    res = np.concatenate([_as_np(rb["is_resolved_mask"]).astype(bool) for rb in rank_batches], axis=0)
    emit_ca = pad & ~in_ctx
    emit_ll = emit_ca & sup_ll_flag
    sup_ca = emit_ca & res
    sup_ll = emit_ll & res
    has_lw = all(rb.get("loss_weight") is not None for rb in rank_batches)
    lw = (
        np.concatenate([_as_np(rb["loss_weight"]).astype(np.float64) for rb in rank_batches], axis=0)
        if has_lw else None
    )
    return {
        "emit_ca": emit_ca,
        "emit_ll": emit_ll,
        "sup_ca": sup_ca,
        "sup_ll": sup_ll,
        "lw": lw,
        "x1_ca": np.concatenate([_as_np(rb["x_1"]["bb_ca"]).astype(np.float64) for rb in rank_batches], axis=0),
        "xt_ca": np.concatenate([_as_np(rb["x_t"]["bb_ca"]).astype(np.float64) for rb in rank_batches], axis=0),
        "t_ca": np.concatenate([_as_np(rb["t"]["bb_ca"]).astype(np.float64) for rb in rank_batches], axis=0),
        "x1_ll": np.concatenate([_as_np(rb["x_1"]["local_latents"]).astype(np.float64) for rb in rank_batches], axis=0),
        "xt_ll": np.concatenate([_as_np(rb["x_t"]["local_latents"]).astype(np.float64) for rb in rank_batches], axis=0),
        "t_ll": np.concatenate([_as_np(rb["t"]["local_latents"]).astype(np.float64) for rb in rank_batches], axis=0),
    }


def _small_npy_dict(batch: dict) -> dict:
    """Convert only the tiny token keys read by ``RO.terms`` to numpy (skips pair tensors)."""
    out = {}
    for k in _SPMD_LOSS_KEYS:
        if k in batch and batch[k] is not None:
            out[k] = _npy_dict(batch[k])
    return out


def device_candidate_scores_spmd(
    glue,
    sk_tt,
    rank_batches: list[dict],
    K: int,
    weights=None,
    pool=None,
    cand_meta: dict | None = None,
) -> list[int]:
    """Select one exploration winner per chip on a ``MeshDevice`` via ``FlowGlue.heads``.

    Evaluates exploration losses (``bb_ca`` + ``local_latents``) in a single vectorized
    float64 pass over all ``world * K`` candidates without touching pair-sized batch keys
    or computing gradients (bit-identical to per-rank ``RO.rfd4_loss``).
    """
    from .mesh_dp import put_sharded, read_sharded

    world = len(rank_batches)
    n_tok = int(rank_batches[0]["padding_mask"].shape[1])
    dh = getattr(glue, "_device_heads", None)
    if dh is None:
        raise RuntimeError("device_candidate_scores_spmd requires glue._device_heads")
    s_raw = getattr(sk_tt, "value", sk_tt)
    n_pad = int(s_raw.shape[1])
    dev = s_raw.device()

    meta = extract_cand_score_meta(rank_batches) if cand_meta is None else cand_meta

    def _up_emit_bool(mask_bool: np.ndarray):
        arr = mask_bool.astype(np.float32)
        if arr.shape[1] < n_pad:
            pad = np.zeros((arr.shape[0], n_pad), dtype=np.float32)
            pad[:, :arr.shape[1]] = arr
            arr = pad
        t_rm = put_sharded(
            torch.from_numpy(np.ascontiguousarray(arr[..., None])), dev,
            dtype=dh.dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        t_tile = ttnn.to_layout(t_rm, ttnn.TILE_LAYOUT)
        if t_tile is not t_rm:
            ttnn.deallocate(t_rm)
        return t_tile

    emit_ca = _up_emit_bool(meta["emit_ca"])
    emit_ll = _up_emit_bool(meta["emit_ll"])
    raw_out = dh(sk_tt, None, insertion=False, sequence=False, masks_dev=(emit_ca, emit_ll, None))
    ttnn.deallocate(emit_ca)
    ttnn.deallocate(emit_ll)
    ca_tt = next(iter(raw_out["bb_ca"].values()))
    ll_tt = next(iter(raw_out["local_latents"].values()))
    ca_raw = getattr(ca_tt, "value", ca_tt)
    ll_raw = getattr(ll_tt, "value", ll_tt)
    ca_all = read_sharded(ca_raw, dev).float().numpy()[:, :n_tok, :3].astype(np.float64)
    ll_all = read_sharded(ll_raw, dev).float().numpy()[:, :n_tok, :32].astype(np.float64)
    ttnn.deallocate(ca_raw)
    ttnn.deallocate(ll_raw)

    loss_ca = _fm_loss_values_only(
        meta["x1_ca"], meta["xt_ca"], ca_all, meta["t_ca"],
        meta["emit_ca"], meta["sup_ca"], meta["lw"],
    )
    loss_ll = _fm_loss_values_only(
        meta["x1_ll"], meta["xt_ll"], ll_all, meta["t_ll"],
        meta["emit_ll"], meta["sup_ll"], meta["lw"],
    )
    sel = loss_ca + loss_ll
    return sel.reshape(world, K).argmin(axis=1).tolist()


def device_winner_loss_spmd(
    glue,
    s_taped,
    rank_bws: list[dict],
    *,
    bridge=None,
    weights=None,
    ref_terms_list: list[dict] | None = None,
    win_indices: list[int] | None = None,
    stacked_bw: dict | None = None,
    stop_at_s: bool = False,
    pool=None,
    head_tracer=None,
    dfolds=None,
    dev_masters=None,
    head_grads_out: dict | None = None,
    head_skip_out: set | None = None,
):
    """Run taped ``FlowGlue.heads`` on a ``MeshDevice`` and backprop all ``world`` chips together.

    Evaluates all ``world`` ranks' non-insertion terms and seeds in a single vectorized float64
    ``RO.terms`` call over only the ~12 token-sized keys (bit-identical to per-rank
    ``RO.rfd4_loss``).

    When ``stop_at_s=True``, wraps ``s_taped``'s underlying device tensor in a fresh local
    ``ag.Tensor(..., requires_grad=True)`` so ``ag.backward`` runs ONLY through ``DeviceHeads``
    and returns ``(mean_total, per_rank_totals, per_rank_terms, per_rank_refs, ds_out)`` where
    ``ds_out`` is the fp32 gradient seed on ``s`` (used by paired ``MeshTrace`` trunk backward).
    When ``head_tracer``, ``dfolds``, and ``dev_masters`` are also supplied, replays paired
    ``DeviceHeads`` fwd/bwd+fold traces inside a dedicated 64 MiB/bank scratch hole disjoint from
    ``win_pair``'s hole and populates ``head_grads_out`` (and ``head_skip_out`` on replay).
    """
    from .mesh_dp import put_sharded, read_sharded, stack_batches

    w = RO.WEIGHTS if weights is None else weights
    world = len(rank_bws)
    if stacked_bw is None:
        stacked_bw = stack_batches(rank_bws)
    n_tok = int(stacked_bw["padding_mask"].shape[1])
    s_raw = getattr(s_taped, "value", s_taped)
    n_pad = int(s_raw.shape[1])
    dev = s_raw.device()

    dh = getattr(glue, "_device_heads", None)
    can_trace_heads = bool(
        stop_at_s
        and head_tracer is not None
        and dfolds is not None
        and dev_masters is not None
        and dh is not None
        and head_grads_out is not None
    )
    run_seq = bool(dh.sequence_readout_enabled) if dh is not None else False
    head_key = head_tracer.head_pair_key(s_raw, dh, run_seq) if can_trace_heads else None
    use_head_replay = bool(can_trace_heads and head_tracer.has_head_pair(head_key))
    want_head_cap = bool(can_trace_heads and not use_head_replay and head_tracer.head_pair_should_capture(head_key))

    cap_masks = None
    cap_refs = None
    if use_head_replay:
        emit_ca, emit_ll, pad_m = dh._upload_masks(stacked_bw, n_pad, need_seq=run_seq)
        ca_t, ll_t, seq_t, ins_t = head_tracer.replay_head_pair_fwd(head_key, s_raw, emit_ca, emit_ll, pad_m)
        for mt in (emit_ca, emit_ll, pad_m):
            if mt is not None:
                ttnn.deallocate(mt)
    else:
        s_local = ag.Tensor(s_raw, requires_grad=True) if stop_at_s else s_taped
        if want_head_cap:
            emit_ca, emit_ll, pad_m = dh._upload_masks(stacked_bw, n_pad, need_seq=run_seq)
            cap_masks = (emit_ca, emit_ll, pad_m)
            with ag.tape():
                o = dh(s_local, None, insertion=True, sequence=run_seq, masks_dev=cap_masks)
        else:
            with ag.tape():
                o = glue.heads(s_local, stacked_bw, insertion=True)
        ca_t = next(iter(o["bb_ca"].values()))
        ll_t = next(iter(o["local_latents"].values()))
        seq_t = o["sequence"]["logits"] if "sequence" in o else None
        ins_t = o["insertion"]["logits"]
        if want_head_cap:
            cap_refs = (
                ttnn.clone(getattr(ca_t, "value", ca_t)),
                ttnn.clone(getattr(ll_t, "value", ll_t)),
                ttnn.clone(getattr(seq_t, "value", seq_t)) if seq_t is not None else None,
                ttnn.clone(getattr(ins_t, "value", ins_t)),
            )

    def read_tok_all(t, d_out: int) -> np.ndarray:
        arr = read_sharded(getattr(t, "value", t), dev).float().numpy()
        return arr[:, :n_tok, :d_out].astype(np.float64)

    ca_all = read_tok_all(ca_t, 3)
    ll_all = read_tok_all(ll_t, 32)
    seq_all = read_tok_all(seq_t, 147) if seq_t is not None else None

    ins_w = float(w.get("insertion_prediction", 1.0))
    ins_loss_np, dz_ins = device_insertion_loss(ins_t, stacked_bw, weight=ins_w)

    # Single vectorized float64 RO.terms call across all `world` winners (keep=1.0 per rank).
    ok_all = {"v_bb_ca": ca_all, "v_local_latents": ll_all}
    if seq_all is not None:
        ok_all["sequence_logits"] = seq_all
    bw_np_all = _small_npy_dict(stacked_bw)
    per = RO.terms(bw_np_all, ok_all, sequence_weight=1.0, signed_volume_weight=1.0)

    seeds_all: dict[str, np.ndarray] = {}
    totals_non_ins = np.zeros(world, dtype=np.float64)
    term_vals: dict[str, np.ndarray | None] = {}
    for name in RO.TERMS:
        wt = float(w.get(name, 0.0))
        if name not in per:
            term_vals[name] = None
            continue
        loss_vec, grads = per[name]
        term_vals[name] = loss_vec
        totals_non_ins += wt * loss_vec
        for gk, g_arr in grads.items():
            sg = wt * g_arr
            seeds_all[gk] = sg if gk not in seeds_all else (seeds_all[gk] + sg)

    ca_seeds = np.zeros((world, n_pad, 3), dtype=np.float32)
    ll_seeds = np.zeros((world, n_pad, 32), dtype=np.float32)
    ca_seeds[:, :n_tok, :3] = seeds_all["v_bb_ca"].astype(np.float32)
    ll_seeds[:, :n_tok, :32] = seeds_all["v_local_latents"].astype(np.float32)
    seq_seeds = None
    have_seq_seed = False
    if seq_t is not None and "sequence_logits" in seeds_all:
        seq_seeds = np.zeros((world, n_pad, 147), dtype=np.float32)
        seq_seeds[:, :n_tok, :147] = seeds_all["sequence_logits"].astype(np.float32)
        have_seq_seed = True
    elif seq_t is not None and (use_head_replay or want_head_cap):
        seq_seeds = np.zeros((world, n_pad, 147), dtype=np.float32)

    per_rank_totals: list[float] = []
    per_rank_terms: list[dict] = []
    per_rank_refs: list[dict] = []
    for r in range(world):
        ins_val = float(ins_loss_np[r])
        per_rank_totals.append(float(totals_non_ins[r]) + ins_w * ins_val)
        terms_r = {
            name: (None if vec is None else float(vec[r]))
            for name, vec in term_vals.items()
        }
        terms_r["insertion_prediction"] = ins_val
        per_rank_terms.append(terms_r)
        ref_r = {}
        if ref_terms_list is not None and r < len(ref_terms_list) and ref_terms_list[r] is not None:
            w_r = int(win_indices[r]) if win_indices is not None else 0
            rt = ref_terms_list[r]
            ref_r = {k: float(rt[k][w_r]) for k in terms_r if k in rt and w_r < len(rt[k])}
        per_rank_refs.append(ref_r)

    d_ca_tt = put_sharded(torch.from_numpy(ca_seeds), dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    d_ll_tt = put_sharded(torch.from_numpy(ll_seeds), dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    d_seq_tt = (
        put_sharded(torch.from_numpy(seq_seeds), dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        if seq_seeds is not None else None
    )

    if use_head_replay:
        d_ins_tt = (
            dz_ins if dz_ins is not None
            else put_sharded(torch.zeros((world, n_pad, n_pad), dtype=torch.float32), dev,
                             dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        )
        ds_out, head_grads = head_tracer.replay_head_pair_bwd(head_key, d_ca_tt, d_ll_tt, d_seq_tt, d_ins_tt)
        for sd in (d_ca_tt, d_ll_tt, d_seq_tt, d_ins_tt):
            if sd is not None:
                ttnn.deallocate(sd)
        head_grads_out.update(head_grads)
        if head_skip_out is not None:
            head_skip_out.update(head_grads.keys())
        mean_total = float(sum(per_rank_totals) / world)
        return mean_total, per_rank_totals, per_rank_terms, per_rank_refs, ds_out

    outs_bwd = [ca_t, ll_t]
    seeds_bwd = [d_ca_tt, d_ll_tt]
    if seq_t is not None and (have_seq_seed or want_head_cap):
        outs_bwd.append(seq_t)
        seeds_bwd.append(d_seq_tt)
    if dz_ins is not None:
        outs_bwd.append(ins_t)
        seeds_bwd.append(dz_ins)
    elif want_head_cap:
        dz_ins = put_sharded(torch.zeros((world, n_pad, n_pad), dtype=torch.float32), dev,
                             dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        outs_bwd.append(ins_t)
        seeds_bwd.append(dz_ins)

    ag.backward(outs_bwd, seeds_bwd)
    del o, ca_t, ll_t, seq_t, ins_t, outs_bwd

    if stop_at_s:
        ds_out = s_local.grad
        s_local.grad = None
        if want_head_cap:
            head_g, _ = dfolds.tape_backward(dev_masters)
            ag.release_pins()
            head_grads_out.update(head_g)
            try:
                head_tracer.capture_head_pair(
                    head_key,
                    dh=dh,
                    dfolds=dfolds,
                    dev_masters=dev_masters,
                    s_val=s_raw,
                    emit_ca=cap_masks[0],
                    emit_ll=cap_masks[1],
                    pad_m=cap_masks[2],
                    ca_ref=cap_refs[0],
                    ll_ref=cap_refs[1],
                    seq_ref=cap_refs[2],
                    ins_ref=cap_refs[3],
                    d_ca_ref=d_ca_tt,
                    d_ll_ref=d_ll_tt,
                    d_seq_ref=d_seq_tt,
                    d_ins_ref=dz_ins,
                    ds_ref=ds_out,
                    head_grads_ref=head_g,
                    run_seq=run_seq,
                )
            finally:
                for t in (*cap_masks, *cap_refs):
                    if t is not None and t.is_allocated():
                        ttnn.deallocate(t)
        for sd in seeds_bwd:
            if sd is not None and sd.is_allocated():
                ttnn.deallocate(sd)
        mean_total = float(sum(per_rank_totals) / world)
        return mean_total, per_rank_totals, per_rank_terms, per_rank_refs, ds_out

    for sd in seeds_bwd:
        if sd is not None and sd.is_allocated():
            ttnn.deallocate(sd)

    if bridge is not None:
        bridge.backward_ups()

    mean_total = float(sum(per_rank_totals) / world)
    return mean_total, per_rank_totals, per_rank_terms, per_rank_refs
