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

import numpy as np
import torch
import ttnn

from .. import autograd as ag
from . import rfd4_objective as RO

__all__ = ["device_insertion_loss", "device_candidate_scores", "device_winner_loss"]


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
        z_np = z_t.detach().cpu().float().numpy()[:, :n_tok, :n_tok].astype(np.float64)
        loss_np, dz_np = RO.insertion_loss(
            z_np,
            {"insertion_source_idx": target_raw, "padding_mask": pad_raw},
        )
        dz_scaled = dz_np * (keep * float(weight))[:, None, None]
        dz_buf = np.zeros((b, n_pad, n_pad), dtype=np.float32)
        dz_buf[:, :n_tok, :n_tok] = dz_scaled.astype(np.float32)
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
