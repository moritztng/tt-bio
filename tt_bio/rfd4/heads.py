"""Device-resident RFD4 output heads and on-card insertion loss.

Runs the four FlowGlue readouts (``bb_ca``, ``local_latents``, ``sequence``,
``insertion``) directly on Tenstorrent in float32 using production ``ttnn``
verbs (``rms_norm``, ``matmul``, ``multiply``). Dispatched in place from
:meth:`tt_bio.rfd4.network.FlowGlue.heads` whenever ``s`` is a device tensor:
with ``tt_bio.autograd.tape()`` off it executes raw ``ttnn`` (byte-identical
A/A); with the tape on, ``taped_ttnn`` shims this module's ``ttnn`` binding
in place without a taped twin.

``device_insertion_loss`` evaluates the ``[b, n, n]`` insertion cross-entropy
and its gradient ``dL/dlogits`` entirely on card in float32 from 1-D index and
mask vectors (~3 KB PCIe total), so the pair-sized ``[b, n, n]`` insertion
logits and gradient never cross PCIe.
"""

from __future__ import annotations

import numpy as np
import torch
import ttnn

from .. import autograd as ag

EPS = 1e-5
_NEG_INF = -1e9


def _f32_ckc():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _pad_1d(arr: np.ndarray, n_pad: int, fill: float = 0.0) -> np.ndarray:
    """Pad [b, n] -> [b, n_pad] along axis 1 with ``fill``."""
    b, n = arr.shape
    if n == n_pad:
        return arr
    out = np.full((b, n_pad), fill, dtype=arr.dtype)
    out[:, :n] = arr
    return out


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _norm_w_tt(t: torch.Tensor, device, dtype):
    """Upload a 1-D [d] RMSNorm weight as [1, 1, 1, d] TILE tensor."""
    return ttnn.from_torch(t.detach().float().reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class DeviceHeads:
    """On-card FlowGlue heads in float32.

    Constructed from a :class:`FlowGlue` instance. When ``state`` is supplied
    (e.g. the leaf dict from ``Folds.recording()``), weights are read through
    ``state["glue.<key>"]`` so ``Folds`` and ``DeviceFolds`` automatically
    track every head weight for on-card ``fold_bwd`` and ``refold``.
    """

    def __init__(self, glue, device, *, state: dict | None = None,
                 dtype=None, compute_kernel_config=None):
        self.device = device
        self.dtype = dtype if dtype is not None else ttnn.float32
        self.ckc = compute_kernel_config if compute_kernel_config is not None else _f32_ckc()
        self.cfg = dict(glue.cfg)
        self.sequence_readout_enabled = bool(glue.sequence_readout_enabled)
        self._insertion_head_scale = float(glue._insertion_head_scale)

        def get_w(key: str, fallback: torch.Tensor) -> torch.Tensor:
            if state is not None and f"glue.{key}" in state:
                return state[f"glue.{key}"]
            return fallback

        def up_norm(w: torch.Tensor):
            return ttnn.from_torch(w, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=device)

        def up_lin(w: torch.Tensor):
            # Linear weight is [out, in]; upload transposed [in, out] for s @ W.
            return ttnn.from_torch(w.t(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=device)

        self.ca_norm_w = up_norm(get_w("ca_linear.0.weight", glue.ca_linear[0].weight))
        self.ca_w = up_lin(get_w("ca_linear.1.weight", glue.ca_linear[1].weight))

        self.ll_norm_w = up_norm(
            get_w("local_latents_linear.0.weight", glue.local_latents_linear[0].weight))
        self.ll_w = up_lin(
            get_w("local_latents_linear.1.weight", glue.local_latents_linear[1].weight))

        if self.sequence_readout_enabled:
            self.seq_norm_w = up_norm(
                get_w("sequence_readout.0.weight", glue.sequence_readout[0].weight))
            self.seq_w = up_lin(
                get_w("sequence_readout.1.weight", glue.sequence_readout[1].weight))
        else:
            self.seq_norm_w = None
            self.seq_w = None

        self.ins_q_w = up_lin(get_w("insertion_q_proj.weight", glue.insertion_q_proj.weight))
        self.ins_k_w = up_lin(get_w("insertion_k_proj.weight", glue.insertion_k_proj.weight))

    def _upload_masks(self, batch: dict, n_pad: int) -> tuple:
        from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks

        emit = compute_flow_emit_masks(batch)

        def up_m(arr) -> object:
            m = _pad_1d(_as_np(arr).astype(np.float32), n_pad, 0.0)
            t = torch.from_numpy(m[..., None].copy())
            return ttnn.from_torch(
                t, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

        emit_ca = up_m(emit["bb_ca"])
        emit_ll = up_m(emit["local_latents"])
        pad_m = up_m(batch["padding_mask"]) if self.sequence_readout_enabled else None
        return emit_ca, emit_ll, pad_m

    def __call__(self, s, batch: dict, *, insertion: bool = True,
                 masks_dev: tuple | None = None) -> dict[str, dict[str, object]]:
        """Run the four heads on device tensor ``s`` ([b, n_pad, 768]).

        Returns the same nested dict schema as :meth:`FlowGlue.heads` with
        device tensors (``ttnn.Tensor`` untaped, ``ag.Tensor`` taped).
        """
        s_raw = getattr(s, "value", s)
        n_pad = int(s_raw.shape[1])
        if masks_dev is None:
            emit_ca, emit_ll, pad_m = self._upload_masks(batch, n_pad)
        else:
            emit_ca, emit_ll, pad_m = masks_dev

        s32 = s if s_raw.dtype == self.dtype else ttnn.typecast(s, self.dtype)
        param = self.cfg["output_parameterization"]

        # 1. bb_ca: RMSNorm(768) -> Linear(768, 3) -> * emit_ca
        ca_h = ttnn.rms_norm(
            s32, weight=self.ca_norm_w, epsilon=EPS, compute_kernel_config=self.ckc)
        ca_v = ttnn.matmul(ca_h, self.ca_w, dtype=self.dtype, compute_kernel_config=self.ckc)
        bb_ca = ttnn.multiply(ca_v, emit_ca)

        # 2. local_latents: RMSNorm(768) -> Linear(768, 32) -> * emit_ll
        ll_h = ttnn.rms_norm(
            s32, weight=self.ll_norm_w, epsilon=EPS, compute_kernel_config=self.ckc)
        ll_v = ttnn.matmul(ll_h, self.ll_w, dtype=self.dtype, compute_kernel_config=self.ckc)
        local_latents = ttnn.multiply(ll_v, emit_ll)

        out: dict[str, dict[str, object]] = {
            "bb_ca": {param["bb_ca"]: bb_ca},
            "local_latents": {param["local_latents"]: local_latents},
        }

        # 3. sequence_readout: RMSNorm(768) -> Linear(768, 147) -> * padding_mask
        if self.sequence_readout_enabled:
            seq_h = ttnn.rms_norm(
                s32, weight=self.seq_norm_w, epsilon=EPS, compute_kernel_config=self.ckc)
            seq_v = ttnn.matmul(
                seq_h, self.seq_w, dtype=self.dtype, compute_kernel_config=self.ckc)
            out["sequence"] = {"logits": ttnn.multiply(seq_v, pad_m)}

        # 4. insertion: q = s @ Wq, k = s @ Wk, logits = (q @ k^T) * (768 ** -0.5)
        if insertion:
            q = ttnn.matmul(s32, self.ins_q_w, dtype=self.dtype, compute_kernel_config=self.ckc)
            k = ttnn.matmul(s32, self.ins_k_w, dtype=self.dtype, compute_kernel_config=self.ckc)
            ins_raw = ttnn.matmul(
                q, k, transpose_b=True, dtype=self.dtype, compute_kernel_config=self.ckc)
            out["insertion"] = {
                "logits": ttnn.multiply(ins_raw, self._insertion_head_scale)
            }

        # Free intermediates when untaped (taped_ttnn intercepts deallocate if pinned).
        if not ag.is_grad_enabled():
            for t in (ca_h, ca_v, ll_h, ll_v):
                ttnn.deallocate(t)
            if self.sequence_readout_enabled:
                ttnn.deallocate(seq_h)
                ttnn.deallocate(seq_v)
            if insertion:
                ttnn.deallocate(q)
                ttnn.deallocate(k)
                ttnn.deallocate(ins_raw)
            if masks_dev is None:
                ttnn.deallocate(emit_ca)
                ttnn.deallocate(emit_ll)
                if pad_m is not None:
                    ttnn.deallocate(pad_m)
                if s32 is not s:
                    ttnn.deallocate(s32)

        return out


def device_insertion_loss(
    logits_tt,
    batch: dict,
    *,
    win: np.ndarray | None = None,
    weight: float = 1.0,
) -> tuple[np.ndarray, object | None]:
    """On-card insertion cross-entropy loss and ``[b, n_pad, n_pad]`` gradient.

    Matches :func:`tt_bio.train.rfd4_objective.insertion_loss` (and the
    ``keep[win] * weight`` scaling from :func:`rfd4_loss`) while keeping the
    pair-sized logits and gradient strictly on device. Only 1-D index/mask
    vectors (~3 KB) are uploaded and ``b`` loss scalars (``4*b`` bytes) are
    downloaded.

    Returns ``(per_example_loss_np [b], dL_dlogits_tt [b, n_pad, n_pad] or None)``.
    When no query rows are active (``target < 0`` everywhere), returns
    ``(zeros(b), None)`` with zero device ops and zero PCIe bytes.
    """
    z = getattr(logits_tt, "value", logits_tt)
    b, n_pad, _ = (int(x) for x in z.shape)
    dev = z.device()

    target_raw = _as_np(batch["insertion_source_idx"]).astype(np.int64)
    pad_raw = _as_np(batch["padding_mask"]).astype(bool)
    n_raw = target_raw.shape[1]

    # Pad to n_pad with inactive sentinels: target=-1 (not query), pad=False (not key).
    target = _pad_1d(target_raw, n_pad, -1)
    pad = _pad_1d(pad_raw, n_pad, False)

    row = target >= 0                         # [b, n_pad] query tokens
    has = row.any(-1)                         # [b]
    if not has.any():
        return np.zeros(b, dtype=np.float64), None

    key_ok = pad & (target < 0)               # [b, n_pad] candidate keys
    nv = row.sum(-1).astype(np.float64)
    snv = np.where(nv > 0, nv, 1.0)
    has_f = has.astype(np.float64)

    if win is None:
        keep = np.ones(b, dtype=np.float64)
    else:
        keep = np.zeros(b, dtype=np.float64)
        keep[np.asarray(win, np.int64)] = 1.0 / len(win)

    loss_row_scale = (row * (has_f / snv)[:, None]).astype(np.float32)         # [b, n_pad]
    dz_row_scale = (loss_row_scale * (keep * float(weight))[:, None]).astype(np.float32)
    # Inactive rows use target = -1 so eq(arange, safe) is 0 across the row.
    safe_idx = np.where(row, target, -1).astype(np.float32)                    # [b, n_pad]
    key_ok_f = key_ok.astype(np.float32)                                       # [b, n_pad]
    key_neg_f = np.where(key_ok, 0.0, _NEG_INF).astype(np.float32)             # [b, n_pad]
    arange_f = np.arange(n_pad, dtype=np.float32).reshape(1, 1, n_pad)

    with ag.no_grad():
        def up3(arr: np.ndarray):
            return ttnn.from_torch(
                torch.from_numpy(np.ascontiguousarray(arr)),
                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
            )

        key_ok_tt = up3(key_ok_f[:, None, :])          # [b, 1, n_pad]
        key_neg_tt = up3(key_neg_f[:, None, :])        # [b, 1, n_pad]
        loss_scale_tt = up3(loss_row_scale[:, :, None])  # [b, n_pad, 1]
        dz_scale_tt = up3(dz_row_scale[:, :, None])    # [b, n_pad, 1]
        safe_tt = up3(safe_idx[:, :, None])            # [b, n_pad, 1]
        arange_tt = up3(arange_f)                      # [1, 1, n_pad]

        # Build [b, n_pad, n_pad] onehot strictly on card via 1-D broadcast eq.
        onehot_raw = ttnn.eq(arange_tt, safe_tt)
        onehot = (onehot_raw if onehot_raw.dtype == ttnn.float32
                  else ttnn.typecast(onehot_raw, ttnn.float32))
        if onehot is not onehot_raw:
            ttnn.deallocate(onehot_raw)

        # zm = where(key_ok[:, None, :], z, -1e9)
        z_masked = ttnn.multiply(z, key_ok_tt)
        zm = ttnn.add(z_masked, key_neg_tt)
        ttnn.deallocate(z_masked)

        # Stable log-softmax and softmax in fp32 along last axis.
        m = ttnn.max(zm, dim=-1, keepdim=True)
        shift = ttnn.subtract(zm, m)
        ttnn.deallocate(zm)
        ttnn.deallocate(m)
        ex = ttnn.exp(shift)
        sum_ex = ttnn.sum(ex, dim=-1, keepdim=True)
        lse = ttnn.log(sum_ex)
        logp = ttnn.subtract(shift, lse)
        p = ttnn.divide(ex, sum_ex)
        ttnn.deallocate(shift)
        ttnn.deallocate(ex)
        ttnn.deallocate(sum_ex)
        ttnn.deallocate(lse)

        # Per-example loss: sum_i (row_scale_i * (-sum_j onehot_ij * logp_ij))
        sel_logp = ttnn.multiply(onehot, logp)
        ttnn.deallocate(logp)
        row_logp = ttnn.sum(sel_logp, dim=-1, keepdim=True)      # [b, n_pad, 1]
        ttnn.deallocate(sel_logp)
        neg_row = ttnn.multiply(row_logp, -1.0)
        ttnn.deallocate(row_logp)
        w_row = ttnn.multiply(neg_row, loss_scale_tt)
        ttnn.deallocate(neg_row)
        loss_tt = ttnn.sum(w_row, dim=1, keepdim=False)          # [b, 1]
        ttnn.deallocate(w_row)
        loss_np = ttnn.to_torch(loss_tt).detach().cpu().numpy().reshape(b).astype(np.float64)
        ttnn.deallocate(loss_tt)

        # Gradient: dz = (p - onehot) * dz_row_scale * key_ok
        diff = ttnn.subtract(p, onehot)
        ttnn.deallocate(p)
        ttnn.deallocate(onehot)
        dz_r = ttnn.multiply(diff, dz_scale_tt)
        ttnn.deallocate(diff)
        dz = ttnn.multiply(dz_r, key_ok_tt)
        ttnn.deallocate(dz_r)

        for t in (key_ok_tt, key_neg_tt, loss_scale_tt, dz_scale_tt, safe_tt, arange_tt):
            ttnn.deallocate(t)

    return loss_np, dz
