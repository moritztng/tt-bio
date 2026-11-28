"""Device-resident RFD4 output heads.

Runs the four FlowGlue readouts (``bb_ca``, ``local_latents``, ``sequence``,
``insertion``) directly on Tenstorrent in float32 using production ``ttnn``
verbs (``rms_norm``, ``matmul``, ``multiply``). Dispatched in place from
:meth:`tt_bio.rfd4.network.FlowGlue.heads` whenever ``s`` is a device tensor:
with taping off it executes raw ``ttnn`` (byte-identical A/A); with taping on,
``taped_ttnn`` shims this module's ``ttnn`` binding in place without a taped
twin.

Training-side on-card losses live in :mod:`tt_bio.train.rfd4_heads` so this
inference module never imports the training stack.
"""

from __future__ import annotations

import numpy as np
import torch
import ttnn

from .. import ops

EPS = 1e-5


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


def _split2_mm(a, b, dtype, ckc, *, transpose_b: bool = False):
    """Exact fp32 matmul via bf16 hi+lo split using production ``ttnn`` verbs.

    Blackhole ``ttnn.matmul`` unpacks ``float32`` operands as ``bfloat16``
    (~4e-3 rel_l2 over the 3-matmul insertion head chain, shifting small ~4e-3
    CE losses by >1.5%). Splitting ``a = a_hi + a_lo`` and ``b = b_hi + b_lo``
    into bf16 pairs recovers ~15-bit mantissa accuracy (~1e-5 rel_l2). Under an
    active tape, ``a_lo`` and ``b_lo`` are computed with the tape suspended so
    they stay detached constants and backward accumulates
    ``g @ (b_hi + b_lo)^T`` and ``(a_hi + a_lo)^T @ g`` cleanly in fp32 without
    bf16 cancellation.
    """
    if dtype != ttnn.float32:
        return ttnn.matmul(
            a, b, transpose_b=transpose_b, dtype=dtype, compute_kernel_config=ckc,
        )
    a_hi = ttnn.typecast(a, ttnn.bfloat16)
    b_hi = ttnn.typecast(b, ttnn.bfloat16)
    with ops.suspend_tape():
        a_raw = getattr(a, "value", a)
        b_raw = getattr(b, "value", b)
        a_hi_raw = getattr(a_hi, "value", a_hi)
        b_hi_raw = getattr(b_hi, "value", b_hi)
        a_hi32 = ttnn.typecast(a_hi_raw, ttnn.float32)
        a_lo32 = ttnn.subtract(a_raw, a_hi32)
        ttnn.deallocate(a_hi32)
        a_lo = ttnn.typecast(a_lo32, ttnn.bfloat16)
        ttnn.deallocate(a_lo32)
        b_hi32 = ttnn.typecast(b_hi_raw, ttnn.float32)
        b_lo32 = ttnn.subtract(b_raw, b_hi32)
        ttnn.deallocate(b_hi32)
        b_lo = ttnn.typecast(b_lo32, ttnn.bfloat16)
        ttnn.deallocate(b_lo32)
    m0 = ttnn.matmul(
        a_hi, b_hi, transpose_b=transpose_b, dtype=dtype, compute_kernel_config=ckc,
    )
    m1 = ttnn.matmul(
        a_hi, b_lo, transpose_b=transpose_b, dtype=dtype, compute_kernel_config=ckc,
    )
    m2 = ttnn.matmul(
        a_lo, b_hi, transpose_b=transpose_b, dtype=dtype, compute_kernel_config=ckc,
    )
    m01 = ttnn.add(m0, m1)
    out = ttnn.add(m01, m2)
    if not ops.taping():
        for t in (a_hi, b_hi, a_lo, b_lo, m0, m1, m2, m01):
            ttnn.deallocate(t)
    return out


class DeviceHeads:
    """On-card FlowGlue heads in float32.

    Constructed from a :class:`FlowGlue` instance. When ``state`` is supplied
    (e.g. the leaf dict from ``Folds.recording()`` or ``DeviceFolds.recording()``),
    weights are read through ``state["glue.<key>"]`` so folds track every head
    weight automatically for on-card ``fold_bwd`` and ``refold``.
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

    def deallocate(self) -> None:
        """Free all device weight tensors held by this module."""
        for attr in ("ca_norm_w", "ca_w", "ll_norm_w", "ll_w",
                     "seq_norm_w", "seq_w", "ins_q_w", "ins_k_w"):
            t = getattr(self, attr, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, attr, None)

    def _upload_masks(self, batch: dict, n_pad: int, *, need_seq: bool = True) -> tuple:
        from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks

        emit = compute_flow_emit_masks(batch)
        multi = getattr(self.device, "get_num_devices", lambda: 1)() > 1
        mm = ttnn.ShardTensorToMesh(self.device, dim=0) if multi else None

        def up_m(arr) -> object:
            m = _pad_1d(_as_np(arr).astype(np.float32), n_pad, 0.0)
            t = torch.from_numpy(m[..., None].copy())
            return ttnn.from_torch(
                t, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device,
                **({"mesh_mapper": mm} if mm is not None else {}))

        emit_ca = up_m(emit["bb_ca"])
        emit_ll = up_m(emit["local_latents"])
        pad_m = up_m(batch["padding_mask"]) if (need_seq and self.sequence_readout_enabled) else None
        return emit_ca, emit_ll, pad_m

    def __call__(self, s, batch: dict, *, insertion: bool = True, sequence: bool = True,
                 masks_dev: tuple | None = None) -> dict[str, dict[str, object]]:
        """Run the heads on device tensor ``s`` ([b, n_pad, 768]).

        Returns the same nested dict schema as :meth:`FlowGlue.heads` with
        device tensors (``ttnn.Tensor`` untaped, ``ag.Tensor`` taped).
        """
        s_raw = getattr(s, "value", s)
        n_pad = int(s_raw.shape[1])
        run_seq = bool(sequence and self.sequence_readout_enabled)
        if masks_dev is None:
            emit_ca, emit_ll, pad_m = self._upload_masks(batch, n_pad, need_seq=run_seq)
        else:
            emit_ca, emit_ll, pad_m = masks_dev

        s32 = s if s_raw.dtype == self.dtype else ttnn.typecast(s, self.dtype)
        param = self.cfg["output_parameterization"]

        # 1. bb_ca: RMSNorm(768) -> Linear(768, 3) -> * emit_ca
        ca_h = ttnn.rms_norm(
            s32, weight=self.ca_norm_w, epsilon=EPS, compute_kernel_config=self.ckc)
        ca_v = _split2_mm(ca_h, self.ca_w, self.dtype, self.ckc)
        bb_ca = ttnn.multiply(ca_v, emit_ca)

        # 2. local_latents: RMSNorm(768) -> Linear(768, 32) -> * emit_ll
        ll_h = ttnn.rms_norm(
            s32, weight=self.ll_norm_w, epsilon=EPS, compute_kernel_config=self.ckc)
        ll_v = _split2_mm(ll_h, self.ll_w, self.dtype, self.ckc)
        local_latents = ttnn.multiply(ll_v, emit_ll)

        out: dict[str, dict[str, object]] = {
            "bb_ca": {param["bb_ca"]: bb_ca},
            "local_latents": {param["local_latents"]: local_latents},
        }

        # 3. sequence_readout: RMSNorm(768) -> Linear(768, 147) -> * padding_mask
        if run_seq:
            seq_h = ttnn.rms_norm(
                s32, weight=self.seq_norm_w, epsilon=EPS, compute_kernel_config=self.ckc)
            seq_v = _split2_mm(seq_h, self.seq_w, self.dtype, self.ckc)
            out["sequence"] = {"logits": ttnn.multiply(seq_v, pad_m)}

        # 4. insertion: q = s @ Wq, k = s @ Wk, logits = (q @ k^T) * (768 ** -0.5)
        if insertion:
            q = _split2_mm(s32, self.ins_q_w, self.dtype, self.ckc)
            k = _split2_mm(s32, self.ins_k_w, self.dtype, self.ckc)
            ins_raw = _split2_mm(q, k, self.dtype, self.ckc, transpose_b=True)
            out["insertion"] = {
                "logits": ttnn.multiply(ins_raw, self._insertion_head_scale)
            }

        # Free intermediates when untaped; under a tape, autograd owns their lifetime.
        if not ops.taping():
            for t in (ca_h, ca_v, ll_h, ll_v):
                ttnn.deallocate(t)
            if run_seq:
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
