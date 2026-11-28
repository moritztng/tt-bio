"""Device-resident SPMD MeshTrace & host-prep offload for multi-chip Blackhole Galaxy RFD4 training.

Why this exists (diagnosed on anx-glx 8x4, 32 chips):
  1. ``FDMeshCommandQueue::write_program_cmds_to_subgrid`` loops serially over all ``world`` chips
     on every ``ttnn`` op (~0.20 ms/op x ~10,000 ops/step = ~2.0 s pure serial host dispatch).
     ``FDMeshCommandQueue::enqueue_trace`` collapses an entire phase to 1 trace issue per chip
     (measured 22.28x speedup on a 100-matmul chain on 8x4, 20.31 ms -> 0.91 ms).
  2. Single-process SPMD runs ``glue.conditioning(32*K)`` and ``glue.init_repr_factory(32*K)``
     for all 32 ranks on the host CPU (~564 ms/step at depth=2). Moving ``cond_factory.linear_out``,
     ``transition_c_1``, ``transition_c_2``, and ``init_repr_factory.linear_out`` onto the 32 chips
     in parallel in exact fp32 HiFi4 eliminates ~520 ms/step with zero numerical regression.

All trace captures share a single fenced ``TraceScratch`` DRAM hole (default 512 MiB/bank =
4 GiB per 32 GiB BH chip), and all folded weights / masters are refreshed in-place so buffer
addresses remain invariant across steps. Opt-out via ``--no-spmd-trace`` for eager A/B parity.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import gc
import math
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch

DEVICE_GLUE_WEIGHTS: frozenset[str] = frozenset({
    "glue.init_repr_factory.linear_out.weight",
    "glue.cond_factory.linear_out.weight",
    "glue.transition_c_1.swish_linear.0.weight",
    "glue.transition_c_1.linear_out.weight",
    "glue.transition_c_2.swish_linear.0.weight",
    "glue.transition_c_2.linear_out.weight",
})


def _safe_dealloc(t, *, keep=None) -> None:
    """Deallocate ``t`` unless it is None or aliases ``keep``'s underlying device buffer."""
    if t is None:
        return
    if keep is not None:
        if t is keep:
            return
        try:
            if t.buffer_address() == keep.buffer_address():
                return
        except Exception:
            pass
    import ttnn
    try:
        if t.is_allocated():
            ttnn.deallocate(t)
    except Exception:
        pass


def _pad_last_32(x: torch.Tensor) -> torch.Tensor:
    rem = int(x.shape[-1]) % 32
    if rem == 0:
        return x
    pad = 32 - rem
    return torch.nn.functional.pad(x, (0, pad))


def _pad_cols_32_dev(w):
    """Pad a 2D ``[M, K]`` TILE_LAYOUT device tensor to ``[M, ceil32(K)]`` with literal 0.0.

    Routes through ROW_MAJOR so physical tile-edge lanes past ``K`` are guaranteed literal
    ``0.0`` even after in-place optimizer updates on ``opt.master``.
    """
    import ttnn

    m, k = int(w.shape[0]), int(w.shape[1])
    rem = k % 32
    if rem == 0:
        return w, False
    pad = 32 - rem
    rm = ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT)
    rm_pad = ttnn.pad(rm, padding=((0, 0), (0, pad)), value=0.0)
    ttnn.deallocate(rm)
    tl_pad = ttnn.to_layout(rm_pad, ttnn.TILE_LAYOUT)
    ttnn.deallocate(rm_pad)
    return tl_pad, True


def _unpad_cols_dev(w_pad, orig_cols: int):
    """Strip column padding from a 2D ``[M, K_pad]`` TILE_LAYOUT tensor back to ``[M, orig_cols]``."""
    import ttnn

    m, k_pad = int(w_pad.shape[0]), int(w_pad.shape[1])
    if k_pad == orig_cols:
        return w_pad
    rm = ttnn.to_layout(w_pad, ttnn.ROW_MAJOR_LAYOUT)
    rm_sl = ttnn.slice(rm, [0, 0], [m, orig_cols])
    ttnn.deallocate(rm)
    tl = ttnn.to_layout(rm_sl, ttnn.TILE_LAYOUT)
    ttnn.deallocate(rm_sl)
    return tl


def _put_replicated_f32(t: torch.Tensor, device):
    import ttnn
    from .mesh_dp import mesh_world

    kw: dict[str, Any] = {
        "dtype": ttnn.float32,
        "layout": ttnn.TILE_LAYOUT,
        "device": device,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }
    if mesh_world(device) > 1:
        kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    return ttnn.from_torch(t.detach().to(torch.float32).contiguous(), **kw)


def _accurate_sigmoid(g):
    """Exact fp32 sigmoid (avoids BH ``ttnn.silu``'s hardcoded ~2e-3 approx LUT)."""
    import ttnn

    mode = getattr(getattr(ttnn, "SigmoidMode", None), "Accurate", None)
    if mode is not None:
        return ttnn.sigmoid(g, mode=mode)
    return ttnn.sigmoid(g)


def _exact_f32_mm(a, b, *, transpose_a: bool = False, transpose_b: bool = False, cfg=None):
    """Exact fp32 matmul via bf16 hi+lo split (eliminates BH TF32 10-bit input truncation)."""
    import ttnn
    from .device_optim import _core_grid, _mm_cg, _split2_bf16

    ah, al = _split2_bf16(a)
    bh, bl = _split2_bf16(b)
    kw = dict(
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        dtype=ttnn.float32,
        compute_kernel_config=cfg,
        **_mm_cg(_core_grid(a.device())),
    )
    p0 = ttnn.matmul(ah, bh, **kw)
    p1 = ttnn.matmul(ah, bl, **kw)
    p2 = ttnn.matmul(al, bh, **kw)
    ttnn.deallocate(ah)
    ttnn.deallocate(al)
    ttnn.deallocate(bh)
    ttnn.deallocate(bl)
    s01 = ttnn.add(p0, p1)
    ttnn.deallocate(p0)
    ttnn.deallocate(p1)
    out = ttnn.add(s01, p2)
    ttnn.deallocate(s01)
    ttnn.deallocate(p2)
    return out


def can_offload_cond_init(glue) -> bool:
    """Return True if ``glue`` matches the standard RFD4 cond/init_repr structure for device offload."""
    try:
        cf = glue.cond_factory
        ir = glue.init_repr_factory
        t1 = glue.transition_c_1
        t2 = glue.transition_c_2
        if getattr(cf, "ret_zero", False) or getattr(ir, "ret_zero", False):
            return False
        if not isinstance(getattr(cf, "ln_out", None), torch.nn.Identity):
            return False
        if not isinstance(getattr(ir, "ln_out", None), torch.nn.Identity):
            return False
        if cf.linear_out.bias is not None or ir.linear_out.bias is not None:
            return False
        if len(cf.feat_creators) < 2:
            return False
        if type(cf.feat_creators[-1]).__name__ != "RecycleRemainingBatchFeat":
            return False
        for tr in (t1, t2):
            if getattr(tr, "use_layer_norm", False):
                return False
            if tr.swish_linear[0].bias is not None or tr.linear_out.bias is not None:
                return False
        return True
    except Exception:
        return False


def _eval_factory_feats(factory, batch: dict, creators: Sequence[Any]) -> torch.Tensor:
    """Evaluate a subset of ``FeatureFactory.feat_creators`` and concatenate + padding-mask."""
    mask = batch["padding_mask"]
    n = int(mask.shape[-1])
    parts = []
    for fc in creators:
        feat = fc(batch)
        if feat.ndim == 2:
            feat = feat.reshape(feat.shape[0], 1, -1).expand(-1, n, -1)
        parts.append(feat)
    feats = torch.cat(parts, dim=-1)
    return factory.apply_padding_mask(feats, mask)


def _fc_expand(feat: torch.Tensor, n: int) -> torch.Tensor:
    if feat.ndim == 2:
        return feat.reshape(feat.shape[0], 1, -1).expand(-1, n, -1)
    return feat


class _OneFactorySplit:
    """Partition a ``FeatureFactory``'s creators into weight-free static vs trainable dynamic columns.

    Static creators (``0`` trainable parameters, excluding ``RecycleRemainingBatchFeat``) are
    safe to evaluate in a background prefetcher thread while the previous step runs on device.
    Dynamic creators (tiny ``nn.Embedding`` layers) run live on the main thread with current
    weights (and with autograd on the winner path).
    """

    def __init__(self, factory, *, exclude_last_recycle: bool = False):
        self.factory = factory
        self.exclude_last_recycle = bool(exclude_last_recycle)
        self.creators = list(factory.feat_creators[:-1] if exclude_last_recycle else factory.feat_creators)
        self.rem_creator = factory.feat_creators[-1] if exclude_last_recycle else None
        self.total_dim: int = 0
        self.static_specs: list[Tuple[Any, int, int]] = []   # (fc, col0, col1)
        self.dynamic_specs: list[Tuple[Any, int, int]] = []  # (fc, col0, col1)
        self.dynamic_cols: list[int] = []
        self._probed = False

    def ensure_probed(self, sample_batch: Mapping[str, Any]) -> None:
        if self._probed:
            return
        n = int(sample_batch["padding_mask"].shape[-1])
        col = 0
        with torch.no_grad():
            for fc in self.creators:
                out = _fc_expand(fc(sample_batch), n)
                w = int(out.shape[-1])
                has_params = any(p.requires_grad for p in fc.parameters()) or (len(list(fc.parameters())) > 0)
                if has_params:
                    self.dynamic_specs.append((fc, col, col + w))
                    self.dynamic_cols.extend(range(col, col + w))
                else:
                    self.static_specs.append((fc, col, col + w))
                col += w
        self.total_dim = col
        self._probed = True

    def build_static_template_into(self, batch: Mapping[str, Any], dst: torch.Tensor) -> None:
        """Write masked static columns directly into pre-allocated ``dst`` ([B, n, width] zeroed)."""
        self.ensure_probed(batch)
        mask = batch["padding_mask"]
        n = int(mask.shape[1])
        with torch.no_grad():
            for fc, c0, c1 in self.static_specs:
                feat = _fc_expand(fc(batch), n).to(torch.float32)
                dst[:, :, c0:c1] = feat
            dst[:, :, :self.total_dim] = self.factory.apply_padding_mask(dst[:, :, :self.total_dim], mask)

    def build_static_template(self, batch: Mapping[str, Any], *, pad_to: int = 0) -> torch.Tensor:
        """Compute ``[B, n, max(total_dim, pad_to)]`` float32 template with static columns filled and masked."""
        self.ensure_probed(batch)
        mask = batch["padding_mask"]
        b, n = int(mask.shape[0]), int(mask.shape[1])
        width = max(self.total_dim, int(pad_to))
        buf = torch.zeros((b, n, width), dtype=torch.float32)
        self.build_static_template_into(batch, buf)
        return buf

    def try_build_static_2d(self, batch: Mapping[str, Any], *, pad_to: int = 0) -> Optional[torch.Tensor]:
        """Return ``[B, max(total_dim, pad_to)]`` if every static creator is constant across valid tokens."""
        self.ensure_probed(batch)
        mask = batch["padding_mask"]
        b = int(mask.shape[0])
        m_bool = mask > 0.5
        first_idx = m_bool.to(torch.int64).argmax(dim=-1)
        b_idx = torch.arange(b, device=mask.device)
        ones_m = m_bool.any(dim=-1, keepdim=True).to(torch.float32)
        width = max(self.total_dim, int(pad_to))
        buf2 = torch.zeros((b, width), dtype=torch.float32)
        with torch.no_grad():
            for fc, c0, c1 in self.static_specs:
                f = fc(batch)
                if f.ndim == 2:
                    f2 = f.to(torch.float32)
                elif f.ndim == 3:
                    f0 = f[b_idx, first_idx]
                    if not bool(((f == f0.unsqueeze(1)) | (~m_bool.unsqueeze(-1))).all()):
                        return None
                    f2 = f0.to(torch.float32)
                else:
                    return None
                buf2[:, c0:c1] = f2
            buf2[:, :self.total_dim] *= ones_m
        return buf2

    def eval_dynamic(self, batch: Mapping[str, Any], *, grad: bool = False) -> Tuple[list[Tuple[torch.Tensor, int, int]], torch.Tensor]:
        """Evaluate ONLY the trainable dynamic creators; returns ``([(feat_i, c0, c1), ...], cat_dyn)``."""
        self.ensure_probed(batch)
        mask = batch["padding_mask"]
        n = int(mask.shape[1])
        ctx = torch.enable_grad() if grad else torch.no_grad()
        parts: list[Tuple[torch.Tensor, int, int]] = []
        with ctx:
            for fc, c0, c1 in self.dynamic_specs:
                feat = _fc_expand(fc(batch), n)
                if feat.dtype != torch.float32:
                    feat = feat.to(torch.float32)
                feat = self.factory.apply_padding_mask(feat, mask)
                parts.append((feat, c0, c1))
        if parts:
            cat_dyn = torch.cat([p[0] for p in parts], dim=-1)
        else:
            b = int(mask.shape[0])
            cat_dyn = torch.zeros((b, n, 0), dtype=torch.float32)
        return parts, cat_dyn

    def fill_into(self, template: torch.Tensor, batch: Mapping[str, Any]) -> torch.Tensor:
        """Clone ``template`` (no-grad) and fill live dynamic columns -> exact full feature tensor."""
        out = template.clone()
        parts, _ = self.eval_dynamic(batch, grad=False)
        for feat, c0, c1 in parts:
            out[:, :, c0:c1] = feat
        return out

    def gather_matrix(self, pad_in: int) -> torch.Tensor:
        """Build static ``[pad_in, ceil32(len(dynamic_cols))]`` 0/1 gather matrix for device VJP."""
        d_dyn = len(self.dynamic_cols)
        d_pad = ((d_dyn + 31) // 32) * 32 if d_dyn > 0 else 32
        mat = torch.zeros((pad_in, d_pad), dtype=torch.float32)
        for j, src_col in enumerate(self.dynamic_cols):
            mat[src_col, j] = 1.0
        return mat


class FactoryStaticSplit:
    """Static/dynamic split for both ``glue.cond_factory`` and ``glue.init_repr_factory``."""

    def __init__(self, glue):
        self.glue = glue
        self.cf = _OneFactorySplit(glue.cond_factory, exclude_last_recycle=True)
        self.ir = _OneFactorySplit(glue.init_repr_factory, exclude_last_recycle=False)

    def ensure_probed(self, sample_batch: Mapping[str, Any]) -> None:
        self.cf.ensure_probed(sample_batch)
        self.ir.ensure_probed(sample_batch)


@dataclass
class PrefetchedStep:
    """Weight-free host features prefetched for a single upcoming SPMD step."""

    step_idx: int
    cf_static: Optional[torch.Tensor]     # [world*K, n, cf_pad] float32 (None when cf_static_2d holds)
    ir_static: torch.Tensor               # [world*K, n, ir_pad] float32 (masked, dyn cols=0)
    static_encs: Optional[list[dict]]     # per-rank ResidentPairBuilder.encode_static outputs
    pair_cond_feats: Optional[list[torch.Tensor]]  # per-rank [K, 512] float32 pair cond features
    cand_meta: dict                       # compact candidate scoring metadata (~12 small arrays)
    cf_static_2d: Optional[torch.Tensor] = None    # [world*K, cf_pad] when cf is row-constant
    pair_stacked: Optional[dict] = None            # pre-stacked uint32 pair index tensors


class SPMDPrefetcher:
    """Single-slot background prefetcher computing step ``t+1``'s weight-free features during step ``t``.

    Computes freshly from ``rank_bts`` on every step (zero cross-step batch caching), touches
    zero trainable weights and zero ``ttnn`` handles, and hides host feature extraction behind
    the pure-C++ ``replay_win_pair_bwd_fold`` + optimizer trace window (~500 ms) with zero GIL
    contention against ``cand_pair`` / ``cand_prep``.
    """

    def __init__(
        self,
        factory_split: FactoryStaticSplit,
        *,
        res_pair=None,
        max_workers: int = 4,
    ):
        self.factory_split = factory_split
        self.res_pair = res_pair
        self._inner = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._bg = ThreadPoolExecutor(max_workers=1)
        self._pending: Optional[Tuple[int, Future]] = None

    def _compute(self, step_idx: int, rank_bts: Sequence[Mapping[str, Any]]) -> PrefetchedStep:
        from .rfd4_heads import extract_cand_score_meta

        self.factory_split.ensure_probed(rank_bts[0])
        fs = self.factory_split
        rp = self.res_pair
        cf_full_in = int(fs.glue.cond_factory.linear_out.weight.shape[1])
        ir_full_in = int(fs.glue.init_repr_factory.linear_out.weight.shape[1])
        cf_pad_in = ((cf_full_in + 31) // 32) * 32
        ir_pad_in = ((ir_full_in + 31) // 32) * 32

        world = len(rank_bts)
        k_local = int(rank_bts[0]["padding_mask"].shape[0])
        n_tok = int(rank_bts[0]["padding_mask"].shape[1])
        # Pre-allocate one contiguous ir_static buffer (eliminates 122 MB torch.cat copy + page faults).
        ir_static = torch.zeros((world * k_local, n_tok, ir_pad_in), dtype=torch.float32)

        def _one_rank(item):
            r_idx, b = item
            with torch.no_grad():
                cf_2d = fs.cf.try_build_static_2d(b, pad_to=cf_pad_in)
                cf_3d = None if cf_2d is not None else fs.cf.build_static_template(b, pad_to=cf_pad_in)
                fs.ir.build_static_template_into(b, ir_static[r_idx * k_local:(r_idx + 1) * k_local])
                st_enc = rp.encode_static(b) if rp is not None else None
                pcf = rp.extract_pair_cond_feats(b) if rp is not None else None
                return cf_2d, cf_3d, st_enc, pcf

        indexed = list(enumerate(rank_bts))
        if world > 1:
            rows = list(self._inner.map(_one_rank, indexed))
        else:
            rows = [_one_rank(indexed[0])]

        if all(r[0] is not None for r in rows):
            cf_static_2d = torch.cat([r[0] for r in rows], dim=0)
            cf_static = None
        else:
            cf_static_2d = None
            cf_pieces = []
            for (r_idx, b), r in zip(indexed, rows):
                if r[1] is not None:
                    cf_pieces.append(r[1])
                else:
                    mask_b = b["padding_mask"]
                    c3 = r[0].unsqueeze(1).expand(k_local, n_tok, cf_pad_in).clone()
                    c3[:, :, :fs.cf.total_dim] = fs.glue.cond_factory.apply_padding_mask(
                        c3[:, :, :fs.cf.total_dim], mask_b,
                    )
                    cf_pieces.append(c3)
            cf_static = torch.cat(cf_pieces, dim=0)

        static_encs = [r[2] for r in rows] if rp is not None else None
        pair_cond_feats = [r[3] for r in rows] if rp is not None else None

        # Pre-stack uint32 pair index tensors in background so candidates_spmd does zero host torch.cat.
        pair_stacked = None
        if static_encs is not None and all(e is not None for e in static_encs):
            e0 = static_encs[0]
            nn2 = e0.n * e0.n
            dedup_all = all(e.dedup for e in static_encs)
            b_c = 1 if dedup_all else e0.k
            if not any(e.fallback_const is not None for e in static_encs):
                def _exp(t: torch.Tensor, e_dedup: bool) -> torch.Tensor:
                    if (not dedup_all) and e_dedup:
                        return t.expand(e0.k, *t.shape[1:]).contiguous()
                    return t

                oh_stacked = []
                for idx_pos, (kind, _) in enumerate(e0.idx_onehot):
                    st = torch.cat(
                        [_exp(e.idx_onehot[idx_pos][1], e.dedup).reshape(1, b_c * nn2) for e in static_encs],
                        dim=0,
                    ).contiguous()
                    oh_stacked.append((kind, st))
                sc_stacked = None
                if rp.misc_w > 0:
                    sc_stacked = torch.cat(
                        [_exp(e.idx_scalar, e.dedup).reshape(1, b_c * nn2) for e in static_encs],
                        dim=0,
                    ).contiguous()
                bins_stacked = torch.cat(
                    [e.bins_step.reshape(1, e0.k * nn2) for e in static_encs], dim=0,
                ).contiguous()
                pair_stacked = {
                    "dedup_all": dedup_all,
                    "onehot": oh_stacked,
                    "scalar": sc_stacked,
                    "bins": bins_stacked,
                }

        cand_meta = extract_cand_score_meta(rank_bts)
        return PrefetchedStep(
            step_idx=step_idx,
            cf_static=cf_static,
            ir_static=ir_static,
            static_encs=static_encs,
            pair_cond_feats=pair_cond_feats,
            cand_meta=cand_meta,
            cf_static_2d=cf_static_2d,
            pair_stacked=pair_stacked,
        )

    def submit(self, step_idx: int, rank_bts: Sequence[Mapping[str, Any]]) -> None:
        if self._pending is not None and self._pending[0] == step_idx:
            return
        self._pending = (step_idx, self._bg.submit(self._compute, step_idx, rank_bts))

    def get(self, step_idx: int, rank_bts: Sequence[Mapping[str, Any]]) -> PrefetchedStep:
        if self._pending is not None and self._pending[0] == step_idx:
            _, fut = self._pending
            self._pending = None
            return fut.result()
        return self._compute(step_idx, rank_bts)

    def close(self) -> None:
        if self._pending is not None:
            try:
                self._pending[1].result()
            except Exception:
                pass
            self._pending = None
        self._bg.shutdown(wait=False)
        self._inner.shutdown(wait=False)


def _transition_bwd_torch(
    d_out: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch analytical reference VJP of masked SwiGLU Transition (for CPU unit tests)."""
    d_out_m = d_out * mask if mask is not None else d_out
    h = x @ w1.T
    half = w1.shape[0] // 2
    a, g = h[..., :half], h[..., half:]
    sig = torch.sigmoid(g)
    sg = g * sig
    sw = sg * a
    d_out_2d = d_out_m.reshape(-1, d_out_m.shape[-1])
    sw_2d = sw.reshape(-1, sw.shape[-1])
    dw2 = d_out_2d.T @ sw_2d
    d_sw = d_out_m @ w2
    d_a = d_sw * sg
    d_sg = d_sw * a
    d_g = d_sg * (sig + sg * (1.0 - sig))
    d_h = torch.cat([d_a, d_g], dim=-1)
    d_h_2d = d_h.reshape(-1, d_h.shape[-1])
    x_2d = x.reshape(-1, x.shape[-1])
    dw1 = d_h_2d.T @ x_2d
    dx = d_h @ w1
    return dx, dw1, dw2


def _eval_dynamic_only(
    split: _OneFactorySplit,
    rank_bts: Sequence[Mapping[str, Any]],
    template: torch.Tensor,
    pool=None,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Fill ONLY the trainable dynamic columns into ``template`` (no-grad, ~5 ms for B=32*K)."""
    out = template if inplace else template.clone()
    if not split.dynamic_specs:
        return out

    off = 0
    for b in rank_bts:
        k = int(b["padding_mask"].shape[0])
        parts, _ = split.eval_dynamic(b, grad=False)
        for feat, c0, c1 in parts:
            out[off:off + k, :, c0:c1] = feat
        off += k
    return out


def cand_conds_and_s0_device(
    glue,
    rank_bts: Sequence[Mapping[str, Any]],
    mk_tok,
    step_depth: int,
    trunk_dtype,
    device,
    *,
    pool=None,
    dev_masters: Optional[Mapping[str, Any]] = None,
    prefetched: Optional[PrefetchedStep] = None,
    factory_split: Optional[FactoryStaticSplit] = None,
) -> Tuple[Any, list[Any]]:
    """Compute grad-off candidate ``(s0_tt, ck_tt)`` directly on ``device`` in exact fp32 HiFi4.

    When ``prefetched`` + ``factory_split`` are supplied, reuses the prefetched weight-free static
    templates and evaluates ONLY the 5 tiny trainable embeddings (~4 ms vs ~168 ms). Reuses live
    device-resident ``dev_masters`` for all 6 projection/transition weights (zero host upload and
    zero stale CPU weight copies when ``DeviceWinnerGlue`` owns those weights).
    """
    import ttnn
    from .. import autograd as ag
    from .mesh_dp import put_sharded

    cf = glue.cond_factory
    ir = glue.init_repr_factory
    t1 = glue.transition_c_1
    t2 = glue.transition_c_2
    rem_creator = cf.feat_creators[-1]

    cf_full_dim = int(cf.linear_out.weight.shape[1])
    cf_pad_dim = ((cf_full_dim + 31) // 32) * 32
    k_local = int(rank_bts[0]["padding_mask"].shape[0])
    n_tok = int(rank_bts[0]["padding_mask"].shape[1])
    world = len(rank_bts)

    cf_base_2d: Optional[torch.Tensor] = None
    cf_base_h: Optional[torch.Tensor] = None
    if prefetched is not None and factory_split is not None:
        base_dim = factory_split.cf.total_dim
        ir_feat_h = _eval_dynamic_only(factory_split.ir, rank_bts, prefetched.ir_static, pool=pool, inplace=True)
        if prefetched.cf_static_2d is not None:
            # Verify the tiny trainable dynamic cf creator(s) are also row-constant across valid tokens.
            cf_2d = prefetched.cf_static_2d.clone()
            if int(cf_2d.shape[-1]) < cf_pad_dim:
                cf_2d = torch.nn.functional.pad(cf_2d, (0, cf_pad_dim - int(cf_2d.shape[-1])))
            dyn_row_const = True
            off = 0
            for b in rank_bts:
                k_b = int(b["padding_mask"].shape[0])
                mask_b = b["padding_mask"]
                m_bool = mask_b > 0.5
                first_idx = m_bool.to(torch.int64).argmax(dim=-1)
                k_idx = torch.arange(k_b, device=mask_b.device)
                ones_m = m_bool.any(dim=-1, keepdim=True).to(torch.float32)
                parts, _ = factory_split.cf.eval_dynamic(b, grad=False)
                for feat, c0, c1 in parts:
                    f0 = feat[k_idx, first_idx]
                    if not bool(((feat == f0.unsqueeze(1)) | (~m_bool.unsqueeze(-1))).all()):
                        dyn_row_const = False
                        break
                    cf_2d[off:off + k_b, c0:c1] = f0 * ones_m
                if not dyn_row_const:
                    break
                off += k_b
            if dyn_row_const:
                cf_base_2d = cf_2d
            else:
                # Rare fallback: materialize 3D cf_base_h from cf_static_2d + dynamic parts.
                mask_all = torch.cat([b["padding_mask"] for b in rank_bts], dim=0)
                c3 = prefetched.cf_static_2d.unsqueeze(1).expand(world * k_local, n_tok, int(prefetched.cf_static_2d.shape[-1])).clone()
                c3[:, :, :base_dim] = cf.apply_padding_mask(c3[:, :, :base_dim], mask_all)
                cf_base_h = _eval_dynamic_only(factory_split.cf, rank_bts, c3, pool=pool, inplace=True)
        else:
            cf_base_h = _eval_dynamic_only(factory_split.cf, rank_bts, prefetched.cf_static, pool=pool, inplace=True)
    else:
        base_creators = list(cf.feat_creators[:-1])
        ir_creators = list(ir.feat_creators)

        def _extract_rank(b: Mapping[str, Any]):
            with torch.no_grad():
                cf_b = _eval_factory_feats(cf, b, base_creators).to(torch.float32)
                ir_b = _eval_factory_feats(ir, b, ir_creators).to(torch.float32)
                return cf_b, ir_b

        if pool is not None and len(rank_bts) > 1:
            extracted = list(pool.map(_extract_rank, rank_bts))
        else:
            extracted = [_extract_rank(b) for b in rank_bts]

        cf_base_h = torch.cat([x[0] for x in extracted], dim=0)
        ir_feat_h = torch.cat([x[1] for x in extracted], dim=0)
        base_dim = int(cf_base_h.shape[-1])

    rem_dim = cf_full_dim - base_dim
    ir_feat_pad = _pad_last_32(ir_feat_h)

    cfg = ag.precise_config()
    m_f32 = None
    m_f32_owned = False
    if mk_tok is not None:
        if mk_tok.dtype == ttnn.float32:
            m_f32 = mk_tok
        else:
            m_f32 = ttnn.typecast(mk_tok, ttnn.float32)
            m_f32_owned = True

    def _w_or_put(master_key: str, host_t: torch.Tensor, *, pad32: bool = False):
        if dev_masters is not None and master_key in dev_masters:
            w = dev_masters[master_key]
            if pad32:
                return _pad_cols_32_dev(w)
            return w, False
        ht = host_t.detach().to(torch.float32)
        if pad32:
            ht = _pad_last_32(ht)
        return _put_replicated_f32(ht, device), True

    with ag.no_grad():
        # 1. s0_tt = (ir_feat @ W_ir^T) * mask
        ir_tt = put_sharded(ir_feat_pad, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        w_ir_tt, own_w_ir = _w_or_put("glue.init_repr_factory.linear_out.weight", ir.linear_out.weight, pad32=True)
        s0_tt = _exact_f32_mm(ir_tt, w_ir_tt, transpose_b=True, cfg=cfg)
        ttnn.deallocate(ir_tt)
        if own_w_ir:
            ttnn.deallocate(w_ir_tt)
        if m_f32 is not None:
            s0_m = ttnn.multiply(s0_tt, m_f32)
            ttnn.deallocate(s0_tt)
            s0_tt = s0_m

        # 2. Conditioning for all recycle cycles
        w_cf_tt, own_w_cf = _w_or_put("glue.cond_factory.linear_out.weight", cf.linear_out.weight, pad32=True)
        w_t1_1, own_t1_1 = _w_or_put("glue.transition_c_1.swish_linear.0.weight", t1.swish_linear[0].weight)
        w_t1_2, own_t1_2 = _w_or_put("glue.transition_c_1.linear_out.weight", t1.linear_out.weight)
        w_t2_1, own_t2_1 = _w_or_put("glue.transition_c_2.swish_linear.0.weight", t2.swish_linear[0].weight)
        w_t2_2, own_t2_2 = _w_or_put("glue.transition_c_2.linear_out.weight", t2.linear_out.weight)
        half = int(w_t1_1.shape[0]) // 2

        rem_seq = [None]
        for cyc in range(1, int(step_depth)):
            rem_seq.append(int(step_depth) - cyc - 1 if glue.condition_on_recycle_remaining else None)

        dummy_b = {
            "padding_mask": rank_bts[0]["padding_mask"][:1, :1],
            "x_t": {"bb_ca": rank_bts[0]["x_t"]["bb_ca"][:1, :1, :]},
        }

        if cf_base_2d is not None:
            # Row-constant fast path: pack all U unique recycle_remaining states into [world*U*K, 1, cf_pad]
            # -> ONE linear_out + ONE transition_c_1 + ONE transition_c_2 on [U*K, 1, 256] per chip (~5 ms!).
            uniq_keys: list[Optional[int]] = []
            cyc_to_u: list[int] = []
            seen_k: dict[Optional[int], int] = {}
            for rem in rem_seq:
                key = None if rem is None else min(int(rem), int(rem_creator.dim) - 1)
                if key not in seen_k:
                    seen_k[key] = len(uniq_keys)
                    uniq_keys.append(key)
                cyc_to_u.append(seen_k[key])
            u_cnt = len(uniq_keys)
            ones_all = torch.cat(
                [(b["padding_mask"] > 0.5).any(dim=-1, keepdim=True).to(torch.float32) for b in rank_bts],
                dim=0,
            ).reshape(world, k_local, 1)  # [world, K, 1]
            cf_u_h = cf_base_2d.reshape(world, 1, k_local, cf_pad_dim).expand(world, u_cnt, k_local, cf_pad_dim).clone()
            with torch.no_grad():
                for ui, key in enumerate(uniq_keys):
                    if key is not None and rem_dim > 0:
                        rv = rem_creator({**dummy_b, "recycle_remaining": key}).to(torch.float32).reshape(1, 1, rem_dim)
                        cf_u_h[:, ui, :, base_dim:base_dim + rem_dim] = rv * ones_all
            cf_u_flat = cf_u_h.reshape(world * u_cnt * k_local, 1, cf_pad_dim).contiguous()
            cf_u_tt = put_sharded(cf_u_flat, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            c0_u = _exact_f32_mm(cf_u_tt, w_cf_tt, transpose_b=True, cfg=cfg)
            ttnn.deallocate(cf_u_tt)

            uk_loc = u_cnt * k_local

            def _transition_1d(x_in, w1, w2):
                h = _exact_f32_mm(x_in, w1, transpose_b=True, cfg=cfg)
                a = ttnn.slice(h, [0, 0, 0], [uk_loc, 1, half])
                g = ttnn.slice(h, [0, 0, half], [uk_loc, 1, 2 * half])
                ttnn.deallocate(h)
                sig = _accurate_sigmoid(g)
                sg = ttnn.multiply(g, sig)
                ttnn.deallocate(sig)
                ttnn.deallocate(g)
                sw = ttnn.multiply(sg, a)
                ttnn.deallocate(sg)
                ttnn.deallocate(a)
                out = _exact_f32_mm(sw, w2, transpose_b=True, cfg=cfg)
                ttnn.deallocate(sw)
                return out

            c1_u = _transition_1d(c0_u, w_t1_1, w_t1_2)
            ttnn.deallocate(c0_u)
            c2_u = _transition_1d(c1_u, w_t2_1, w_t2_2)
            ttnn.deallocate(c1_u)
            if trunk_dtype != ttnn.float32:
                c2_out = ttnn.typecast(c2_u, trunk_dtype)
                ttnn.deallocate(c2_u)
            else:
                c2_out = c2_u

            d_c = int(c2_out.shape[-1])
            if mk_tok is not None:
                if mk_tok.dtype == trunk_dtype:
                    bcast_m = mk_tok
                    own_bm = False
                else:
                    bcast_m = ttnn.typecast(mk_tok, trunk_dtype)
                    own_bm = True
            else:
                ones_h = torch.ones((1, n_tok, 1), dtype=torch.float32)
                bcast_m = ttnn.from_torch(
                    ones_h, dtype=trunk_dtype, layout=ttnn.TILE_LAYOUT,
                    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                own_bm = True

            u_devs = []
            for ui in range(u_cnt):
                if u_cnt == 1:
                    cu_1 = c2_out
                else:
                    cu_1 = ttnn.slice(c2_out, [ui * k_local, 0, 0], [(ui + 1) * k_local, 1, d_c])
                cu_full = ttnn.multiply(bcast_m, cu_1)
                if cu_1 is not c2_out:
                    ttnn.deallocate(cu_1)
                u_devs.append(cu_full)
            ttnn.deallocate(c2_out)
            if own_bm:
                ttnn.deallocate(bcast_m)

            # Build per-cycle list, cloning any repeated unique rem so caller owns distinct buffers.
            used_u: set[int] = set()
            ck_tt = []
            for ui in cyc_to_u:
                if ui not in used_u:
                    used_u.add(ui)
                    ck_tt.append(u_devs[ui])
                else:
                    ck_tt.append(ttnn.clone(u_devs[ui]))
        else:
            if int(cf_base_h.shape[-1]) < cf_pad_dim:
                cf_base_pad = torch.nn.functional.pad(cf_base_h, (0, cf_pad_dim - int(cf_base_h.shape[-1])))
            else:
                cf_base_pad = cf_base_h
            cf_tt = put_sharded(cf_base_pad, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            c_base = _exact_f32_mm(cf_tt, w_cf_tt, transpose_b=True, cfg=cfg)
            ttnn.deallocate(cf_tt)

            def _transition(x_in, w1, w2):
                h = _exact_f32_mm(x_in, w1, transpose_b=True, cfg=cfg)
                a = ttnn.slice(h, [0, 0, 0], [k_local, n_tok, half])
                g = ttnn.slice(h, [0, 0, half], [k_local, n_tok, 2 * half])
                ttnn.deallocate(h)
                sig = _accurate_sigmoid(g)
                sg = ttnn.multiply(g, sig)
                ttnn.deallocate(sig)
                ttnn.deallocate(g)
                sw = ttnn.multiply(sg, a)
                ttnn.deallocate(sg)
                ttnn.deallocate(a)
                out = _exact_f32_mm(sw, w2, transpose_b=True, cfg=cfg)
                ttnn.deallocate(sw)
                if m_f32 is not None:
                    om = ttnn.multiply(out, m_f32)
                    ttnn.deallocate(out)
                    out = om
                return out

            cache_by_rem: dict[Optional[int], Any] = {}
            ck_tt = []
            for rem in rem_seq:
                key = None if rem is None else min(int(rem), int(rem_creator.dim) - 1)
                if key in cache_by_rem:
                    ck_tt.append(ttnn.clone(cache_by_rem[key]))
                    continue
                if key is None:
                    c_in = ttnn.multiply(c_base, m_f32) if m_f32 is not None else ttnn.clone(c_base)
                else:
                    with torch.no_grad():
                        rv = rem_creator({**dummy_b, "recycle_remaining": key}).to(torch.float32).reshape(rem_dim)
                        e_rem = torch.zeros((1, 1, cf_pad_dim), dtype=torch.float32)
                        e_rem[0, 0, base_dim:base_dim + rem_dim] = rv
                    e_rem_tt = _put_replicated_f32(e_rem, device)
                    bias_tt = _exact_f32_mm(e_rem_tt, w_cf_tt, transpose_b=True, cfg=cfg)
                    ttnn.deallocate(e_rem_tt)
                    c_sum = ttnn.add(c_base, bias_tt)
                    ttnn.deallocate(bias_tt)
                    if m_f32 is not None:
                        c_in = ttnn.multiply(c_sum, m_f32)
                        ttnn.deallocate(c_sum)
                    else:
                        c_in = c_sum
                c1 = _transition(c_in, w_t1_1, w_t1_2)
                ttnn.deallocate(c_in)
                c2 = _transition(c1, w_t2_1, w_t2_2)
                ttnn.deallocate(c1)
                if trunk_dtype != ttnn.float32:
                    c_out = ttnn.typecast(c2, trunk_dtype)
                    ttnn.deallocate(c2)
                else:
                    c_out = c2
                cache_by_rem[key] = c_out
                ck_tt.append(c_out)

            ttnn.deallocate(c_base)

        for wt, owned in (
            (w_cf_tt, own_w_cf),
            (w_t1_1, own_t1_1), (w_t1_2, own_t1_2),
            (w_t2_1, own_t2_1), (w_t2_2, own_t2_2),
        ):
            if owned:
                ttnn.deallocate(wt)
        if m_f32_owned and m_f32 is not None:
            ttnn.deallocate(m_f32)

    return s0_tt, ck_tt


class DeviceWinnerGlue:
    """Device-resident forward + analytical fp32 backward for the 6 winner host-glue projections.

    Replaces the 32-rank CPU ``init_repr.linear_out``, ``cond_factory.linear_out``,
    ``transition_c_1``, and ``transition_c_2`` forward + ``torch.autograd.backward`` (~285 ms/step
    + 33 MB D2H) with exact fp32 HiFi4 device ops across all 32 chips in parallel (~8 ms + 5.7 MB
    D2H for only the 5 tiny CPU embeddings).
    """

    _W_KEYS = (
        "glue.init_repr_factory.linear_out.weight",
        "glue.cond_factory.linear_out.weight",
        "glue.transition_c_1.swish_linear.0.weight",
        "glue.transition_c_1.linear_out.weight",
        "glue.transition_c_2.swish_linear.0.weight",
        "glue.transition_c_2.linear_out.weight",
    )

    def __init__(self, glue, factory_split: FactoryStaticSplit, device, *, tracer: Optional["SPMDMeshTracer"] = None):
        self.glue = glue
        self.split = factory_split
        self.device = device
        self.tracer = tracer
        self._p_ir_dyn = None
        self._p_cf_dyn = None
        self._ir_pad_in = 0
        self._cf_pad_in = 0
        self._ir_orig_in = int(glue.init_repr_factory.linear_out.weight.shape[1])
        self._cf_orig_in = int(glue.cond_factory.linear_out.weight.shape[1])
        self._traces: dict[tuple, Any] = {}

    def _trace_key(self, n_tok: int, m1_tok, trunk_dtype, dev_masters: Mapping[str, Any]) -> tuple:
        def _addr(t):
            return t.buffer_address() if hasattr(t, "buffer_address") else id(t)

        return (
            int(n_tok),
            m1_tok is not None,
            None if m1_tok is None else str(m1_tok.dtype),
            str(trunk_dtype),
            tuple(_addr(dev_masters[k]) for k in self._W_KEYS),
        )

    def _ensure_gather_mats(self, sample_batch: Mapping[str, Any]) -> None:
        if self._p_ir_dyn is not None:
            return
        self.split.ensure_probed(sample_batch)
        self._ir_pad_in = ((self._ir_orig_in + 31) // 32) * 32
        self._cf_pad_in = ((self._cf_orig_in + 31) // 32) * 32
        self._p_ir_dyn = _put_replicated_f32(self.split.ir.gather_matrix(self._ir_pad_in), self.device)
        self._p_cf_dyn = _put_replicated_f32(self.split.cf.gather_matrix(self._cf_pad_in), self.device)

    def _fwd_device_ops(
        self,
        ir_tt,
        cf_tt,
        m1_tok,
        n_tok: int,
        trunk_dtype,
        dev_masters: Mapping[str, Any],
    ):
        """Pure device-op forward body shared identically by eager and MeshTrace capture."""
        import ttnn
        from .. import autograd as ag

        cfg = ag.precise_config()
        m_f32 = None
        m_f32_owned = False
        if m1_tok is not None:
            if m1_tok.dtype == ttnn.float32:
                m_f32 = m1_tok
            else:
                m_f32 = ttnn.typecast(m1_tok, ttnn.float32)
                m_f32_owned = True

        w_ir_pad, own_w_ir = _pad_cols_32_dev(dev_masters[self._W_KEYS[0]])
        w_cf_pad, own_w_cf = _pad_cols_32_dev(dev_masters[self._W_KEYS[1]])

        s_i_raw = _exact_f32_mm(ir_tt, w_ir_pad, transpose_b=True, cfg=cfg)
        if m_f32 is not None:
            s_i_dev = ttnn.multiply(s_i_raw, m_f32)
            ttnn.deallocate(s_i_raw)
        else:
            s_i_dev = s_i_raw

        c0_raw = _exact_f32_mm(cf_tt, w_cf_pad, transpose_b=True, cfg=cfg)
        if m_f32 is not None:
            c0_dev = ttnn.multiply(c0_raw, m_f32)
            ttnn.deallocate(c0_raw)
        else:
            c0_dev = c0_raw

        w_t1_1 = dev_masters[self._W_KEYS[2]]
        w_t1_2 = dev_masters[self._W_KEYS[3]]
        w_t2_1 = dev_masters[self._W_KEYS[4]]
        w_t2_2 = dev_masters[self._W_KEYS[5]]
        half = int(w_t1_1.shape[0]) // 2
        b_loc = int(c0_dev.shape[0])

        def _tr_fwd_save(x_in, w1, w2):
            h = _exact_f32_mm(x_in, w1, transpose_b=True, cfg=cfg)
            a = ttnn.slice(h, [0, 0, 0], [b_loc, n_tok, half])
            g = ttnn.slice(h, [0, 0, half], [b_loc, n_tok, 2 * half])
            ttnn.deallocate(h)
            sig = _accurate_sigmoid(g)
            sg = ttnn.multiply(g, sig)
            ttnn.deallocate(g)
            sw = ttnn.multiply(sg, a)
            out = _exact_f32_mm(sw, w2, transpose_b=True, cfg=cfg)
            if m_f32 is not None:
                om = ttnn.multiply(out, m_f32)
                ttnn.deallocate(out)
                out = om
            return out, (a, sig, sg, sw)

        c1_dev, st1 = _tr_fwd_save(c0_dev, w_t1_1, w_t1_2)
        c2_f32, st2 = _tr_fwd_save(c1_dev, w_t2_1, w_t2_2)
        if trunk_dtype != ttnn.float32:
            c_out_dev = ttnn.typecast(c2_f32, trunk_dtype)
            ttnn.deallocate(c2_f32)
        else:
            c_out_dev = c2_f32

        return (
            s_i_dev, c_out_dev, c0_dev, c1_dev, st1, st2,
            w_ir_pad, own_w_ir, w_cf_pad, own_w_cf, m_f32, m_f32_owned,
        )

    def _bwd_device_ops(
        self,
        ds_i,
        dc_out,
        *,
        ir_tt,
        cf_tt,
        w_ir_pad,
        w_cf_pad,
        c0_dev,
        c1_dev,
        st1,
        st2,
        m_f32,
        dev_masters: Mapping[str, Any],
        consume_saved: bool,
    ):
        """Pure device-op backward body shared identically by eager and MeshTrace capture."""
        import ttnn
        from .. import autograd as ag

        cfg = ag.precise_config()
        w_t1_1 = dev_masters[self._W_KEYS[2]]
        w_t1_2 = dev_masters[self._W_KEYS[3]]
        w_t2_1 = dev_masters[self._W_KEYS[4]]
        w_t2_2 = dev_masters[self._W_KEYS[5]]

        def _mm2d_t_a(a_3d, b_3d):
            bn = int(a_3d.shape[0]) * int(a_3d.shape[1])
            a2 = ttnn.reshape(a_3d, [bn, int(a_3d.shape[2])])
            b2 = ttnn.reshape(b_3d, [bn, int(b_3d.shape[2])])
            out = _exact_f32_mm(a2, b2, transpose_a=True, cfg=cfg)
            _safe_dealloc(a2, keep=a_3d)
            _safe_dealloc(b2, keep=b_3d)
            return out

        def _tr_bwd(d_out, x_in, st, w1, w2):
            a, sig, sg, sw = st
            d_out_f32 = d_out if d_out.dtype == ttnn.float32 else ttnn.typecast(d_out, ttnn.float32)
            own_df = d_out_f32 is not d_out
            if m_f32 is not None:
                d_out_m = ttnn.multiply(d_out_f32, m_f32)
                if own_df:
                    ttnn.deallocate(d_out_f32)
            else:
                d_out_m = d_out_f32 if own_df else ttnn.clone(d_out_f32)

            dw2 = _mm2d_t_a(d_out_m, sw)
            d_sw = _exact_f32_mm(d_out_m, w2, cfg=cfg)
            ttnn.deallocate(d_out_m)

            d_a = ttnn.multiply(d_sw, sg)
            d_sg = ttnn.multiply(d_sw, a)
            ttnn.deallocate(d_sw)

            one_m_sig = ttnn.rsub(sig, 1.0)
            sg_term = ttnn.multiply(sg, one_m_sig)
            ttnn.deallocate(one_m_sig)
            dsilu = ttnn.add(sig, sg_term)
            ttnn.deallocate(sg_term)
            d_g = ttnn.multiply(d_sg, dsilu)
            ttnn.deallocate(d_sg)
            ttnn.deallocate(dsilu)

            d_h = ttnn.concat([d_a, d_g], dim=-1)
            ttnn.deallocate(d_a)
            ttnn.deallocate(d_g)

            dw1 = _mm2d_t_a(d_h, x_in)
            d_x = _exact_f32_mm(d_h, w1, cfg=cfg)
            ttnn.deallocate(d_h)
            if consume_saved:
                for t in (a, sig, sg, sw):
                    ttnn.deallocate(t)
            return d_x, dw1, dw2

        d_c1, dw_t2_1, dw_t2_2 = _tr_bwd(dc_out, c1_dev, st2, w_t2_1, w_t2_2)
        if consume_saved:
            ttnn.deallocate(c1_dev)
        d_c0, dw_t1_1, dw_t1_2 = _tr_bwd(d_c1, c0_dev, st1, w_t1_1, w_t1_2)
        ttnn.deallocate(d_c1)
        if consume_saved:
            ttnn.deallocate(c0_dev)

        if m_f32 is not None:
            d_c0_m = ttnn.multiply(d_c0, m_f32)
            ttnn.deallocate(d_c0)
        else:
            d_c0_m = d_c0
        dw_cf_pad = _mm2d_t_a(d_c0_m, cf_tt)
        if consume_saved:
            ttnn.deallocate(cf_tt)
        dw_cf = _unpad_cols_dev(dw_cf_pad, self._cf_orig_in)
        if dw_cf is not dw_cf_pad:
            ttnn.deallocate(dw_cf_pad)

        w_cf_dyn = _exact_f32_mm(w_cf_pad, self._p_cf_dyn, cfg=cfg)
        d_cf_dyn_dev = _exact_f32_mm(d_c0_m, w_cf_dyn, cfg=cfg)
        ttnn.deallocate(w_cf_dyn)
        ttnn.deallocate(d_c0_m)

        ds_f32 = ds_i if ds_i.dtype == ttnn.float32 else ttnn.typecast(ds_i, ttnn.float32)
        own_dsf = ds_f32 is not ds_i
        if m_f32 is not None:
            d_s_m = ttnn.multiply(ds_f32, m_f32)
            if own_dsf:
                ttnn.deallocate(ds_f32)
        else:
            d_s_m = ds_f32 if own_dsf else ttnn.clone(ds_f32)
        dw_ir_pad = _mm2d_t_a(d_s_m, ir_tt)
        if consume_saved:
            ttnn.deallocate(ir_tt)
        dw_ir = _unpad_cols_dev(dw_ir_pad, self._ir_orig_in)
        if dw_ir is not dw_ir_pad:
            ttnn.deallocate(dw_ir_pad)

        w_ir_dyn = _exact_f32_mm(w_ir_pad, self._p_ir_dyn, cfg=cfg)
        d_ir_dyn_dev = _exact_f32_mm(d_s_m, w_ir_dyn, cfg=cfg)
        ttnn.deallocate(w_ir_dyn)
        ttnn.deallocate(d_s_m)

        return (dw_ir, dw_cf, dw_t1_1, dw_t1_2, dw_t2_1, dw_t2_2), d_ir_dyn_dev, d_cf_dyn_dev

    def _capture_pair(
        self,
        key: tuple,
        *,
        ir_src,
        cf_src,
        m1_src,
        ds_src,
        dc_src,
        n_tok: int,
        trunk_dtype,
        dev_masters: Mapping[str, Any],
    ) -> bool:
        """Capture paired (fwd, bwd) traces with persistent outside-hole saved activation slots."""
        import ttnn
        from .. import autograd as ag

        tracer = self.tracer
        if tracer is None or not tracer.enabled or not tracer._hole_ready:
            return False

        owned_local: list[Any] = []

        def _mk(t):
            c = ttnn.clone(t)
            ttnn.copy(t, c)
            owned_local.append(c)
            return c

        try:
            with ag.no_grad():
                ir_slot = _mk(ir_src)
                cf_slot = _mk(cf_src)
                m_slot = _mk(m1_src) if m1_src is not None else None
                ds_slot = _mk(ds_src)
                dc_slot = _mk(dc_src)

                # Eager warmup: allocates persistent activation & output slots + warms all kernels/copies.
                (
                    s_w, c_w, c0_w, c1_w, st1_w, st2_w,
                    wir_w, own_wir, wcf_w, own_wcf, mf_w, own_mf,
                ) = self._fwd_device_ops(ir_slot, cf_slot, m_slot, n_tok, trunk_dtype, dev_masters)

                s_i_out = _mk(s_w)
                c_out = _mk(c_w)
                c0_slot = _mk(c0_w)
                c1_slot = _mk(c1_w)
                st1_slots = tuple(_mk(t) for t in st1_w)
                st2_slots = tuple(_mk(t) for t in st2_w)
                wir_slot = _mk(wir_w) if own_wir else None
                wcf_slot = _mk(wcf_w) if own_wcf else None
                mf_slot = _mk(mf_w) if own_mf else None
                for t in (s_w, c_w, c0_w, c1_w, *st1_w, *st2_w):
                    ttnn.deallocate(t)
                if own_wir:
                    ttnn.deallocate(wir_w)
                if own_wcf:
                    ttnn.deallocate(wcf_w)
                if own_mf:
                    ttnn.deallocate(mf_w)

                dws_w, dir_w, dcf_w = self._bwd_device_ops(
                    ds_slot, dc_slot,
                    ir_tt=ir_slot, cf_tt=cf_slot,
                    w_ir_pad=(wir_slot if wir_slot is not None else dev_masters[self._W_KEYS[0]]),
                    w_cf_pad=(wcf_slot if wcf_slot is not None else dev_masters[self._W_KEYS[1]]),
                    c0_dev=c0_slot, c1_dev=c1_slot, st1=st1_slots, st2=st2_slots,
                    m_f32=(mf_slot if mf_slot is not None else m_slot),
                    dev_masters=dev_masters,
                    consume_saved=False,
                )
                dw_out = {k: _mk(g) for k, g in zip(self._W_KEYS, dws_w)}
                d_ir_dyn_out = _mk(dir_w)
                d_cf_dyn_out = _mk(dcf_w)
                for t in (*dws_w, dir_w, dcf_w):
                    ttnn.deallocate(t)
                ttnn.synchronize_device(self.device)

            def _cap_fwd():
                with ag.no_grad():
                    (
                        s_t, c_t, c0_t, c1_t, st1_t, st2_t,
                        wir_t, own_wi, wcf_t, own_wc, mf_t, own_m,
                    ) = self._fwd_device_ops(ir_slot, cf_slot, m_slot, n_tok, trunk_dtype, dev_masters)
                    ttnn.copy(s_t, s_i_out)
                    ttnn.copy(c_t, c_out)
                    ttnn.copy(c0_t, c0_slot)
                    ttnn.copy(c1_t, c1_slot)
                    for src, dst in zip(st1_t, st1_slots):
                        ttnn.copy(src, dst)
                    for src, dst in zip(st2_t, st2_slots):
                        ttnn.copy(src, dst)
                    if own_wi and wir_slot is not None:
                        ttnn.copy(wir_t, wir_slot)
                        ttnn.deallocate(wir_t)
                    if own_wc and wcf_slot is not None:
                        ttnn.copy(wcf_t, wcf_slot)
                        ttnn.deallocate(wcf_t)
                    if own_m and mf_slot is not None:
                        ttnn.copy(mf_t, mf_slot)
                        ttnn.deallocate(mf_t)
                    for t in (s_t, c_t, c0_t, c1_t, *st1_t, *st2_t):
                        ttnn.deallocate(t)

            def _cap_bwd():
                with ag.no_grad():
                    dws_t, dir_t, dcf_t = self._bwd_device_ops(
                        ds_slot, dc_slot,
                        ir_tt=ir_slot, cf_tt=cf_slot,
                        w_ir_pad=(wir_slot if wir_slot is not None else dev_masters[self._W_KEYS[0]]),
                        w_cf_pad=(wcf_slot if wcf_slot is not None else dev_masters[self._W_KEYS[1]]),
                        c0_dev=c0_slot, c1_dev=c1_slot, st1=st1_slots, st2=st2_slots,
                        m_f32=(mf_slot if mf_slot is not None else m_slot),
                        dev_masters=dev_masters,
                        consume_saved=False,
                    )
                    for k, src in zip(self._W_KEYS, dws_t):
                        ttnn.copy(src, dw_out[k])
                        ttnn.deallocate(src)
                    ttnn.copy(dir_t, d_ir_dyn_out)
                    ttnn.deallocate(dir_t)
                    ttnn.copy(dcf_t, d_cf_dyn_out)
                    ttnn.deallocate(dcf_t)

            tids = tracer._capture_runs([_cap_fwd, _cap_bwd])
            tracer._track(*owned_local)
            self._traces[key] = {
                "tid_fwd": tids[0],
                "tid_bwd": tids[1],
                "ir_slot": ir_slot,
                "cf_slot": cf_slot,
                "m_slot": m_slot,
                "s_i_out": s_i_out,
                "c_out": c_out,
                "ds_slot": ds_slot,
                "dc_slot": dc_slot,
                "ds_dtype": str(ds_src.dtype),
                "dc_dtype": str(dc_src.dtype),
                "dw_out": dw_out,
                "d_ir_dyn_out": d_ir_dyn_out,
                "d_cf_dyn_out": d_cf_dyn_out,
            }
            return True
        except Exception as exc:
            for t in owned_local:
                _safe_dealloc(t)
            self._traces[key] = "eager"
            tracer.stats["refused"] += 1
            tracer.refusal_reasons.append(f"win_glue:{type(exc).__name__}:{exc}")
            return False

    def forward(
        self,
        stacked_bw: Mapping[str, Any],
        win_indices: Sequence[int],
        k_local: int,
        *,
        prefetched: Optional[PrefetchedStep],
        m1_tok,
        recycle_remaining: Optional[int],
        trunk_dtype,
        dev_masters: Mapping[str, Any],
    ) -> Tuple[Any, Any, dict]:
        """Run winner ``init_repr`` and ``conditioning`` on device and save compact activations."""
        import ttnn
        from .. import autograd as ag
        from .mesh_dp import put_sharded

        self._ensure_gather_mats(stacked_bw)
        world = len(win_indices)
        mask = stacked_bw["padding_mask"]
        n_tok = int(mask.shape[1])

        if prefetched is not None:
            flat_idx = torch.tensor([r * k_local + int(w) for r, w in enumerate(win_indices)], dtype=torch.long)
            ir_pad = prefetched.ir_static.index_select(0, flat_idx)
            if prefetched.cf_static is not None:
                cf_buf = prefetched.cf_static.index_select(0, flat_idx)
            else:
                cf_row = prefetched.cf_static_2d.index_select(0, flat_idx)
                cf_buf = cf_row.unsqueeze(1).expand(world, n_tok, int(cf_row.shape[-1])).clone()
                cf_buf[:, :, :self.split.cf.total_dim] = self.glue.cond_factory.apply_padding_mask(
                    cf_buf[:, :, :self.split.cf.total_dim], mask,
                )
            if int(ir_pad.shape[-1]) < self._ir_pad_in:
                ir_pad = torch.nn.functional.pad(ir_pad, (0, self._ir_pad_in - int(ir_pad.shape[-1])))
            if int(cf_buf.shape[-1]) < self._cf_pad_in:
                cf_buf = torch.nn.functional.pad(cf_buf, (0, self._cf_pad_in - int(cf_buf.shape[-1])))
        else:
            ir_pad = self.split.ir.build_static_template(stacked_bw, pad_to=self._ir_pad_in)
            cf_buf = self.split.cf.build_static_template(stacked_bw, pad_to=self._cf_pad_in)

        # Live trainable embeddings ONLY (5 tiny nn.Embeddings with autograd enabled, ~2 ms total!).
        ir_parts, ir_dyn_cat = self.split.ir.eval_dynamic(stacked_bw, grad=True)
        cf_parts, cf_dyn_cat = self.split.cf.eval_dynamic(stacked_bw, grad=True)
        with torch.no_grad():
            for feat, c0, c1 in ir_parts:
                ir_pad[:, :, c0:c1] = feat.detach()
            for feat, c0, c1 in cf_parts:
                cf_buf[:, :, c0:c1] = feat.detach()

            # Fill recycle_remaining column(s) at [base_dim : base_dim + rem_dim].
            base_dim = self.split.cf.total_dim
            rem_dim = self._cf_orig_in - base_dim
            if rem_dim > 0:
                if recycle_remaining is not None:
                    rem_creator = self.split.cf.rem_creator
                    rv = rem_creator({**stacked_bw, "recycle_remaining": recycle_remaining}).to(torch.float32)
                    rv = _fc_expand(rv, n_tok)
                    rv = self.glue.cond_factory.apply_padding_mask(rv, mask)
                    cf_buf[:, :, base_dim:base_dim + rem_dim] = rv
                else:
                    cf_buf[:, :, base_dim:base_dim + rem_dim] = 0.0

        with ag.no_grad():
            ir_tt = put_sharded(ir_pad, self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            cf_tt = put_sharded(cf_buf, self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

            tkey = (
                self._trace_key(n_tok, m1_tok, trunk_dtype, dev_masters)
                if (self.tracer is not None and self.tracer.enabled) else None
            )
            tr = self._traces.get(tkey) if tkey is not None else None
            if isinstance(tr, dict):
                ttnn.copy(ir_tt, tr["ir_slot"])
                ttnn.copy(cf_tt, tr["cf_slot"])
                ttnn.deallocate(ir_tt)
                ttnn.deallocate(cf_tt)
                if tr["m_slot"] is not None:
                    ttnn.copy(m1_tok, tr["m_slot"])
                ttnn.execute_trace(self.device, tr["tid_fwd"], cq_id=0, blocking=False)
                self.tracer.stats["win_glue_replay"] += 1
                s_i_dev = ttnn.clone(tr["s_i_out"])
                c_out_dev = ttnn.clone(tr["c_out"])
                saved = {
                    "traced": tr,
                    "ir_dyn_cat": ir_dyn_cat,
                    "cf_dyn_cat": cf_dyn_cat,
                    "world": world,
                    "n_tok": n_tok,
                }
                return s_i_dev, c_out_dev, saved

            want_cap = (
                tkey is not None
                and tr is None
                and self.tracer._hole_ready
            )
            cap_ir = ttnn.clone(ir_tt) if want_cap else None
            cap_cf = ttnn.clone(cf_tt) if want_cap else None
            cap_m1 = (ttnn.clone(m1_tok) if (want_cap and m1_tok is not None) else None)

            (
                s_i_dev, c_out_dev, c0_dev, c1_dev, st1, st2,
                w_ir_pad, own_w_ir, w_cf_pad, own_w_cf, m_f32, m_f32_owned,
            ) = self._fwd_device_ops(ir_tt, cf_tt, m1_tok, n_tok, trunk_dtype, dev_masters)

        saved = {
            "ir_tt": ir_tt,
            "cf_tt": cf_tt,
            "w_ir_pad": w_ir_pad,
            "own_w_ir": own_w_ir,
            "w_cf_pad": w_cf_pad,
            "own_w_cf": own_w_cf,
            "c0_dev": c0_dev,
            "c1_dev": c1_dev,
            "st1": st1,
            "st2": st2,
            "m_f32": m_f32,
            "m_f32_owned": m_f32_owned,
            "ir_dyn_cat": ir_dyn_cat,
            "cf_dyn_cat": cf_dyn_cat,
            "world": world,
            "n_tok": n_tok,
        }
        if want_cap:
            saved["_cap"] = (tkey, cap_ir, cap_cf, cap_m1, trunk_dtype)
        return s_i_dev, c_out_dev, saved

    def backward(
        self,
        ds_i,
        dc_out,
        saved: dict,
        dev_masters: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compute exact per-chip device grads for all 6 weights + backprop the 5 CPU embeddings."""
        import ttnn
        from .. import autograd as ag
        from .mesh_dp import read_sharded

        n_tok = saved["n_tok"]
        world = saved["world"]

        if "traced" in saved:
            tr = saved["traced"]
            with ag.no_grad():
                ttnn.copy(ds_i, tr["ds_slot"])
                ttnn.copy(dc_out, tr["dc_slot"])
                ttnn.execute_trace(self.device, tr["tid_bwd"], cq_id=0, blocking=False)
                d_ir_pad_h = read_sharded(tr["d_ir_dyn_out"], self.device).float().reshape(world, n_tok, -1)
                d_cf_pad_h = read_sharded(tr["d_cf_dyn_out"], self.device).float().reshape(world, n_tok, -1)
                out_grads = {k: ttnn.clone(tr["dw_out"][k]) for k in self._W_KEYS}
        else:
            with ag.no_grad():
                dws, d_ir_dyn_dev, d_cf_dyn_dev = self._bwd_device_ops(
                    ds_i, dc_out,
                    ir_tt=saved["ir_tt"],
                    cf_tt=saved["cf_tt"],
                    w_ir_pad=saved["w_ir_pad"],
                    w_cf_pad=saved["w_cf_pad"],
                    c0_dev=saved["c0_dev"],
                    c1_dev=saved["c1_dev"],
                    st1=saved["st1"],
                    st2=saved["st2"],
                    m_f32=saved["m_f32"],
                    dev_masters=dev_masters,
                    consume_saved=True,
                )
                if saved["own_w_cf"]:
                    ttnn.deallocate(saved["w_cf_pad"])
                if saved["own_w_ir"]:
                    ttnn.deallocate(saved["w_ir_pad"])
                if saved["m_f32_owned"] and saved["m_f32"] is not None:
                    ttnn.deallocate(saved["m_f32"])

            cap_info = saved.pop("_cap", None)
            if cap_info is not None:
                tkey, cap_ir, cap_cf, cap_m1, trunk_dtype = cap_info
                try:
                    self._capture_pair(
                        tkey,
                        ir_src=cap_ir, cf_src=cap_cf, m1_src=cap_m1,
                        ds_src=ds_i, dc_src=dc_out,
                        n_tok=n_tok, trunk_dtype=trunk_dtype, dev_masters=dev_masters,
                    )
                finally:
                    for t in (cap_ir, cap_cf, cap_m1):
                        _safe_dealloc(t)

            d_ir_pad_h = read_sharded(d_ir_dyn_dev, self.device).float().reshape(world, n_tok, -1)
            d_cf_pad_h = read_sharded(d_cf_dyn_dev, self.device).float().reshape(world, n_tok, -1)
            ttnn.deallocate(d_ir_dyn_dev)
            ttnn.deallocate(d_cf_dyn_dev)
            out_grads = dict(zip(self._W_KEYS, dws))

        leaves: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        n_ir_dyn = len(self.split.ir.dynamic_cols)
        n_cf_dyn = len(self.split.cf.dynamic_cols)
        if n_ir_dyn > 0 and saved["ir_dyn_cat"].requires_grad:
            leaves.append(saved["ir_dyn_cat"])
            grads.append(d_ir_pad_h[:, :, :n_ir_dyn].to(saved["ir_dyn_cat"].dtype))
        if n_cf_dyn > 0 and saved["cf_dyn_cat"].requires_grad:
            leaves.append(saved["cf_dyn_cat"])
            grads.append(d_cf_pad_h[:, :, :n_cf_dyn].to(saved["cf_dyn_cat"].dtype))
        if leaves:
            torch.autograd.backward(leaves, grads)

        saved.clear()
        return out_grads

    def release(self) -> None:
        _safe_dealloc(self._p_ir_dyn)
        _safe_dealloc(self._p_cf_dyn)
        self._p_ir_dyn = None
        self._p_cf_dyn = None
        self._traces.clear()


class SPMDMeshTracer:
    """Manages fenced ``MeshTrace`` captures across all static phases of a multi-chip SPMD step.

    All captures share a single ``TraceScratch`` DRAM hole so intermediates written during replay
    can never alias live eager tensors (or saved activations between paired ``win_fwd`` and
    ``win_bwd_fold``). Any capture refusal gracefully marks only that key eager.
    """

    def __init__(
        self,
        device,
        *,
        enabled: bool = True,
        hole_per_bank: int = 1600 << 20,
        trace_win_pair: bool = True,
        trace_muon_ns: bool = True,
    ):
        self.device = device
        self.enabled = bool(enabled)
        self.trace_win_pair = bool(trace_win_pair)
        self.trace_muon_ns = bool(trace_muon_ns)
        self.hole_per_bank = int(hole_per_bank)
        self._hole_ready = not self.trace_win_pair
        self._scratch = None
        self._head_scratch = None
        self._head_hole_per_bank = 64 << 20
        self._tids: list[Any] = []
        self._owned_slots: list[Any] = []
        self._owned_ids: set[int] = set()
        self._refold_state: Any = 0  # 0=unseen, "eager", or tid
        self._untaped_seen: dict[tuple, int] = {}
        self._untaped_traces: dict[tuple, Any] = {}  # key -> "eager" or dict(tid, ...)
        self._win_seen: dict[tuple, int] = {}
        self._win_grad_meta: dict[tuple, Tuple[list[str], list[str]]] = {}
        self._win_traces: dict[tuple, Any] = {}  # key -> "eager" or dict(tid_fwd, tid_bwd, ...)
        self._head_seen: dict[tuple, int] = {}
        self._head_traces: dict[tuple, Any] = {}
        self._ns_seen: dict[tuple, int] = {}
        self._ns_traces: dict[tuple, Any] = {}
        self._inplace_seen: dict[tuple, int] = {}
        self._inplace_traces: dict[tuple, Any] = {}
        self._inplace_slots: dict[tuple, list[Any]] = {}
        self.stats = {
            "refold_replay": 0,
            "untaped_replay": 0,
            "win_pair_replay": 0,
            "win_glue_replay": 0,
            "head_pair_replay": 0,
            "opt_body_replay": 0,
            "ag_unpack_replay": 0,
            "ns_replay": 0,
            "captures": 0,
            "refused": 0,
        }
        self.refusal_reasons: list[str] = []

    def _get_scratch(self):
        if self._scratch is None:
            from ..tenstorrent import TraceScratch
            self._scratch = TraceScratch(self.device)
        return self._scratch

    def _get_head_scratch(self):
        if self._head_scratch is None:
            from ..tenstorrent import TraceScratch
            self._head_scratch = TraceScratch(self.device)
        return self._head_scratch

    def _capture_runs(self, runs: Sequence[Callable[[], Any]], settle: Optional[Callable[[], None]] = None):
        sc = self._get_scratch()
        need = self.hole_per_bank // 2
        size = self.hole_per_bank
        tids = sc.capture(need, size, list(runs), settle=settle)
        if tids is None:
            raise RuntimeError(f"TraceScratch could not reserve {size >> 20} MiB/bank hole")
        self._tids.extend(tids)
        self.stats["captures"] += len(tids)
        return tids

    def _capture_head_runs(self, runs: Sequence[Callable[[], Any]], settle: Optional[Callable[[], None]] = None):
        """Capture ``DeviceHeads`` into its own 64 MiB/bank hole disjoint from ``win_pair``'s hole."""
        sc = self._get_head_scratch()
        need = self._head_hole_per_bank // 2
        size = self._head_hole_per_bank
        tids = sc.capture(need, size, list(runs), settle=settle)
        if tids is None:
            raise RuntimeError(f"Head TraceScratch could not reserve {size >> 20} MiB/bank hole")
        self._tids.extend(tids)
        self.stats["captures"] += len(tids)
        return tids

    def _track(self, *tensors):
        for t in tensors:
            if t is not None:
                self._owned_slots.append(t)
                self._owned_ids.add(id(t))

    # ---- T1: In-place refold + _t1_w refresh ----
    def run_refold(self, dfolds, dev_masters: Mapping[str, Any], dev_hook: Optional[Callable] = None) -> None:
        import ttnn

        if not self.enabled or self._refold_state == "eager":
            dfolds.refold(dev_masters, on_refold=dev_hook, in_place=self.enabled)
            return

        if self._refold_state == 0:
            # Step 0 warm eager pass (populates lazy _t1_w and compiles every refold kernel).
            dfolds.refold(dev_masters, on_refold=dev_hook, in_place=True)
            self._refold_state = 1
            return

        if self._refold_state == 1:
            if not self._hole_ready:
                dfolds.refold(dev_masters, on_refold=dev_hook, in_place=True)
                return
            ttnn.synchronize_device(self.device)
            try:
                tids = self._capture_runs([lambda: dfolds.refold(dev_masters, on_refold=dev_hook, in_place=True)])
                self._refold_state = tids[0]
            except Exception as exc:
                self._refold_state = "eager"
                self.stats["refused"] += 1
                self.refusal_reasons.append(f"refold:{type(exc).__name__}:{exc}")
                dfolds.refold(dev_masters, on_refold=dev_hook, in_place=True)
                return
            ttnn.execute_trace(self.device, self._refold_state, cq_id=0, blocking=False)
            self.stats["refold_replay"] += 1
            return

        ttnn.execute_trace(self.device, self._refold_state, cq_id=0, blocking=False)
        self.stats["refold_replay"] += 1

    # ---- T2 & T3: Untaped trunk forward (cand_trunk & win_pre), factored per recycle ----
    def _run_untaped_pass(
        self,
        tag: str,
        trunk,
        s_init,
        z_init,
        c_one,
        masks: Sequence[Any],
        zd,
        low,
        prev=None,
    ):
        """Run one recycle cycle matching ``Trunk.__call__`` bit-for-bit under MeshTrace."""
        import ttnn
        from .. import tenstorrent as T

        has_prev = prev is not None
        key = (
            tag,
            "pN" if has_prev else "p0",
            tuple(s_init.shape), str(s_init.dtype),
            tuple(z_init.shape), str(z_init.dtype),
            tuple(c_one.shape), str(c_one.dtype),
            str(zd), str(low),
            tuple(m is not None for m in masks),
        )

        def _call(si, zi, ci, mk_a, pr_a):
            with T.lever_scope(trunk._levers):
                zb = None
                if pr_a is None:
                    s_in, z_in = ttnn.clone(si), ttnn.clone(zi)
                else:
                    s_in = trunk._recycle(si, pr_a[0], trunk.s_norm, trunk.s_proj, trunk.residual_dtype)
                    z_rec = trunk._recycle(zi, pr_a[1], trunk.z_norm, trunk.z_proj, zd, low=low)
                    if low is not None:
                        z_in, zb = z_rec
                    else:
                        z_in = z_rec
                return trunk.run_pass(s_in, z_in, ci, mk_a, zb=zb)

        tr = self._untaped_traces.get(key)
        if tr == "eager":
            return _call(s_init, z_init, c_one, masks, prev)

        if isinstance(tr, dict):
            ttnn.copy(s_init, tr["s_in"])
            ttnn.copy(z_init, tr["z_in"])
            ttnn.copy(c_one, tr["c_in"])
            if has_prev:
                ttnn.copy(prev[0], tr["prev_s"])
                ttnn.copy(prev[1], tr["prev_z"])
            for m, slot in zip(masks, tr["mk_in"]):
                if slot is not None:
                    ttnn.copy(m, slot)
            ttnn.execute_trace(self.device, tr["tid"], cq_id=0, blocking=False)
            self.stats["untaped_replay"] += 1
            return ttnn.clone(tr["s_out"]), ttnn.clone(tr["z_out"])

        seen = self._untaped_seen.get(key, 0)
        if seen == 0 or not self._hole_ready:
            self._untaped_seen[key] = 1
            return _call(s_init, z_init, c_one, masks, prev)

        # Call 2: allocate fixed slots outside the hole, run once to size/populate outputs, capture.
        s_in = ttnn.clone(s_init)
        z_in = ttnn.clone(z_init)
        c_in = ttnn.clone(c_one)
        prev_s = ttnn.clone(prev[0]) if has_prev else None
        prev_z = ttnn.clone(prev[1]) if has_prev else None
        prev_in = (prev_s, prev_z) if has_prev else None
        mk_in = [None if m is None else ttnn.clone(m) for m in masks]
        s_w, z_w = _call(s_in, z_in, c_in, mk_in, prev_in)
        s_out = ttnn.clone(s_w)
        z_out = ttnn.clone(z_w)
        # Warm ttnn.copy kernels eagerly for every output shape/dtype before entering trace capture.
        ttnn.copy(s_w, s_out)
        ttnn.copy(z_w, z_out)
        ttnn.synchronize_device(self.device)

        def _cap():
            sc, zc = _call(s_in, z_in, c_in, mk_in, prev_in)
            ttnn.copy(sc, s_out)
            ttnn.copy(zc, z_out)
            ttnn.deallocate(sc)
            ttnn.deallocate(zc)

        try:
            tids = self._capture_runs([_cap])
            self._track(s_in, z_in, c_in, prev_s, prev_z, *mk_in, s_out, z_out)
            self._untaped_traces[key] = dict(
                tid=tids[0], s_in=s_in, z_in=z_in, c_in=c_in,
                prev_s=prev_s, prev_z=prev_z, mk_in=mk_in,
                s_out=s_out, z_out=z_out,
            )
        except Exception as exc:
            for t in (s_in, z_in, c_in, prev_s, prev_z, *mk_in, s_out, z_out):
                _safe_dealloc(t)
            self._untaped_traces[key] = "eager"
            self.stats["refused"] += 1
            self.refusal_reasons.append(f"untaped:{tag}:{key[1]}:{type(exc).__name__}:{exc}")

        return s_w, z_w

    def run_untaped_trunk(
        self,
        tag: str,
        trunk,
        s,
        z,
        cs: Sequence[Any],
        masks: Sequence[Any],
        *,
        return_penultimate: bool = False,
    ):
        import ttnn
        from .. import tenstorrent as T
        from ..rfd4.trunk import _masked_copy

        if not self.enabled:
            if return_penultimate:
                return trunk(s, z, cs, masks, return_penultimate=True)
            return trunk(s, z, cs, masks)

        with T.lever_scope(trunk._levers):
            res_mask = masks[4]
            zd = trunk.pair_dtype(masks)
            res_pair_mask_c = masks[5] if zd == trunk.residual_dtype else masks[3]
            s_init = _masked_copy(s, res_mask, trunk.residual_dtype, alias=True)
            z_init = _masked_copy(z, res_pair_mask_c, zd, alias=True)
            low = trunk.dtype if zd == ttnn.float32 and trunk.dtype == ttnn.bfloat16 else None
            do_final_mask = res_mask is not None and T.lever("mask_at_mix")

        cur_s, cur_z = self._run_untaped_pass(
            tag, trunk, s_init, z_init, cs[0], masks, zd, low, prev=None,
        )
        penult = (None, None)
        last = len(cs) - 1
        for i in range(1, len(cs)):
            nxt_s, nxt_z = self._run_untaped_pass(
                tag, trunk, s_init, z_init, cs[i], masks, zd, low, prev=(cur_s, cur_z),
            )
            if return_penultimate and i == last:
                penult = (cur_s, cur_z)
            else:
                ttnn.deallocate(cur_s)
                ttnn.deallocate(cur_z)
            cur_s, cur_z = nxt_s, nxt_z

        if do_final_mask:
            with T.lever_scope(trunk._levers):
                ms = _masked_copy(cur_s, res_mask, trunk.residual_dtype)
                mz = _masked_copy(cur_z, res_pair_mask_c, zd)
                ttnn.deallocate(cur_s)
                ttnn.deallocate(cur_z)
                cur_s, cur_z = ms, mz
                if penult[0] is not None:
                    pm = masks[5] if penult[1].dtype == trunk.residual_dtype else masks[3]
                    ps = _masked_copy(penult[0], res_mask, penult[0].dtype)
                    pz = _masked_copy(penult[1], pm, penult[1].dtype)
                    ttnn.deallocate(penult[0])
                    ttnn.deallocate(penult[1])
                    penult = (ps, pz)

        for t, given in ((s_init, s), (z_init, z)):
            if t is not given:
                ttnn.deallocate(t)

        if return_penultimate:
            return cur_s, cur_z, penult
        return cur_s, cur_z

    # ---- T4: Paired win_last forward + (tape_bwd + dfolds.tape_backward) ----
    def win_pair_key(self, s_i_val, z_in_val, c_vals: Sequence[Any], m1: Sequence[Any], prev) -> tuple:
        return (
            prev is not None,
            tuple(m is not None for m in m1),
            tuple(s_i_val.shape), str(s_i_val.dtype),
            tuple(z_in_val.shape), str(z_in_val.dtype),
            len(c_vals),
        )

    def has_win_pair(self, key: tuple) -> bool:
        return self.enabled and self.trace_win_pair and isinstance(self._win_traces.get(key), dict)

    def find_win_pair_donor(self, key: tuple) -> Optional[dict]:
        """Return an existing captured ``win_pair`` trace matching shapes/dtypes (different ``prev``)."""
        if not self.enabled or not self.trace_win_pair or self._win_traces.get(key) == "eager":
            return None
        if key in self._win_traces:
            return None
        for other_key, tr in self._win_traces.items():
            if isinstance(tr, dict) and other_key[1:] == key[1:]:
                return tr
        return None

    def _warm_in_hole(
        self,
        runs: Sequence[Callable[[], Any]],
        settle: Optional[Callable[[], None]] = None,
    ) -> None:
        """Compile uncached kernels for a new variant inside the existing ``TraceScratch`` hole."""
        import ttnn

        sc = self._scratch
        if sc is None or sc.hole is None:
            return
        ballast = sc._fill()
        try:
            for t, _ in sc.held:
                ttnn.deallocate(t)
            sc.held = []
            for run in runs:
                run()
            ttnn.synchronize_device(self.device)
        finally:
            if settle is not None:
                settle()
            sc.held += sc._fill()
            for t, _ in ballast:
                ttnn.deallocate(t)

    def record_win_pair_meta(self, key: tuple, grad_names: Sequence[str], idle: Sequence[str]) -> None:
        """Record exact gradient-name and idle sets from an eager ``win_pair`` pass."""
        self._win_grad_meta[key] = (sorted(grad_names), list(idle))

    def capture_win_pair_direct(
        self,
        key: tuple,
        *,
        donor: dict,
        levers,
        winner,
        dfolds,
        dev_masters: Mapping[str, Any],
        s_i_val,
        z_in_val,
        c_vals: Sequence[Any],
        m1: Sequence[Any],
        prev,
    ) -> bool:
        """Capture a second ``win_pair`` variant (e.g. ``prev=True``) directly inside the open hole.

        Reuses all shared persistent I/O & gradient slots from ``donor`` (allocating only
        ``prev_slots`` and any cycle-only gradient slots missing from ``donor``) so multi-depth
        schedules never run a second eager pass outside the ``TraceScratch`` hole and never
        OOM/refuse after the first variant opens the hole.
        """
        import ttnn
        from .. import autograd as ag
        from .. import tenstorrent as T

        if not self.enabled or key in self._win_traces:
            return False

        s_i_slot = donor["s_i"]
        z_in_slot = donor["z_in"]
        c_slots = donor["c_in"]
        m1_slots = donor["m1_in"]
        s_out_slot = donor["s_out"]
        ds_out_slot = donor["ds_out"]
        d_s_i_slot = donor["d_s_i"]
        d_z_in_slot = donor["d_z_in"]
        d_c_slots = donor["d_c"]
        donor_grads = donor["dev_grads"]

        # Populate donor input slots with valid finite values from the current step.
        ttnn.copy(s_i_val, s_i_slot)
        ttnn.copy(z_in_val, z_in_slot)
        for cv, slot in zip(c_vals, c_slots):
            ttnn.copy(cv, slot)
        for mv, slot in zip(m1, m1_slots):
            if slot is not None:
                ttnn.copy(mv, slot)
        prev_slots = None if prev is None else (ttnn.clone(prev[0]), ttnn.clone(prev[1]))
        ttnn.synchronize_device(self.device)

        held: dict[str, Any] = {}
        extra_grad_slots: list[Any] = []

        def _run_fwd():
            with T.lever_scope(levers), ag.tape():
                s_leaf = ag.Tensor(s_i_slot, requires_grad=True)
                z_leaf = ag.Tensor(z_in_slot, requires_grad=True)
                c_leaves = [ag.Tensor(cs, requires_grad=True) for cs in c_slots]
                s_c, z_c = winner(s_leaf, z_leaf, c_leaves, m1_slots, prev=prev_slots)
            ttnn.copy(s_c.value, s_out_slot)
            held.update(
                s_c=s_c, z_c=z_c,
                s_leaf=s_leaf, z_leaf=z_leaf, c_leaves=c_leaves,
            )

        def _settle():
            held.clear()
            ag.release_pins()
            gc.collect()

        # Ensure exact grad_names + idle are known AND all kernels for this variant are compiled.
        meta = self._win_grad_meta.get(key)
        need_warm = (self._win_seen.get(key, 0) == 0) or (meta is None)
        if need_warm:
            probed: dict[str, Any] = {}

            def _warm_bwd_probe():
                s_c = held["s_c"]
                z_c = held["z_c"]
                s_leaf = held["s_leaf"]
                z_leaf = held["z_leaf"]
                c_leaves = held["c_leaves"]
                seed = ttnn.clone(ds_out_slot)
                ag.backward([s_c], [seed])
                ttnn.deallocate(seed)
                _safe_dealloc(getattr(s_c, "value", s_c))
                _safe_dealloc(getattr(z_c, "value", z_c))
                _safe_dealloc(getattr(s_leaf.grad, "value", s_leaf.grad))
                s_leaf.grad = None
                _safe_dealloc(getattr(z_leaf.grad, "value", z_leaf.grad))
                z_leaf.grad = None
                for cl in c_leaves:
                    _safe_dealloc(getattr(cl.grad, "value", cl.grad))
                    cl.grad = None
                dg_c, idle_c = dfolds.tape_backward(dev_masters)
                probed["grad_names"] = sorted(dg_c.keys())
                probed["idle"] = list(idle_c)
                for g in dg_c.values():
                    _safe_dealloc(g)

            try:
                self._warm_in_hole([_run_fwd, _warm_bwd_probe], settle=_settle)
                self._win_seen[key] = 1
                meta = (probed["grad_names"], probed["idle"])
                self._win_grad_meta[key] = meta
            except Exception as exc:
                _settle()
                if prev_slots is not None:
                    for t in prev_slots:
                        _safe_dealloc(t)
                self._win_traces[key] = "eager"
                self.stats["refused"] += 1
                self.refusal_reasons.append(f"win_pair_direct_warm:{type(exc).__name__}:{exc}")
                return False

        grad_names, idle_list = meta
        # Reuse donor slots for shared grads; allocate fresh slots ONLY for cycle-only grads.
        dev_grad_slots: dict[str, Any] = {}
        for nm in grad_names:
            if nm in donor_grads:
                dev_grad_slots[nm] = donor_grads[nm]
            else:
                slot = ttnn.clone(dev_masters[nm])
                ttnn.copy(dev_masters[nm], slot)  # warm ttnn.copy for this shape eagerly
                dev_grad_slots[nm] = slot
                extra_grad_slots.append(slot)
        if extra_grad_slots:
            ttnn.synchronize_device(self.device)

        def _run_bwd_fold():
            s_c = held["s_c"]
            z_c = held["z_c"]
            s_leaf = held["s_leaf"]
            z_leaf = held["z_leaf"]
            c_leaves = held["c_leaves"]
            seed = ttnn.clone(ds_out_slot)
            ag.backward([s_c], [seed])
            ttnn.deallocate(seed)
            _safe_dealloc(getattr(s_c, "value", s_c))
            _safe_dealloc(getattr(z_c, "value", z_c))
            gs = getattr(s_leaf.grad, "value", s_leaf.grad)
            ttnn.copy(gs, d_s_i_slot)
            ttnn.deallocate(gs)
            s_leaf.grad = None
            gz = getattr(z_leaf.grad, "value", z_leaf.grad)
            ttnn.copy(gz, d_z_in_slot)
            ttnn.deallocate(gz)
            z_leaf.grad = None
            for cl, dcs in zip(c_leaves, d_c_slots):
                gc_ = getattr(cl.grad, "value", cl.grad)
                ttnn.copy(gc_, dcs)
                ttnn.deallocate(gc_)
                cl.grad = None
            dg_c, _idle_c = dfolds.tape_backward(dev_masters)
            for nm in grad_names:
                g = dg_c.pop(nm)
                ttnn.copy(g, dev_grad_slots[nm])
                ttnn.deallocate(g)
            for g in dg_c.values():
                _safe_dealloc(g)

        try:
            tids = self._capture_runs([_run_fwd, _run_bwd_fold], settle=_settle)
            if prev_slots is not None:
                self._track(*prev_slots)
            if extra_grad_slots:
                self._track(*extra_grad_slots)
            self._win_traces[key] = dict(
                tid_fwd=tids[0],
                tid_bwd=tids[1],
                s_i=s_i_slot,
                z_in=z_in_slot,
                c_in=c_slots,
                prev_in=prev_slots,
                m1_in=m1_slots,
                s_out=s_out_slot,
                ds_out=ds_out_slot,
                d_s_i=d_s_i_slot,
                d_z_in=d_z_in_slot,
                d_c=d_c_slots,
                dev_grads=dev_grad_slots,
                idle=list(idle_list),
            )
            return True
        except Exception as exc:
            _settle()
            if prev_slots is not None:
                for t in prev_slots:
                    _safe_dealloc(t)
            for t in extra_grad_slots:
                _safe_dealloc(t)
            self._win_traces[key] = "eager"
            self.stats["refused"] += 1
            self.refusal_reasons.append(f"win_pair_direct:{type(exc).__name__}:{exc}")
            return False

    def win_pair_should_capture(self, key: tuple) -> bool:
        if not self.enabled or not self.trace_win_pair or self._win_traces.get(key) == "eager":
            return False
        if key in self._win_traces:
            return False
        seen = self._win_seen.get(key, 0)
        if seen == 0:
            self._win_seen[key] = 1
            return False
        return True

    def replay_win_pair_fwd(self, key: tuple, s_i_val, z_in_val, c_vals: Sequence[Any], m1: Sequence[Any], prev):
        import ttnn

        tr = self._win_traces[key]
        ttnn.copy(s_i_val, tr["s_i"])
        ttnn.copy(z_in_val, tr["z_in"])
        for cv, slot in zip(c_vals, tr["c_in"]):
            ttnn.copy(cv, slot)
        if prev is not None and tr["prev_in"] is not None:
            ttnn.copy(prev[0], tr["prev_in"][0])
            ttnn.copy(prev[1], tr["prev_in"][1])
        for mv, slot in zip(m1, tr["m1_in"]):
            if slot is not None:
                ttnn.copy(mv, slot)
        ttnn.execute_trace(self.device, tr["tid_fwd"], cq_id=0, blocking=False)
        return tr["s_out"]

    def replay_win_pair_bwd_fold(self, key: tuple, ds_out):
        import ttnn

        tr = self._win_traces[key]
        ttnn.copy(ds_out, tr["ds_out"])
        if id(ds_out) not in self._owned_ids:
            ttnn.deallocate(ds_out)
        ttnn.execute_trace(self.device, tr["tid_bwd"], cq_id=0, blocking=False)
        self.stats["win_pair_replay"] += 1
        return tr["d_s_i"], tr["d_z_in"], tr["d_c"], tr["dev_grads"], tr["idle"]

    def capture_win_pair(
        self,
        key: tuple,
        *,
        levers,
        winner,
        dfolds,
        dev_masters: Mapping[str, Any],
        s_i_val,
        z_in_val,
        c_vals: Sequence[Any],
        m1: Sequence[Any],
        prev,
        s_out_ref,
        ds_out_ref,
        d_s_i_ref,
        d_z_in_ref,
        d_c_refs: Sequence[Any],
        trunk_dev_grads_ref: Mapping[str, Any],
        idle_ref: Sequence[Any],
    ) -> bool:
        """Capture paired ``(tid_fwd, tid_bwd)`` sharing one fenced hole with live saved activations."""
        import ttnn
        from .. import autograd as ag
        from .. import tenstorrent as T

        if not self.enabled or key in self._win_traces:
            return False

        # Allocate persistent input/output slots OUTSIDE the TraceScratch hole.
        s_i_slot = ttnn.clone(s_i_val)
        z_in_slot = ttnn.clone(z_in_val)
        c_slots = [ttnn.clone(cv) for cv in c_vals]
        prev_slots = None if prev is None else (ttnn.clone(prev[0]), ttnn.clone(prev[1]))
        m1_slots = [None if mv is None else ttnn.clone(mv) for mv in m1]
        s_out_slot = ttnn.clone(s_out_ref)
        ds_out_slot = ttnn.clone(ds_out_ref)
        d_s_i_slot = ttnn.clone(d_s_i_ref)
        d_z_in_slot = ttnn.clone(d_z_in_ref)
        d_c_slots = [ttnn.clone(dc) for dc in d_c_refs]
        grad_names = sorted(trunk_dev_grads_ref.keys())
        dev_grad_slots = {nm: ttnn.clone(trunk_dev_grads_ref[nm]) for nm in grad_names}
        # Warm ttnn.copy kernels eagerly for every output shape/dtype before trace capture.
        ttnn.copy(s_out_ref, s_out_slot)
        ttnn.copy(d_s_i_ref, d_s_i_slot)
        ttnn.copy(d_z_in_ref, d_z_in_slot)
        for dc, dcs in zip(d_c_refs, d_c_slots):
            ttnn.copy(dc, dcs)
        for nm in grad_names:
            ttnn.copy(trunk_dev_grads_ref[nm], dev_grad_slots[nm])
        ttnn.synchronize_device(self.device)

        held: dict[str, Any] = {}

        def _run_fwd():
            with T.lever_scope(levers), ag.tape():
                s_leaf = ag.Tensor(s_i_slot, requires_grad=True)
                z_leaf = ag.Tensor(z_in_slot, requires_grad=True)
                c_leaves = [ag.Tensor(cs, requires_grad=True) for cs in c_slots]
                s_c, z_c = winner(s_leaf, z_leaf, c_leaves, m1_slots, prev=prev_slots)
            ttnn.copy(s_c.value, s_out_slot)
            held.update(
                s_c=s_c, z_c=z_c,
                s_leaf=s_leaf, z_leaf=z_leaf, c_leaves=c_leaves,
            )

        def _run_bwd_fold():
            s_c = held["s_c"]
            z_c = held["z_c"]
            s_leaf = held["s_leaf"]
            z_leaf = held["z_leaf"]
            c_leaves = held["c_leaves"]
            seed = ttnn.clone(ds_out_slot)
            ag.backward([s_c], [seed])
            ttnn.deallocate(seed)
            _safe_dealloc(getattr(s_c, "value", s_c))
            _safe_dealloc(getattr(z_c, "value", z_c))
            gs = getattr(s_leaf.grad, "value", s_leaf.grad)
            ttnn.copy(gs, d_s_i_slot)
            ttnn.deallocate(gs)
            s_leaf.grad = None
            gz = getattr(z_leaf.grad, "value", z_leaf.grad)
            ttnn.copy(gz, d_z_in_slot)
            ttnn.deallocate(gz)
            z_leaf.grad = None
            for cl, dcs in zip(c_leaves, d_c_slots):
                gc_ = getattr(cl.grad, "value", cl.grad)
                ttnn.copy(gc_, dcs)
                ttnn.deallocate(gc_)
                cl.grad = None
            dg_c, _idle_c = dfolds.tape_backward(dev_masters)
            for nm in grad_names:
                g = dg_c.pop(nm)
                ttnn.copy(g, dev_grad_slots[nm])
                ttnn.deallocate(g)
            for g in dg_c.values():
                _safe_dealloc(g)

        def _settle():
            tape_cm = held.pop("tape_cm", None)
            lev_cm = held.pop("lev_cm", None)
            held.clear()
            if tape_cm is not None:
                try:
                    tape_cm.__exit__(None, None, None)
                except Exception:
                    pass
            if lev_cm is not None:
                try:
                    lev_cm.__exit__(None, None, None)
                except Exception:
                    pass
            ag.release_pins()
            gc.collect()

        try:
            tids = self._capture_runs([_run_fwd, _run_bwd_fold], settle=_settle)
            self._hole_ready = True
            self._track(
                s_i_slot, z_in_slot, *c_slots, *(prev_slots or ()), *m1_slots,
                s_out_slot, ds_out_slot, d_s_i_slot, d_z_in_slot, *d_c_slots,
                *dev_grad_slots.values(),
            )
            self._win_traces[key] = dict(
                tid_fwd=tids[0],
                tid_bwd=tids[1],
                s_i=s_i_slot,
                z_in=z_in_slot,
                c_in=c_slots,
                prev_in=prev_slots,
                m1_in=m1_slots,
                s_out=s_out_slot,
                ds_out=ds_out_slot,
                d_s_i=d_s_i_slot,
                d_z_in=d_z_in_slot,
                d_c=d_c_slots,
                dev_grads=dev_grad_slots,
                idle=list(idle_ref),
            )
            return True
        except Exception as exc:
            _settle()
            for t in (
                s_i_slot, z_in_slot, *c_slots, *(prev_slots or ()), *m1_slots,
                s_out_slot, ds_out_slot, d_s_i_slot, d_z_in_slot, *d_c_slots,
                *dev_grad_slots.values(),
            ):
                _safe_dealloc(t)
            self._hole_ready = True
            self._win_traces[key] = "eager"
            self.stats["refused"] += 1
            self.refusal_reasons.append(f"win_pair:{type(exc).__name__}:{exc}")
            return False

    # ---- T4b: Paired MeshTrace for DeviceHeads (disjoint 64 MiB/bank hole, safe mid-win_pair) ----
    @staticmethod
    def head_pair_key(s_raw, dh, run_seq: bool) -> tuple:
        def _addr(t):
            return t.buffer_address() if hasattr(t, "buffer_address") else id(t)

        w_attrs = ("ca_norm_w", "ca_w", "ll_norm_w", "ll_w", "seq_norm_w", "seq_w", "ins_q_w", "ins_k_w")
        w_addrs = tuple(_addr(getattr(dh, a)) for a in w_attrs if getattr(dh, a, None) is not None)
        return (
            tuple(int(d) for d in s_raw.shape),
            str(s_raw.dtype),
            bool(run_seq),
            w_addrs,
        )

    def has_head_pair(self, key: tuple) -> bool:
        return isinstance(self._head_traces.get(key), dict)

    def head_pair_should_capture(self, key: tuple) -> bool:
        if not self.enabled or not self._hole_ready or self._head_traces.get(key) == "eager":
            return False
        return key not in self._head_traces

    def replay_head_pair_fwd(self, key: tuple, s_raw, emit_ca, emit_ll, pad_m):
        import ttnn

        tr = self._head_traces[key]
        ttnn.copy(s_raw, tr["s_in"])
        ttnn.copy(emit_ca, tr["emit_ca"])
        ttnn.copy(emit_ll, tr["emit_ll"])
        if tr["pad_m"] is not None and pad_m is not None:
            ttnn.copy(pad_m, tr["pad_m"])
        ttnn.execute_trace(self.device, tr["tid_fwd"], cq_id=0, blocking=False)
        self.stats["head_pair_replay"] += 1
        return tr["ca_out"], tr["ll_out"], tr["seq_out"], tr["ins_out"]

    def replay_head_pair_bwd(self, key: tuple, d_ca, d_ll, d_seq, d_ins):
        import ttnn

        tr = self._head_traces[key]
        ttnn.copy(d_ca, tr["d_ca"])
        ttnn.copy(d_ll, tr["d_ll"])
        if tr["d_seq"] is not None:
            ttnn.copy(d_seq, tr["d_seq"])
        ttnn.copy(d_ins, tr["d_ins"])
        ttnn.execute_trace(self.device, tr["tid_bwd"], cq_id=0, blocking=False)
        return tr["ds_out"], tr["head_grads"]

    def capture_head_pair(
        self,
        key: tuple,
        *,
        dh,
        dfolds,
        dev_masters: Mapping[str, Any],
        s_val,
        emit_ca,
        emit_ll,
        pad_m,
        ca_ref,
        ll_ref,
        seq_ref,
        ins_ref,
        d_ca_ref,
        d_ll_ref,
        d_seq_ref,
        d_ins_ref,
        ds_ref,
        head_grads_ref: Mapping[str, Any],
        run_seq: bool,
    ) -> bool:
        """Capture paired (DeviceHeads fwd, bwd+fold) into disjoint 64 MiB/bank ``_head_scratch``."""
        import gc
        import ttnn
        from .. import autograd as ag

        if not self.enabled or not self._hole_ready:
            return False

        owned_local: list[Any] = []

        def _mk(t):
            if t is None:
                return None
            c = ttnn.clone(t)
            ttnn.copy(t, c)
            owned_local.append(c)
            return c

        held: dict[str, Any] = {}

        def _settle():
            held.clear()
            ag.release_pins()
            gc.collect()

        try:
            s_in = _mk(s_val)
            eca_in = _mk(emit_ca)
            ell_in = _mk(emit_ll)
            pm_in = _mk(pad_m) if run_seq else None
            ca_out = _mk(ca_ref)
            ll_out = _mk(ll_ref)
            seq_out = _mk(seq_ref) if run_seq else None
            ins_out = _mk(ins_ref)
            d_ca_in = _mk(d_ca_ref)
            d_ll_in = _mk(d_ll_ref)
            d_seq_in = _mk(d_seq_ref) if run_seq else None
            d_ins_in = _mk(d_ins_ref)
            ds_out = _mk(ds_ref)
            grad_names = sorted(head_grads_ref.keys())
            head_grad_slots = {nm: _mk(head_grads_ref[nm]) for nm in grad_names}
            ttnn.synchronize_device(self.device)

            def _run_fwd():
                with ag.tape():
                    s_leaf = ag.Tensor(s_in, requires_grad=True)
                    o = dh(s_leaf, None, insertion=True, sequence=run_seq, masks_dev=(eca_in, ell_in, pm_in))
                ca_t = next(iter(o["bb_ca"].values()))
                ll_t = next(iter(o["local_latents"].values()))
                seq_t = o["sequence"]["logits"] if run_seq else None
                ins_t = o["insertion"]["logits"]
                ttnn.copy(getattr(ca_t, "value", ca_t), ca_out)
                ttnn.copy(getattr(ll_t, "value", ll_t), ll_out)
                if run_seq:
                    ttnn.copy(getattr(seq_t, "value", seq_t), seq_out)
                ttnn.copy(getattr(ins_t, "value", ins_t), ins_out)
                held.update(s_leaf=s_leaf, ca_t=ca_t, ll_t=ll_t, seq_t=seq_t, ins_t=ins_t)

            def _run_bwd_fold():
                s_leaf = held["s_leaf"]
                outs = [held["ca_t"], held["ll_t"]]
                seeds = [ttnn.clone(d_ca_in), ttnn.clone(d_ll_in)]
                if run_seq:
                    outs.append(held["seq_t"])
                    seeds.append(ttnn.clone(d_seq_in))
                outs.append(held["ins_t"])
                seeds.append(ttnn.clone(d_ins_in))
                ag.backward(outs, seeds)
                for sd in seeds:
                    ttnn.deallocate(sd)
                for ot in outs:
                    _safe_dealloc(getattr(ot, "value", ot))
                gs = getattr(s_leaf.grad, "value", s_leaf.grad)
                ttnn.copy(gs, ds_out)
                ttnn.deallocate(gs)
                s_leaf.grad = None
                dg_c, _ = dfolds.tape_backward(dev_masters)
                for nm in grad_names:
                    g = dg_c.pop(nm)
                    ttnn.copy(g, head_grad_slots[nm])
                    ttnn.deallocate(g)
                for g in dg_c.values():
                    _safe_dealloc(g)

            # Warm kernels + ttnn.copy eagerly once before capture (prevents cold JIT inside capture).
            _run_fwd()
            _run_bwd_fold()
            _settle()
            ttnn.synchronize_device(self.device)

            tids = self._capture_head_runs([_run_fwd, _run_bwd_fold], settle=_settle)
            self._track(*owned_local)
            self._head_traces[key] = {
                "tid_fwd": tids[0],
                "tid_bwd": tids[1],
                "s_in": s_in,
                "emit_ca": eca_in,
                "emit_ll": ell_in,
                "pad_m": pm_in,
                "ca_out": ca_out,
                "ll_out": ll_out,
                "seq_out": seq_out,
                "ins_out": ins_out,
                "d_ca": d_ca_in,
                "d_ll": d_ll_in,
                "d_seq": d_seq_in,
                "d_ins": d_ins_in,
                "ds_out": ds_out,
                "head_grads": head_grad_slots,
            }
            return True
        except Exception as exc:
            _settle()
            for t in owned_local:
                _safe_dealloc(t)
            self._head_traces[key] = "eager"
            self.stats["refused"] += 1
            self.refusal_reasons.append(f"head_pair:{type(exc).__name__}:{exc}")
            return False

    # ---- T5a: Traced batch of sharded Muon Newton-Schulz loops (zero dynamic scalars) ----
    def run_muon_ns_batch(
        self,
        key: tuple,
        u_list: Sequence[Any],
        run_fn: Callable[[Sequence[Any]], list[Any]],
    ) -> list[Any]:
        import ttnn

        if not self.enabled or not self.trace_muon_ns:
            return run_fn(u_list)

        tr = self._ns_traces.get(key)
        if tr == "eager":
            return run_fn(u_list)

        if isinstance(tr, dict):
            for u, slot in zip(u_list, tr["u_in"]):
                ttnn.copy(u, slot)
            ttnn.execute_trace(self.device, tr["tid"], cq_id=0, blocking=False)
            self.stats["ns_replay"] += 1
            return [ttnn.clone(o) for o in tr["o_out"]]

        seen = self._ns_seen.get(key, 0)
        if seen == 0 or not self._hole_ready:
            self._ns_seen[key] = 1
            return run_fn(u_list)

        u_in = [ttnn.clone(u) for u in u_list]
        o_warm = run_fn(u_in)
        o_out = [ttnn.clone(o) for o in o_warm]
        # Warm ttnn.copy kernels for every bucket shape/dtype before trace capture.
        for ow, oo in zip(o_warm, o_out):
            ttnn.copy(ow, oo)
        ttnn.synchronize_device(self.device)

        def _cap():
            os_c = run_fn(u_in)
            for oc, slot in zip(os_c, o_out):
                ttnn.copy(oc, slot)
                ttnn.deallocate(oc)

        try:
            tids = self._capture_runs([_cap])
            self._track(*u_in, *o_out)
            self._ns_traces[key] = dict(tid=tids[0], u_in=u_in, o_out=o_out)
        except Exception as exc:
            for t in (*u_in, *o_out):
                _safe_dealloc(t)
            self._ns_traces[key] = "eager"
            self.stats["refused"] += 1
            self.refusal_reasons.append(f"muon_ns:{type(exc).__name__}:{exc}")
        return o_warm

    # ---- T5b: Generic in-place MeshTrace (SPMDZero1Step body + AG unpack) ----
    def run_in_place_trace(
        self,
        tag: str,
        key: tuple,
        inputs: Sequence[Any],
        body_fn: Callable[[Sequence[Any]], None],
        *,
        extra_owned: Sequence[Any] = (),
        warm_key: Any = None,
        share_slots_key: Any = None,
    ) -> bool:
        """Warm, capture, and replay an in-place MeshTrace mutating persistent device state.

        Semantics guarantee ``body_fn`` executes on hardware EXACTLY ONCE per call (never twice
        on capture — ``begin_trace_capture`` records without executing, then ``execute_trace``
        runs the recorded trace once for the current step). Returns ``True`` when handled (warm,
        capture+execute, or replay) and ``False`` when disabled/refused so caller runs eager.

        Optional ``warm_key`` decouples kernel-warmth tracking from exact ``key`` so variant
        masks (e.g. ``opt_body`` at ``d==1`` vs ``d>=2``) capture on first encounter once the
        tag's kernels are warm. Optional ``share_slots_key`` reuses persistent staging ``slots``
        across non-overlapping variants with identical input tensor signatures (zero extra DRAM).
        """
        import ttnn

        if not self.enabled:
            return False
        full_key = (tag, key)
        w_key = full_key if warm_key is None else (tag, warm_key)
        tr = self._inplace_traces.get(full_key)
        if tr == "eager":
            return False

        if isinstance(tr, dict):
            for src, dst in zip(inputs, tr["slots"]):
                ttnn.copy(src, dst)
            ttnn.execute_trace(self.device, tr["tid"], cq_id=0, blocking=False)
            stat_k = f"{tag}_replay"
            self.stats[stat_k] = self.stats.get(stat_k, 0) + 1
            return True

        seen = self._inplace_seen.get(w_key, 0)
        if seen == 0 or not self._hole_ready:
            # Warm step: executes body_fn ONCE eagerly, compiling all kernels & ttnn.copy shapes.
            body_fn(inputs)
            self._inplace_seen[w_key] = 1
            return True

        s_key = None if share_slots_key is None else (tag, share_slots_key)
        shared_existing = self._inplace_slots.get(s_key) if s_key is not None else None
        if shared_existing is not None:
            slots = shared_existing
            owns_new_slots = False
        else:
            slots = [ttnn.clone(x) for x in inputs]
            owns_new_slots = True
        for x, sl in zip(inputs, slots):
            ttnn.copy(x, sl)
        ttnn.synchronize_device(self.device)

        try:
            tids = self._capture_runs([lambda: body_fn(slots)])
            if owns_new_slots:
                self._track(*slots)
                if s_key is not None:
                    self._inplace_slots[s_key] = slots
            if extra_owned:
                self._track(*extra_owned)
            self._inplace_traces[full_key] = dict(tid=tids[0], slots=slots)
        except Exception as exc:
            if owns_new_slots:
                for t in slots:
                    _safe_dealloc(t)
            self._inplace_traces[full_key] = "eager"
            self.stats["refused"] += 1
            self.refusal_reasons.append(f"{tag}:{type(exc).__name__}:{exc}")
            # Capture failed before hardware ran; apply this step once eagerly.
            body_fn(inputs)
            return True

        ttnn.execute_trace(self.device, tids[0], cq_id=0, blocking=False)
        stat_k = f"{tag}_replay"
        self.stats[stat_k] = self.stats.get(stat_k, 0) + 1
        return True

    def release(self) -> None:
        import ttnn

        for tid in self._tids:
            try:
                ttnn.release_trace(self.device, tid)
            except Exception:
                pass
        self._tids.clear()
        for t in self._owned_slots:
            _safe_dealloc(t)
        self._owned_slots.clear()
        self._owned_ids.clear()
        if self._scratch is not None:
            try:
                self._scratch.release()
            except Exception:
                pass
            self._scratch = None
        if self._head_scratch is not None:
            try:
                self._head_scratch.release()
            except Exception:
                pass
            self._head_scratch = None
        self._refold_state = 0
        self._untaped_traces.clear()
        self._win_traces.clear()
        self._head_traces.clear()
        self._ns_traces.clear()
        self._inplace_traces.clear()
        self._inplace_seen.clear()
        self._inplace_slots.clear()
