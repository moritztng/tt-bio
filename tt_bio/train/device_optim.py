"""Device-resident optimizer: bucketed Muon (split2/fp32 Newton-Schulz) + AdamW + clip + EMA on card.

Every master weight, momentum buffer, Adam moment, and EMA shadow lives on the device in float32
TILE_LAYOUT, packed into shape buckets ``(k, ...)`` so all ~558 RFD4 parameters update in ~30
batched kernel groups instead of ~11,000 per-parameter dispatches.

``step(dev_grads)`` consumes ``{name: dev_fp32_grad}`` already on card (from
:class:`~tt_bio.train.folds.DeviceFolds` and taped device modules), optionally reduces them across
data-parallel ranks via ``reduce_fn(dev_grads)`` on device, clips Muon and Adam groups separately,
runs batched Newton-Schulz and elementwise updates on card, and updates the bucketed EMA shadow in
place.

Zero parameter-sized or pair-sized tensors cross PCIe during ``step()``: only one tiny
per-parameter squared gradient norm vector (~2.2 KB for RFD4) is read back to compute the
global clip coefficients and exact dead-gradient set.

Precision (D34 / D89):
    Plain bf16 Newton-Schulz is chaotic on near-rank-1 real momentum matrices (D34). Keeping
    X, G, P in float32 and evaluating each matmul as a 2-part bf16 hi/lo split (``ah@bh`` at HiFi4
    + ``ah@bl + al@bh`` at HiFi2, all accumulating into fp32 dest with ``packer_l1_acc=True``),
    plus exact split-bf16 Frobenius/clip reductions (avoiding ``ttnn.sum``'s 2.7e-4 TF32 unpack
    and masking padded tile tails with exact 0.0), clears the 20-step ``rel_l2 <= 1e-4`` bar
    against a float64 host reference optimizer across all RFD4 weights.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Optional, Tuple

from .optim import Muon

__all__ = [
    "DeviceEMA",
    "DeviceOptimizer",
    "newton_schulz_device",
    "rfd4_device_optimizer",
    "transpose_exact_f32",
]

_ONES_COL_CACHE: Dict[Tuple[int, int], object] = {}
_ONES_ROW_CACHE: Dict[Tuple[int, int], object] = {}


def _safe_dealloc(t, keep=None) -> None:
    """Deallocate ``t`` unless it is None or shares storage with ``keep``."""
    import ttnn

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
    if t.is_allocated():
        ttnn.deallocate(t)


def _clone_exact_f32(t):
    import ttnn

    return ttnn.clone(t)


def _slice_exact_f32(t, beg, fin):
    import ttnn

    return ttnn.slice(t, beg, fin)


def _concat_dim0_exact_f32(ts: list):
    """Bit-exact dim-0 concat of 3-D+ fp32 TILE tensors (chunked for large buckets)."""
    import ttnn

    if len(ts) == 1:
        return _clone_exact_f32(ts[0])
    if len(ts) <= 24:
        return ttnn.concat(ts, dim=0)
    chunks = [_concat_dim0_exact_f32(ts[i:i + 24]) for i in range(0, len(ts), 24)]
    out = _concat_dim0_exact_f32(chunks)
    for c in chunks:
        ttnn.deallocate(c)
    return out


def transpose_exact_f32(t):
    """Bit-exact last-2-dim transpose of a float32 TILE tensor via 3-part bf16 decomposition.

    Direct ``ttnn.transpose`` on float32 (and ``ttnn.bitcast(..., float32)`` on ttnn 0.68.0)
    truncates mantissa bits to TF32. Splitting an IEEE float32 (24 significand bits) into three
    8-bit-significand bf16 slices ``hi + mid + lo``, transposing each losslessly in bf16 TILE
    layout, and summing back in float32 preserves all 24 bits bit-exactly (0.0 max error).
    """
    import ttnn

    if len(t.shape) < 2:
        return _clone_exact_f32(t)
    bf16, f32 = ttnn.bfloat16, ttnn.float32
    hi = ttnn.typecast(t, bf16)
    hi_f = ttnn.typecast(hi, f32)
    r1 = ttnn.subtract(t, hi_f)
    ttnn.deallocate(hi_f)
    mid = ttnn.typecast(r1, bf16)
    mid_f = ttnn.typecast(mid, f32)
    r2 = ttnn.subtract(r1, mid_f)
    ttnn.deallocate(r1)
    ttnn.deallocate(mid_f)
    lo = ttnn.typecast(r2, bf16)
    ttnn.deallocate(r2)

    hi_t = ttnn.transpose(hi, -2, -1)
    ttnn.deallocate(hi)
    hi_tf = ttnn.typecast(hi_t, f32)
    ttnn.deallocate(hi_t)

    mid_t = ttnn.transpose(mid, -2, -1)
    ttnn.deallocate(mid)
    mid_tf = ttnn.typecast(mid_t, f32)
    ttnn.deallocate(mid_t)

    s1 = ttnn.add(hi_tf, mid_tf)
    ttnn.deallocate(hi_tf)
    ttnn.deallocate(mid_tf)

    lo_t = ttnn.transpose(lo, -2, -1)
    ttnn.deallocate(lo)
    lo_tf = ttnn.typecast(lo_t, f32)
    ttnn.deallocate(lo_t)

    out = ttnn.add(s1, lo_tf)
    ttnn.deallocate(s1)
    ttnn.deallocate(lo_tf)
    return out


def _ckc(fidelity):
    import ttnn

    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _split2_bf16(x):
    import ttnn

    hi = ttnn.typecast(x, ttnn.bfloat16)
    hi_f32 = ttnn.typecast(hi, ttnn.float32)
    rem = ttnn.subtract(x, hi_f32)
    ttnn.deallocate(hi_f32)
    lo = ttnn.typecast(rem, ttnn.bfloat16)
    ttnn.deallocate(rem)
    return hi, lo


def _dev_key(device):
    try:
        if hasattr(device, "get_device_ids"):
            return ("mesh", tuple(device.get_device_ids()))
        if hasattr(device, "id"):
            return ("dev", int(device.id()))
    except Exception:
        pass
    return id(device)


def _ones_col_bf16(n: int, device):
    """Cached ``[n, 1]`` bf16 TILE column of ones (padded rows ``n..n_pad-1`` are exact 0.0)."""
    import torch
    import ttnn

    key = (_dev_key(device), int(n))
    hit = _ONES_COL_CACHE.get(key)
    if hit is not None and hit.is_allocated():
        return hit
    t = ttnn.from_torch(
        torch.ones(int(n), 1, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    _ONES_COL_CACHE[key] = t
    return t


def _ones_row_bf16(m: int, device):
    """Cached ``[1, m]`` bf16 TILE row of ones (padded cols ``m..m_pad-1`` are exact 0.0)."""
    import torch
    import ttnn

    key = (_dev_key(device), int(m))
    hit = _ONES_ROW_CACHE.get(key)
    if hit is not None and hit.is_allocated():
        return hit
    t = ttnn.from_torch(
        torch.ones(1, int(m), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    _ONES_ROW_CACHE[key] = t
    return t


def _sum_sq_last2_exact_f32(x):
    """Exact per-slice sum of squares over the last two dims of ``[k, m, n]`` -> ``[k, 1, 1]`` fp32.

    Uses 2-part bf16 matmuls against exact ``ones`` vectors (0.0 in padded tile tails) with
    HiFi4 + fp32_dest_acc: eliminates ``ttnn.sum``'s ~2.7e-4 TF32 unpack truncation AND masks
    out padded rows/cols on non-32-aligned shapes such as ``(12, 384)`` or ``(147, 768)``.
    """
    import ttnn

    dev = x.device()
    m, n = int(x.shape[-2]), int(x.shape[-1])
    ckc = _ckc(ttnn.MathFidelity.HiFi4)
    f32 = ttnn.float32

    x2 = ttnn.square(x)
    hi, lo = _split2_bf16(x2)
    ttnn.deallocate(x2)

    ones_n = _ones_col_bf16(n, dev)
    r_hi = ttnn.matmul(hi, ones_n, dtype=f32, compute_kernel_config=ckc)
    r_lo = ttnn.matmul(lo, ones_n, dtype=f32, compute_kernel_config=ckc)
    ttnn.deallocate(hi)
    ttnn.deallocate(lo)
    row_sum = ttnn.add(r_hi, r_lo)      # [k, m, 1]
    ttnn.deallocate(r_hi)
    ttnn.deallocate(r_lo)

    if m == 1:
        return row_sum

    r2_hi, r2_lo = _split2_bf16(row_sum)
    ttnn.deallocate(row_sum)
    ones_m = _ones_col_bf16(m, dev)
    s_hi = ttnn.matmul(r2_hi, ones_m, transpose_a=True, dtype=f32, compute_kernel_config=ckc)
    s_lo = ttnn.matmul(r2_lo, ones_m, transpose_a=True, dtype=f32, compute_kernel_config=ckc)
    ttnn.deallocate(r2_hi)
    ttnn.deallocate(r2_lo)
    out = ttnn.add(s_hi, s_lo)          # [k, 1, 1]
    ttnn.deallocate(s_hi)
    ttnn.deallocate(s_lo)
    return out


def _mm_sym_split2(ah, al, *, transpose_b: bool, ckc_hi, ckc_lo):
    """Symmetric product ``A @ A^T`` (``transpose_b=True``) or symmetric ``G @ G``
    (``transpose_b=False``) using 2-part bf16 split: ``ah@ah + ah@al + al@ah``.

    Computes the transpose cross-term ``al@ah`` directly as a single matmul instead of
    transposing ``ah@al`` via 3-part bf16 split (saves 15 ops/call x 10 calls/bucket).
    """
    import ttnn

    f32 = ttnn.float32
    p0 = ttnn.matmul(ah, ah, transpose_b=transpose_b, dtype=f32, compute_kernel_config=ckc_hi)
    p1 = ttnn.matmul(ah, al, transpose_b=transpose_b, dtype=f32, compute_kernel_config=ckc_lo)
    p1t = ttnn.matmul(al, ah, transpose_b=transpose_b, dtype=f32, compute_kernel_config=ckc_lo)
    cross = ttnn.add(p1, p1t)
    ttnn.deallocate(p1)
    ttnn.deallocate(p1t)
    out = ttnn.add(p0, cross)
    ttnn.deallocate(p0)
    ttnn.deallocate(cross)
    return out


def _mm_split2(ah, al, bh, bl, *, ckc_hi, ckc_lo):
    """General 2-part bf16 split matmul ``ah@bh + ah@bl + al@bh`` in fp32."""
    import ttnn

    f32 = ttnn.float32
    p0 = ttnn.matmul(ah, bh, dtype=f32, compute_kernel_config=ckc_hi)
    p1 = ttnn.matmul(ah, bl, dtype=f32, compute_kernel_config=ckc_lo)
    s1 = ttnn.add(p0, p1)
    ttnn.deallocate(p0)
    ttnn.deallocate(p1)
    p2 = ttnn.matmul(al, bh, dtype=f32, compute_kernel_config=ckc_lo)
    out = ttnn.add(s1, p2)
    ttnn.deallocate(s1)
    ttnn.deallocate(p2)
    return out


def newton_schulz_device(x, *, steps: int = 5, coeffs: Tuple[float, float, float] = (3.4445, -4.775, 2.0315),
                         eps: float = 1e-7, mode: str = "split2"):
    """Batched Newton-Schulz orthogonalization on card.

    Args:
        x: device float32 TILE tensor of shape ``(k, m, n)`` with ``m <= n``. Not deallocated by caller;
           returns a new device float32 tensor and frees its own intermediates.
        steps: number of quintic Newton-Schulz iterations (default 5).
        coeffs: polynomial coefficients ``(a, b, c)``.
        eps: Frobenius norm clamp floor.
        mode: ``"split2"`` (2-part bf16 split matmul, HiFi4 main + HiFi2 cross-terms, default),
              ``"split2_hifi4"`` (HiFi4 on all 3 terms), or ``"fp32"`` (direct fp32 matmul).
    """
    import ttnn

    if steps >= 100:
        raise ValueError("Muon ns_steps must be below 100")
    a, b, c = (float(v) for v in coeffs)
    f32 = ttnn.float32
    ckc_hi = _ckc(ttnn.MathFidelity.HiFi4)
    ckc_lo = ckc_hi if mode == "split2_hifi4" else _ckc(ttnn.MathFidelity.HiFi2)

    ss = _sum_sq_last2_exact_f32(x)
    nrm = ttnn.sqrt(ss)
    ttnn.deallocate(ss)
    clamped = ttnn.clamp(nrm, min=float(eps))
    ttnn.deallocate(nrm)
    inv = ttnn.reciprocal(clamped)
    ttnn.deallocate(clamped)
    cur = ttnn.multiply(x, inv)
    ttnn.deallocate(inv)

    for _ in range(steps):
        if mode == "fp32":
            g = ttnn.matmul(cur, cur, transpose_b=True, dtype=f32, compute_kernel_config=ckc_hi)
            gg = ttnn.matmul(g, g, dtype=f32, compute_kernel_config=ckc_hi)
            bg = ttnn.multiply(g, b)
            cgg = ttnn.multiply(gg, c)
            ttnn.deallocate(g)
            ttnn.deallocate(gg)
            p = ttnn.add(bg, cgg)
            ttnn.deallocate(bg)
            ttnn.deallocate(cgg)
            px = ttnn.matmul(p, cur, dtype=f32, compute_kernel_config=ckc_hi)
            ttnn.deallocate(p)
        elif mode in ("split2", "split2_hifi4"):
            xh, xl = _split2_bf16(cur)
            g = _mm_sym_split2(xh, xl, transpose_b=True, ckc_hi=ckc_hi, ckc_lo=ckc_lo)
            gh, gl = _split2_bf16(g)
            gg = _mm_sym_split2(gh, gl, transpose_b=False, ckc_hi=ckc_hi, ckc_lo=ckc_lo)
            ttnn.deallocate(gh)
            ttnn.deallocate(gl)
            bg = ttnn.multiply(g, b)
            cgg = ttnn.multiply(gg, c)
            ttnn.deallocate(g)
            ttnn.deallocate(gg)
            p = ttnn.add(bg, cgg)
            ttnn.deallocate(bg)
            ttnn.deallocate(cgg)
            ph, pl = _split2_bf16(p)
            ttnn.deallocate(p)
            px = _mm_split2(ph, pl, xh, xl, ckc_hi=ckc_hi, ckc_lo=ckc_lo)
            ttnn.deallocate(ph)
            ttnn.deallocate(pl)
            ttnn.deallocate(xh)
            ttnn.deallocate(xl)
        else:
            raise ValueError(f"unknown newton_schulz_device mode {mode!r}")

        ax = ttnn.multiply(cur, a)
        nxt = ttnn.add(ax, px)
        ttnn.deallocate(cur)
        ttnn.deallocate(ax)
        ttnn.deallocate(px)
        cur = nxt

    return cur


def _to_dev_f32(v, device):
    import numpy as np
    import torch
    import ttnn

    if isinstance(v, ttnn.Tensor):
        if v.dtype == ttnn.float32 and v.layout == ttnn.TILE_LAYOUT:
            return _clone_exact_f32(v)
        t = ttnn.to_layout(v, ttnn.TILE_LAYOUT) if v.layout != ttnn.TILE_LAYOUT else v
        out = ttnn.typecast(t, ttnn.float32) if t.dtype != ttnn.float32 else _clone_exact_f32(t)
        if t is not v:
            ttnn.deallocate(t)
        return out
    if isinstance(v, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
    else:
        t = v.detach().to(torch.float32).contiguous()
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _canon_3d_shape(shp: Tuple[int, ...]) -> Tuple[int, int, int]:
    """Canonical 3-D ``(1, m, n)`` TILE shape so dim 0 is always tile-aligned for packing."""
    if len(shp) == 1:
        return (1, 1, shp[0])
    if len(shp) == 2:
        return (1, shp[0], shp[1])
    lead = 1
    for d in shp[:-2]:
        lead *= d
    return (lead, shp[-2], shp[-1])


class _Bucket:
    __slots__ = (
        "kind", "shape", "canon", "names", "k",
        "master_b", "state1_b", "state2_b", "ema_b", "_zero1",
    )

    def __init__(self, kind: str, shape: Tuple[int, ...], names: Tuple[str, ...]):
        self.kind = kind          # "muon" or "adam"
        self.shape = shape
        self.canon = _canon_3d_shape(shape)   # (s0, m, n) where s0==1 for 1D/2D
        self.names = names
        self.k = len(names)
        self.master_b = None
        self.state1_b = None      # momentum_buffer (muon) or exp_avg (adam)
        self.state2_b = None      # None (muon) or exp_avg_sq (adam)
        self.ema_b = None
        self._zero1 = None


class DeviceEMA:
    """Exponential moving average kept wholly on card in float32 TILE_LAYOUT.

    Supports both bucketed updates (owned by :class:`DeviceOptimizer`, updating all ~558
    RFD4 parameters in ~30 batched ops/step) and standalone dict updates.
    """

    def __init__(self, decay: float = 0.999, buckets: Optional[list] = None):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")
        self.decay = float(decay)
        self.n_averaged = 0
        self._buckets = buckets
        self.shadow: Dict[str, object] = {}

    def update(self, master: Mapping[str, object] | None = None) -> None:
        import ttnn

        if self._buckets is not None:
            if self.n_averaged == 0:
                for bk in self._buckets:
                    bk.ema_b = _clone_exact_f32(bk.master_b)
            else:
                w = float(1.0 - self.decay)
                for bk in self._buckets:
                    e = bk.ema_b
                    diff = ttnn.subtract(bk.master_b, e)
                    scaled = ttnn.multiply(diff, w)
                    ttnn.deallocate(diff)
                    nxt = ttnn.add(e, scaled)
                    ttnn.deallocate(scaled)
                    ttnn.deallocate(e)
                    bk.ema_b = nxt
            self.n_averaged += 1
            return

        assert master is not None
        if self.n_averaged == 0:
            self.shadow = {n: _clone_exact_f32(v) for n, v in master.items()}
        else:
            w = float(1.0 - self.decay)
            for n, v in master.items():
                e = self.shadow[n]
                diff = ttnn.subtract(v, e)
                scaled = ttnn.multiply(diff, w)
                ttnn.deallocate(diff)
                nxt = ttnn.add(e, scaled)
                ttnn.deallocate(scaled)
                ttnn.deallocate(e)
                self.shadow[n] = nxt
        self.n_averaged += 1

    def host_shadow(self) -> Dict[str, "np.ndarray"]:
        import numpy as np
        from .mesh_dp import read_rep0

        if self._buckets is not None:
            out: Dict[str, np.ndarray] = {}
            for bk in self._buckets:
                if bk.ema_b is None:
                    continue
                arr = read_rep0(bk.ema_b).float().numpy()
                s0, m, n = bk.canon
                for i, name in enumerate(bk.names):
                    sl = arr[i * s0:(i + 1) * s0, :m, :n]
                    out[name] = np.ascontiguousarray(sl.reshape(bk.shape), dtype=np.float32)
            return out
        return {n: read_rep0(v).float().numpy().astype(np.float32, copy=False)
                for n, v in self.shadow.items()}

    def deallocate(self) -> None:
        import ttnn

        if self._buckets is not None:
            for bk in self._buckets:
                if bk.ema_b is not None:
                    if bk.ema_b.is_allocated():
                        ttnn.deallocate(bk.ema_b)
                    bk.ema_b = None
        for v in self.shadow.values():
            if v is not None and v.is_allocated():
                ttnn.deallocate(v)
        self.shadow.clear()


class DeviceOptimizer:
    """Device-resident bucketed Muon + AdamW + per-group clip + EMA optimizer.

    Contract exposed to ``DeviceFolds`` (``r4t-perf``) and device all-reduce (``r4t-scale``):
        * ``self.master``: ``{upstream_name: ttnn.Tensor}`` in float32 TILE_LAYOUT on ``device``
          with each parameter's exact upstream shape.
        * ``self.ema``: optional :class:`DeviceEMA` updated on card in place.
        * ``step(grads, *, reduce_fn=None, disabled=(), free_grads=False)``: accepts
          ``{upstream_name: ttnn.Tensor | None}`` in float32 TILE_LAYOUT on ``device``.
          If ``reduce_fn`` is given, it runs on ``grads`` before norm/clipping so device
          all-reduce stays on card.
    """

    def __init__(
        self,
        initial_weights: Mapping[str, object],
        device,
        *,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_norm: float = 1.0,
        schedule: Optional[Callable[[int], float]] = None,
        muon: Optional[Muon] = None,
        ema_decay: Optional[float] = 0.999,
        ns_mode: str = "split2",
    ):
        import ttnn

        self.device = device
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.clip_norm = float(clip_norm)
        self.schedule = schedule
        self.muon = muon
        self.ns_mode = str(ns_mode)
        self.steps = 0
        self.beta1_pow = 1.0
        self.beta2_pow = 1.0
        self.last_grad_norm = 0.0
        self.last_grad_norm_muon = 0.0
        self.last_dead_grads: frozenset[str] = frozenset()

        self.shapes: Dict[str, Tuple[int, ...]] = {}
        self.master: Dict[str, object] = {}
        for name, val in initial_weights.items():
            dt = _to_dev_f32(val, device)
            self.master[name] = dt
            self.shapes[name] = tuple(int(d) for d in dt.shape)

        self.muon_names = frozenset(
            n for n, shp in self.shapes.items()
            if muon is not None and muon.takes(n, shp)
        )
        self.adam_names = tuple(n for n in self.master if n not in self.muon_names)

        # Build exact-shape buckets for both Muon and Adam.
        muon_by_shp: Dict[Tuple[int, ...], list] = {}
        for n in self.master:
            if n in self.muon_names:
                muon_by_shp.setdefault(self.shapes[n], []).append(n)
        adam_by_shp: Dict[Tuple[int, ...], list] = {}
        for n in self.adam_names:
            adam_by_shp.setdefault(self.shapes[n], []).append(n)

        self._muon_buckets: list[_Bucket] = []
        self._adam_buckets: list[_Bucket] = []

        def pack_bucket(kind: str, shp: Tuple[int, ...], names: list) -> _Bucket:
            bk = _Bucket(kind, shp, tuple(names))
            s0, m, n_dim = bk.canon
            pieces = []
            for nm in bk.names:
                t = self.master[nm]
                p = ttnn.reshape(t, (s0, m, n_dim))
                pieces.append(p)
            bk.master_b = _concat_dim0_exact_f32(pieces)
            for nm, p in zip(bk.names, pieces):
                _safe_dealloc(p, keep=self.master[nm])
            return bk

        for shp, nms in sorted(muon_by_shp.items()):
            self._muon_buckets.append(pack_bucket("muon", shp, nms))
        for shp, nms in sorted(adam_by_shp.items()):
            self._adam_buckets.append(pack_bucket("adam", shp, nms))

        self._all_buckets = self._muon_buckets + self._adam_buckets
        # Backwards-compatible summary attribute (list of ((m, n), [(name, flipped), ...]))
        self._muon_groups = [
            ((min(bk.shape), max(bk.shape)),
             [(nm, bk.shape[0] > bk.shape[1]) for nm in bk.names])
            for bk in self._muon_buckets
        ]
        self.ema = DeviceEMA(ema_decay, buckets=self._all_buckets) if ema_decay is not None else None

    def clip_coef(self, gnorm: float) -> float:
        return (min(1.0, self.clip_norm / gnorm)
                if (self.clip_norm > 0.0 and gnorm > 0.0) else 1.0)

    def _bucket_zero1(self, bk: _Bucket):
        import ttnn

        if bk._zero1 is None or not bk._zero1.is_allocated():
            bk._zero1 = ttnn.zeros(list(bk.canon), dtype=ttnn.float32,
                                   layout=ttnn.TILE_LAYOUT, device=self.device)
        return bk._zero1

    def _pack_grads(self, bk: _Bucket, grads: Mapping[str, object], dis: set):
        """Pack bucket grads into ``[k*s0, m, n]``. Returns ``(g_b, active_flags)`` or ``(None, [])``."""
        import ttnn

        s0, m, n_dim = bk.canon
        flags = []
        any_active = False
        for nm in bk.names:
            ok = (nm not in dis) and (grads.get(nm) is not None)
            flags.append(ok)
            if ok:
                any_active = True
        if not any_active:
            return None, flags

        pieces = []
        owned = []
        z1 = None
        for nm, ok in zip(bk.names, flags):
            if ok:
                g = grads[nm]
                p = ttnn.reshape(g, (s0, m, n_dim))
                pieces.append(p)
                owned.append((p, g))
            else:
                if z1 is None:
                    z1 = self._bucket_zero1(bk)
                pieces.append(z1)
        g_b = _concat_dim0_exact_f32(pieces)
        for p, orig in owned:
            _safe_dealloc(p, keep=orig)
        return g_b, flags

    def _unpack_master(self, bk: _Bucket, *, in_place: bool = False) -> None:
        """Refresh ``self.master[name]`` individual slices from ``bk.master_b``."""
        import ttnn

        s0, m, n_dim = bk.canon
        for i, nm in enumerate(bk.names):
            if bk.k == 1:
                sl = _clone_exact_f32(bk.master_b)
            else:
                sl = _slice_exact_f32(bk.master_b, [i * s0, 0, 0], [(i + 1) * s0, m, n_dim])
            new_m = ttnn.reshape(sl, list(bk.shape))
            _safe_dealloc(sl, keep=new_m)
            old_m = self.master[nm]
            if in_place and old_m is not None:
                if new_m is not old_m:
                    ttnn.copy(new_m, old_m)
                    _safe_dealloc(new_m, keep=old_m)
            else:
                self.master[nm] = new_m
                _safe_dealloc(old_m, keep=new_m)

    def _mask_tensor(self, flags: list, s0: int):
        """Build ``[k*s0, 1, 1]`` float32 active and inactive masks on card (tiny, 4*k bytes)."""
        import torch
        import ttnn

        vals = []
        for f in flags:
            vals.extend([1.0 if f else 0.0] * s0)
        m_act = ttnn.from_torch(
            torch.tensor(vals, dtype=torch.float32).reshape(-1, 1, 1),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        m_inact = ttnn.from_torch(
            torch.tensor([1.0 - v for v in vals], dtype=torch.float32).reshape(-1, 1, 1),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        return m_act, m_inact

    def _group_norms_and_dead(self, muon_packed: list, adam_packed: list) -> tuple[float, float, set[str]]:
        """Compute exact per-group squared L2 norms AND detect missing/zero device gradients.

        Reads back ONE concatenated ``[<=N_params, 1, 1]`` float32 vector (~2.2 KB for RFD4)
        holding exact per-parameter squared gradient norms instead of running per-bucket
        device binary-tree reductions + two scalar readbacks.
        """
        import numpy as np
        import ttnn

        parts = []
        meta = []
        dead: set[str] = set()
        total_slices = 0

        for group_kind, packed in (("muon", muon_packed), ("adam", adam_packed)):
            for bk, g_b, flags in packed:
                if g_b is None:
                    dead.update(bk.names)
                    continue
                sq = _sum_sq_last2_exact_f32(g_b)  # [bk.k * s0, 1, 1] fp32
                parts.append(sq)
                meta.append((group_kind, bk, flags))
                total_slices += bk.k * bk.canon[0]

        if not parts:
            return 0.0, 0.0, dead

        if len(parts) == 1:
            all_sq = parts[0]
        else:
            all_sq = _concat_dim0_exact_f32(parts)
            for p in parts:
                ttnn.deallocate(p)

        from .mesh_dp import read_rep0

        vals = read_rep0(all_sq).float().numpy().reshape(-1)
        ttnn.deallocate(all_sq)
        if vals.size != total_slices:
            raise RuntimeError(
                f"DeviceOptimizer: expected {total_slices} per-param squared norms, got {vals.size}")

        sq_muon = 0.0
        sq_adam = 0.0
        off = 0
        for group_kind, bk, flags in meta:
            s0 = bk.canon[0]
            for i, (nm, ok) in enumerate(zip(bk.names, flags)):
                sl = vals[off + i * s0:off + (i + 1) * s0]
                if not ok:
                    dead.add(nm)
                else:
                    p_sq = float(np.sum(sl, dtype=np.float64))
                    if p_sq == 0.0:
                        dead.add(nm)
                    if group_kind == "muon":
                        sq_muon += p_sq
                    else:
                        sq_adam += p_sq
            off += bk.k * s0
        return sq_muon, sq_adam, dead

    def _step_muon_bucket(self, bk: _Bucket, g_b, flags: list, clip_muon: float, lr_muon: float,
                          *, in_place_masters: bool = False) -> None:
        import ttnn

        if g_b is None:
            return
        mu = self.muon
        one_minus_mu = float(1.0 - mu.momentum)
        all_active = all(flags)

        g_eff = ttnn.multiply(g_b, float(clip_muon)) if clip_muon != 1.0 else g_b

        if bk.state1_b is None:
            buf_new = ttnn.multiply(g_eff, one_minus_mu)
        else:
            diff = ttnn.subtract(g_eff, bk.state1_b)
            scaled = ttnn.multiply(diff, one_minus_mu)
            ttnn.deallocate(diff)
            buf_cand = ttnn.add(bk.state1_b, scaled)
            ttnn.deallocate(scaled)
            if all_active:
                ttnn.deallocate(bk.state1_b)
                buf_new = buf_cand
            else:
                m_act, m_inact = self._mask_tensor(flags, bk.canon[0])
                p1 = ttnn.multiply(buf_cand, m_act)
                p2 = ttnn.multiply(bk.state1_b, m_inact)
                ttnn.deallocate(buf_cand)
                ttnn.deallocate(bk.state1_b)
                buf_new = ttnn.add(p1, p2)
                ttnn.deallocate(p1)
                ttnn.deallocate(p2)
                ttnn.deallocate(m_act)
                ttnn.deallocate(m_inact)
        bk.state1_b = buf_new

        if mu.nesterov:
            d2 = ttnn.subtract(buf_new, g_eff)
            s2 = ttnn.multiply(d2, one_minus_mu)
            ttnn.deallocate(d2)
            u_b = ttnn.subtract(buf_new, s2)
            ttnn.deallocate(s2)
        else:
            u_b = _clone_exact_f32(buf_new)

        if g_eff is not g_b:
            ttnn.deallocate(g_eff)

        flipped = bk.shape[0] > bk.shape[1]
        if flipped:
            u_ns = transpose_exact_f32(u_b)
            ttnn.deallocate(u_b)
        else:
            u_ns = u_b

        o_ns = newton_schulz_device(
            u_ns, steps=mu.ns_steps, coeffs=mu.ns_coefficients, eps=mu.eps, mode=self.ns_mode
        )
        ttnn.deallocate(u_ns)

        if flipped:
            o_b = transpose_exact_f32(o_ns)
            ttnn.deallocate(o_ns)
        else:
            o_b = o_ns

        decay_scale = float(1.0 - lr_muon * mu.weight_decay)
        step_scale = float(-lr_muon * mu.ratio(bk.shape))
        t1 = ttnn.multiply(bk.master_b, decay_scale)
        t2 = ttnn.multiply(o_b, step_scale)
        ttnn.deallocate(o_b)
        cand = ttnn.add(t1, t2)
        ttnn.deallocate(t1)
        ttnn.deallocate(t2)

        if all_active:
            ttnn.deallocate(bk.master_b)
            bk.master_b = cand
        else:
            m_act, m_inact = self._mask_tensor(flags, bk.canon[0])
            p1 = ttnn.multiply(cand, m_act)
            p2 = ttnn.multiply(bk.master_b, m_inact)
            ttnn.deallocate(cand)
            ttnn.deallocate(bk.master_b)
            bk.master_b = ttnn.add(p1, p2)
            ttnn.deallocate(p1)
            ttnn.deallocate(p2)
            ttnn.deallocate(m_act)
            ttnn.deallocate(m_inact)

        self._unpack_master(bk, in_place=in_place_masters)

    def _step_adam_bucket(self, bk: _Bucket, g_b, dis: set,
                          clip_adam: float, lr: float, bc1: float, bc2: float,
                          *, in_place_masters: bool = False) -> None:
        import ttnn

        b1, b2 = self.beta1, self.beta2
        om1, om2 = float(1.0 - b1), float(1.0 - b2)
        enabled = [nm not in dis for nm in bk.names]
        if not any(enabled):
            return
        all_enabled = all(enabled)

        # If every parameter in the bucket has g is None, never had moments, and wd == 0: exact no-op.
        if g_b is None and bk.state1_b is None and self.weight_decay == 0.0:
            return

        if g_b is None:
            if bk.state1_b is not None:
                m_cand = ttnn.multiply(bk.state1_b, b1)
                v_cand = ttnn.multiply(bk.state2_b, b2)
            else:
                m_cand = v_cand = None
        else:
            g_eff = ttnn.multiply(g_b, float(clip_adam)) if clip_adam != 1.0 else g_b
            g2 = ttnn.square(g_eff)
            if bk.state1_b is None:
                m_cand = ttnn.multiply(g_eff, om1)
                v_cand = ttnn.multiply(g2, om2)
            else:
                mb = ttnn.multiply(bk.state1_b, b1)
                mg = ttnn.multiply(g_eff, om1)
                m_cand = ttnn.add(mb, mg)
                ttnn.deallocate(mb)
                ttnn.deallocate(mg)
                vb = ttnn.multiply(bk.state2_b, b2)
                vg = ttnn.multiply(g2, om2)
                v_cand = ttnn.add(vb, vg)
                ttnn.deallocate(vb)
                ttnn.deallocate(vg)
            ttnn.deallocate(g2)
            if g_eff is not g_b:
                ttnn.deallocate(g_eff)

        if m_cand is not None:
            if all_enabled or bk.state1_b is None:
                if bk.state1_b is not None:
                    ttnn.deallocate(bk.state1_b)
                    ttnn.deallocate(bk.state2_b)
                if not all_enabled:
                    m_act, _ = self._mask_tensor(enabled, bk.canon[0])
                    m_new = ttnn.multiply(m_cand, m_act)
                    v_new = ttnn.multiply(v_cand, m_act)
                    ttnn.deallocate(m_cand)
                    ttnn.deallocate(v_cand)
                    ttnn.deallocate(m_act)
                    bk.state1_b, bk.state2_b = m_new, v_new
                else:
                    bk.state1_b, bk.state2_b = m_cand, v_cand
            else:
                m_act, m_inact = self._mask_tensor(enabled, bk.canon[0])
                m1 = ttnn.multiply(m_cand, m_act)
                m2 = ttnn.multiply(bk.state1_b, m_inact)
                v1 = ttnn.multiply(v_cand, m_act)
                v2 = ttnn.multiply(bk.state2_b, m_inact)
                ttnn.deallocate(m_cand)
                ttnn.deallocate(v_cand)
                ttnn.deallocate(bk.state1_b)
                ttnn.deallocate(bk.state2_b)
                bk.state1_b = ttnn.add(m1, m2)
                bk.state2_b = ttnn.add(v1, v2)
                ttnn.deallocate(m1)
                ttnn.deallocate(m2)
                ttnn.deallocate(v1)
                ttnn.deallocate(v2)
                ttnn.deallocate(m_act)
                ttnn.deallocate(m_inact)

        if bk.state1_b is None:
            cand = ttnn.multiply(bk.master_b, float(1.0 - lr * self.weight_decay))
        else:
            m_hat = ttnn.multiply(bk.state1_b, float(1.0 / bc1))
            v_hat = ttnn.multiply(bk.state2_b, float(1.0 / bc2))
            sqv = ttnn.sqrt(v_hat)
            ttnn.deallocate(v_hat)
            den = ttnn.add(sqv, self.eps)
            ttnn.deallocate(sqv)
            ratio = ttnn.divide(m_hat, den)
            ttnn.deallocate(m_hat)
            ttnn.deallocate(den)
            if self.weight_decay != 0.0:
                wd_term = ttnn.multiply(bk.master_b, self.weight_decay)
                comb = ttnn.add(ratio, wd_term)
                ttnn.deallocate(ratio)
                ttnn.deallocate(wd_term)
                ratio = comb
            upd = ttnn.multiply(ratio, lr)
            ttnn.deallocate(ratio)
            cand = ttnn.subtract(bk.master_b, upd)
            ttnn.deallocate(upd)

        if all_enabled:
            ttnn.deallocate(bk.master_b)
            bk.master_b = cand
        else:
            m_act, m_inact = self._mask_tensor(enabled, bk.canon[0])
            p1 = ttnn.multiply(cand, m_act)
            p2 = ttnn.multiply(bk.master_b, m_inact)
            ttnn.deallocate(cand)
            ttnn.deallocate(bk.master_b)
            bk.master_b = ttnn.add(p1, p2)
            ttnn.deallocate(p1)
            ttnn.deallocate(p2)
            ttnn.deallocate(m_act)
            ttnn.deallocate(m_inact)

        self._unpack_master(bk, in_place=in_place_masters)

    def step(
        self,
        grads: Mapping[str, object],
        *,
        reduce_fn: Optional[Callable[[Mapping[str, object]], Mapping[str, object]]] = None,
        disabled=(),
        free_grads: bool = False,
        in_place_masters: bool = False,
        free_skip: Optional[set] = None,
    ) -> dict:
        """Run one optimizer step wholly on card."""
        import ttnn

        orig_grads = grads
        if reduce_fn is not None:
            grads = reduce_fn(grads)
        dis = set(disabled)
        skip = set(free_skip) if free_skip is not None else ()

        lr = self.lr if self.schedule is None else float(self.schedule(self.steps))
        lr_muon = (
            lr if (self.muon is None or self.muon.schedule is None)
            else float(self.muon.schedule(self.steps))
        )
        self.steps += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2
        bc1 = 1.0 - self.beta1_pow
        bc2 = 1.0 - self.beta2_pow

        muon_packed = [(bk, *self._pack_grads(bk, grads, dis)) for bk in self._muon_buckets]
        adam_packed = [(bk, *self._pack_grads(bk, grads, dis)) for bk in self._adam_buckets]

        if free_grads:
            for nm, g in grads.items():
                if g is not None and g.is_allocated() and not (orig_grads is grads and nm in skip):
                    ttnn.deallocate(g)
            if orig_grads is not grads:
                for nm, g in orig_grads.items():
                    if g is not None and g.is_allocated() and nm not in skip:
                        ttnn.deallocate(g)

        sq_muon, sq_adam, dead_grads = self._group_norms_and_dead(muon_packed, adam_packed)
        self.last_dead_grads = frozenset(dead_grads)
        if self._muon_buckets:
            gnorm_muon = math.sqrt(max(sq_muon, 0.0))
            gnorm_adam = math.sqrt(max(sq_adam, 0.0))
            clip_muon = self.clip_coef(gnorm_muon)
            clip_adam = self.clip_coef(gnorm_adam)
            self.last_grad_norm_muon = gnorm_muon
            self.last_grad_norm = gnorm_adam
        else:
            gnorm_adam = math.sqrt(max(sq_adam, 0.0))
            gnorm_muon = 0.0
            clip_adam = clip_muon = self.clip_coef(gnorm_adam)
            self.last_grad_norm = gnorm_adam
            self.last_grad_norm_muon = 0.0

        for bk, g_b, flags in muon_packed:
            self._step_muon_bucket(bk, g_b, flags, clip_muon, lr_muon, in_place_masters=in_place_masters)
            if g_b is not None:
                ttnn.deallocate(g_b)

        for bk, g_b, _flags in adam_packed:
            self._step_adam_bucket(bk, g_b, dis, clip_adam, lr, bc1, bc2, in_place_masters=in_place_masters)
            if g_b is not None:
                ttnn.deallocate(g_b)

        if self.ema is not None:
            self.ema.update()

        return {
            "step": self.steps,
            "lr": lr,
            "lr_muon": lr_muon,
            "grad_norm": gnorm_adam,
            "grad_norm_muon": gnorm_muon,
            "clip": clip_adam,
            "clip_muon": clip_muon,
            "dead_grads": dead_grads,
        }

    def _unpack_bucket_host(self, bk: _Bucket, dev_t) -> Dict[str, "np.ndarray"]:
        import numpy as np
        from .mesh_dp import read_rep0

        if dev_t is None:
            return {nm: np.zeros(bk.shape, dtype=np.float32) for nm in bk.names}
        arr = read_rep0(dev_t).float().numpy()
        s0, m, n = bk.canon
        out: Dict[str, np.ndarray] = {}
        for i, name in enumerate(bk.names):
            sl = arr[i * s0:(i + 1) * s0, :m, :n]
            out[name] = np.ascontiguousarray(sl.reshape(bk.shape), dtype=np.float32)
        return out

    def _pack_host_arrays(self, bk: _Bucket, arrays: Mapping[str, object], *, fill_zero: bool = False):
        import numpy as np
        import torch
        import ttnn

        s0, m, n_dim = bk.canon
        zero = np.zeros((s0, m, n_dim), dtype=np.float32) if fill_zero else None
        pieces = []
        for nm in bk.names:
            if nm in arrays:
                pieces.append(np.asarray(arrays[nm], dtype=np.float32).reshape(s0, m, n_dim))
            elif fill_zero:
                pieces.append(zero)
            else:
                raise KeyError(nm)
        stacked = np.concatenate(pieces, axis=0)
        t = torch.from_numpy(np.ascontiguousarray(stacked, dtype=np.float32))
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)

    def host_master(self) -> Dict[str, "np.ndarray"]:
        """Read all fp32 masters back to host numpy (checkpointing / parity grading, never per step)."""
        out: Dict[str, "np.ndarray"] = {}
        for bk in self._all_buckets:
            out.update(self._unpack_bucket_host(bk, bk.master_b))
        return out

    def state_families(self) -> Dict[str, Dict[str, "np.ndarray"]]:
        """Per-parameter host fp32 state dicts matching :meth:`AdamW.state_families` for checkpoints."""
        exp_avg: Dict[str, "np.ndarray"] = {}
        exp_avg_sq: Dict[str, "np.ndarray"] = {}
        for bk in self._adam_buckets:
            exp_avg.update(self._unpack_bucket_host(bk, bk.state1_b))
            exp_avg_sq.update(self._unpack_bucket_host(bk, bk.state2_b))
        fams: Dict[str, Dict[str, "np.ndarray"]] = {
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
        }
        if self._muon_buckets:
            mom: Dict[str, "np.ndarray"] = {}
            for bk in self._muon_buckets:
                mom.update(self._unpack_bucket_host(bk, bk.state1_b))
            fams["momentum_buffer"] = mom
        if self.ema is not None and self.ema.n_averaged > 0:
            fams["ema_shadow"] = self.ema.host_shadow()
        return fams

    def state_dict(self) -> dict:
        d = {
            "steps": self.steps,
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "clip_norm": self.clip_norm,
            "beta1_pow": self.beta1_pow,
            "beta2_pow": self.beta2_pow,
        }
        if self.ema is not None:
            d["ema_decay"] = self.ema.decay
            d["ema_n_averaged"] = self.ema.n_averaged
        return d

    def load_state_dict(self, d: dict) -> None:
        for k in ("steps", "lr", "beta1", "beta2", "eps", "weight_decay", "clip_norm",
                  "beta1_pow", "beta2_pow"):
            if k in d:
                setattr(self, k, d[k])
        if self.ema is not None and "ema_n_averaged" in d:
            self.ema.n_averaged = int(d["ema_n_averaged"])

    def load_host_state(
        self,
        masters: Mapping[str, object],
        families: Mapping[str, Mapping[str, object]] = (),
    ) -> None:
        """Upload checkpoint masters, optimizer moments, and EMA shadow back onto device."""
        fams = dict(families) if families else {}
        have, want = set(masters), set(self.master)
        if have != want:
            raise ValueError(
                f"DeviceOptimizer.load_host_state mismatch: missing {sorted(want - have)}, "
                f"unexpected {sorted(have - want)}")
        for bk in self._all_buckets:
            new_m = self._pack_host_arrays(bk, masters)
            _safe_dealloc(bk.master_b, keep=new_m)
            bk.master_b = new_m
            self._unpack_master(bk)
            if bk.kind == "muon":
                mom = fams.get("momentum_buffer")
                if mom and any(nm in mom for nm in bk.names):
                    new_s1 = self._pack_host_arrays(bk, mom, fill_zero=True)
                    _safe_dealloc(bk.state1_b, keep=new_s1)
                    bk.state1_b = new_s1
                else:
                    _safe_dealloc(bk.state1_b)
                    bk.state1_b = None
            else:
                ea = fams.get("exp_avg")
                if ea and any(nm in ea for nm in bk.names):
                    new_s1 = self._pack_host_arrays(bk, ea, fill_zero=True)
                    _safe_dealloc(bk.state1_b, keep=new_s1)
                    bk.state1_b = new_s1
                else:
                    _safe_dealloc(bk.state1_b)
                    bk.state1_b = None
                eas = fams.get("exp_avg_sq")
                if eas and any(nm in eas for nm in bk.names):
                    new_s2 = self._pack_host_arrays(bk, eas, fill_zero=True)
                    _safe_dealloc(bk.state2_b, keep=new_s2)
                    bk.state2_b = new_s2
                else:
                    _safe_dealloc(bk.state2_b)
                    bk.state2_b = None
            if self.ema is not None:
                es = fams.get("ema_shadow")
                if es and any(nm in es for nm in bk.names):
                    new_e = self._pack_host_arrays(bk, es, fill_zero=True)
                    _safe_dealloc(bk.ema_b, keep=new_e)
                    bk.ema_b = new_e
                else:
                    _safe_dealloc(bk.ema_b)
                    bk.ema_b = None

    def drop_unowned(self) -> None:
        """No-op for unsharded DeviceOptimizer; matches :class:`AdamW` checkpoint contract."""
        return

    def master_digest(self) -> str:
        """SHA-256 prefix of sorted fp32 master bytes."""
        import hashlib

        hm = self.host_master()
        h = hashlib.sha256()
        for k in sorted(hm):
            h.update(hm[k].tobytes())
        return h.hexdigest()[:16]

    def deallocate(self) -> None:
        """Free all device tensors owned by this optimizer (masters, buckets, moments, EMA)."""
        import ttnn

        if self.ema is not None:
            self.ema.deallocate()
        for t in self.master.values():
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
        self.master.clear()
        for bk in self._all_buckets:
            for attr in ("master_b", "state1_b", "state2_b", "ema_b", "_zero1"):
                t = getattr(bk, attr, None)
                if t is not None and t.is_allocated():
                    ttnn.deallocate(t)
                setattr(bk, attr, None)


def rfd4_device_optimizer(
    initial_weights: Mapping[str, object],
    device,
    *,
    lr: float = 1e-4,
    ema_decay: Optional[float] = 0.999,
    ns_mode: str = "split2",
    ns_steps: int = 5,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> DeviceOptimizer:
    """Construct :class:`DeviceOptimizer` matching :func:`tt_bio.train.rfd4_objective.update_rule`."""
    from .rfd4_objective import MUON_EXCLUDE, UPDATE_RULE

    return DeviceOptimizer(
        initial_weights,
        device,
        lr=lr,
        betas=betas,
        eps=1e-8,
        weight_decay=0.0,
        clip_norm=float(UPDATE_RULE["clip"]),
        muon=Muon(exclude=MUON_EXCLUDE, adjust_lr_fn="match_rms_adamw", ns_steps=ns_steps),
        ema_decay=ema_decay,
        ns_mode=ns_mode,
    )
