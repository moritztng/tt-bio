"""Device-resident optimizer: Muon (fp32/split2 Newton-Schulz) + AdamW + per-group clip + EMA on card.

Every master weight, momentum buffer, Adam moment, and EMA shadow lives on the device in float32
TILE_LAYOUT. ``step(dev_grads)`` consumes ``{name: dev_fp32_grad}`` already on the card (from
:class:`~tt_bio.train.folds.DeviceFolds` and taped device modules), optionally reduces them across
data-parallel ranks via ``reduce_fn(dev_grads)`` on device, clips Muon and Adam groups separately,
runs batched Newton-Schulz and elementwise updates on card, and updates the EMA shadow in place.

Zero parameter-sized or pair-sized tensors cross PCIe during ``step()``: only the two per-group
squared gradient norm scalars (8 bytes) are read back to compute the global clip coefficients.

Precision (D34 / D89):
    Plain bf16 Newton-Schulz is chaotic on near-rank-1 real momentum matrices (D34). Keeping
    X, G, P in float32 and evaluating each matmul as a 2-part bf16 hi/lo split (``ah@bh`` at HiFi4
    + ``ah@bl + al@bh`` at HiFi2, all accumulating into fp32 dest with ``packer_l1_acc=True``)
    reaches Blackhole FPU hardware floor (~4.7e-4/matmul vs ~1.48e-3 for native fp32 TF32
    truncation) and clears the 20-step ``rel_l2 <= 1e-4`` bar against a float64 host reference
    optimizer across all RFD4 weights.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Optional, Tuple

from .optim import Muon

__all__ = ["DeviceEMA", "DeviceOptimizer", "newton_schulz_device", "transpose_exact_f32"]


def transpose_exact_f32(t):
    """Bit-exact last-2-dim transpose of a float32 TILE tensor via uint32 bitcast.

    ``ttnn.transpose`` on float32 truncates mantissa bits (~TF32); bitcasting to uint32 moves
    the raw 32-bit words without touching the FPU.
    """
    import ttnn

    if len(t.shape) < 2:
        return ttnn.clone(t)
    u = ttnn.bitcast(t, ttnn.uint32)
    ut = ttnn.transpose(u, -2, -1)
    return ttnn.bitcast(ut, ttnn.float32)


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


def _mm_sym_split2(ah, al, *, transpose_b: bool, ckc_hi, ckc_lo):
    """Symmetric product ``A @ A^T`` (``transpose_b=True``) or ``G @ G`` (symmetric ``G``, ``transpose_b=False``)
    using 2-part bf16 split: ``ah@ah + (ah@al) + (ah@al)^T`` (2 matmuls instead of 3)."""
    import ttnn

    f32 = ttnn.float32
    p0 = ttnn.matmul(ah, ah, transpose_b=transpose_b, dtype=f32, compute_kernel_config=ckc_hi)
    p1 = ttnn.matmul(ah, al, transpose_b=transpose_b, dtype=f32, compute_kernel_config=ckc_lo)
    p1t = transpose_exact_f32(p1)
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
           this function returns a new device float32 tensor and frees its own intermediates.
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

    x2 = ttnn.multiply(x, x)
    s1 = ttnn.sum(x2, dim=-1, keepdim=True)
    ttnn.deallocate(x2)
    ss = ttnn.sum(s1, dim=-2, keepdim=True)
    ttnn.deallocate(s1)
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


class DeviceEMA:
    """Exponential moving average kept wholly on card in float32 TILE_LAYOUT.

    Matches :class:`~tt_bio.train.optim.EMA` (upstream ``SimpleEMA``, decay 0.999): the first
    ``update`` clones the stepped masters, and every subsequent step computes
    ``shadow <- shadow + (1 - decay) * (master - shadow)`` on card.
    """

    def __init__(self, decay: float = 0.999):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")
        self.decay = float(decay)
        self.n_averaged = 0
        self.shadow: Dict[str, object] = {}

    def update(self, master: Mapping[str, object]) -> None:
        import ttnn

        if self.n_averaged == 0:
            self.shadow = {n: ttnn.clone(v) for n, v in master.items()}
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
        import ttnn

        return {n: ttnn.to_torch(v).float().numpy().astype(np.float32, copy=False)
                for n, v in self.shadow.items()}


def _to_dev_f32(v, device):
    import numpy as np
    import torch
    import ttnn

    if isinstance(v, ttnn.Tensor):
        if v.dtype == ttnn.float32 and v.layout == ttnn.TILE_LAYOUT:
            return ttnn.clone(v)
        t = ttnn.to_layout(v, ttnn.TILE_LAYOUT) if v.layout != ttnn.TILE_LAYOUT else v
        out = ttnn.typecast(t, ttnn.float32) if t.dtype != ttnn.float32 else ttnn.clone(t)
        if t is not v:
            ttnn.deallocate(t)
        return out
    if isinstance(v, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
    else:
        t = v.detach().to(torch.float32).contiguous()
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _sum_sq_scalar_tensor(g):
    """Reduce a device fp32 TILE tensor ``g`` to a single-element device fp32 tensor holding ``sum(g^2)``."""
    import ttnn

    sq = ttnn.multiply(g, g)
    cur = sq
    nd = len(cur.shape)
    for dim in range(nd - 1, -1, -1):
        nxt = ttnn.sum(cur, dim=dim, keepdim=True)
        ttnn.deallocate(cur)
        cur = nxt
    if tuple(int(d) for d in cur.shape) != (1,):
        flat = ttnn.reshape(cur, (1,))
        if flat is not cur:
            ttnn.deallocate(cur)
        cur = flat
    return cur


class DeviceOptimizer:
    """Device-resident Muon + AdamW + per-group clip + EMA optimizer.

    Contract exposed to ``DeviceFolds`` (``r4t-perf``) and device all-reduce (``r4t-scale``):
        * ``self.master``: ``{upstream_name: ttnn.Tensor}`` in float32 TILE_LAYOUT on ``device``
          with each parameter's exact upstream shape.
        * ``self.ema``: optional :class:`DeviceEMA` with ``self.ema.shadow`` in the same layout.
        * ``step(grads, *, reduce_fn=None, disabled=())``: accepts ``{upstream_name: ttnn.Tensor | None}``
          in float32 TILE_LAYOUT on ``device``. If ``reduce_fn`` is given, it is invoked on ``grads``
          before norm/clipping so device all-reduce runs in place on the card gradients.
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

        # Group Muon parameters by canonical wide shape (min(r, c), max(r, c)) for batched NS.
        groups: Dict[Tuple[int, int], list] = {}
        for n in self.master:
            if n in self.muon_names:
                r, c = self.shapes[n]
                flipped = r > c
                key = (c, r) if flipped else (r, c)
                groups.setdefault(key, []).append((n, flipped))
        self._muon_groups = sorted(groups.items())

        self.momentum_buffer: Dict[str, object] = {}
        self.exp_avg: Dict[str, object] = {}
        self.exp_avg_sq: Dict[str, object] = {}
        self.ema = DeviceEMA(ema_decay) if ema_decay is not None else None

    def clip_coef(self, gnorm: float) -> float:
        return (min(1.0, self.clip_norm / gnorm)
                if (self.clip_norm > 0.0 and gnorm > 0.0) else 1.0)

    def _group_norm_sq(self, names, grads: Mapping[str, object], disabled: set) -> float:
        """Compute sum of squared L2 norms of active gradients in ``names`` on card; reads back 1 float."""
        import ttnn

        acc = None
        for n in names:
            if n in disabled:
                continue
            g = grads.get(n)
            if g is None:
                continue
            s = _sum_sq_scalar_tensor(g)
            if acc is None:
                acc = s
            else:
                nxt = ttnn.add(acc, s)
                ttnn.deallocate(acc)
                ttnn.deallocate(s)
                acc = nxt
        if acc is None:
            return 0.0
        val = float(ttnn.to_torch(acc).float().reshape(-1)[0].item())
        ttnn.deallocate(acc)
        return val

    def _step_muon_group(self, m: int, n_dim: int, entries: list, grads: Mapping[str, object],
                         disabled: set, clip_muon: float, lr_muon: float) -> None:
        import ttnn

        mu = self.muon
        one_minus_mu = float(1.0 - mu.momentum)
        active = []
        u_list = []

        for name, flipped in entries:
            if name in disabled:
                continue
            g = grads.get(name)
            if g is None:
                continue
            if clip_muon != 1.0:
                g_eff = ttnn.multiply(g, float(clip_muon))
            else:
                g_eff = g

            buf = self.momentum_buffer.get(name)
            if buf is None:
                buf_new = ttnn.multiply(g_eff, one_minus_mu)
            else:
                diff = ttnn.subtract(g_eff, buf)
                scaled = ttnn.multiply(diff, one_minus_mu)
                ttnn.deallocate(diff)
                buf_new = ttnn.add(buf, scaled)
                ttnn.deallocate(scaled)
                ttnn.deallocate(buf)
            self.momentum_buffer[name] = buf_new

            if mu.nesterov:
                d2 = ttnn.subtract(buf_new, g_eff)
                s2 = ttnn.multiply(d2, one_minus_mu)
                ttnn.deallocate(d2)
                u = ttnn.subtract(buf_new, s2)
                ttnn.deallocate(s2)
            else:
                u = ttnn.clone(buf_new)

            if g_eff is not g:
                ttnn.deallocate(g_eff)

            if flipped:
                ut = transpose_exact_f32(u)
                ttnn.deallocate(u)
                u = ut
            u3 = ttnn.reshape(u, (1, m, n_dim))
            if u3 is not u:
                ttnn.deallocate(u)
            u_list.append(u3)
            active.append((name, flipped))

        if not active:
            return

        if len(u_list) == 1:
            u_batch = u_list[0]
            o_batch = newton_schulz_device(
                u_batch, steps=mu.ns_steps, coeffs=mu.ns_coefficients, eps=mu.eps, mode=self.ns_mode
            )
            ttnn.deallocate(u_batch)
        else:
            u_batch = ttnn.concat(u_list, dim=0)
            for u3 in u_list:
                ttnn.deallocate(u3)
            o_batch = newton_schulz_device(
                u_batch, steps=mu.ns_steps, coeffs=mu.ns_coefficients, eps=mu.eps, mode=self.ns_mode
            )
            ttnn.deallocate(u_batch)

        decay_scale = float(1.0 - lr_muon * mu.weight_decay)
        for idx, (name, flipped) in enumerate(active):
            if len(active) == 1:
                o_slice = o_batch
            else:
                o_slice = ttnn.slice(o_batch, [idx, 0, 0], [idx + 1, m, n_dim])
            o2 = ttnn.reshape(o_slice, (m, n_dim))
            if o2 is not o_slice and o_slice is not o_batch:
                ttnn.deallocate(o_slice)
            if flipped:
                ot = transpose_exact_f32(o2)
                if o2 is not o_batch:
                    ttnn.deallocate(o2)
                o2 = ot
            step_scale = float(-lr_muon * mu.ratio(self.shapes[name]))
            th = self.master[name]
            t1 = ttnn.multiply(th, decay_scale)
            t2 = ttnn.multiply(o2, step_scale)
            if o2 is not o_batch:
                ttnn.deallocate(o2)
            th_new = ttnn.add(t1, t2)
            ttnn.deallocate(t1)
            ttnn.deallocate(t2)
            ttnn.deallocate(th)
            self.master[name] = th_new

        ttnn.deallocate(o_batch)

    def _step_adam_one(self, name: str, g, clip_adam: float, lr: float, bc1: float, bc2: float) -> None:
        import ttnn

        b1, b2 = self.beta1, self.beta2
        om1, om2 = float(1.0 - b1), float(1.0 - b2)
        m = self.exp_avg.get(name)
        v = self.exp_avg_sq.get(name)

        if g is None:
            # Zero-grad rule (optim.py:514-560): decay existing moments and step on momentum.
            if m is None and self.weight_decay == 0.0:
                return
            if m is not None:
                m_new = ttnn.multiply(m, b1)
                v_new = ttnn.multiply(v, b2)
                ttnn.deallocate(m)
                ttnn.deallocate(v)
                self.exp_avg[name] = m_new
                self.exp_avg_sq[name] = v_new
            else:
                m_new = v_new = None
        else:
            g_eff = ttnn.multiply(g, float(clip_adam)) if clip_adam != 1.0 else g
            g2 = ttnn.multiply(g_eff, g_eff)
            if m is None:
                m_new = ttnn.multiply(g_eff, om1)
                v_new = ttnn.multiply(g2, om2)
            else:
                mb = ttnn.multiply(m, b1)
                mg = ttnn.multiply(g_eff, om1)
                m_new = ttnn.add(mb, mg)
                ttnn.deallocate(mb)
                ttnn.deallocate(mg)
                ttnn.deallocate(m)
                vb = ttnn.multiply(v, b2)
                vg = ttnn.multiply(g2, om2)
                v_new = ttnn.add(vb, vg)
                ttnn.deallocate(vb)
                ttnn.deallocate(vg)
                ttnn.deallocate(v)
            ttnn.deallocate(g2)
            if g_eff is not g:
                ttnn.deallocate(g_eff)
            self.exp_avg[name] = m_new
            self.exp_avg_sq[name] = v_new

        th = self.master[name]
        if m_new is None:
            th_new = ttnn.multiply(th, float(1.0 - lr * self.weight_decay))
            ttnn.deallocate(th)
            self.master[name] = th_new
            return

        m_hat = ttnn.multiply(m_new, float(1.0 / bc1))
        v_hat = ttnn.multiply(v_new, float(1.0 / bc2))
        sqv = ttnn.sqrt(v_hat)
        ttnn.deallocate(v_hat)
        den = ttnn.add(sqv, self.eps)
        ttnn.deallocate(sqv)
        ratio = ttnn.divide(m_hat, den)
        ttnn.deallocate(m_hat)
        ttnn.deallocate(den)
        if self.weight_decay != 0.0:
            wd_term = ttnn.multiply(th, self.weight_decay)
            comb = ttnn.add(ratio, wd_term)
            ttnn.deallocate(ratio)
            ttnn.deallocate(wd_term)
            ratio = comb
        upd = ttnn.multiply(ratio, lr)
        ttnn.deallocate(ratio)
        th_new = ttnn.subtract(th, upd)
        ttnn.deallocate(upd)
        ttnn.deallocate(th)
        self.master[name] = th_new

    def step(
        self,
        grads: Mapping[str, object],
        *,
        reduce_fn: Optional[Callable[[Mapping[str, object]], Mapping[str, object]]] = None,
        disabled=(),
        free_grads: bool = False,
    ) -> dict:
        """Run one optimizer step wholly on card.

        Args:
            grads: ``{name: dev_fp32_grad | None}`` on ``self.device``.
            reduce_fn: optional device all-reduce hook (``r4t-scale``) transforming ``grads`` in place.
            disabled: parameter names excluded from gradient norm and update this step.
            free_grads: if True, deallocate every non-None device gradient tensor after consuming it.
        """
        import ttnn

        if reduce_fn is not None:
            grads = reduce_fn(grads)
        dis = set(disabled)

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

        if self.muon_names:
            sq_muon = self._group_norm_sq(self.muon_names, grads, dis)
            sq_adam = self._group_norm_sq(self.adam_names, grads, dis)
            gnorm_muon = math.sqrt(max(sq_muon, 0.0))
            gnorm_adam = math.sqrt(max(sq_adam, 0.0))
            clip_muon = self.clip_coef(gnorm_muon)
            clip_adam = self.clip_coef(gnorm_adam)
            self.last_grad_norm_muon = gnorm_muon
            self.last_grad_norm = gnorm_adam
        else:
            sq_all = self._group_norm_sq(self.adam_names, grads, dis)
            gnorm_adam = math.sqrt(max(sq_all, 0.0))
            gnorm_muon = 0.0
            clip_adam = clip_muon = self.clip_coef(gnorm_adam)
            self.last_grad_norm = gnorm_adam
            self.last_grad_norm_muon = 0.0

        for (m, n_dim), entries in self._muon_groups:
            self._step_muon_group(m, n_dim, entries, grads, dis, clip_muon, lr_muon)

        for name in self.adam_names:
            if name in dis:
                continue
            self._step_adam_one(name, grads.get(name), clip_adam, lr, bc1, bc2)

        if self.ema is not None:
            self.ema.update(self.master)

        if free_grads:
            for g in grads.values():
                if g is not None:
                    ttnn.deallocate(g)

        return {
            "step": self.steps,
            "lr": lr,
            "lr_muon": lr_muon,
            "grad_norm": gnorm_adam,
            "grad_norm_muon": gnorm_muon,
            "clip": clip_adam,
            "clip_muon": clip_muon,
        }

    def host_master(self) -> Dict[str, "np.ndarray"]:
        """Read all fp32 masters back to host numpy (for checkpointing or parity grading, never per step)."""
        import numpy as np
        import ttnn

        return {n: ttnn.to_torch(v).float().numpy().astype(np.float32, copy=False)
                for n, v in self.master.items()}

    def master_digest(self) -> str:
        """SHA-256 prefix of sorted fp32 master bytes (reads masters once; call at end of run, not per step)."""
        import hashlib

        hm = self.host_master()
        h = hashlib.sha256()
        for k in sorted(hm):
            h.update(hm[k].tobytes())
        return h.hexdigest()[:16]
