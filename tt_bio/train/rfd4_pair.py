"""Training-side taped wrapper for RFD4 PairOnCard (r4t-perf P2).

Lives under `tt_bio/train/` so inference module `tt_bio/rfd4/pair.py` never imports `autograd`.
"""
from __future__ import annotations

import torch
import ttnn

from .. import autograd as ag
from ..rfd4.pair import PairInputs, PairOnCard


def pair_taped(pc: PairOnCard, p: PairInputs, bridge, batch: dict, dtype=None):
    """The training winner's z_init: the program `pc.__call__` runs, taped. Its backward stays on the card
    and hands the builder's projection and norm weights and the AdaLN's gamma and beta their gradients, so
    neither the pair nor its gradient crosses PCIe and the host never runs a pair-sized backward.

        n1 = x r1,  a = n1 w,  b = a r2,  z = b gamma + beta
        dgamma = sum g b,  dbeta = sum g,  da = r2 (g gamma - b mean(g gamma b)),  dw = sum da n1
        dx = r1 (da w - n1 mean(da w n1)),  dW = dx^T F
    """
    b = int(p.bins.shape[0])
    assert b == 1, "one taped winner a step"
    z = pc(p, 0)
    if dtype is not None and z.dtype != dtype:
        z32, z = z, ttnn.typecast(z, dtype)
        ttnn.deallocate(z32)
    fac = pc.split.builder.init_repr_factory
    W, lnw = fac.linear_out.weight, fac.ln_out.weight
    n, c = int(p.bins.shape[1]), int(W.shape[0])
    leaves = [bridge.up(t, ttnn.float32) for t in (W, lnw.view(1, c), p.gamma.reshape(1, c), p.beta.reshape(1, c))]
    if not any(t.requires_grad for t in leaves):
        return ag.Tensor(z)
    e1, e2 = float(fac.ln_out.eps), float(pc.split.builder.adaln.norm.eps)
    feats = pc.split.features(batch).reshape(n * n, -1)
    k = int(feats.shape[-1])
    kp = -(-k // 32) * 32
    F = ttnn.from_torch(torch.nn.functional.pad(feats, [0, kp - k]).contiguous(), dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT, device=pc.device)
    const = pc._const[2]
    bins = p.bins
    row = lambda v: ttnn.from_torch(v.detach().float().reshape(1, 1, 1, c), dtype=ttnn.float32,
                                    layout=ttnn.TILE_LAYOUT, device=pc.device)
    w_tt, g_tt = row(lnw), row(p.gamma)

    def make():
        def bw(g):
            cfg = ag.precise_config()
            mean = lambda v: ag._row_mean(v, cfg, c)
            if g.dtype != ttnn.float32:
                g = ttnn.typecast(g, ttnn.float32)
            x = pc._x(const, bins)
            r1 = ttnn.rsqrt(ttnn.add(mean(ttnn.multiply(x, x)), e1))
            n1 = ttnn.multiply(x, r1)
            ttnn.deallocate(x)
            a = ttnn.multiply(n1, w_tt)
            r2 = ttnn.rsqrt(ttnn.add(mean(ttnn.multiply(a, a)), e2))
            bb = ttnn.multiply(a, r2)
            ttnn.deallocate(a)
            Wl, Ll, Gl, Bl = leaves
            if Bl.requires_grad:
                Bl.add_grad(ag._sum_leading(g, Bl.value.shape))
            if Gl.requires_grad:
                Gl.add_grad(ag._sum_leading(ttnn.multiply(g, bb), Gl.value.shape))
            db = ttnn.multiply(g, g_tt)
            da = ttnn.multiply(ttnn.subtract(db, ttnn.multiply(bb, mean(ttnn.multiply(db, bb)))), r2)
            for t in (db, bb, r2):
                ttnn.deallocate(t)
            if Ll.requires_grad:
                Ll.add_grad(ag._sum_leading(ttnn.multiply(da, n1), Ll.value.shape))
            if Wl.requires_grad:
                dn1 = ttnn.multiply(da, w_tt)
                dx = ttnn.multiply(ttnn.subtract(dn1, ttnn.multiply(n1, mean(ttnn.multiply(dn1, n1)))), r1)
                dW = ttnn.matmul(ttnn.reshape(dx, [n * n, c]), F, transpose_a=True, dtype=ttnn.float32,
                                 compute_kernel_config=cfg)
                Wl.add_grad(ttnn.slice(dW, [0, 0], [c, k]) if kp != k else dW)
                for t in (dn1, dx):
                    ttnn.deallocate(t)
            for t in (da, n1, r1, F, w_tt, g_tt):
                ttnn.deallocate(t)
        return bw

    return ag._tape(z, leaves, make, reads=())


def _ckc_hifi4():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _ckc_hifi2():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _split2_bf16(x):
    hi = ttnn.typecast(x, ttnn.bfloat16)
    hi_f = ttnn.typecast(hi, ttnn.float32)
    rem = ttnn.subtract(x, hi_f)
    ttnn.deallocate(hi_f)
    lo = ttnn.typecast(rem, ttnn.bfloat16)
    ttnn.deallocate(rem)
    return hi, lo


def _split3_bf16(x):
    bf16, f32 = ttnn.bfloat16, ttnn.float32
    hi = ttnn.typecast(x, bf16)
    hi_f = ttnn.typecast(hi, f32)
    r1 = ttnn.subtract(x, hi_f)
    ttnn.deallocate(hi_f)
    mid = ttnn.typecast(r1, bf16)
    mid_f = ttnn.typecast(mid, f32)
    r2 = ttnn.subtract(r1, mid_f)
    ttnn.deallocate(r1)
    ttnn.deallocate(mid_f)
    lo = ttnn.typecast(r2, bf16)
    ttnn.deallocate(r2)
    return hi, mid, lo


class DeviceTrainPair:
    """Device-resident training pair builder (r4t-perf P3): zero pair-sized PCIe in fwd + bwd.

    Lives purely under ``tt_bio/train/`` so inference ``tt_bio/rfd4/pair.py`` stays byte-identical.
    Each step:
      1. Uploads compact ``F_const`` (``[n*n, 96]`` bf16 = 12.6 MB; all 9 constant pair features
         are exact in ``{-1, 0, +1}``) and ``bins`` (``[K*n, n]`` uint32 = 1.05 MB) ONCE.
      2. Reads ``W`` (``[384, 107]``) and ``ln_out.weight`` (``[384]``) directly from
         ``opt.master`` on card (0 host sync, 0 PairSplit/PairOnCard rebuild).
      3. Computes ``proj = F_const @ W_const^T`` via 3-part exact bf16 matmul on card (~1-2 ms,
         ~1e-7 rel_l2 vs float64), gathers ``x = proj + W_step[bins]`` via ``tt_bio.pair_build``
         (``unit=True``, exact fp32), and runs both RMSNorms on card using split2 dot products
         (immune to ``ttnn.sum``'s 2.7e-4 TF32 unpack bug).
      4. In winner backward, computes ``dW_const = dx^T @ F_const`` and
         ``dW_step = dx^T @ onehot(bins_win)`` directly on card (0 MB ``F`` upload) and writes
         ``dW`` and ``d_lnw`` directly into ``self.grads`` on card for ``DeviceOptimizer``.
    """

    W_KEY = "glue.pair_repr_builder.init_repr_factory.linear_out.weight"
    LNW_KEY = "glue.pair_repr_builder.init_repr_factory.ln_out.weight"

    def __init__(self, builder, device):
        from ..rfd4.pair import STEP_FEATURE

        self.builder = builder
        self.device = device
        factory = builder.init_repr_factory
        self.c = int(factory.linear_out.weight.shape[0])
        self.total_dim = int(factory.linear_out.weight.shape[1])
        self.e1 = float(factory.ln_out.eps)
        self.e2 = float(builder.adaln.norm.eps)

        self.const_creators = []
        const_cols = []
        step_info = []
        start = 0
        for cr in factory.feat_creators:
            stop = start + cr.get_dim()
            if type(cr).__name__ == STEP_FEATURE:
                step_info.append((cr, start, stop))
            else:
                self.const_creators.append(cr)
                const_cols.extend(range(start, stop))
            start = stop
        if len(step_info) != 1:
            raise ValueError(f"expected one {STEP_FEATURE}, found {len(step_info)}")
        self.step_creator, self.step_a, self.step_b = step_info[0]
        self.nbins = self.step_b - self.step_a
        self.const_dim = len(const_cols)
        assert const_cols == list(range(self.step_b, self.total_dim)) or const_cols == list(range(0, self.step_a)), (
            "DeviceTrainPair expects contiguous const feature columns"
        )
        self.const_a = const_cols[0]
        self.const_b = const_cols[-1] + 1
        self.const_kpad = -(-self.const_dim // 32) * 32
        self.nbins_pad = -(-self.nbins // 32) * 32

        self._ones_c = ttnn.from_torch(
            torch.ones(self.c, 1, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        eye = torch.zeros(self.nbins, self.nbins_pad, dtype=torch.bfloat16)
        eye[torch.arange(self.nbins), torch.arange(self.nbins)] = 1.0
        self._eye_nb = ttnn.from_torch(
            eye, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
        )
        self._unit_cache: dict[tuple[int, int], tuple] = {}
        self._step_state = None
        self.grads: dict[str, object] = {}

    def _unit_tensors(self, b: int, n: int):
        key = (b, n)
        hit = self._unit_cache.get(key)
        if hit is not None and all(t.is_allocated() for t in hit):
            return hit
        ones_s = ttnn.ones([b * n, n], dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        ones_wg = ttnn.ones([b, 32, self.c], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        zeros_b = ttnn.zeros([b, 32, self.c], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        hit = (ones_s, ones_wg, zeros_b)
        self._unit_cache[key] = hit
        return hit

    def _exact_row_mean(self, v):
        """Exact fp32 row mean over last dim ``c`` via 2-D split2 bf16 matmul (immune to TF32 sum)."""
        ckc = _ckc_hifi2()
        shp = [int(d) for d in v.shape]
        rows = 1
        for d in shp[:-1]:
            rows *= d
        v2d = ttnn.reshape(v, [rows, shp[-1]]) if len(shp) > 2 else v
        hi, lo = _split2_bf16(v2d)
        if v2d is not v and v2d.buffer_address() != v.buffer_address():
            ttnn.deallocate(v2d)
        s0 = ttnn.matmul(hi, self._ones_c, dtype=ttnn.float32, compute_kernel_config=ckc)
        s1 = ttnn.matmul(lo, self._ones_c, dtype=ttnn.float32, compute_kernel_config=ckc)
        ttnn.deallocate(hi)
        ttnn.deallocate(lo)
        s = ttnn.add(s0, s1)
        ttnn.deallocate(s0)
        ttnn.deallocate(s1)
        m2d = ttnn.multiply(s, 1.0 / float(self.c))
        ttnn.deallocate(s)
        if len(shp) > 2:
            out_shp = shp[:-1] + [1]
            out = ttnn.reshape(m2d, out_shp)
            if out.buffer_address() != m2d.buffer_address():
                ttnn.deallocate(m2d)
            return out
        return m2d

    @staticmethod
    def _exact_rsqrt(v):
        """Exact fp32 rsqrt (avoids ttnn.rsqrt's ~2.5e-3 default fast-approx LUT error)."""
        sq = ttnn.sqrt(v)
        out = ttnn.reciprocal(sq)
        ttnn.deallocate(sq)
        return out

    def _fast_row_mean(self, v):
        """All-core fp32 row mean over last dim ``c`` (~0.5 ms vs 8 ms 1-col matmul)."""
        s = ttnn.sum(v, dim=-1, keepdim=True, compute_kernel_config=_ckc_hifi4())
        out = ttnn.multiply(s, 1.0 / float(self.c))
        ttnn.deallocate(s)
        return out

    def _fast_row_mean_sq(self, x):
        x2 = ttnn.multiply(x, x)
        out = self._fast_row_mean(x2)
        ttnn.deallocate(x2)
        return out

    def _fast_row_mean_prod(self, a, b):
        ab = ttnn.multiply(a, b)
        out = self._fast_row_mean(ab)
        ttnn.deallocate(ab)
        return out

    def _bins(self, batch: dict) -> torch.Tensor:
        c = self.step_creator
        x = batch["x_t"]["bb_ca"]
        d = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)
        return torch.bucketize(d, torch.linspace(c.min_dist, c.max_dist, c.dim - 1, device=x.device))

    def modulation(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        ada = self.builder.adaln
        nc = ada.norm_cond(self.builder.cond_factory(batch))
        return ada.to_gamma(nc).float(), ada.to_beta(nc).float()

    def release_step(self) -> None:
        if self._step_state is None:
            return
        st = self._step_state
        self._step_state = None
        for k in ("proj", "table", "f_const_hi", "f_const_lo", "bins_tt"):
            t = st.get(k)
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)

    def release(self) -> None:
        self.release_step()
        for t in (self._ones_c, self._eye_nb):
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
        for tup in self._unit_cache.values():
            for t in tup:
                if t is not None and t.is_allocated():
                    ttnn.deallocate(t)
        self._unit_cache.clear()

    def pop_grads(self) -> dict[str, object]:
        out = dict(self.grads)
        self.grads.clear()
        return out

    def prepare_batch(self, batch: dict, dev_masters: dict) -> None:
        """Upload compact F_const + bins ONCE and build exact fp32 ``proj`` and ``table`` on card."""
        from .device_optim import transpose_exact_f32

        self.release_step()
        n = int(batch["padding_mask"].shape[-1])
        parts = []
        for cr in self.const_creators:
            f = cr(batch)
            if f.ndim == 2:
                f = f.reshape(f.shape[0], 1, 1, -1).expand(-1, n, n, -1)
            parts.append(f.float())
        b = int(parts[0].shape[0])
        dedup = b == 1 or all(torch.equal(f, f[:1].expand_as(f)) for f in parts)
        if not dedup:
            raise ValueError("DeviceTrainPair requires identical constant pair features across batch candidates")
        f_const = torch.cat([p[:1] for p in parts], dim=-1).reshape(n * n, self.const_dim)
        f_bf16 = f_const.bfloat16()
        exact_bf16 = bool(torch.equal(f_const, f_bf16.float()))
        pad_cols = self.const_kpad - self.const_dim
        if pad_cols:
            f_hi_t = torch.nn.functional.pad(f_bf16, [0, pad_cols]).contiguous()
        else:
            f_hi_t = f_bf16.contiguous()
        f_const_hi = ttnn.from_torch(f_hi_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        f_const_lo = None
        if not exact_bf16:
            f_rem = (f_const - f_bf16.float()).bfloat16()
            f_lo_t = (torch.nn.functional.pad(f_rem, [0, pad_cols]).contiguous()
                      if pad_cols else f_rem.contiguous())
            f_const_lo = ttnn.from_torch(f_lo_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Slice W_full directly from dev_masters on card (0 PCIe bytes)
        w_full = dev_masters[self.W_KEY]
        w_rm = ttnn.to_layout(w_full, ttnn.ROW_MAJOR_LAYOUT)
        w_step_rm = ttnn.slice(w_rm, [0, self.step_a], [self.c, self.step_b])
        w_const_rm = ttnn.slice(w_rm, [0, self.const_a], [self.c, self.const_b])
        ttnn.deallocate(w_rm)

        # table [nbins, c] fp32 ROW_MAJOR for pair_build exact gather
        w_step_tl = ttnn.to_layout(w_step_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(w_step_rm)
        table_tl = transpose_exact_f32(w_step_tl)
        ttnn.deallocate(w_step_tl)
        table = ttnn.to_layout(table_tl, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(table_tl)

        # proj = F_const @ W_const^T via 3-part exact bf16 split on W_const (118 KB!)
        if pad_cols:
            w_const_pad = ttnn.pad(w_const_rm, [(0, 0), (0, pad_cols)], 0.0)
            ttnn.deallocate(w_const_rm)
        else:
            w_const_pad = w_const_rm
        w_const_tl = ttnn.to_layout(w_const_pad, ttnn.TILE_LAYOUT)
        ttnn.deallocate(w_const_pad)
        w0, w1, w2 = _split3_bf16(w_const_tl)
        ttnn.deallocate(w_const_tl)
        ckc4 = _ckc_hifi4()
        ckc2 = _ckc_hifi2()
        wt0 = ttnn.transpose(w0, -2, -1); ttnn.deallocate(w0)
        wt1 = ttnn.transpose(w1, -2, -1); ttnn.deallocate(w1)
        wt2 = ttnn.transpose(w2, -2, -1); ttnn.deallocate(w2)

        p0 = ttnn.matmul(f_const_hi, wt0, dtype=ttnn.float32, compute_kernel_config=ckc4)
        p1 = ttnn.matmul(f_const_hi, wt1, dtype=ttnn.float32, compute_kernel_config=ckc2)
        s01 = ttnn.add(p0, p1); ttnn.deallocate(p0); ttnn.deallocate(p1)
        p2 = ttnn.matmul(f_const_hi, wt2, dtype=ttnn.float32, compute_kernel_config=ckc2)
        proj_2d = ttnn.add(s01, p2); ttnn.deallocate(s01); ttnn.deallocate(p2)
        if f_const_lo is not None:
            q0 = ttnn.matmul(f_const_lo, wt0, dtype=ttnn.float32, compute_kernel_config=ckc2)
            q1 = ttnn.matmul(f_const_lo, wt1, dtype=ttnn.float32, compute_kernel_config=ckc2)
            qs = ttnn.add(q0, q1); ttnn.deallocate(q0); ttnn.deallocate(q1)
            nxt = ttnn.add(proj_2d, qs); ttnn.deallocate(proj_2d); ttnn.deallocate(qs)
            proj_2d = nxt
        for wt in (wt0, wt1, wt2):
            ttnn.deallocate(wt)

        proj = ttnn.reshape(proj_2d, [1, n, n, self.c])
        if proj.buffer_address() != proj_2d.buffer_address():
            ttnn.deallocate(proj_2d)

        bins_host = self._bins(batch)  # [K, n, n] int64
        k_cand = int(bins_host.shape[0])
        bins_tt = ttnn.from_torch(
            bins_host.reshape(k_cand * n, n).to(torch.int32).contiguous(),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
        )
        self._step_state = dict(
            n=n, k=k_cand, proj=proj, table=table,
            f_const_hi=f_const_hi, f_const_lo=f_const_lo,
            bins_tt=bins_tt, ln_w=dev_masters[self.LNW_KEY],
        )

    def _gather_x(self, proj, table, bins_tt, b: int, n: int):
        from ..pair_build import pair_build

        ones_s, ones_wg, zeros_b = self._unit_tensors(b, n)
        x = pair_build(proj, table, bins_tt, ones_s, ones_wg, zeros_b)
        if x is None:
            raise RuntimeError(f"pair_build declined DeviceTrainPair inputs: b={b}, n={n}, c={self.c}")
        return x

    def forward_candidates(self, gamma: torch.Tensor, beta: torch.Tensor, dtype=None):
        """Grad-off candidate z_init ``[K, n, n, c]`` built wholly on card from prepared state."""
        st = self._step_state
        assert st is not None, "call prepare_batch() first"
        b, n, c = st["k"], st["n"], self.c
        ckc4 = _ckc_hifi4()
        x = self._gather_x(st["proj"], st["table"], st["bins_tt"], b, n)
        g_tt = ttnn.from_torch(gamma.detach().float().reshape(b, 1, 1, c).contiguous(),
                               dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        b_tt = ttnn.from_torch(beta.detach().float().reshape(b, 1, 1, c).contiguous(),
                               dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        a = ttnn.rms_norm(x, weight=st["ln_w"], epsilon=self.e1, compute_kernel_config=ckc4)
        ttnn.deallocate(x)
        bb = ttnn.rms_norm(a, epsilon=self.e2, compute_kernel_config=ckc4)
        ttnn.deallocate(a)
        scaled = ttnn.multiply(bb, g_tt)
        ttnn.deallocate(bb); ttnn.deallocate(g_tt)
        z = ttnn.add(scaled, b_tt)
        ttnn.deallocate(scaled); ttnn.deallocate(b_tt)
        if dtype is not None and z.dtype != dtype:
            z32, z = z, ttnn.typecast(z, dtype)
            ttnn.deallocate(z32)
        return z

    def forward_winner_taped(self, win_idx: int, gamma: torch.Tensor, beta: torch.Tensor,
                             bridge, dtype=None):
        """Taped winner z_init ``[1, n, n, c]``; backward computes dW + d_lnw wholly on card."""
        st = self._step_state
        assert st is not None, "call prepare_batch() first"
        n, c = st["n"], self.c
        ckc4 = _ckc_hifi4()
        bins_win = ttnn.slice(st["bins_tt"], [win_idx * n, 0], [(win_idx + 1) * n, n])
        x = self._gather_x(st["proj"], st["table"], bins_win, 1, n)
        ln_w_master = st["ln_w"]
        g_tt = ttnn.from_torch(gamma.detach().float().reshape(1, 1, 1, c).contiguous(),
                               dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        b_tt = ttnn.from_torch(beta.detach().float().reshape(1, 1, 1, c).contiguous(),
                               dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        a = ttnn.rms_norm(x, weight=ln_w_master, epsilon=self.e1, compute_kernel_config=ckc4)
        ttnn.deallocate(x)
        bb = ttnn.rms_norm(a, epsilon=self.e2, compute_kernel_config=ckc4)
        ttnn.deallocate(a)
        scaled = ttnn.multiply(bb, g_tt)
        ttnn.deallocate(bb)
        z = ttnn.add(scaled, b_tt)
        ttnn.deallocate(scaled); ttnn.deallocate(b_tt)
        if dtype is not None and z.dtype != dtype:
            z32, z = z, ttnn.typecast(z, dtype)
            ttnn.deallocate(z32)

        Gl = bridge.up(gamma.reshape(1, c), ttnn.float32)
        Bl = bridge.up(beta.reshape(1, c), ttnn.float32)
        # Keep a requires_grad leaf so ag._tape attaches bw even if gamma/beta are detached in unit tests.
        leaves = [Gl, Bl, ag.Tensor(ln_w_master, requires_grad=True)]
        proj = st["proj"]
        table = st["table"]
        f_hi = st["f_const_hi"]
        f_lo = st["f_const_lo"]

        def make():
            def bw(g):
                ckc4_b = _ckc_hifi4()
                ckc2_b = _ckc_hifi2()
                if g.dtype != ttnn.float32:
                    g32 = ttnn.typecast(g, ttnn.float32)
                else:
                    g32 = g
                x_b = self._gather_x(proj, table, bins_win, 1, n)
                w_row = ttnn.reshape(ln_w_master, [1, 1, 1, c])
                m1_b = self._fast_row_mean_sq(x_b)
                v1_b = ttnn.add(m1_b, self.e1); ttnn.deallocate(m1_b)
                r1_b = self._exact_rsqrt(v1_b); ttnn.deallocate(v1_b)
                n1_b = ttnn.multiply(x_b, r1_b)
                ttnn.deallocate(x_b)
                a_b = ttnn.multiply(n1_b, w_row)
                m2_b = self._fast_row_mean_sq(a_b)
                v2_b = ttnn.add(m2_b, self.e2); ttnn.deallocate(m2_b)
                r2_b = self._exact_rsqrt(v2_b); ttnn.deallocate(v2_b)
                bb_b = ttnn.multiply(a_b, r2_b)
                ttnn.deallocate(a_b)

                if Bl.requires_grad:
                    Bl.add_grad(ag._sum_leading(g32, Bl.value.shape))
                if Gl.requires_grad:
                    gb = ttnn.multiply(g32, bb_b)
                    Gl.add_grad(ag._sum_leading(gb, Gl.value.shape))
                    ttnn.deallocate(gb)

                db = ttnn.multiply(g32, g_tt)
                if g32 is not g:
                    ttnn.deallocate(g32)
                m_db = self._fast_row_mean_prod(db, bb_b)
                bb_m = ttnn.multiply(bb_b, m_db)
                ttnn.deallocate(m_db)
                sub_db = ttnn.subtract(db, bb_m)
                ttnn.deallocate(db); ttnn.deallocate(bb_m); ttnn.deallocate(bb_b)
                da = ttnn.multiply(sub_db, r2_b)
                ttnn.deallocate(sub_db); ttnn.deallocate(r2_b)

                # d_lnw stays on card in self.grads[LNW_KEY]
                dan1 = ttnn.multiply(da, n1_b)
                d_lnw = ag._sum_leading(dan1, [c])
                ttnn.deallocate(dan1)
                self.grads[self.LNW_KEY] = d_lnw

                # dx -> dW_const + dW_step wholly on card in self.grads[W_KEY]
                dn1 = ttnn.multiply(da, w_row)
                ttnn.deallocate(da)
                if w_row.buffer_address() != ln_w_master.buffer_address():
                    ttnn.deallocate(w_row)
                m_dn = self._fast_row_mean_prod(dn1, n1_b)
                n1_m = ttnn.multiply(n1_b, m_dn)
                ttnn.deallocate(m_dn); ttnn.deallocate(n1_b)
                sub_dn = ttnn.subtract(dn1, n1_m)
                ttnn.deallocate(dn1); ttnn.deallocate(n1_m)
                dx = ttnn.multiply(sub_dn, r1_b)
                ttnn.deallocate(sub_dn); ttnn.deallocate(r1_b)

                dx_2d = ttnn.reshape(dx, [n * n, c])
                dx_hi, dx_lo = _split2_bf16(dx_2d)
                if dx_2d.buffer_address() != dx.buffer_address():
                    ttnn.deallocate(dx_2d)
                ttnn.deallocate(dx)

                # dW_const = dx^T @ F_const [c, const_kpad] -> slice [c, const_dim]
                c0 = ttnn.matmul(dx_hi, f_hi, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=ckc4_b)
                c1 = ttnn.matmul(dx_lo, f_hi, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=ckc2_b)
                dc_pad = ttnn.add(c0, c1); ttnn.deallocate(c0); ttnn.deallocate(c1)
                if f_lo is not None:
                    c2 = ttnn.matmul(dx_hi, f_lo, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=ckc2_b)
                    nxt = ttnn.add(dc_pad, c2); ttnn.deallocate(dc_pad); ttnn.deallocate(c2)
                    dc_pad = nxt
                dc_rm = ttnn.to_layout(dc_pad, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(dc_pad)
                dw_const_rm = ttnn.slice(dc_rm, [0, 0], [c, self.const_dim])
                ttnn.deallocate(dc_rm)

                # dW_step = dx^T @ onehot(bins_win) [c, nbins_pad] -> slice [c, nbins]
                bins_flat = ttnn.reshape(bins_win, [1, n * n])
                oh_3d = ttnn.embedding(bins_flat, self._eye_nb, layout=ttnn.TILE_LAYOUT,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if bins_flat.buffer_address() != bins_win.buffer_address():
                    ttnn.deallocate(bins_flat)
                oh_2d = ttnn.reshape(oh_3d, [n * n, self.nbins_pad])
                s0 = ttnn.matmul(dx_hi, oh_2d, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=ckc4_b)
                s1 = ttnn.matmul(dx_lo, oh_2d, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=ckc2_b)
                if oh_2d.buffer_address() != oh_3d.buffer_address():
                    ttnn.deallocate(oh_2d)
                ttnn.deallocate(oh_3d)
                ttnn.deallocate(dx_hi); ttnn.deallocate(dx_lo)
                ds_pad = ttnn.add(s0, s1); ttnn.deallocate(s0); ttnn.deallocate(s1)
                ds_rm = ttnn.to_layout(ds_pad, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(ds_pad)
                dw_step_rm = ttnn.slice(ds_rm, [0, 0], [c, self.nbins])
                ttnn.deallocate(ds_rm)

                pieces = [dw_step_rm, dw_const_rm] if self.step_a < self.const_a else [dw_const_rm, dw_step_rm]
                dw_rm = ttnn.concat(pieces, dim=1)
                ttnn.deallocate(dw_step_rm); ttnn.deallocate(dw_const_rm)
                dw_tl = ttnn.to_layout(dw_rm, ttnn.TILE_LAYOUT)
                ttnn.deallocate(dw_rm)
                self.grads[self.W_KEY] = dw_tl

                ttnn.deallocate(g_tt)
                ttnn.deallocate(bins_win)
                self.release_step()
            return bw

        return ag._tape(z, leaves, make, reads=())
