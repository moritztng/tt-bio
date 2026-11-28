"""Training-side taped wrapper for RFD4 PairOnCard (r4t-perf P2) and zero-pair-PCIe
ResidentPairBuilder (r4t-resident D89).

Lives under `tt_bio/train/` so inference module `tt_bio/rfd4/pair.py` never imports `autograd`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .. import autograd as ag
from ..rfd4.pair import STEP_FEATURE, PairInputs, PairOnCard, split_bf16


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


@dataclass
class EncodedPair:
    """Compact batch-only integer indices for on-card pair construction (zero [N,N,C] floats)."""

    k: int
    n: int
    bins_step: torch.Tensor        # [k, n*n] int32 (0..step_dim)
    idx_onehot: list               # list[(name, [b_c, n*n] int32)]
    idx_scalar: torch.Tensor       # [b_c, n*n] int32 (packed binary/ternary)
    dedup: bool
    gamma: torch.Tensor            # [k, c] fp32
    beta: torch.Tensor             # [k, c] fp32
    fallback_const: torch.Tensor | None = None  # optional [b_c, n*n, const_w] bf16 if non-compact


class _PairStepCache:
    """Device-resident pair tensors shared between the candidate pass and winner taped pass."""

    __slots__ = ("k", "n", "dedup", "C", "F_const", "F_step_k", "ws_parts")

    def __init__(self, k: int, n: int, dedup: bool, C, F_const, F_step_k, ws_parts):
        self.k = k
        self.n = n
        self.dedup = dedup
        self.C = C
        self.F_const = F_const
        self.F_step_k = F_step_k
        self.ws_parts = ws_parts

    def release(self) -> None:
        for attr in ("C", "F_const", "F_step_k"):
            t = getattr(self, attr, None)
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
            setattr(self, attr, None)
        if self.ws_parts is not None:
            for t in self.ws_parts:
                if t is not None and t.is_allocated():
                    ttnn.deallocate(t)
            self.ws_parts = None


def _permute_exact_f32(t, dims: tuple[int, ...]):
    """Bit-exact ``permute`` of a float32 TILE tensor via 3-part bf16 decomposition.

    Direct ``ttnn.permute`` on float32 (and ``uint32`` bitcast round-trip on ttnn 0.68.0)
    truncates mantissa bits (~5.9e-4 rel_l2). Splitting an IEEE float32 (24 significand bits)
    into three 8-bit bf16 slices ``hi + mid + lo``, permuting each losslessly in bf16 TILE
    layout, and summing back in float32 preserves all 24 bits bit-exactly (0.0 error).
    """
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

    hp = ttnn.permute(hi, dims)
    ttnn.deallocate(hi)
    mp = ttnn.permute(mid, dims)
    ttnn.deallocate(mid)
    lp = ttnn.permute(lo, dims)
    ttnn.deallocate(lo)

    h32 = ttnn.typecast(hp, f32)
    ttnn.deallocate(hp)
    m32 = ttnn.typecast(mp, f32)
    ttnn.deallocate(mp)
    l32 = ttnn.typecast(lp, f32)
    ttnn.deallocate(lp)

    hm = ttnn.add(h32, m32)
    ttnn.deallocate(h32)
    ttnn.deallocate(m32)
    out = ttnn.add(hm, l32)
    ttnn.deallocate(hm)
    ttnn.deallocate(l32)
    return out


def _exact_fp32_row_mean(x, c: int, cfg=None):
    """Exact IEEE-754 fp32 mean along last dim of ``[b, n1, n2, c]`` (zero TF32 / FPU loss).

    ``ttnn.sum`` unpacks fp32 to TF32 (~2.7e-4 rel_l2) and even a 2-part bf16 matmul
    against a ones column loses ~1.4e-4 when channels span 400x dynamic range (post-``lnw``
    squares). This reduces ``c -> 32`` via tile-aligned SFPU fp32 adds, bit-exact permutes
    the single remaining tile to ``[b, 32, n1, n2]``, reduces ``32 -> 1`` via 5 pure SFPU
    fp32 adds, and bit-exact permutes back to ``[b, n1, n2, 1]`` (~2.7e-8 rel_l2, 3.5 ms at
    B=1, pure TILE_LAYOUT, 0 PCIe bytes).
    """
    b, n1, n2, ch = (int(d) for d in x.shape)
    if ch != c or ch % 32 != 0:
        raise ValueError(f"_exact_fp32_row_mean expects last dim == {c} (multiple of 32), got {x.shape}")

    cur = x
    owned = False
    while ch > 32:
        n_tiles = ch // 32
        if n_tiles == 3:
            a = ttnn.slice(cur, [0, 0, 0, 0], [b, n1, n2, 32])
            b_part = ttnn.slice(cur, [0, 0, 0, 32], [b, n1, n2, 64])
            c_part = ttnn.slice(cur, [0, 0, 0, 64], [b, n1, n2, 96])
            ab = ttnn.add(a, b_part)
            ttnn.deallocate(a)
            ttnn.deallocate(b_part)
            nxt = ttnn.add(ab, c_part)
            ttnn.deallocate(ab)
            ttnn.deallocate(c_part)
        else:
            half = (n_tiles // 2) * 32
            a = ttnn.slice(cur, [0, 0, 0, 0], [b, n1, n2, half])
            b_part = ttnn.slice(cur, [0, 0, 0, half], [b, n1, n2, 2 * half])
            s = ttnn.add(a, b_part)
            ttnn.deallocate(a)
            ttnn.deallocate(b_part)
            if 2 * half < ch:
                rem = ttnn.slice(cur, [0, 0, 0, 2 * half], [b, n1, n2, ch])
                nxt = ttnn.concat([s, rem], dim=-1)
                ttnn.deallocate(s)
                ttnn.deallocate(rem)
            else:
                nxt = s
        if owned:
            ttnn.deallocate(cur)
        cur, owned = nxt, True
        ch = int(cur.shape[-1])

    p = _permute_exact_f32(cur, (0, 3, 1, 2))
    if owned:
        ttnn.deallocate(cur)
    cur = p
    k = 32
    while k > 1:
        half = k // 2
        a = ttnn.slice(cur, [0, 0, 0, 0], [b, half, n1, n2])
        b_part = ttnn.slice(cur, [0, half, 0, 0], [b, k, n1, n2])
        nxt = ttnn.add(a, b_part)
        ttnn.deallocate(a)
        ttnn.deallocate(b_part)
        ttnn.deallocate(cur)
        cur = nxt
        k = half
    s_col = _permute_exact_f32(cur, (0, 2, 3, 1))
    ttnn.deallocate(cur)
    out = ttnn.multiply(s_col, 1.0 / float(c))
    ttnn.deallocate(s_col)
    return out


def _rsqrt_exact_f32(s):
    """Exact IEEE-754 fp32 rsqrt via accurate SFPU seed + 1 Newton-Raphson step."""
    y = ttnn.rsqrt(s, fast_and_approximate_mode=False)
    yy = ttnn.multiply(y, y)
    syy = ttnn.multiply(s, yy)
    ttnn.deallocate(yy)
    nh = ttnn.multiply(syy, -0.5)
    ttnn.deallocate(syy)
    fac = ttnn.add(nh, 1.5)
    ttnn.deallocate(nh)
    out = ttnn.multiply(y, fac)
    ttnn.deallocate(y)
    ttnn.deallocate(fac)
    return out


def _norm_adaln_on_card(x, lnw_tt, gamma_tt, beta_tt, e1: float, e2: float, c: int, cfg, cg=None,
                        *, return_saved: bool = False):
    """Run PairReprBuilder's RMSNorm -> ln_out -> AdaLN RMSNorm -> gamma/beta strictly on card."""
    x2 = ttnn.multiply(x, x)
    mx2 = _exact_fp32_row_mean(x2, c, cfg)
    ttnn.deallocate(x2)
    s1 = ttnn.add(mx2, e1)
    ttnn.deallocate(mx2)
    r1 = _rsqrt_exact_f32(s1)
    ttnn.deallocate(s1)
    n1 = ttnn.multiply(x, r1)
    if not return_saved:
        ttnn.deallocate(r1)
    a = ttnn.multiply(n1, lnw_tt)
    if not return_saved:
        ttnn.deallocate(n1)
    a2 = ttnn.multiply(a, a)
    ma2 = _exact_fp32_row_mean(a2, c, cfg)
    ttnn.deallocate(a2)
    s2 = ttnn.add(ma2, e2)
    ttnn.deallocate(ma2)
    r2 = _rsqrt_exact_f32(s2)
    ttnn.deallocate(s2)
    bb = ttnn.multiply(a, r2)
    ttnn.deallocate(a)
    if not return_saved:
        ttnn.deallocate(r2)
    z1 = ttnn.multiply(bb, gamma_tt)
    if not return_saved:
        ttnn.deallocate(bb)
    z = ttnn.add(z1, beta_tt)
    ttnn.deallocate(z1)
    if return_saved:
        return z, n1, r1, bb, r2
    return z


class ResidentPairBuilder:
    """Device-resident RFD4 pair builder (fwd + taped bwd) with zero pair-sized PCIe transfers.

    Replaces the 100.66 MB ``PairOnCard._constant`` upload, the 33.55 MB ``pair_taped`` ``F``
    upload, the per-step host ``PairSplit``/``PairOnCard`` rebuild, and CPU ``constant``/``scales``
    compute with ~1.83 MB of integer feature indices per step (batch input only):
      - ``bins_step`` ([K, N, N] uint32, 1.05 MB for K=4)
      - one-hot const indices (e.g. TokenBondPathLength 0..64, BondOrder 0..5: 0.26 MB each)
      - bit-packed scalar const index (7 binary + 1 ternary feature: 0.26 MB)

    Reconstructs exact ``{-1, 0, 1}`` features in bf16 TILE layout via ``ttnn.embedding``,
    computes ``C = F_const @ W_const^T`` and ``rows = F_step @ W_step^T`` on card in split-bf16
    -> fp32, runs both RMSNorms + AdaLN on card, and reuses the resident ``F_const`` + ``F_step``
    for the winner's taped backward ``dW = dx^T @ F`` (0 extra PCIe bytes).
    """

    def __init__(self, builder, device):
        fac = builder.init_repr_factory
        if builder.cond_factory is None or fac.ret_zero:
            raise ValueError("ResidentPairBuilder requires AdaLN and active pair features")
        self.builder = builder
        self.device = device
        self.c = int(fac.linear_out.weight.shape[0])
        self.k_orig = int(fac.linear_out.weight.shape[1])
        self.e1 = float(fac.ln_out.eps)
        self.e2 = float(builder.adaln.norm.eps)

        # Classify upstream feature creators without hardcoding column offsets.
        self.step_creator = None
        self.step_orig_cols: list[int] = []
        self.onehot_creators: list[tuple[object, list[int]]] = []
        self.scalar_creators: list[tuple[object, list[int], bool]] = []
        off = 0
        for fc in fac.feat_creators:
            d = int(fc.get_dim())
            cols = list(range(off, off + d))
            name = type(fc).__name__
            if name == STEP_FEATURE:
                self.step_creator = fc
                self.step_orig_cols = cols
            elif d >= 4:
                self.onehot_creators.append((fc, cols))
            else:
                is_ternary = ("CanonicalPolymerOrdering" in name)
                self.scalar_creators.append((fc, cols, is_ternary))
            off += d
        if self.step_creator is None or off != self.k_orig:
            raise ValueError(f"Unexpected pair feature layout: step={self.step_creator}, off={off}, k={self.k_orig}")

        # Pack columns into 32-aligned blocks:
        #   [0 : step_w]             step one-hot (dim 30 -> 32)
        #   [step_w : step_w+cw]     const blocks: large 1-hot (>=32) get own 32-aligned block;
        #                            small 1-hot (<32) + scalars share one 32-aligned misc block.
        self.step_dim = len(self.step_orig_cols)
        self.step_w = -(-self.step_dim // 32) * 32

        self.large_oh: list[tuple[object, list[int], int, int]] = []   # (fc, orig_cols, blk_off, blk_w)
        self.small_oh: list[tuple[object, list[int], int]] = []        # (fc, orig_cols, misc_col_off)
        self.scalar_meta: list[tuple[object, list[int], bool, int]] = []  # (fc, orig_cols, is_tern, misc_col_off)

        pack_to_orig: dict[int, int] = {}
        for i, oc in enumerate(self.step_orig_cols):
            pack_to_orig[i] = oc

        cur_const = 0
        for fc, cols in self.onehot_creators:
            d = len(cols)
            if d >= 32:
                bw = -(-d // 32) * 32
                for i, oc in enumerate(cols):
                    pack_to_orig[self.step_w + cur_const + i] = oc
                self.large_oh.append((fc, cols, cur_const, bw))
                cur_const += bw

        misc_off = cur_const
        misc_cursor = 0
        for fc, cols in self.onehot_creators:
            d = len(cols)
            if d < 32:
                for i, oc in enumerate(cols):
                    pack_to_orig[self.step_w + misc_off + misc_cursor + i] = oc
                self.small_oh.append((fc, cols, misc_cursor))
                misc_cursor += d

        self._scalar_bin_slots: list[int] = []
        self._scalar_tern_slot: int | None = None
        for fc, cols, is_tern in self.scalar_creators:
            for j, oc in enumerate(cols):
                slot = misc_cursor + j
                pack_to_orig[self.step_w + misc_off + slot] = oc
                if is_tern and self._scalar_tern_slot is None:
                    self._scalar_tern_slot = slot
                else:
                    self._scalar_bin_slots.append(slot)
            self.scalar_meta.append((fc, cols, is_tern, misc_cursor))
            misc_cursor += len(cols)

        self.misc_w = -(-misc_cursor // 32) * 32 if misc_cursor > 0 else 0
        self.const_w = cur_const + self.misc_w
        self.k_pack = self.step_w + self.const_w
        assert self.k_pack % 32 == 0 and self.const_w % 32 == 0 and self.step_w % 32 == 0

        # Host permutation indices (pack_col -> orig_col) and device inverse permutation [k_pack, k_orig_pad]
        self._pack_cols = sorted(pack_to_orig.keys())
        self._orig_cols = [pack_to_orig[p] for p in self._pack_cols]

        k_orig_pad = -(-self.k_orig // 32) * 32
        self._k_orig_pad = k_orig_pad
        p_inv = torch.zeros(self.k_pack, k_orig_pad, dtype=torch.bfloat16)
        for pc, oc in pack_to_orig.items():
            p_inv[pc, oc] = 1.0
        self._P_pack_to_orig = ttnn.from_torch(
            p_inv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )

        # Static ROW_MAJOR bf16 embedding tables for ttnn.embedding(..., layout=TILE_LAYOUT)
        def _up_tbl(t: torch.Tensor):
            return ttnn.from_torch(t.bfloat16().contiguous(), dtype=ttnn.bfloat16,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        tbl_step = torch.zeros(self.step_dim + 1, self.step_w, dtype=torch.bfloat16)
        tbl_step[torch.arange(self.step_dim), torch.arange(self.step_dim)] = 1.0
        self._tbl_step = _up_tbl(tbl_step)

        self._tbl_large_oh = []
        for _fc, cols, _boff, bw in self.large_oh:
            d = len(cols)
            t = torch.zeros(d + 1, bw, dtype=torch.bfloat16)
            t[torch.arange(d), torch.arange(d)] = 1.0
            self._tbl_large_oh.append(_up_tbl(t))

        self._tbl_small_oh = []
        for _fc, cols, moff in self.small_oh:
            d = len(cols)
            t = torch.zeros(d + 1, self.misc_w, dtype=torch.bfloat16)
            t[torch.arange(d), moff + torch.arange(d)] = 1.0
            self._tbl_small_oh.append(_up_tbl(t))

        n_bin = len(self._scalar_bin_slots)
        n_tern = 3 if self._scalar_tern_slot is not None else 1
        n_states = (1 << n_bin) * n_tern
        assert n_states <= 4096, f"Too many packed scalar states: {n_states}"
        tbl_s = torch.zeros(n_states, self.misc_w, dtype=torch.bfloat16)
        tern_vals = (0.0, 1.0, -1.0)
        for st in range(n_states):
            bmask = st & ((1 << n_bin) - 1)
            for bi, slot in enumerate(self._scalar_bin_slots):
                if (bmask >> bi) & 1:
                    tbl_s[st, slot] = 1.0
            if self._scalar_tern_slot is not None:
                tv = st >> n_bin
                tbl_s[st, self._scalar_tern_slot] = tern_vals[tv]
        self._tbl_scalar = _up_tbl(tbl_s)
        self._n_bin = n_bin
        self._bins_grid = torch.linspace(
            self.step_creator.min_dist, self.step_creator.max_dist, self.step_dim - 1
        )

    def modulation(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-design AdaLN `(gamma, beta)` [b, c] fp32 (with autograd when enabled)."""
        ada = self.builder.adaln
        nc = ada.norm_cond(self.builder.cond_factory(batch))
        return ada.to_gamma(nc).float(), ada.to_beta(nc).float()

    def extract_pair_cond_feats(self, batch: dict) -> torch.Tensor | None:
        """Extract weight-free static features for ``builder.cond_factory`` (returns None if non-static)."""
        cf = self.builder.cond_factory
        if cf is None or getattr(cf, "ret_zero", False):
            return None
        if any(any(True for _ in fc.parameters()) for fc in cf.feat_creators):
            return None
        with torch.no_grad():
            parts = [fc(batch) for fc in cf.feat_creators]
            if not (getattr(cf, "compact", False) and all(p.ndim == 2 for p in parts)):
                return None
            feats = torch.cat(parts, dim=-1)
            return cf.apply_padding_mask(feats, batch["padding_mask"])

    def modulation_from_cond_feats(
        self,
        cond_feats: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute AdaLN `(gamma, beta)` [B, c] from pre-extracted static ``cond_feats`` [B, 512]."""
        cf = self.builder.cond_factory
        ada = self.builder.adaln
        projected = cf.linear_out(cond_feats)
        features_proc = cf.ln_out(projected).to(projected.dtype)
        c_cond = cf.apply_padding_mask(features_proc, padding_mask)
        nc = ada.norm_cond(c_cond)
        return ada.to_gamma(nc).float(), ada.to_beta(nc).float()

    def _bins_step(self, batch: dict) -> torch.Tensor:
        x = batch["x_t"]["bb_ca"]
        d = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)
        grid = self._bins_grid.to(device=x.device, dtype=d.dtype)
        return torch.bucketize(d, grid).to(torch.int32)

    @staticmethod
    def _slice_b1(batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v[:1]
            elif isinstance(v, dict):
                out[k] = {sk: (sv[:1] if torch.is_tensor(sv) else sv) for sk, sv in v.items()}
            else:
                out[k] = v
        return out

    @staticmethod
    def _is_const_dedup(batch: dict, k: int) -> bool:
        if k <= 1:
            return True
        for key in ("padding_mask", "bond_mask", "token_bond_path_length", "canonical_polymer_ordering"):
            v = batch.get(key)
            if torch.is_tensor(v) and v.shape[0] == k and not torch.equal(v, v[:1].expand_as(v)):
                return False
        return True

    def encode_static(self, batch: dict) -> EncodedPair:
        """Encode weight-independent pair indices (safe to run ahead-of-time in a background prefetcher)."""
        n = int(batch["padding_mask"].shape[-1])
        k = int(batch["padding_mask"].shape[0])
        with torch.no_grad():
            bins_step = self._bins_step(batch).reshape(k, n * n).contiguous()

            dedup = self._is_const_dedup(batch, k)
            cb = self._slice_b1(batch) if dedup else batch
            b_c = 1 if dedup else k

            def _eval_fc(fc):
                f = fc(cb)
                if f.ndim == 2:
                    f = f.reshape(f.shape[0], 1, 1, -1).expand(-1, n, n, -1)
                return f.float()

            def _fast_onehot_idx(f: torch.Tensor, d: int):
                """Single-pass one-hot verification & index extraction (equivalent to exact {0,1} & sum<=1)."""
                mx, idx = f.max(-1)
                s = f.sum(-1)
                if bool(((mx == 0) | (mx == 1)).all() and (s == mx).all() and (f.min() >= 0)):
                    return torch.where(mx > 0.5, idx, d).reshape(b_c, n * n).to(torch.int32).contiguous()
                return None

            idx_oh = []
            compact_ok = True
            raw_large = []
            for fc, cols, _boff, bw in self.large_oh:
                f = _eval_fc(fc)
                idx_t = _fast_onehot_idx(f, len(cols))
                if idx_t is not None:
                    idx_oh.append(("large", idx_t))
                else:
                    compact_ok = False
                raw_large.append((f, bw))

            raw_small = []
            for fc, cols, moff in self.small_oh:
                f = _eval_fc(fc)
                idx_t = _fast_onehot_idx(f, len(cols))
                if idx_t is not None:
                    idx_oh.append(("small", idx_t))
                else:
                    compact_ok = False
                raw_small.append((f, moff))

            idx_s = torch.zeros(b_c, n, n, dtype=torch.int32)
            bit_i = 0
            raw_scalar = []
            for fc, cols, is_tern, moff in self.scalar_meta:
                f = _eval_fc(fc)
                raw_scalar.append((f, moff))
                for j in range(len(cols)):
                    ch = f[..., j]
                    if is_tern:
                        if not bool(((ch == 0) | (ch == 1) | (ch == -1)).all()):
                            compact_ok = False
                        else:
                            tv = torch.where(ch > 0.5, 1, torch.where(ch < -0.5, 2, 0)).to(torch.int32)
                            idx_s = idx_s + (tv << self._n_bin)
                    else:
                        if not bool(((ch == 0) | (ch == 1)).all()):
                            compact_ok = False
                        else:
                            idx_s = idx_s + ((ch > 0.5).to(torch.int32) << bit_i)
                        bit_i += 1

            fallback_const = None
            if not compact_ok:
                fb = torch.zeros(b_c, n, n, self.const_w, dtype=torch.float32)
                off_c = 0
                for f, bw in raw_large:
                    fb[..., off_c:off_c + f.shape[-1]] = f
                    off_c += bw
                for f, moff in raw_small:
                    fb[..., off_c + moff:off_c + moff + f.shape[-1]] = f
                for f, moff in raw_scalar:
                    fb[..., off_c + moff:off_c + moff + f.shape[-1]] = f
                fallback_const = fb.reshape(b_c, n * n, self.const_w).bfloat16().contiguous()

            return EncodedPair(
                k=k, n=n, bins_step=bins_step, idx_onehot=idx_oh,
                idx_scalar=idx_s.reshape(b_c, n * n).contiguous(),
                dedup=dedup, gamma=None, beta=None, fallback_const=fallback_const,
            )

    def encode_step(self, batch: dict) -> EncodedPair:
        """Encode one training step's batch into compact integer indices (no [N, N, C] floats)."""
        enc = self.encode_static(batch)
        with torch.no_grad():
            enc.gamma, enc.beta = self.modulation(batch)
        return enc

    def _pack_weights_on_card(self):
        """Pack current ``linear_out.weight`` [c, k_orig] into exact 3-part bf16 weights on card."""
        w_host = self.builder.init_repr_factory.linear_out.weight.detach().float()
        w_pack = torch.zeros(self.c, self.k_pack, dtype=torch.float32)
        w_pack[:, self._pack_cols] = w_host[:, self._orig_cols]

        multi = getattr(self.device, "get_num_devices", lambda: 1)() > 1

        def _up_bf16_rep(p: torch.Tensor):
            pc = p.contiguous()
            if multi and hasattr(ttnn, "to_layout") and pc.ndim >= 2 and int(pc.shape[-1]) % 32 == 0:
                rm = ttnn.from_torch(pc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
                tl = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)
                if tl is not rm:
                    ttnn.deallocate(rm)
                return tl
            return ttnn.from_torch(pc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        ws_parts = tuple(_up_bf16_rep(p) for p in split_bf16(w_pack[:, :self.step_w].contiguous()))
        wc_parts = tuple(_up_bf16_rep(p) for p in split_bf16(w_pack[:, self.step_w:].contiguous()))
        return ws_parts, wc_parts

    def _mm_f_wt(self, F_bf16, w_parts, cfg):
        """Compute ``F_bf16 @ sum(w_parts)^T`` in fp32 (HiFi4 + fp32_dest_acc, full core grid)."""
        from .device_optim import _core_grid

        cg = _core_grid(self.device)
        acc = None
        for wp in w_parts:
            p = ttnn.matmul(F_bf16, wp, transpose_b=True, dtype=ttnn.float32,
                            compute_kernel_config=cfg, core_grid=cg)
            if acc is None:
                acc = p
            else:
                nxt = ttnn.add(acc, p)
                ttnn.deallocate(acc)
                ttnn.deallocate(p)
                acc = nxt
        return acc

    def _build_f_const(self, enc: EncodedPair):
        """Reconstruct exact ``[b_c, n*n, const_w]`` bf16 TILE const features strictly on card."""
        b_c = 1 if enc.dedup else enc.k
        nn2 = enc.n * enc.n
        if enc.fallback_const is not None:
            return ttnn.from_torch(enc.fallback_const, dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=self.device)

        def _embed(idx_t: torch.Tensor, tbl):
            idx_tt = ttnn.from_torch(
                idx_t.reshape(1, b_c * nn2).contiguous(),
                dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
            )
            e = ttnn.embedding(idx_tt, tbl, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(idx_tt)
            return e

        blocks = []
        li, si = 0, 0
        misc_acc = None
        for kind, idx_t in enc.idx_onehot:
            if kind == "large":
                blocks.append(_embed(idx_t, self._tbl_large_oh[li]))
                li += 1
            else:
                e = _embed(idx_t, self._tbl_small_oh[si])
                si += 1
                if misc_acc is None:
                    misc_acc = e
                else:
                    nxt = ttnn.add(misc_acc, e)
                    ttnn.deallocate(misc_acc)
                    ttnn.deallocate(e)
                    misc_acc = nxt

        if self.misc_w > 0:
            es = _embed(enc.idx_scalar, self._tbl_scalar)
            if misc_acc is None:
                misc_acc = es
            else:
                nxt = ttnn.add(misc_acc, es)
                ttnn.deallocate(misc_acc)
                ttnn.deallocate(es)
                misc_acc = nxt
            blocks.append(misc_acc)

        if len(blocks) == 1:
            f_c = blocks[0]
        else:
            f_c = ttnn.concat(blocks, dim=-1)
            for bl in blocks:
                ttnn.deallocate(bl)
        return f_c  # [1, b_c * n*n, const_w] bf16 TILE

    def candidates(self, enc: EncodedPair, dtype=None) -> tuple[object, _PairStepCache]:
        """Build grad-off candidate ``z_init`` [K, n, n, c] on card and cache const state for winner."""
        from .device_optim import _core_grid, _safe_dealloc

        with ag.no_grad():
            cfg = ag.precise_config()
            cg = _core_grid(self.device)
            k, n, c = enc.k, enc.n, self.c
            b_c = 1 if enc.dedup else k
            nn2 = n * n

            ws_parts, wc_parts = self._pack_weights_on_card()
            F_const = self._build_f_const(enc)  # [1, b_c * nn2, const_w] bf16 TILE
            C_flat = self._mm_f_wt(F_const, wc_parts, cfg)  # [1, b_c * nn2, c] fp32
            for t in wc_parts:
                ttnn.deallocate(t)
            C = ttnn.reshape(C_flat, (b_c, n, n, c))
            _safe_dealloc(C_flat, keep=C)

            bins_tt = ttnn.from_torch(
                enc.bins_step.reshape(1, k * nn2).contiguous(),
                dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
            )
            F_step_k = ttnn.embedding(
                bins_tt, self._tbl_step, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, k * nn2, step_w] bf16 TILE
            ttnn.deallocate(bins_tt)

            lnw = self.builder.init_repr_factory.ln_out.weight.detach().float()
            if dtype == ttnn.bfloat16:
                # Grad-off bf16 candidate pass: 1-part bf16 step projection + fused ttnn.rms_norm (~35 ms faster).
                rows_flat = ttnn.matmul(F_step_k, ws_parts[0], transpose_b=True, dtype=ttnn.bfloat16,
                                        compute_kernel_config=cfg, core_grid=cg)
                rows = ttnn.reshape(rows_flat, (k, n, n, c))
                _safe_dealloc(rows_flat, keep=rows)
                C16 = ttnn.typecast(C, ttnn.bfloat16)
                x = ttnn.add(rows, C16)
                ttnn.deallocate(rows)
                ttnn.deallocate(C16)
                lnw_1d = ttnn.from_torch(lnw.reshape(c).contiguous(), dtype=ttnn.bfloat16,
                                         layout=ttnn.TILE_LAYOUT, device=self.device)
                gam_tt = ttnn.from_torch(enc.gamma.detach().float().reshape(k, 1, 1, c).contiguous(),
                                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
                bet_tt = ttnn.from_torch(enc.beta.detach().float().reshape(k, 1, 1, c).contiguous(),
                                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
                a = ttnn.rms_norm(x, epsilon=self.e1, weight=lnw_1d, compute_kernel_config=cfg)
                ttnn.deallocate(x)
                ttnn.deallocate(lnw_1d)
                bb = ttnn.rms_norm(a, epsilon=self.e2, compute_kernel_config=cfg)
                ttnn.deallocate(a)
                z1 = ttnn.multiply(bb, gam_tt)
                ttnn.deallocate(bb)
                ttnn.deallocate(gam_tt)
                z = ttnn.add(z1, bet_tt)
                ttnn.deallocate(z1)
                ttnn.deallocate(bet_tt)
            else:
                rows_flat = self._mm_f_wt(F_step_k, ws_parts, cfg)  # [1, k * nn2, c] fp32
                rows = ttnn.reshape(rows_flat, (k, n, n, c))
                _safe_dealloc(rows_flat, keep=rows)
                x = ttnn.add(rows, C)
                ttnn.deallocate(rows)
                lnw_tt = ttnn.from_torch(lnw.reshape(1, 1, 1, c).contiguous(), dtype=ttnn.float32,
                                         layout=ttnn.TILE_LAYOUT, device=self.device)
                gam_tt = ttnn.from_torch(enc.gamma.detach().float().reshape(k, 1, 1, c).contiguous(),
                                         dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
                bet_tt = ttnn.from_torch(enc.beta.detach().float().reshape(k, 1, 1, c).contiguous(),
                                         dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
                z = _norm_adaln_on_card(x, lnw_tt, gam_tt, bet_tt, self.e1, self.e2, c, cfg)
                for t in (x, lnw_tt, gam_tt, bet_tt):
                    ttnn.deallocate(t)
                if dtype is not None and z.dtype != dtype:
                    z32, z = z, ttnn.typecast(z, dtype)
                    ttnn.deallocate(z32)

        cache = _PairStepCache(k, n, enc.dedup, C, F_const, F_step_k, ws_parts)
        return z, cache

    def winner_taped(self, cache: _PairStepCache, win_idx: int, gamma_w: torch.Tensor,
                     beta_w: torch.Tensor, bridge, dtype=None):
        """Taped winner ``z_init`` [1, n, n, c] and backward strictly on card (0 pair PCIe bytes).

        Consumes and releases ``cache`` (reusing ``C``, ``F_const``, and ``F_step_k[win_idx]``).
        """
        from .device_optim import _core_grid, _safe_dealloc, _split2_bf16

        cfg = ag.precise_config()
        cg = _core_grid(self.device)
        n, c = cache.n, self.c
        nn2 = n * n
        w = int(win_idx)
        fac = self.builder.init_repr_factory
        W, lnw = fac.linear_out.weight, fac.ln_out.weight

        with ag.no_grad():
            # Slice winner's F_step [1, nn2, step_w] and F_const [1, nn2, const_w] on card (tile-aligned: 65536 % 32 == 0).
            F_step_w = ttnn.slice(cache.F_step_k, [0, w * nn2, 0], [1, (w + 1) * nn2, self.step_w])
            _safe_dealloc(cache.F_step_k, keep=F_step_w)
            cache.F_step_k = None

            if cache.dedup:
                F_const_w = cache.F_const
                C_w = cache.C
                cache.F_const = None
                cache.C = None
            else:
                F_const_w = ttnn.slice(cache.F_const, [0, w * nn2, 0], [1, (w + 1) * nn2, self.const_w])
                _safe_dealloc(cache.F_const, keep=F_const_w)
                cache.F_const = None
                C_w = ttnn.slice(cache.C, [w, 0, 0, 0], [w + 1, n, n, c])
                _safe_dealloc(cache.C, keep=C_w)
                cache.C = None

            rows_flat = self._mm_f_wt(F_step_w, cache.ws_parts, cfg)
            for t in cache.ws_parts:
                ttnn.deallocate(t)
            cache.ws_parts = None

            rows_w = ttnn.reshape(rows_flat, (1, n, n, c))
            _safe_dealloc(rows_flat, keep=rows_w)
            x_w = ttnn.add(rows_w, C_w)
            ttnn.deallocate(rows_w)
            ttnn.deallocate(C_w)

            row_t = lambda v: ttnn.from_torch(v.detach().float().reshape(1, 1, 1, c).contiguous(),
                                              dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
            w_tt, g_tt, b_tt = row_t(lnw), row_t(gamma_w), row_t(beta_w)
            z, n1, r1, bb, r2 = _norm_adaln_on_card(
                x_w, w_tt, g_tt, b_tt, self.e1, self.e2, c, cfg, cg, return_saved=True,
            )
            ttnn.deallocate(x_w)
            ttnn.deallocate(b_tt)
            if dtype is not None and z.dtype != dtype:
                z32, z = z, ttnn.typecast(z, dtype)
                ttnn.deallocate(z32)

        leaves = [bridge.up(t, ttnn.float32) for t in (W, lnw.view(1, c), gamma_w.reshape(1, c), beta_w.reshape(1, c))]
        if not any(t.requires_grad for t in leaves):
            for t in (n1, r1, bb, r2, F_step_w, F_const_w, w_tt, g_tt):
                ttnn.deallocate(t)
            return ag.Tensor(z)

        k_orig, k_orig_pad, k_pack = self.k_orig, self._k_orig_pad, self.k_pack
        p_inv = self._P_pack_to_orig

        def make():
            def bw(g):
                mean = lambda v: ag._row_mean(v, cfg, c)
                g_owned = False
                if g.dtype != ttnn.float32:
                    g = ttnn.typecast(g, ttnn.float32)
                    g_owned = True

                Wl, Ll, Gl, Bl = leaves
                if Bl.requires_grad:
                    Bl.add_grad(ag._sum_leading(g, Bl.value.shape))
                if Gl.requires_grad:
                    gb = ttnn.multiply(g, bb)
                    Gl.add_grad(ag._sum_leading(gb, Gl.value.shape))
                    ttnn.deallocate(gb)
                db = ttnn.multiply(g, g_tt)
                if g_owned:
                    ttnn.deallocate(g)
                dbb = ttnn.multiply(db, bb)
                mdbb = mean(dbb)
                ttnn.deallocate(dbb)
                b_proj = ttnn.multiply(bb, mdbb)
                ttnn.deallocate(mdbb)
                db_sub = ttnn.subtract(db, b_proj)
                ttnn.deallocate(b_proj)
                da = ttnn.multiply(db_sub, r2)
                for t in (db, db_sub, bb, r2):
                    ttnn.deallocate(t)

                if Ll.requires_grad:
                    dan1 = ttnn.multiply(da, n1)
                    Ll.add_grad(ag._sum_leading(dan1, Ll.value.shape))
                    ttnn.deallocate(dan1)
                if Wl.requires_grad:
                    dn1 = ttnn.multiply(da, w_tt)
                    dn1n1 = ttnn.multiply(dn1, n1)
                    mdn = mean(dn1n1)
                    ttnn.deallocate(dn1n1)
                    n1_proj = ttnn.multiply(n1, mdn)
                    ttnn.deallocate(mdn)
                    dn1_sub = ttnn.subtract(dn1, n1_proj)
                    ttnn.deallocate(n1_proj)
                    dx = ttnn.multiply(dn1_sub, r1)
                    ttnn.deallocate(dn1)
                    ttnn.deallocate(dn1_sub)

                    F_all = ttnn.concat([F_step_w, F_const_w], dim=-1)  # [1, nn2, k_pack] bf16 TILE
                    ttnn.deallocate(F_step_w)
                    ttnn.deallocate(F_const_w)
                    F_2d = ttnn.reshape(F_all, [nn2, k_pack])
                    dx_2d = ttnn.reshape(dx, [nn2, c])
                    dx_hi, dx_lo = _split2_bf16(dx_2d)
                    _safe_dealloc(dx_2d, keep=dx)
                    ttnn.deallocate(dx)

                    dw0 = ttnn.matmul(dx_hi, F_2d, transpose_a=True, dtype=ttnn.float32,
                                      compute_kernel_config=cfg, core_grid=cg)
                    dw1 = ttnn.matmul(dx_lo, F_2d, transpose_a=True, dtype=ttnn.float32,
                                      compute_kernel_config=cfg, core_grid=cg)
                    ttnn.deallocate(dx_hi)
                    ttnn.deallocate(dx_lo)
                    _safe_dealloc(F_2d, keep=F_all)
                    ttnn.deallocate(F_all)
                    dW_pack = ttnn.add(dw0, dw1)
                    ttnn.deallocate(dw0)
                    ttnn.deallocate(dw1)

                    # Exact inverse column permutation on card: [c, k_pack] @ [k_pack, k_orig_pad] -> [c, k_orig]
                    dp_hi, dp_lo = _split2_bf16(dW_pack)
                    ttnn.deallocate(dW_pack)
                    u0 = ttnn.matmul(dp_hi, p_inv, dtype=ttnn.float32,
                                     compute_kernel_config=cfg, core_grid=cg)
                    u1 = ttnn.matmul(dp_lo, p_inv, dtype=ttnn.float32,
                                     compute_kernel_config=cfg, core_grid=cg)
                    ttnn.deallocate(dp_hi)
                    ttnn.deallocate(dp_lo)
                    dW_pad = ttnn.add(u0, u1)
                    ttnn.deallocate(u0)
                    ttnn.deallocate(u1)
                    if k_orig_pad != k_orig:
                        dW = ttnn.slice(dW_pad, [0, 0], [c, k_orig])
                        _safe_dealloc(dW_pad, keep=dW)
                    else:
                        dW = dW_pad
                    Wl.add_grad(dW)
                else:
                    ttnn.deallocate(F_step_w)
                    ttnn.deallocate(F_const_w)

                for t in (da, n1, r1, w_tt, g_tt):
                    ttnn.deallocate(t)
            return bw

        return ag._tape(z, leaves, make, reads=())

    def _build_f_const_spmd(self, encs: list[EncodedPair], dedup_all: bool, *, pair_stacked: dict | None = None):
        """Reconstruct sharded ``[1, b_c * n*n, const_w]`` bf16 TILE const features across a MeshDevice."""
        from .mesh_dp import put_sharded

        k, n = encs[0].k, encs[0].n
        b_c = 1 if dedup_all else k
        nn2 = n * n

        def _expand_b(t: torch.Tensor, e_dedup: bool) -> torch.Tensor:
            if (not dedup_all) and e_dedup:
                return t.expand(k, *t.shape[1:]).contiguous()
            return t

        if any(e.fallback_const is not None for e in encs):
            pieces = [_expand_b(e.fallback_const.reshape(1 if e.dedup else k, nn2, self.const_w), e.dedup)
                      .reshape(1, b_c * nn2, self.const_w) for e in encs]
            return put_sharded(torch.cat(pieces, dim=0), self.device,
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        def _embed_stacked(stacked: torch.Tensor, tbl):
            idx_tt = put_sharded(stacked, self.device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            e = ttnn.embedding(idx_tt, tbl, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(idx_tt)
            return e

        def _embed_spmd(idx_list: list[torch.Tensor], tbl):
            stacked = torch.cat([t.reshape(1, b_c * nn2) for t in idx_list], dim=0).contiguous()
            return _embed_stacked(stacked, tbl)

        blocks = []
        li, si = 0, 0
        misc_acc = None
        if pair_stacked is not None:
            for kind, stacked in pair_stacked["onehot"]:
                if kind == "large":
                    blocks.append(_embed_stacked(stacked, self._tbl_large_oh[li]))
                    li += 1
                else:
                    e_t = _embed_stacked(stacked, self._tbl_small_oh[si])
                    si += 1
                    if misc_acc is None:
                        misc_acc = e_t
                    else:
                        nxt = ttnn.add(misc_acc, e_t)
                        ttnn.deallocate(misc_acc)
                        ttnn.deallocate(e_t)
                        misc_acc = nxt
            if self.misc_w > 0 and pair_stacked["scalar"] is not None:
                es = _embed_stacked(pair_stacked["scalar"], self._tbl_scalar)
                if misc_acc is None:
                    misc_acc = es
                else:
                    nxt = ttnn.add(misc_acc, es)
                    ttnn.deallocate(misc_acc)
                    ttnn.deallocate(es)
                    misc_acc = nxt
                blocks.append(misc_acc)
        else:
            for idx_pos, (kind, _) in enumerate(encs[0].idx_onehot):
                idx_list = [_expand_b(e.idx_onehot[idx_pos][1], e.dedup) for e in encs]
                if kind == "large":
                    blocks.append(_embed_spmd(idx_list, self._tbl_large_oh[li]))
                    li += 1
                else:
                    e_t = _embed_spmd(idx_list, self._tbl_small_oh[si])
                    si += 1
                    if misc_acc is None:
                        misc_acc = e_t
                    else:
                        nxt = ttnn.add(misc_acc, e_t)
                        ttnn.deallocate(misc_acc)
                        ttnn.deallocate(e_t)
                        misc_acc = nxt

            if self.misc_w > 0:
                sc_list = [_expand_b(e.idx_scalar, e.dedup) for e in encs]
                es = _embed_spmd(sc_list, self._tbl_scalar)
                if misc_acc is None:
                    misc_acc = es
                else:
                    nxt = ttnn.add(misc_acc, es)
                    ttnn.deallocate(misc_acc)
                    ttnn.deallocate(es)
                    misc_acc = nxt
                blocks.append(misc_acc)

        if len(blocks) == 1:
            f_c = blocks[0]
        else:
            f_c = ttnn.concat(blocks, dim=-1)
            for bl in blocks:
                ttnn.deallocate(bl)
        return f_c  # [1, b_c * n*n, const_w] bf16 TILE per chip

    def candidates_spmd(
        self,
        encs: list[EncodedPair],
        dtype=None,
        *,
        pair_stacked: dict | None = None,
    ) -> tuple[object, _PairStepCache]:
        """SPMD ``candidates`` across a ``MeshDevice``: builds per-chip ``[K, n, n, c]`` on card."""
        from .device_optim import _core_grid, _mm_cg, _safe_dealloc
        from .mesh_dp import put_sharded

        with ag.no_grad():
            cfg = ag.precise_config()
            cg = _core_grid(self.device)
            cg_kw = _mm_cg(cg)
            k, n, c = encs[0].k, encs[0].n, self.c
            dedup_all = pair_stacked["dedup_all"] if pair_stacked is not None else all(e.dedup for e in encs)
            b_c = 1 if dedup_all else k
            nn2 = n * n

            ws_parts, wc_parts = self._pack_weights_on_card()
            F_const = self._build_f_const_spmd(encs, dedup_all, pair_stacked=pair_stacked)
            C_flat = self._mm_f_wt(F_const, wc_parts, cfg)
            for t in wc_parts:
                ttnn.deallocate(t)
            C = ttnn.reshape(C_flat, (b_c, n, n, c))
            _safe_dealloc(C_flat, keep=C)

            if pair_stacked is not None:
                bins_stacked = pair_stacked["bins"]
            else:
                bins_stacked = torch.cat([e.bins_step.reshape(1, k * nn2) for e in encs], dim=0).contiguous()
            bins_tt = put_sharded(bins_stacked, self.device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            F_step_k = ttnn.embedding(
                bins_tt, self._tbl_step, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(bins_tt)

            lnw = self.builder.init_repr_factory.ln_out.weight.detach().float()
            gam_all = torch.cat([e.gamma.detach().float().reshape(k, 1, 1, c) for e in encs], dim=0)
            bet_all = torch.cat([e.beta.detach().float().reshape(k, 1, 1, c) for e in encs], dim=0)

            if dtype == ttnn.bfloat16:
                # Grad-off bf16 candidate pass: 1-part bf16 step projection + fused ttnn.rms_norm.
                rows_flat = ttnn.matmul(
                    F_step_k, ws_parts[0], transpose_b=True, dtype=ttnn.bfloat16,
                    compute_kernel_config=cfg, **cg_kw,
                )
                rows = ttnn.reshape(rows_flat, (k, n, n, c))
                _safe_dealloc(rows_flat, keep=rows)
                C16 = ttnn.typecast(C, ttnn.bfloat16)
                x = ttnn.add(rows, C16)
                ttnn.deallocate(rows)
                ttnn.deallocate(C16)
                lnw_1d = ttnn.from_torch(
                    lnw.reshape(c).contiguous(), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=self.device,
                )
                gam_tt = put_sharded(gam_all, self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                bet_tt = put_sharded(bet_all, self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                a = ttnn.rms_norm(x, epsilon=self.e1, weight=lnw_1d, compute_kernel_config=cfg)
                ttnn.deallocate(x)
                ttnn.deallocate(lnw_1d)
                bb = ttnn.rms_norm(a, epsilon=self.e2, compute_kernel_config=cfg)
                ttnn.deallocate(a)
                z1 = ttnn.multiply(bb, gam_tt)
                ttnn.deallocate(bb)
                ttnn.deallocate(gam_tt)
                z = ttnn.add(z1, bet_tt)
                ttnn.deallocate(z1)
                ttnn.deallocate(bet_tt)
            else:
                rows_flat = self._mm_f_wt(F_step_k, ws_parts, cfg)
                rows = ttnn.reshape(rows_flat, (k, n, n, c))
                _safe_dealloc(rows_flat, keep=rows)
                x = ttnn.add(rows, C)
                ttnn.deallocate(rows)
                lnw_tt = ttnn.from_torch(
                    lnw.reshape(1, 1, 1, c).contiguous(), dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT, device=self.device,
                )
                gam_tt = put_sharded(gam_all, self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
                bet_tt = put_sharded(bet_all, self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
                z = _norm_adaln_on_card(x, lnw_tt, gam_tt, bet_tt, self.e1, self.e2, c, cfg, cg)
                for t in (x, lnw_tt, gam_tt, bet_tt):
                    ttnn.deallocate(t)
                if dtype is not None and z.dtype != dtype:
                    z32, z = z, ttnn.typecast(z, dtype)
                    ttnn.deallocate(z32)

        cache = _PairStepCache(k, n, dedup_all, C, F_const, F_step_k, ws_parts)
        return z, cache

    def winner_taped_spmd(
        self,
        cache: _PairStepCache,
        encs: list[EncodedPair],
        win_indices: list[int],
        gamma_w: torch.Tensor,
        beta_w: torch.Tensor,
        bridge,
        dtype=None,
    ):
        """SPMD taped winner ``z_init`` [1, n, n, c] per chip with heterogeneous ``win_indices``."""
        from .device_optim import _core_grid, _mm_cg, _safe_dealloc, _split2_bf16
        from .mesh_dp import put_sharded, select_per_rank_candidate

        cfg = ag.precise_config()
        cg = _core_grid(self.device)
        cg_kw = _mm_cg(cg)
        world = len(win_indices)
        n, c = cache.n, self.c
        nn2 = n * n
        fac = self.builder.init_repr_factory
        W, lnw = fac.linear_out.weight, fac.ln_out.weight

        with ag.no_grad():
            if len(set(win_indices)) == 1:
                w = int(win_indices[0])
                F_step_w = ttnn.slice(cache.F_step_k, [0, w * nn2, 0], [1, (w + 1) * nn2, self.step_w])
                _safe_dealloc(cache.F_step_k, keep=F_step_w)
                cache.F_step_k = None
            else:
                if cache.F_step_k is not None and cache.F_step_k.is_allocated():
                    ttnn.deallocate(cache.F_step_k)
                cache.F_step_k = None
                win_bins = torch.cat(
                    [encs[r].bins_step[int(win_indices[r]):int(win_indices[r]) + 1].reshape(1, nn2)
                     for r in range(world)],
                    dim=0,
                ).contiguous()
                win_bins_tt = put_sharded(win_bins, self.device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                F_step_w = ttnn.embedding(
                    win_bins_tt, self._tbl_step, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(win_bins_tt)

            if cache.dedup:
                F_const_w = cache.F_const
                C_w = cache.C
                cache.F_const = None
                cache.C = None
            else:
                fc_3d = ttnn.reshape(cache.F_const, (cache.k, nn2, self.const_w))
                F_const_w = select_per_rank_candidate(fc_3d, win_indices, self.device)
                _safe_dealloc(fc_3d, keep=cache.F_const)
                _safe_dealloc(cache.F_const, keep=F_const_w)
                cache.F_const = None
                C_w = select_per_rank_candidate(cache.C, win_indices, self.device)
                _safe_dealloc(cache.C, keep=C_w)
                cache.C = None

            rows_flat = self._mm_f_wt(F_step_w, cache.ws_parts, cfg)
            for t in cache.ws_parts:
                ttnn.deallocate(t)
            cache.ws_parts = None

            rows_w = ttnn.reshape(rows_flat, (1, n, n, c))
            _safe_dealloc(rows_flat, keep=rows_w)
            x_w = ttnn.add(rows_w, C_w)
            ttnn.deallocate(rows_w)
            ttnn.deallocate(C_w)

            w_tt = ttnn.from_torch(lnw.detach().float().reshape(1, 1, 1, c).contiguous(),
                                   dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
            g_tt = put_sharded(gamma_w.detach().float().reshape(world, 1, 1, c),
                               self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            b_tt = put_sharded(beta_w.detach().float().reshape(world, 1, 1, c),
                               self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            z, n1, r1, bb, r2 = _norm_adaln_on_card(
                x_w, w_tt, g_tt, b_tt, self.e1, self.e2, c, cfg, cg, return_saved=True,
            )
            ttnn.deallocate(x_w)
            ttnn.deallocate(b_tt)
            if dtype is not None and z.dtype != dtype:
                z32, z = z, ttnn.typecast(z, dtype)
                ttnn.deallocate(z32)

        leaves = [
            bridge.up_replicated(W, ttnn.float32),
            bridge.up_replicated(lnw.view(1, c), ttnn.float32),
            bridge.up(gamma_w.reshape(world, c), ttnn.float32),
            bridge.up(beta_w.reshape(world, c), ttnn.float32),
        ]
        if not any(t.requires_grad for t in leaves):
            for t in (n1, r1, bb, r2, F_step_w, F_const_w, w_tt, g_tt):
                ttnn.deallocate(t)
            return ag.Tensor(z)

        k_orig, k_orig_pad, k_pack = self.k_orig, self._k_orig_pad, self.k_pack
        p_inv = self._P_pack_to_orig

        def make():
            def bw(g):
                mean = lambda v: ag._row_mean(v, cfg, c)
                g_owned = False
                if g.dtype != ttnn.float32:
                    g = ttnn.typecast(g, ttnn.float32)
                    g_owned = True

                Wl, Ll, Gl, Bl = leaves
                if Bl.requires_grad:
                    Bl.add_grad(ag._sum_leading(g, Bl.value.shape))
                if Gl.requires_grad:
                    gb = ttnn.multiply(g, bb)
                    Gl.add_grad(ag._sum_leading(gb, Gl.value.shape))
                    ttnn.deallocate(gb)
                db = ttnn.multiply(g, g_tt)
                if g_owned:
                    ttnn.deallocate(g)
                dbb = ttnn.multiply(db, bb)
                mdbb = mean(dbb)
                ttnn.deallocate(dbb)
                b_proj = ttnn.multiply(bb, mdbb)
                ttnn.deallocate(mdbb)
                db_sub = ttnn.subtract(db, b_proj)
                ttnn.deallocate(b_proj)
                da = ttnn.multiply(db_sub, r2)
                for t in (db, db_sub, bb, r2):
                    ttnn.deallocate(t)

                if Ll.requires_grad:
                    dan1 = ttnn.multiply(da, n1)
                    Ll.add_grad(ag._sum_leading(dan1, Ll.value.shape))
                    ttnn.deallocate(dan1)
                if Wl.requires_grad:
                    dn1 = ttnn.multiply(da, w_tt)
                    dn1n1 = ttnn.multiply(dn1, n1)
                    mdn = mean(dn1n1)
                    ttnn.deallocate(dn1n1)
                    n1_proj = ttnn.multiply(n1, mdn)
                    ttnn.deallocate(mdn)
                    dn1_sub = ttnn.subtract(dn1, n1_proj)
                    ttnn.deallocate(n1_proj)
                    dx = ttnn.multiply(dn1_sub, r1)
                    ttnn.deallocate(dn1)
                    ttnn.deallocate(dn1_sub)

                    F_all = ttnn.concat([F_step_w, F_const_w], dim=-1)
                    ttnn.deallocate(F_step_w)
                    ttnn.deallocate(F_const_w)
                    F_2d = ttnn.reshape(F_all, [nn2, k_pack])
                    dx_2d = ttnn.reshape(dx, [nn2, c])
                    dx_hi, dx_lo = _split2_bf16(dx_2d)
                    _safe_dealloc(dx_2d, keep=dx)
                    ttnn.deallocate(dx)

                    dw0 = ttnn.matmul(dx_hi, F_2d, transpose_a=True, dtype=ttnn.float32,
                                      compute_kernel_config=cfg, **cg_kw)
                    dw1 = ttnn.matmul(dx_lo, F_2d, transpose_a=True, dtype=ttnn.float32,
                                      compute_kernel_config=cfg, **cg_kw)
                    ttnn.deallocate(dx_hi)
                    ttnn.deallocate(dx_lo)
                    _safe_dealloc(F_2d, keep=F_all)
                    ttnn.deallocate(F_all)
                    dW_pack = ttnn.add(dw0, dw1)
                    ttnn.deallocate(dw0)
                    ttnn.deallocate(dw1)

                    dp_hi, dp_lo = _split2_bf16(dW_pack)
                    ttnn.deallocate(dW_pack)
                    u0 = ttnn.matmul(dp_hi, p_inv, dtype=ttnn.float32, compute_kernel_config=cfg, **cg_kw)
                    u1 = ttnn.matmul(dp_lo, p_inv, dtype=ttnn.float32, compute_kernel_config=cfg, **cg_kw)
                    ttnn.deallocate(dp_hi)
                    ttnn.deallocate(dp_lo)
                    dW_pad = ttnn.add(u0, u1)
                    ttnn.deallocate(u0)
                    ttnn.deallocate(u1)
                    if k_orig_pad != k_orig:
                        dW = ttnn.slice(dW_pad, [0, 0], [c, k_orig])
                        _safe_dealloc(dW_pad, keep=dW)
                    else:
                        dW = dW_pad
                    Wl.add_grad(dW)
                else:
                    ttnn.deallocate(F_step_w)
                    ttnn.deallocate(F_const_w)

                for t in (da, n1, r1, w_tt, g_tt):
                    ttnn.deallocate(t)
            return bw

        return ag._tape(z, leaves, make, reads=())

    def deallocate(self) -> None:
        for t in [self._P_pack_to_orig, self._tbl_step, self._tbl_scalar,
                  *self._tbl_large_oh, *self._tbl_small_oh]:
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
