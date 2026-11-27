"""Training-side taped wrapper for RFD4 PairOnCard (r4t-perf P2) and zero-pair-PCIe
ResidentPairBuilder (r4t-resident D89).

Lives under `tt_bio/train/` so inference module `tt_bio/rfd4/pair.py` never imports `autograd`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .. import autograd as ag
from ..rfd4.pair import STEP_FEATURE, PairInputs, PairOnCard


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

    __slots__ = ("k", "n", "dedup", "C", "F_const", "F_step_k", "ws_h", "ws_l")

    def __init__(self, k: int, n: int, dedup: bool, C, F_const, F_step_k, ws_h, ws_l):
        self.k = k
        self.n = n
        self.dedup = dedup
        self.C = C
        self.F_const = F_const
        self.F_step_k = F_step_k
        self.ws_h = ws_h
        self.ws_l = ws_l

    def release(self) -> None:
        for attr in ("C", "F_const", "F_step_k", "ws_h", "ws_l"):
            t = getattr(self, attr, None)
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
            setattr(self, attr, None)


def _norm_adaln_on_card(x, lnw_tt, gamma_tt, beta_tt, e1: float, e2: float, c: int, cfg):
    """Run PairReprBuilder's RMSNorm -> ln_out -> AdaLN RMSNorm -> gamma/beta strictly on card."""
    mean = lambda v: ag._row_mean(v, cfg, c)
    x2 = ttnn.multiply(x, x)
    mx2 = mean(x2)
    ttnn.deallocate(x2)
    r1 = ttnn.rsqrt(ttnn.add(mx2, e1))
    ttnn.deallocate(mx2)
    n1 = ttnn.multiply(x, r1)
    ttnn.deallocate(r1)
    a = ttnn.multiply(n1, lnw_tt)
    ttnn.deallocate(n1)
    a2 = ttnn.multiply(a, a)
    ma2 = mean(a2)
    ttnn.deallocate(a2)
    r2 = ttnn.rsqrt(ttnn.add(ma2, e2))
    ttnn.deallocate(ma2)
    bb = ttnn.multiply(a, r2)
    ttnn.deallocate(a)
    ttnn.deallocate(r2)
    z1 = ttnn.multiply(bb, gamma_tt)
    ttnn.deallocate(bb)
    z = ttnn.add(z1, beta_tt)
    ttnn.deallocate(z1)
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

    def encode_step(self, batch: dict) -> EncodedPair:
        """Encode one training step's batch into compact integer indices (no [N, N, C] floats)."""
        n = int(batch["padding_mask"].shape[-1])
        k = int(batch["padding_mask"].shape[0])
        with torch.no_grad():
            bins_step = self._bins_step(batch).reshape(k, n * n).contiguous()
            gamma, beta = self.modulation(batch)

            dedup = self._is_const_dedup(batch, k)
            cb = self._slice_b1(batch) if dedup else batch
            b_c = 1 if dedup else k

            def _eval_fc(fc):
                f = fc(cb)
                if f.ndim == 2:
                    f = f.reshape(f.shape[0], 1, 1, -1).expand(-1, n, n, -1)
                return f.float()

            idx_oh = []
            compact_ok = True
            raw_large = []
            for fc, cols, _boff, bw in self.large_oh:
                f = _eval_fc(fc)
                d = len(cols)
                is_1h = bool(((f == 0) | (f == 1)).all() and (f.sum(-1) <= 1).all())
                if is_1h:
                    has = f.sum(-1) > 0.5
                    idx = torch.where(has, f.argmax(-1), torch.tensor(d, dtype=torch.long))
                    idx_oh.append(("large", idx.reshape(b_c, n * n).to(torch.int32).contiguous()))
                else:
                    compact_ok = False
                raw_large.append((f, bw))

            raw_small = []
            for fc, cols, moff in self.small_oh:
                f = _eval_fc(fc)
                d = len(cols)
                is_1h = bool(((f == 0) | (f == 1)).all() and (f.sum(-1) <= 1).all())
                if is_1h:
                    has = f.sum(-1) > 0.5
                    idx = torch.where(has, f.argmax(-1), torch.tensor(d, dtype=torch.long))
                    idx_oh.append(("small", idx.reshape(b_c, n * n).to(torch.int32).contiguous()))
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
                dedup=dedup, gamma=gamma, beta=beta, fallback_const=fallback_const,
            )

    def _pack_weights_on_card(self):
        """Pack current ``linear_out.weight`` [c, k_orig] into split-bf16 step & const weights on card."""
        from .device_optim import _split2_bf16

        w_host = self.builder.init_repr_factory.linear_out.weight.detach().float()
        w_pack = torch.zeros(self.c, self.k_pack, dtype=torch.float32)
        w_pack[:, self._pack_cols] = w_host[:, self._orig_cols]

        ws = ttnn.from_torch(w_pack[:, :self.step_w].contiguous(), dtype=ttnn.float32,
                             layout=ttnn.TILE_LAYOUT, device=self.device)
        wc = ttnn.from_torch(w_pack[:, self.step_w:].contiguous(), dtype=ttnn.float32,
                             layout=ttnn.TILE_LAYOUT, device=self.device)
        ws_h, ws_l = _split2_bf16(ws)
        wc_h, wc_l = _split2_bf16(wc)
        ttnn.deallocate(ws)
        ttnn.deallocate(wc)
        return ws_h, ws_l, wc_h, wc_l

    def _mm_f_wt(self, F_bf16, w_hi, w_lo, cfg):
        """Compute ``F_bf16 @ (w_hi + w_lo)^T`` in fp32 (HiFi4 + fp32_dest_acc)."""
        p0 = ttnn.matmul(F_bf16, w_hi, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=cfg)
        p1 = ttnn.matmul(F_bf16, w_lo, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=cfg)
        out = ttnn.add(p0, p1)
        ttnn.deallocate(p0)
        ttnn.deallocate(p1)
        return out

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
        from .device_optim import _safe_dealloc

        with ag.no_grad():
            cfg = ag.precise_config()
            k, n, c = enc.k, enc.n, self.c
            b_c = 1 if enc.dedup else k
            nn2 = n * n

            ws_h, ws_l, wc_h, wc_l = self._pack_weights_on_card()
            F_const = self._build_f_const(enc)  # [1, b_c * nn2, const_w] bf16 TILE
            C_flat = self._mm_f_wt(F_const, wc_h, wc_l, cfg)  # [1, b_c * nn2, c] fp32
            ttnn.deallocate(wc_h)
            ttnn.deallocate(wc_l)
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

            rows_flat = self._mm_f_wt(F_step_k, ws_h, ws_l, cfg)  # [1, k * nn2, c] fp32
            rows = ttnn.reshape(rows_flat, (k, n, n, c))
            _safe_dealloc(rows_flat, keep=rows)

            x = ttnn.add(rows, C)  # broadcasts C [1, n, n, c] over [k, n, n, c]
            ttnn.deallocate(rows)

            lnw = self.builder.init_repr_factory.ln_out.weight.detach().float()
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

        cache = _PairStepCache(k, n, enc.dedup, C, F_const, F_step_k, ws_h, ws_l)
        return z, cache

    def winner_taped(self, cache: _PairStepCache, win_idx: int, gamma_w: torch.Tensor,
                     beta_w: torch.Tensor, bridge, dtype=None):
        """Taped winner ``z_init`` [1, n, n, c] and backward strictly on card (0 pair PCIe bytes).

        Consumes and releases ``cache`` (reusing ``C``, ``F_const``, and ``F_step_k[win_idx]``).
        """
        from .device_optim import _safe_dealloc, _split2_bf16

        cfg = ag.precise_config()
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

            rows_flat = self._mm_f_wt(F_step_w, cache.ws_h, cache.ws_l, cfg)
            ttnn.deallocate(cache.ws_h)
            ttnn.deallocate(cache.ws_l)
            cache.ws_h = cache.ws_l = None

            rows_w = ttnn.reshape(rows_flat, (1, n, n, c))
            _safe_dealloc(rows_flat, keep=rows_w)
            x_w = ttnn.add(rows_w, C_w)
            ttnn.deallocate(rows_w)
            ttnn.deallocate(C_w)

            row_t = lambda v: ttnn.from_torch(v.detach().float().reshape(1, 1, 1, c).contiguous(),
                                              dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
            w_tt, g_tt, b_tt = row_t(lnw), row_t(gamma_w), row_t(beta_w)
            z = _norm_adaln_on_card(x_w, w_tt, g_tt, b_tt, self.e1, self.e2, c, cfg)
            ttnn.deallocate(b_tt)
            if dtype is not None and z.dtype != dtype:
                z32, z = z, ttnn.typecast(z, dtype)
                ttnn.deallocate(z32)

        leaves = [bridge.up(t, ttnn.float32) for t in (W, lnw.view(1, c), gamma_w.reshape(1, c), beta_w.reshape(1, c))]
        if not any(t.requires_grad for t in leaves):
            for t in (x_w, F_step_w, F_const_w, w_tt, g_tt):
                ttnn.deallocate(t)
            return ag.Tensor(z)

        e1, e2 = self.e1, self.e2
        k_orig, k_orig_pad, k_pack = self.k_orig, self._k_orig_pad, self.k_pack
        p_inv = self._P_pack_to_orig

        def make():
            def bw(g):
                mean = lambda v: ag._row_mean(v, cfg, c)
                g_owned = False
                if g.dtype != ttnn.float32:
                    g = ttnn.typecast(g, ttnn.float32)
                    g_owned = True
                x2 = ttnn.multiply(x_w, x_w)
                mx2 = mean(x2)
                ttnn.deallocate(x2)
                r1 = ttnn.rsqrt(ttnn.add(mx2, e1))
                ttnn.deallocate(mx2)
                n1 = ttnn.multiply(x_w, r1)
                ttnn.deallocate(x_w)
                a = ttnn.multiply(n1, w_tt)
                a2 = ttnn.multiply(a, a)
                ma2 = mean(a2)
                ttnn.deallocate(a2)
                r2 = ttnn.rsqrt(ttnn.add(ma2, e2))
                ttnn.deallocate(ma2)
                bb = ttnn.multiply(a, r2)
                ttnn.deallocate(a)

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

                    dw0 = ttnn.matmul(dx_hi, F_2d, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=cfg)
                    dw1 = ttnn.matmul(dx_lo, F_2d, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=cfg)
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
                    u0 = ttnn.matmul(dp_hi, p_inv, dtype=ttnn.float32, compute_kernel_config=cfg)
                    u1 = ttnn.matmul(dp_lo, p_inv, dtype=ttnn.float32, compute_kernel_config=cfg)
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
