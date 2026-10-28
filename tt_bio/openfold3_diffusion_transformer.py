"""OF3 token-level DiffusionTransformer device port (P8, leg 3).

OF3 ``DiffusionTransformer`` (AF3 Algorithm 23, non-cross-attention path): a 24-block
stack of ``DiffusionTransformerBlock``, each block = ``AttentionPairBias`` (with
``use_ada_layer_norm=True``) + ``ConditionedTransitionBlock``.

Per block (``a`` evolves, ``s``/``z``/``mask`` are fixed across the stack):

    a_ln = AdaLN(a, s)                                   # c_a=768, c_s=384
    z_b  = permute(linear_z(LN_z(z)), [0,3,1,2])         # [1,16,N,N] per-block pair bias
    o    = MHA(a_ln, a_ln, biases=[mask_bias, z_b])       # 16 heads, head_dim=48
    o    = linear_o(o * sigmoid(linear_g(a_ln)))         # query gate (mha), output proj
    o    = o * sigmoid(linear_ada_out(s))                 # APB output gate
    a    = a + o
    a_t  = AdaLN_t(a, s)
    tr   = linear_out(silu(linear_a(a_t)) * linear_b(a_t)) * sigmoid(linear_g_t(s))
    a    = a + (tr * tok_mask)

``AttentionPairBias`` here is the non-cross variant: full N-token self-attention (no
windowing, no block gather), with a ``layer_norm_z`` + ``linear_z`` pair-bias
projection. Where that ``layer_norm_z`` lives is the one architectural difference
between the two checkpoints this module serves. OF3-preview2 keeps a weight-only copy
inside every block; OpenBind (upstream ``v0.5.0``, ``MODEL_VERSION`` 2.0.0) hoists it
into a single pre-stack norm, the same shape the cross-attention ``AtomTransformer``
has always used. That is the *only* weight-key difference between the two checkpoints
(4887 shared tensors, zero shape mismatches, 48 per-block ``layer_norm_z`` keys traded
for 1 shared one), so the state dict selects the variant on its own and no flag is
threaded in. The shared ``AdaLN``
math (a = sigmoid(Wg(LN_s(s))) * LN(a) + Ws(LN_s(s))) is identical to the
``AtomTransformer``'s, so the conditioning submodules reuse ``tenstorrent.AdaLN`` via
``remap_of3_adaln`` -- no primitive is reimplemented. The SwiGLU transition is the
same math the trunk transition and the ``AtomTransformer`` conditioned transition use.

head_dim=48 is not tile-aligned, so the q/k/v projections are fused and padded to
head_dim=64 (16 heads -> 1024/head-group, fused qkv -> 3072), then
``nlp_create_qkv_heads`` splits the padded heads and ``nlp_concat_heads`` joins them, pad lanes
kept: the gate and output projection are re-laned to 1024 with zeros there. Attention itself is
MANUAL (matmul QK^T + scale + mask, fp32 numerically-stable softmax, matmul attn@V), NOT the fused
``scaled_dot_product_attention``: the fused SDPA does its softmax in bf16, and its
per-block error (~0.998) compounds to ~0.967 over the 24-block stack; a CPU bf16
control with an fp32 softmax holds 0.99996 over the same stack, isolating the softmax
precision as the lever. Computing the softmax in fp32 (scores cast up, softmax, cast
back) closes the gap -- the device stack tracks the reference at >=0.9998 at every
block. ``linear_z`` is a plain linear (no head_dim scaling, unlike the protenix pair
bias).

Two correctness details the port had to get right, both invisible at the block level
and fatal at the stack level: (1) the additive attention mask must cover the
tile-padded key positions -- ``from_torch`` pads *storage* with 0 (="unmasked" for an
additive mask), so a logical-N mask leaves the tile-padded keys unmasked and they leak
garbage into valid queries, compounding to a ~0.30 stack PCC collapse; the stack pads
its inputs to the tile width and marks padded keys -1e9; (2) the fp32 softmax above.

Golden: ``scripts/of3_diffusion_transformer_golden.py`` instantiates the full
``DiffusionModule``, forward-hooks ``diffusion_transformer`` to capture its exact
``(a, s, z, mask)`` input and ``a`` output (block 0 + full 24-block stack), and adds
``diffusion_transformer_real`` to ``~/of3_ref_out.pkl`` -- so the device port is
PCC-gated against the exact reference artifacts, isolating the device block precision
from the atom-encoder/conditioning host math.
"""
from __future__ import annotations


import torch
import ttnn

from . import ops
from . import tenstorrent as _T
from .tenstorrent import AdaLN, CORE_GRID_MAIN, Module, _cached, _dtype, batched_matmul, host_f64_softmax_site, pair_row_blocks, row_block_after_refusal, site_softmax, softmax_ckc
from .openfold3_atom_transformer import remap_of3_adaln
from .token_axis import TILE, bucketed_width
from .eltwise_fusion import gated_add, scale_add, mask_add
from .addcmul_at import addcmul_at

C_A = 768
C_S = 384
C_Z = 128
N_HEADS = 16
HEAD_DIM = 48
PADDED_HEAD_DIM = 64  # -48 % 32 = 16 -> 64 (tile-aligned)
N_BLOCKS = 24

# Pair shapes whose single-pass pair-bias projection DRAM refused, and the block they settled at.
_PAIR_BIAS_ROWS_REFUSED: dict = {}


def _sub(sd: dict, prefix: str) -> dict:
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def _pad_single(t: ttnn.Tensor, padded_N: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Device [S, N, C] -> [S, padded_N, C] (zero-padded), logical width padded_N."""
    N = t.shape[1]
    if padded_N == N:
        return t
    x = ttnn.to_torch(t)  # [1, N, C] (logical, to_torch strips tile padding)
    x = torch.nn.functional.pad(x, (0, 0, 0, padded_N - N), value=0.0)
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=t.device(), dtype=dtype)


def _pad_pair(t: ttnn.Tensor, padded_N: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Device [1, N, N, C] -> [1, padded_N, padded_N, C] (zero-padded)."""
    N = t.shape[1]
    if padded_N == N:
        return t
    x = ttnn.to_torch(t)
    x = torch.nn.functional.pad(x, (0, 0, 0, padded_N - N, 0, padded_N - N), value=0.0)
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=t.device(), dtype=dtype)


class _DiTBlock(Module):
    """One OF3 DiffusionTransformerBlock (Algorithm 23, non-cross path)."""

    def __init__(self, sd_block: dict, compute_kernel_config, ln_z_w=None):
        super().__init__({}, compute_kernel_config)  # weights loaded via _w_tt, not torch_to_tt
        self._act_dtype = _dtype(ttnn.bfloat16)
        self._w = sd_block
        self._wc: dict = {}
        # OFF. `__call__` typecasts the scores to fp32 on the line above the call, so this is
        # the site where the config pays: 12.50x accuracy against float64 for 1.55x cost at
        # [1,16,128,128], 144 calls a fold, +0.72 s against a 2.28 s A/A fold floor on qb2 card 0
        # and 144 x 6.3 us bounding the real cost at 0.91 ms. It stays off because `of3t-softmax`
        # measured the gain invisible in the structure (0.3237 A against a 0.6250 A seed floor),
        # not because it is expensive. perf/of3t_fwdkcfg/.
        self._softmax_ckc = softmax_ckc("openfold3.diffusion_transformer")
        self._softmax_f64 = host_f64_softmax_site("openfold3.diffusion_transformer")
        # dit_sdpa32 (normal-mode lever, shared with Protenix-v2's token DiT): the attention as one
        # `_sdpa32` program instead of matmul, scale_add, softmax and matmul. The bf16 stack
        # (diffusion_bf16) takes it too: bf16 qkv and an fp32 mask, the operands dit_qkv16 already
        # feeds it, and an fp32 output the out-projection reads.
        self.sdpa32 = _T.lever("dit_sdpa32")

        apb = "attention_pair_bias."
        self.adaln_a = AdaLN(False, remap_of3_adaln(_sub(self._w, apb + "layer_norm_a")),
                             compute_kernel_config)
        # OpenBind shares one layer_norm_z across the stack and passes it in; OF3-preview2
        # carries one per block.
        self.ln_z_w = ln_z_w if ln_z_w is not None else self._w_tt(apb + "layer_norm_z.weight", False)
        self.w_lin_z = self._w_tt(apb + "linear_z.weight")
        self.w_ada_out = self._w_tt(apb + "linear_ada_out.weight")
        self.b_ada_out = self._w_tt(apb + "linear_ada_out.bias", False)

        # Fused padded qkv: cat([linear_q, linear_k, linear_v]) with head_dim 48->64.
        wq = self._w[apb + "mha.linear_q.weight"]
        bq = self._w[apb + "mha.linear_q.bias"]
        wk = self._w[apb + "mha.linear_k.weight"]
        wv = self._w[apb + "mha.linear_v.weight"]
        qkv_w = torch.cat([wq, wk, wv], dim=0)  # [3*768, 768]
        qkv_w = qkv_w.reshape(3 * N_HEADS, HEAD_DIM, -1)
        qkv_w = torch.nn.functional.pad(qkv_w, (0, 0, 0, PADDED_HEAD_DIM - HEAD_DIM), value=0)
        qkv_w = qkv_w.reshape(3 * N_HEADS * PADDED_HEAD_DIM, -1)  # [3072, 768]
        self.qkv_w = ttnn.from_torch(qkv_w.t().contiguous(), layout=ttnn.TILE_LAYOUT,
                                     device=self.device, dtype=self._act_dtype)
        qb = bq.reshape(N_HEADS, HEAD_DIM)
        qb = torch.nn.functional.pad(qb, (0, PADDED_HEAD_DIM - HEAD_DIM), value=0)
        qb = qb.reshape(N_HEADS * PADDED_HEAD_DIM)
        qkv_b = torch.cat([qb, torch.zeros(2 * N_HEADS * PADDED_HEAD_DIM, dtype=qb.dtype)])
        self.qkv_b = ttnn.from_torch(qkv_b, layout=ttnn.TILE_LAYOUT, device=self.device,
                                     dtype=self._act_dtype)

        # The gate and the output projection on the padded head lanes, so the heads come back
        # with one nlp_concat_heads (see `_pad_head_lanes`): v is zero in those lanes, so is o,
        # and linear_o's matching rows are zero.
        self.w_g = self._w_lanes(apb + "mha.linear_g.weight", -1)
        self.w_o = self._w_lanes(apb + "mha.linear_o.weight", 0)

        ct = "conditioned_transition."
        self.adaln_t = AdaLN(False, remap_of3_adaln(_sub(self._w, ct + "layer_norm")),
                             compute_kernel_config)
        self.w_la = self._w_tt(ct + "swiglu.linear_a.weight")
        self.w_lb = self._w_tt(ct + "swiglu.linear_b.weight")
        self.w_lout = self._w_tt(ct + "linear_out.weight")
        self.w_lg = self._w_tt(ct + "linear_g.weight")
        self.b_lg = self._w_tt(ct + "linear_g.bias", False)

        # dit_mm16 / dit_b8 (Protenix-v2's levers, `dit_lowp`): the block's linears off the
        # adaLN outputs at lower operand formats. The s-path linears keep the stream's format.
        self.lp = _T.dit_lowp(self._act_dtype, compute_kernel_config)
        if self.lp:
            for k in ("qkv_w", "w_g", "w_o", "w_la", "w_lb", "w_lout"):
                setattr(self, k, self._round(getattr(self, k), self.lp.w))
            self.qkv_b = self._round(self.qkv_b, ttnn.bfloat16)

        # dit_s16: the six s-path linears (both adaLNs' scale and shift, the two s gates) on bf16
        # operands with fp32 accumulation at HiFi4, written fp32. The adaLNs read one parameter-free
        # LN(s) per call that the stack shares, with their gammas folded into the weights on the host.
        # On an fp32 DiT these linears are math-bound at fp32 operands.
        self.s16 = None
        # adaln_mod on top of dit_s16: every s-path sigmoid runs in its linear's epilogue, so each
        # adaLN is LN + one addcmul (`_SigTerms`), the APB residual one addcmul and the transition
        # gate a plain multiply: 72 fewer ops per step at 24 blocks, single rounding of the products.
        self.sig16 = False
        if self._act_dtype == ttnn.float32 and _T.lever("dit_s16"):
            self.sig16 = _T.lever("adaln_mod")
            bf = lambda w: ttnn.from_torch(w.float().contiguous(), dtype=ttnn.bfloat16,  # noqa: E731
                                           layout=ttnn.TILE_LAYOUT, device=self.device)
            ada = []
            for ln in (self.adaln_a, self.adaln_t):
                w_sc, b_sc, w_sb = ln.folded_weights()
                ada.append((bf(w_sc), bf(b_sc.reshape(1, -1)), bf(w_sb)))
            ckc = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)
            self.s16 = (ada, (bf(self._w[apb + "linear_ada_out.weight"].t()),
                              bf(self._w[apb + "linear_ada_out.bias"].reshape(1, -1))),
                        (bf(self._w[ct + "linear_g.weight"].t()),
                         bf(self._w[ct + "linear_g.bias"].reshape(1, -1))), ckc)

        # dit_fuse: the adaLNs read one parameter-free LN(s) per stack call (`OF3DiffusionTransformer.shared_s`)
        # with their gammas folded into the weights on the host, as dit_s16 does. Every s-path sigmoid runs in
        # its linear's epilogue, so each adaLN modulates in one addcmul, the APB gated residual is one addcmul
        # and, where the token mask is all ones, so is the transition's gated, masked residual; `_sdpa32` writes
        # the attention at the stack's dtype itself (no typecast). 33 -> 23 ttnn calls a block, at either dtype.
        # On the bf16 DiT (the `--fast` stack) it is not bit-exact: each fused product skips one bf16 rounding
        # (which `gated_add` declines in bf16 for exactly that reason), the sigmoid reads the fp32 accumulator
        # instead of its bf16 rounding, and LN(s) * gamma is rounded once inside W. On the fp32 DiT (OpenBind-0
        # normal) the skipped roundings are fp32 ones and the epilogue sigmoid is `mm_sigmoid`'s fp32 one.
        # dit_b8 on dit_fuse: the fused block reads the six weights rounded to bfp8 above through `_lin`, its
        # activations stay bf16 (the fused ops write them) and the precision is `_lin`'s (dit_hifi2 on Wormhole).
        self.fuse = None
        if self.sdpa32 and self.s16 is None and _T.lever("dit_fuse"):
            self.lp = None
            bf = lambda w: ttnn.from_torch(w.float().contiguous(), dtype=self._act_dtype,  # noqa: E731
                                           layout=ttnn.TILE_LAYOUT, device=self.device)
            ada = []
            for ln in (self.adaln_a, self.adaln_t):
                w_sc, b_sc, w_sb = ln.folded_weights()
                ada.append((bf(w_sc), bf(b_sc.reshape(1, -1)), bf(w_sb)))
            self.fuse = ada

        # dit_hifi2: the bf16 DiT's linears at HiFi2, everything else as the model's config (read per call).
        # Wormhole only: graded there (fast, both models), where HiFi4 leaves the linears math-bound; flat on p150a.
        self._wormhole = _T.is_wormhole()
        c = compute_kernel_config
        self.ckc_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=getattr(c, "math_approx_mode", True),
            fp32_dest_acc_en=getattr(c, "fp32_dest_acc_en", True), packer_l1_acc=getattr(c, "packer_l1_acc", True))

    def _fuse_adaln(self, a, s_hat, i, hoist=()):
        """adaLN under dit_fuse: `s_hat W_sb + LN(a) * sigmoid(s_hat W_sc + b)`, one addcmul.
        `hoist` (dit_shoist): the stack's wide (sc, sb, gates); this block's columns are read in place."""
        if hoist:
            an = ttnn.layer_norm(a, epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
            off = (2 * self.idx + i) * C_A
            out = addcmul_at(hoist[1], an, hoist[0], C_A, x_off=off, z_off=off)
            if out is not None:
                ttnn.deallocate(an)
                return out
            ttnn.deallocate(an)
        w_sc, b_sc, w_sb = self.fuse[i]
        l1 = self.l1(a)
        sc = self._lin(s_hat, w_sc, bias=b_sc, activation="sigmoid", **l1)
        sb = self._lin(s_hat, w_sb, **l1)
        an = ttnn.layer_norm(a, epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
        out = ttnn.addcmul(sb, an, sc, value=1.0)
        for t in (sc, sb, an):
            ttnn.deallocate(t)
        return out

    def _fused_mask(self, z, mask_bias):
        """`_sdpa32_mask` at the format `dit_mask16` / `dit_mask8` name (`sdpa32_mask_lowp`): the attention reads the
        [1, 16, N, N] mask in every block of every step (16 MB fp32 at 512 tokens), the rollout builds it once per
        block. On a p150a the bf16 A/B was noise (< 0.1 s), so Blackhole keeps fp32 and its digest."""
        return _T.sdpa32_mask_lowp(self._sdpa32_mask(z, mask_bias))

    def _call_fused(self, a, s, z, mask_bias, tok_mask_col, cache, s_hat, tok_ones, hoist=()):
        """`__call__` under dit_fuse (`_sdpa32`): the same block in 23 ttnn calls, 17 with `hoist`."""
        l1 = self.l1(a)
        lin = lambda *a, **k: self._lin(*a, **l1, **k)  # noqa: E731
        a_ln = self._fuse_adaln(a, s_hat, 0, hoist)
        bias = _cached(cache, (id(self), "pair_bias32"), lambda: self._fused_mask(z, mask_bias))
        qkv = lin(a_ln, self.qkv_w, bias=self.qkv_b)
        qkv = ttnn.unsqueeze(qkv, 1)
        o = _T._sdpa32_fused(qkv, bias, HEAD_DIM ** -0.5, N_HEADS, out_dtype=self._act_dtype,
                             memory_config=l1.get("memory_config"))
        ttnn.deallocate(qkv)
        if cache is None:
            ttnn.deallocate(bias)
        S, n = o.shape[0], o.shape[2]
        o = ttnn.reshape(o, (S, n, N_HEADS * PADDED_HEAD_DIM))
        g = lin(a_ln, self.w_g)
        ttnn.deallocate(a_ln)
        o2 = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g); ttnn.deallocate(o)
        o = lin(o2, self.w_o)
        ttnn.deallocate(o2)
        a2 = addcmul_at(a, o, hoist[2], C_A, z_off=2 * self.idx * C_A) if hoist else None
        if a2 is None:
            og = lin(s, self.w_ada_out, bias=self.b_ada_out, activation="sigmoid")
            a2 = ttnn.addcmul(a, o, og, value=1.0)
            ttnn.deallocate(og)
        ttnn.deallocate(o)
        a = a2

        a_t = self._fuse_adaln(a, s_hat, 1, hoist)
        b1 = lin(a_t, self.w_la, activation="silu")
        b2 = lin(a_t, self.w_lb)
        ttnn.deallocate(a_t)
        bb = ttnn.multiply(b1, b2)
        ttnn.deallocate(b1); ttnn.deallocate(b2)
        out = lin(bb, self.w_lout)
        ttnn.deallocate(bb)
        if hoist and tok_ones:
            a2 = addcmul_at(a, out, hoist[2], C_A, z_off=(2 * self.idx + 1) * C_A)
            if a2 is not None:
                ttnn.deallocate(out)
                return a2
        lg = lin(s, self.w_lg, bias=self.b_lg, activation="sigmoid")
        if tok_ones:          # x * 1 is exact: the mask multiply is an identity
            a2 = ttnn.addcmul(a, out, lg, value=1.0)
        else:
            gated = ttnn.multiply(out, lg)
            a2 = mask_add(a, gated, tok_mask_col)
            ttnn.deallocate(gated)
        ttnn.deallocate(out); ttnn.deallocate(lg)
        return a2

    def _s16_lin(self, x, wb, activation=None):
        w, b = wb
        if activation == "sigmoid":
            activation = _T.mm_sigmoid(self.s16[3])
        return ttnn.linear(x, w, bias=b, dtype=ttnn.float32, compute_kernel_config=self.s16[3],
                           core_grid=CORE_GRID_MAIN, activation=activation)

    def _s16_terms(self, s16, i):
        w_sc, b_sc, w_sb = self.s16[0][i]
        if self.sig16:
            return _T._SigTerms((self._s16_lin(s16[1], (w_sc, b_sc), "sigmoid"),
                                 self._s16_lin(s16[1], (w_sb, None))))
        return self._s16_lin(s16[1], (w_sc, b_sc)), self._s16_lin(s16[1], (w_sb, None))

    def _w_tt(self, key, transpose=True):
        v = self._wc.get((key, transpose))
        if v is None:
            w = self._w[key]
            v = ttnn.from_torch(w.t().contiguous() if transpose else w,
                                layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self._act_dtype)
            self._wc[(key, transpose)] = v
        return v

    def _w_lanes(self, key, axis):
        w = _T._pad_head_lanes(self._w[key].t(), N_HEADS, HEAD_DIM, PADDED_HEAD_DIM, axis)
        return ttnn.from_torch(w.contiguous(), layout=ttnn.TILE_LAYOUT, device=self.device,
                               dtype=self._act_dtype)

    def _round(self, t, dtype):
        """A device weight rounded via bf16 on the host to `dtype`, once."""
        return ttnn.from_torch(ttnn.to_torch(t).float().to(torch.bfloat16), dtype=dtype,
                               layout=ttnn.TILE_LAYOUT, device=self.device)

    def l1(self, a):
        """dit_l1 (on dit_fuse, read per call): where every per-step tensor of the block is written. The stack
        puts the stream and the conditioning there once a call (`OF3DiffusionTransformer.__call__`), and the
        elementwise ops write where their first input lives. Weights and cached pair biases stay DRAM.

        Only where the block's live set fits beside the consumers' circular buffers. Priced in stream-sized
        tensors `a` [S, rows, 768]: a, s and s_hat for the call, then at most the transition's a_t, b1, b2
        (two each) and their product (two), or the attention's a_ln, qkv (four, padded heads) and o; 12 with
        slack. Decided on the shape alone, so every block of a call and the stack agree."""
        if not (self.fuse and _T.lever("dit_l1")):
            return {}
        elem = 4 if self._act_dtype == ttnn.float32 else 2
        fits = _T._l1_fits(_T._padded_bytes(tuple(a.shape), elem), 12.0, _T._PAIR_L1_CONSUMER_RESERVE)
        return {"memory_config": ttnn.L1_MEMORY_CONFIG} if fits else {}

    def _lin(self, x, w, bias=None, activation=None, **kw):
        hifi2 = self._wormhole and self._act_dtype == ttnn.bfloat16 and _T.lever("dit_hifi2")
        ckc = self.ckc_hifi2 if hifi2 else self.compute_kernel_config
        if activation == "sigmoid":
            activation = _T.mm_sigmoid(ckc, f32=self._act_dtype == ttnn.float32)
        return ops.linear(x, w, bias=bias, activation=activation, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN,
                          **kw)

    def _lin_lp(self, x, w, dtype, bias=None, activation=None):
        """A linear off an adaLN output: written at `dtype` under `self.lp`, else the shared path."""
        if self.lp is None:
            return self._lin(x, w, bias, activation)
        return self.lp.linear(x, w, bias, dtype=dtype, activation=activation)

    def _z_bias(self, z):
        """linear_z(LN_z(z)), per pair position."""
        zn = ttnn.layer_norm(z, weight=self.ln_z_w, epsilon=1e-5,
                             compute_kernel_config=self.compute_kernel_config)
        zb = self._lin(zn, self.w_lin_z)
        ttnn.deallocate(zn)
        return zb

    def _pair_bias(self, z, mask_bias):
        """LN_z(z) -> linear_z -> [1,16,N,N] + mask_bias. Pure function of the
        conditioning pair, so the rollout computes it once per block, not once per step.
        The fp32 norm is a whole second pair (1207959552 B at 1536 tokens), so a refusal
        re-runs the projection in row blocks and only the 16-channel bias is ever whole."""
        zb = row_block_after_refusal(
            _PAIR_BIAS_ROWS_REFUSED, tuple(z.padded_shape), lambda: self._z_bias(z),
            lambda rows: pair_row_blocks(self._z_bias, (z,), rows),
            rows=256, tag="of3 dit pair bias")          # [1, N, N, 16]
        zb = ttnn.to_layout(zb, ttnn.ROW_MAJOR_LAYOUT)
        zb = ttnn.permute(zb, (0, 3, 1, 2))             # [1, 16, N, N]
        zb = ttnn.to_layout(zb, ttnn.TILE_LAYOUT)
        return ttnn.add_(zb, mask_bias)                 # + mask_bias [1,1,1,N]

    def _attend(self, q, k, v, bias, cache):
        # Manual attention with an fp32 softmax: the fused SDPA does softmax in bf16,
        # and its per-block error (~0.998) compounds to ~0.967 over 24 blocks. A CPU
        # bf16 control with an fp32 softmax holds 0.99996 over the same stack, so the
        # softmax precision is the lever. Compute scores in bf16 (fp32 dest acc), cast
        # up for a numerically-stable fp32 softmax, cast back for attn@V.
        scale = HEAD_DIM ** -0.5
        sc = batched_matmul(q, ttnn.permute(k, (0, 1, 3, 2)),
                            compute_kernel_config=self.compute_kernel_config)
        sc = scale_add(sc, scale, bias)
        ttnn.deallocate(q); ttnn.deallocate(k)
        if cache is None:
            ttnn.deallocate(bias)
        sc = ttnn.typecast(sc, ttnn.float32)
        attn = site_softmax(sc, dim=-1, numeric_stable=True,
                            compute_kernel_config=self._softmax_ckc,
                            host_f64=self._softmax_f64)
        ttnn.deallocate(sc)
        attn = ttnn.typecast(attn, self._act_dtype)
        o = batched_matmul(attn, v, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(attn); ttnn.deallocate(v)
        return o

    def _sdpa32_mask(self, z, mask_bias):
        """The pair bias as `_sdpa32`'s fp32 mask. `_sdpa32` scales mask and scores together, so
        the bias is divided by the scale; the token axis is padded to `sdpa32_rows`."""
        zb = self._pair_bias(z, mask_bias)
        b = ttnn.multiply(zb, HEAD_DIM ** 0.5, dtype=ttnn.float32)
        ttnn.deallocate(zb)
        m = _T.sdpa32_mask(b)
        ttnn.deallocate(b)
        return m

    def _attend32(self, q, k, v, mask, cache):
        """softmax(q k^T * scale + bias) v as one `_sdpa32` program; the stack padded the token
        axis to `sdpa32_rows`."""
        o = _T._sdpa32(q, k, v, mask, HEAD_DIM ** -0.5)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        if cache is None:
            ttnn.deallocate(mask)
        return o

    def __call__(self, a, s, z, mask_bias, tok_mask_col, cache=None, s16=None, tok_ones=False):
        """`s16`: the stack's `shared_s`. `tok_ones`: the token mask is 1 on every row (dit_fuse)."""
        if self.fuse:
            return self._call_fused(a, s, z, mask_bias, tok_mask_col, cache, s16[0], tok_ones, s16[1:])
        lin, lp = self._lin, self.lp
        # Under lp each adaLN writes lp.act, read only by linears; the transition's hidden linears
        # write bf16, their product lp.mid, and every linear into the stream writes lp.out.
        act, out_dt = (lp.act, lp.out) if lp else (None, None)
        mid = {"dtype": lp.mid} if lp else {}
        # AdaLN-conditioned a.
        st = self._s16_terms(s16, 0) if s16 else None
        a_ln = self.adaln_a(a, s, dtype=act, s_terms=st)
        if st:
            ttnn.deallocate(st[0]); ttnn.deallocate(st[1])

        if self.sdpa32:
            bias = _cached(cache, (id(self), "pair_bias32"),
                           lambda: self._sdpa32_mask(z, mask_bias))
        else:
            bias = _cached(cache, (id(self), "pair_bias"),
                           lambda: self._pair_bias(z, mask_bias))

        # Fused padded qkv -> heads; `_sdpa32` reads them at lp.qkv (dit_qkv16: bf16), output fp32.
        qkv = self._lin_lp(a_ln, self.qkv_w, lp.qkv if lp and self.sdpa32 else out_dt,
                           bias=self.qkv_b)                              # [1, N, 3072]
        qkv = ttnn.unsqueeze(qkv, 1)
        if self.sdpa32:
            # The heads read and written in place inside the one attention program, so
            # nlp_create_qkv_heads and nlp_concat_heads (63 us a block on Blackhole) do not run.
            # Same arithmetic: digests unchanged on OpenFold3 and OpenBind-0 at 512 tokens.
            o = _T._sdpa32_fused(qkv, bias, HEAD_DIM ** -0.5, N_HEADS)
            ttnn.deallocate(qkv)
            if cache is None:
                ttnn.deallocate(bias)
            S, n = o.shape[0], o.shape[2]
            o = ttnn.reshape(o, (S, n, N_HEADS * PADDED_HEAD_DIM))
        else:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=N_HEADS, num_kv_heads=N_HEADS, transpose_k_heads=False)
            ttnn.deallocate(qkv)
            o = (self._attend32 if self.sdpa32 else self._attend)(q, k, v, bias, cache)
            S, n = o.shape[0], o.shape[2]
            o = ttnn.reshape(ttnn.experimental.nlp_concat_heads(o),
                             (S, n, N_HEADS * PADDED_HEAD_DIM))  # [S, N, 1024], pad lanes zero
        if self.lp is None and o.dtype != self._act_dtype:
            o = ttnn.typecast(o, self._act_dtype)        # bf16 stack: _sdpa32 writes fp32
        # Query gate (flat == per-head: g.view(N,H,d) * o(H,N,d) == flat multiply).
        # The gate at o's format: a multiply whose b carries a fused sigmoid is wrong on mixed
        # operand formats (Protenix-v2's dit_mm16 grade, perf/spd_difflin/k1_check.py).
        g = self._lin_lp(a_ln, self.w_g, o.dtype)
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], **mid)
        ttnn.deallocate(g)
        o = self._lin_lp(o, self.w_o, out_dt)           # [1, N, 768]
        # APB output gate from s.
        sig = "sigmoid" if s16 and self.sig16 else None
        og = (self._s16_lin(s16[0], self.s16[1], sig) if s16
              else lin(s, self.w_ada_out, bias=self.b_ada_out))
        ttnn.deallocate(a_ln)
        if sig:
            a2 = gated_add(a, o, og)
        else:
            o = ttnn.multiply(o, og, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            a2 = ttnn.add(a, o)
        ttnn.deallocate(og); ttnn.deallocate(o)
        a = a2

        # Conditioned SwiGLU transition.
        st = self._s16_terms(s16, 1) if s16 else None
        a_t = self.adaln_t(a, s, dtype=act, s_terms=st)
        if st:
            ttnn.deallocate(st[0]); ttnn.deallocate(st[1])
        b1 = self._lin_lp(a_t, self.w_la, ttnn.bfloat16, activation="silu")
        b2 = self._lin_lp(a_t, self.w_lb, ttnn.bfloat16)
        ttnn.deallocate(a_t)
        bb = ttnn.multiply(b1, b2, **mid)
        ttnn.deallocate(b1); ttnn.deallocate(b2)
        out = self._lin_lp(bb, self.w_lout, out_dt)
        ttnn.deallocate(bb)
        lg = self._s16_lin(s16[0], self.s16[2], sig) if s16 else lin(s, self.w_lg, bias=self.b_lg)
        out = ttnn.multiply(out, lg, **({} if sig else
                                        {"input_tensor_b_activations": [ttnn.UnaryOpType.SIGMOID]}))
        ttnn.deallocate(lg)
        a = mask_add(a, out, tok_mask_col)
        ttnn.deallocate(out)
        return a


class OF3DiffusionTransformer(Module):
    """OF3 ``DiffusionTransformer`` (Algorithm 23, non-cross path) on device.

    Inputs (device bf16):
        a:        [S, N, 768]   token single (evolving)
        s:        [S or 1, N, 384] conditioning single (si); at 1 it broadcasts
        z:        [1, N, N, 128] conditioning pair (zij, shared by every sample)
        token_mask:   [1, N]    shared
        tok_mask_col: [1, N, 1] token mask for transition masking, shared
    Returns [S, N, 768].

    **S is the sample axis and the stack is written once for every S.** Only ``a`` has to
    differ between samples. ``z`` and the masks are pure functions of the trunk output and
    ``s`` of the step's noise level, which the sampler's samples share, so they keep a
    leading dim of 1 and broadcast: AdaLN's conditioning half and the two ``s`` gates then
    run once per block instead of once per sample. The per-block pair
    bias is the expensive shared term -- ``[1, 16, N, N]``, cached, and added to the
    ``[S, 16, N, N]`` scores by broadcast, never replicated, which is what keeps the sample
    axis nearly free on DRAM (Protenix measured 1.9 GB per replicated copy at 1095 tokens,
    ``protenix.py:1462``).

    The same is true of Protenix's token DiT, which has carried a sample axis since its
    multiplicity path shipped, and of the AdaLN and matmul verbs underneath: a batched
    matmul against a 2-D weight and a broadcasting binary both already reduce their
    gradient over the leading axis (``taped_ttnn._v_matmul``, ``autograd._reduce_to``), so
    the backward needs no sample-axis case either.
    """

    #: The capability gate every AF3-family sampler on this fleet uses. OFF is the S=1 path,
    #: which is bit-identical to what shipped before the axis existed; a caller that batches
    #: must check the flag rather than assume, because a batched call that the hardware
    #: refuses has to fall back to the loop rather than fail the fold.
    supports_multiplicity = True

    def __init__(self, state_dict, compute_kernel_config, n_blocks=N_BLOCKS):
        super().__init__(state_dict, compute_kernel_config)
        self._act_dtype = _dtype(ttnn.bfloat16)
        self._w = {k: v for k, v in self.weights.data.items()}
        # The checkpoint picks the variant: OpenBind carries one shared pre-stack
        # layer_norm_z, OF3-preview2 carries one per block. Nothing else differs. The shared
        # norm is applied inside each block's pair bias rather than once for the stack: the
        # biases are cached for the rollout, so a stack-level normed pair would sit beside them
        # for all 200 steps (1207959552 B at 1536 tokens) for the sake of 23 layer norms that
        # run once per sample. Same op on the same input, so the same bits.
        shared_ln = self._w.get("layer_norm_z.weight")
        ln_z_w = None if shared_ln is None else ttnn.from_torch(
            shared_ln, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self._act_dtype)
        self.blocks = [_DiTBlock(_sub(self._w, f"blocks.{b}"), compute_kernel_config,
                                 ln_z_w=ln_z_w)
                       for b in range(n_blocks)]
        self.n_blocks = n_blocks
        self._tok_ones: dict = {}
        for b, blk in enumerate(self.blocks):
            blk.idx = b
        # dit_shoist (on dit_fuse): every block's s-path linears read the same `s` / `s_hat` within a
        # call, so the stack runs them as three wide linears, 48 x 768 columns each: the adaLN
        # sigmoid scales (block b's attention adaLN at column 2b x 768, its transition adaLN at
        # 2b + 1), the adaLN shifts, and the two sigmoid gates (APB output at 2b, transition at
        # 2b + 1). The blocks read their columns in place (`addcmul_at`): 144 [N,384]x[384,768]
        # linears a step become 3, 0.48 s of a 512-token p150a fold's device time (cen68).
        self.hoist = None
        if self.blocks[0].fuse and self._act_dtype == ttnn.bfloat16 and _T.lever("dit_shoist"):
            bf = lambda w: ttnn.from_torch(w.float().contiguous(), dtype=ttnn.bfloat16,  # noqa: E731
                                           layout=ttnn.TILE_LAYOUT, device=self.device)
            sc_w, sc_b, sb_w, g_w, g_b = [], [], [], [], []
            for blk in self.blocks:
                for ln in (blk.adaln_a, blk.adaln_t):
                    w_sc, b_sc, w_sb = ln.folded_weights()
                    sc_w.append(w_sc); sc_b.append(b_sc.reshape(-1)); sb_w.append(w_sb)
                for k in ("attention_pair_bias.linear_ada_out", "conditioned_transition.linear_g"):
                    g_w.append(blk._w[k + ".weight"].t()); g_b.append(blk._w[k + ".bias"].reshape(-1))
            cat = lambda xs: torch.cat([x.float() for x in xs], dim=-1)  # noqa: E731
            self.hoist = (bf(cat(sc_w)), bf(cat(sc_b).reshape(1, -1)), bf(cat(sb_w)),
                          bf(cat(g_w)), bf(cat(g_b).reshape(1, -1)))

    def _masks(self, token_mask, N, padded_N, rows):
        """``(mask_bias [1,1,1,padded_N], tok_mask_col at rows or ())``: -1e9 on padded and masked
        keys; the column mask is ``()`` when the caller's ``tok_mask_col`` already fits."""
        tok = ttnn.to_torch(token_mask).float().reshape(-1)  # [N]
        # Whether the transition's mask multiply is an identity on every row the blocks compute.
        self._tok_ones[rows] = bool(rows == N and (tok == 1.0).all())
        mb_t = torch.full((padded_N,), -1e9, dtype=torch.float32)
        mb_t[:N] = (tok - 1.0) * 1e9
        mask_bias = ttnn.from_torch(mb_t.reshape(1, 1, 1, padded_N), layout=ttnn.TILE_LAYOUT,
                                    device=self.device, dtype=self._act_dtype)
        if rows == N:
            return mask_bias, ()
        tmc_p = torch.zeros(1, rows, 1, dtype=torch.float32)
        tmc_p[0, :N, 0] = tok
        return mask_bias, ttnn.from_torch(tmc_p, layout=ttnn.TILE_LAYOUT, device=self.device,
                                          dtype=self._act_dtype)

    def _wide_lin(self, x, w, bias=None, activation=None):
        """One of dit_shoist's wide [rows, 384] x [384, 36864] linears. On Wormhole, up to 512 rows, a 2D multicast
        program whose output block (2 x 18 tiles) fits L1: ttnn's auto program for this shape takes 1146 us at
        1000 MHz, this one 922 us at the same fidelity and accumulation, rel. RMS error vs fp32 equal (shw, op_probe
        sw). Else ttnn's choice."""
        blk = self.blocks[0]
        g = x.device().compute_with_storage_grid_size()
        mt, nt = -(-x.padded_shape[-2] // TILE), w.padded_shape[-1] // TILE
        pm, pn = -(-mt // g.y), -(-nt // g.x)
        if not (blk._wormhole and pm <= 2 and pn % 18 == 0 and w.padded_shape[-2] % (4 * TILE) == 0):
            return blk._lin(x, w, bias=bias, activation=activation)
        hifi2 = blk._act_dtype == ttnn.bfloat16 and _T.lever("dit_hifi2")
        act = None if activation is None else ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(g.x, g.y), in0_block_w=4, out_subblock_h=pm, out_subblock_w=2,
            out_block_h=pm, out_block_w=18, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
            fused_activation=act, fuse_batch=True)
        return ttnn.linear(x, w, bias=bias, program_config=pc,
                           compute_kernel_config=blk.ckc_hifi2 if hifi2 else blk.compute_kernel_config)

    def shared_s(self, s_d):
        """What every block of one call reads off ``s`` and the stack computes once (``s16`` in
        ``_DiTBlock.__call__``), or None. The caller frees the tensors after the last block."""
        if self.blocks[0].fuse:
            s_hat = ttnn.layer_norm(s_d, epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
            if self.hoist is None:
                return (s_hat,)
            sc_w, sc_b, sb_w, g_w, g_b = self.hoist
            lin = self._wide_lin
            return (s_hat, lin(s_hat, sc_w, sc_b, "sigmoid"), lin(s_hat, sb_w),
                    lin(s_d, g_w, g_b, "sigmoid"))
        if self.blocks[0].s16:
            return (ttnn.typecast(s_d, ttnn.bfloat16),
                    ttnn.typecast(ttnn.layer_norm(s_d, epsilon=1e-5,
                                                  compute_kernel_config=self.compute_kernel_config),
                                  ttnn.bfloat16))
        return None

    def __call__(self, a, s, z, token_mask, tok_mask_col, cache=None):
        # Pad to the tile-aligned logical width so the SDPA's tiled key extent is
        # fully masked. from_torch pads *storage* with 0 (="unmasked" for an additive
        # mask), so a logical-N mask leaves the tile-padded keys (N -> ceil(N/32)*32)
        # unmasked and they leak garbage into valid queries, compounding across the
        # 24 blocks (stack PCC collapses to ~0.3). Padding the inputs to the tile
        # width here and marking padded keys -1e9 in mask_bias closes the leak; valid
        # positions are unaffected (padded queries' outputs are stripped at readout,
        # padded keys are masked out of every valid query's softmax).
        N = token_mask.shape[-1]
        S = a.shape[0]
        padded_N = bucketed_width(N, TILE)
        rows = _T.sdpa32_rows(padded_N) if self.blocks[0].sdpa32 else padded_N
        # The masks are functions of token_mask alone, so a rollout builds them once. Building
        # them reads the mask back to the host, which on every step stalled the host until the
        # atom encoder had drained, and a traced step cannot hold a read or an upload at all.
        mask_bias, tmc_d = _cached(cache, (id(self), "masks", rows),
                                   lambda: self._masks(token_mask, N, padded_N, rows))
        if isinstance(tmc_d, tuple):
            tmc_d = tok_mask_col
        if padded_N == N:
            a_d, s_d, z_d = a, s, z
        else:
            a_d = _pad_single(a, padded_N, self._act_dtype)
            s_d = _pad_single(s, padded_N, self._act_dtype)
            z_d = _pad_pair(z, padded_N, self._act_dtype)
        # dit_sdpa32 runs at sdpa32_rows. The stream is padded to it here, once a call, not q, k
        # and v in every block: rows are independent everywhere but the attention, and its mask
        # (`_sdpa32_mask`, built from the padded_N bias) hides the new keys. The same padding
        # Protenix's token DiT does.
        if rows != padded_N:
            a_d, s_d = (ttnn.pad(x, [(0, 0), (0, rows - padded_N), (0, 0)], value=0.0)
                        for x in (a_d, s_d))
        l1 = self.blocks[0].l1(a_d)
        if l1:
            # dit_l1: copies, so the caller's tensors are untouched; ours are freed below.
            a_d, s_d = (ttnn.to_memory_config(x, l1["memory_config"]) for x in (a_d, s_d))
        x = a_d
        s16 = self.shared_s(s_d)
        for blk in self.blocks:
            x = blk(x, s_d, z_d, mask_bias, tmc_d, cache=cache, s16=s16,
                    tok_ones=self._tok_ones.get(rows, False))
        for t in s16 or ():
            ttnn.deallocate(t)
        if l1:
            ttnn.deallocate(s_d)
            if x is not a_d:
                ttnn.deallocate(a_d)
            x, x_l1 = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), x
            ttnn.deallocate(x_l1)
        if rows == N:
            return x
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.slice(x, [0, 0, 0], [S, N, C_A])
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)
