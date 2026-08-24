"""X-Cell on Tenstorrent (ttnn): set-level perturbation prediction.

A from-scratch ttnn implementation scored against `tt_bio.xcell_reference`, which is itself a
transcription of the preprint's Appendix A (there is no upstream inference code to port -- see
that module's docstring). Structure follows `esmc.py`, the closest shipped model: a plain
transformer LM over a long token axis, built on `tenstorrent.Module` / `WeightScope` /
`get_device`. Weight keys are exactly the reference's `state_dict` keys, so the same checkpoint
loads into both and every component is directly comparable.

WHERE THE TOKEN AXES ARE, AND WHY ONLY ONE OF THEM IS PADDED
------------------------------------------------------------
X-Cell has three candidate axes and they need three different answers. All three were settled by
measurement on a p150 (card 2, ttnn 0.68.0), not by reading the source -- PLAYBOOKS §MODEL 2b.

**The gene axis (G+1, the real token axis): run it ragged, no pad, no mask.**
`ttnn.transformer.scaled_dot_product_attention` masks its own ragged tail *when the caller
supplies no additive bias*. Measured bias-free against torch fp32: relative error 0.068 at S=98,
0.073 at 129, 0.062 at 450, against 0.028 at 32 and 0.040 at 64 -- the same bf16 noise floor at
ragged lengths as at aligned ones, not the 70x that an unmasked tail costs. The known defect
(`token-axis-must-bucket-to-multiple-of-32`) needs a caller bias covering only the logical length;
with no bias there is nothing for the padded columns to enter the softmax at. So this axis is
IMMUNE BY ROUTE and padding it would be strictly worse: a bucketed gene axis *must* then carry a
mask, `mask_shape[2] == q_shape[2]` refuses the cheap key-only `[N,1,1,K]` broadcast bias, and a
full `[N,1,Q,K]` bias at G=4000 is a 32 MB tensor per call to protect an axis that measures clean
without it. Padding without masking is not an option either: measured 3.1x the reference error
(0.1245 vs 0.0400 at G=98 over a 128 bucket), which is the defect itself.

Per-length recompilation, the other reason to bucket, does not bite here the way it does on ESMC:
a protein LM sees a new length per sequence, whereas X-Cell's gene axis is fixed by the dataset
and the `n_genes` subsample for the whole run, so one program serves every call.

**The context axis (6 priors): always pad to a full tile and always mask.**
Six keys can never be a multiple of 32, so this axis is ragged at every single call. Bias-free it
also measures clean (unmasked and masked agree to the last bit at C=6). But a *missing* prior
source has to be masked out of the attention, and that means a caller bias, which is exactly the
condition the defect needs. Rather than branch on "is anything missing" -- an opt-in guard, which
PLAYBOOKS forbids -- this pads the context to 32 and supplies a full `[N,1,Q,32]` bias on every
call, with -1e9 on the padded columns and on absent sources alike. One path, default-safe, and at
32 keys against a 4001-token query the mask is ~256 kB, not 32 MB.

**The cell-set axis (S, folded into batch): a batch dim, not a token axis.**
Appendix A.1.1 reshapes `(B, S, G')` to `(B*S, G')` and nothing in the forward pass mixes cells:
the perturbation context is replicated across the set, the CLS embedding is per cell, the output
is per cell. The set-level part of X-Cell is its training objective, not its inference graph. So
no reduce runs over this axis, `TILE_LAYOUT` does not tile it, and it is padded only to keep the
compiled program count down, never for correctness. Said plainly because "bucket both token axes"
would otherwise look unfinished here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn

from tt_bio.tenstorrent import (
    Module,
    TorchWrapper,
    Weights,
    _dtype,
    get_device,
)
from tt_bio.xcell_reference import (
    PRIOR_SOURCES,
    REVEAL_FRACTIONS,
    VALUE_CLIP,
    XCELL_MINI,
    XCELL_ULTRA,
    XCellConfig,
)

# The cross-attention context is 6 tokens and therefore always ragged; pad it to one full tile
# and mask, on every call. See the module docstring.
CONTEXT_TILE = 32
NEG = -1e9

# The cell-set axis is a batch dim, so this multiple buys compiled-program reuse, not correctness.
SET_BUCKET = 32

_ROW = lambda x: x.reshape(1, -1)          # torch nn.Linear bias -> ttnn row vector
_ASIS = lambda x: x                        # embedding tables are indexed, not matmul'd


class RMSNorm(Module):
    """A17. Learnable scale, no bias."""

    def __init__(self, eps: float, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.weight = self.torch_to_tt("weight")
        self.eps = eps

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps,
                             compute_kernel_config=self.compute_kernel_config)


class LayerNorm(Module):
    def __init__(self, eps: float, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.weight = self.torch_to_tt("weight")
        self.bias = self.torch_to_tt("bias")
        self.eps = eps

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps,
                               compute_kernel_config=self.compute_kernel_config)


class ValueEncoder(Module):
    """A3: LayerNorm(W2 ReLU(W1 clip(x, -inf, 512)) + b2).

    `W1` is d x 1, so `W1 x` is a scalar times a d-vector, not a contraction. Running it as a
    broadcast multiply of `[N,G,1]` against `[1,1,d]` keeps it exact and skips a K=1 matmul, whose
    contraction would be 31/32 tile padding and would depend on that padding being zero.
    """

    def __init__(self, cfg: XCellConfig, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.w1 = self.torch_to_tt("fc1.weight", transform=lambda x: x.reshape(1, 1, -1))
        self.fc2_w = self.torch_to_tt("fc2.weight")
        self.fc2_b = self.torch_to_tt("fc2.bias", transform=_ROW)
        self.norm = LayerNorm(cfg.norm_eps, self.scope("norm"), ck)

    def __call__(self, x_col: ttnn.Tensor) -> ttnn.Tensor:
        h = ttnn.multiply(x_col, self.w1)          # [N,G,1] x [1,1,d] -> [N,G,d]
        h = ttnn.relu(h)
        h = self._lin(h, self.fc2_w, bias=self.fc2_b)
        return self.norm(h)


class InputEmbedding(Module):
    """A2, A4, A5: gene identity + value + perturbation mask, with the learned CLS prepended."""

    def __init__(self, cfg: XCellConfig, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.cfg = cfg
        self.gene = self.torch_to_tt("gene.weight", transform=_ASIS)
        self.gene_norm = LayerNorm(cfg.norm_eps, self.scope("gene_norm"), ck)
        self.value = ValueEncoder(cfg, self.scope("value"), ck)
        self.mask_tbl = self.torch_to_tt("mask.weight", transform=_ASIS)
        self.mask_norm = LayerNorm(cfg.norm_eps, self.scope("mask_norm"), ck)
        self.cls = self.torch_to_tt("cls.weight", transform=_ASIS)

    def raw_gene(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """The UN-normalized gene embedding: what the tied head reads (A26)."""
        return ttnn.embedding(tokens, self.gene, layout=ttnn.TILE_LAYOUT,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def __call__(self, values: ttnn.Tensor, tokens: ttnn.Tensor,
                 pert_mask: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        raw = self.raw_gene(tokens)
        e = self.gene_norm(raw)
        m = self.mask_norm(ttnn.embedding(pert_mask, self.mask_tbl, layout=ttnn.TILE_LAYOUT,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG))
        h = ttnn.add(ttnn.add(e, self.value(values)), m)
        ttnn.deallocate(e); ttnn.deallocate(m)
        # A5: position 0 is the learned CLS vector and gets no value or mask term.
        n = h.shape[0]
        cls = ttnn.repeat(ttnn.reshape(self.cls, (1, 1, -1)), (n, 1, 1))
        out = ttnn.concat([cls, h], dim=1)
        ttnn.deallocate(cls); ttnn.deallocate(h)
        return out, raw


class PriorProjection(Module):
    """A6-A10: two-layer LeakyReLU with a LayerNorm output, one instance per source row."""

    def __init__(self, cfg: XCellConfig, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.fc1_w = self.torch_to_tt("fc1.weight")
        self.fc1_b = self.torch_to_tt("fc1.bias", transform=_ROW)
        self.fc2_w = self.torch_to_tt("fc2.weight")
        self.fc2_b = self.torch_to_tt("fc2.bias", transform=_ROW)
        self.norm = LayerNorm(cfg.norm_eps, self.scope("norm"), ck)
        self.slope = cfg.leaky_slope

    def __call__(self, z: ttnn.Tensor) -> ttnn.Tensor:
        h = ttnn.leaky_relu(self._lin(z, self.fc1_w, bias=self.fc1_b), self.slope)
        h = self._lin(h, self.fc2_w, bias=self.fc2_b)
        return self.norm(h)


class SelfAttention(Module):
    """A12-A16 over the gene axis. Bias-free SDPA at the true length: see the module docstring."""

    def __init__(self, cfg: XCellConfig, *, qk_norm: bool, bias: bool, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.q_w = self.torch_to_tt("q.weight", dtype=_dtype())
        self.k_w = self.torch_to_tt("k.weight", dtype=_dtype())
        self.v_w = self.torch_to_tt("v.weight", dtype=_dtype())
        self.o_w = self.torch_to_tt("o.weight", dtype=_dtype())
        if bias:
            self.q_b = self.torch_to_tt("q.bias", transform=_ROW)
            self.k_b = self.torch_to_tt("k.bias", transform=_ROW)
            self.v_b = self.torch_to_tt("v.bias", transform=_ROW)
            self.o_b = self.torch_to_tt("o.bias", transform=_ROW)
        else:
            self.q_b = self.k_b = self.v_b = self.o_b = None
        self.q_norm = RMSNorm(cfg.norm_eps, self.scope("q_norm"), ck) if qk_norm else None
        self.k_norm = RMSNorm(cfg.norm_eps, self.scope("k_norm"), ck) if qk_norm else None

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        q = self._heads(self._lin(x, self.q_w, bias=self.q_b))
        k = self._heads(self._lin(x, self.k_w, bias=self.k_b))
        v = self._heads(self._lin(x, self.v_w, bias=self.v_b))
        if self.q_norm is not None:
            qn = self.q_norm(q); ttnn.deallocate(q); q = qn
            kn = self.k_norm(k); ttnn.deallocate(k); k = kn
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=self.d_head ** -0.5)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        m = self._merge_heads(o)
        ttnn.deallocate(o)
        out = self._lin(m, self.o_w, bias=self.o_b)
        ttnn.deallocate(m)
        return out

    def _heads(self, t: ttnn.Tensor) -> ttnn.Tensor:
        n, s, _ = t.shape
        return ttnn.permute(ttnn.reshape(t, (n, s, self.n_heads, self.d_head)), (0, 2, 1, 3))


class CrossAttention(Module):
    """A21. Query from the gene representations, key/value from the padded 6-token context."""

    def __init__(self, cfg: XCellConfig, *, bias: bool, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.q_w = self.torch_to_tt("q.weight", dtype=_dtype())
        self.k_w = self.torch_to_tt("k.weight", dtype=_dtype())
        self.v_w = self.torch_to_tt("v.weight", dtype=_dtype())
        self.o_w = self.torch_to_tt("o.weight", dtype=_dtype())
        if bias:
            self.q_b = self.torch_to_tt("q.bias", transform=_ROW)
            self.k_b = self.torch_to_tt("k.bias", transform=_ROW)
            self.v_b = self.torch_to_tt("v.bias", transform=_ROW)
            self.o_b = self.torch_to_tt("o.bias", transform=_ROW)
        else:
            self.q_b = self.k_b = self.v_b = self.o_b = None

    def __call__(self, x: ttnn.Tensor, context: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        q = self._heads(self._lin(x, self.q_w, bias=self.q_b), x.shape[1])
        k = self._heads(self._lin(context, self.k_w, bias=self.k_b), context.shape[1])
        v = self._heads(self._lin(context, self.v_w, bias=self.v_b), context.shape[1])
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=self.d_head ** -0.5)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        m = self._merge_heads(o)
        ttnn.deallocate(o)
        out = self._lin(m, self.o_w, bias=self.o_b)
        ttnn.deallocate(m)
        return out

    def _heads(self, t: ttnn.Tensor, s: int) -> ttnn.Tensor:
        n = t.shape[0]
        return ttnn.permute(ttnn.reshape(t, (n, s, self.n_heads, self.d_head)), (0, 2, 1, 3))


class ReluFFN(Module):
    """Mini's 1x ReLU feed-forward (Table A2: FFN dim 512 = 1 x d, activation ReLU)."""

    def __init__(self, state_dict: Weights, ck, *, prefix: str = ""):
        super().__init__(state_dict, ck)
        p = prefix
        self.fc1_w = self.torch_to_tt(f"{p}fc1.weight", dtype=_dtype())
        self.fc1_b = self.torch_to_tt(f"{p}fc1.bias", transform=_ROW)
        self.fc2_w = self.torch_to_tt(f"{p}fc2.weight", dtype=_dtype())
        self.fc2_b = self.torch_to_tt(f"{p}fc2.bias", transform=_ROW)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = ttnn.relu(self._lin(x, self.fc1_w, bias=self.fc1_b))
        out = self._lin(h, self.fc2_w, bias=self.fc2_b)
        ttnn.deallocate(h)
        return out


class SwiGLUFFN(Module):
    """A18: W_down(SiLU(W_gate x) * W_up x), no biases. Ultra's feed-forward."""

    def __init__(self, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.gate_w = self.torch_to_tt("gate.weight", dtype=_dtype())
        self.up_w = self.torch_to_tt("up.weight", dtype=_dtype())
        self.down_w = self.torch_to_tt("down.weight", dtype=_dtype())

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        g = ttnn.silu(self._lin(x, self.gate_w))
        u = self._lin(x, self.up_w)
        p = ttnn.multiply(g, u)
        ttnn.deallocate(g); ttnn.deallocate(u)
        out = self._lin(p, self.down_w)
        ttnn.deallocate(p)
        return out


class ModernBlock(Module):
    """A19-A22: pre-RMSNorm self-attention + SwiGLU, and at a cross layer a second pair."""

    def __init__(self, cfg: XCellConfig, *, cross: bool, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.attn_norm = RMSNorm(cfg.norm_eps, self.scope("attn_norm"), ck)
        self.attn = SelfAttention(cfg, qk_norm=True, bias=False,
                                  state_dict=self.scope("attn"), ck=ck)
        self.ffn_norm = RMSNorm(cfg.norm_eps, self.scope("ffn_norm"), ck)
        self.ffn = SwiGLUFFN(self.scope("ffn"), ck)
        self.cross = None
        if cross:
            self.cross_norm = RMSNorm(cfg.norm_eps, self.scope("cross_norm"), ck)
            self.cross = CrossAttention(cfg, bias=False, state_dict=self.scope("cross"), ck=ck)
            self.cross_ffn_norm = RMSNorm(cfg.norm_eps, self.scope("cross_ffn_norm"), ck)
            self.cross_ffn = SwiGLUFFN(self.scope("cross_ffn"), ck)

    def __call__(self, x, context, bias):
        x = ttnn.add(x, self.attn(self.attn_norm(x)))
        x = ttnn.add(x, self.ffn(self.ffn_norm(x)))
        if self.cross is not None:
            x = ttnn.add(x, self.cross(self.cross_norm(x), context, bias))
            x = ttnn.add(x, self.cross_ffn(self.cross_ffn_norm(x)))
        return x


class LegacyBlock(Module):
    """Mini's Post-LN block: residual first, LayerNorm after, biases throughout (Table A2)."""

    def __init__(self, cfg: XCellConfig, *, cross: bool, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.attn = SelfAttention(cfg, qk_norm=False, bias=True,
                                  state_dict=self.scope("attn"), ck=ck)
        self.attn_norm = LayerNorm(cfg.norm_eps, self.scope("attn_norm"), ck)
        self.ffn = ReluFFN(self.weights, ck)
        self.ffn_norm = LayerNorm(cfg.norm_eps, self.scope("ffn_norm"), ck)
        self.cross = None
        if cross:
            self.cross = CrossAttention(cfg, bias=True, state_dict=self.scope("cross"), ck=ck)
            self.cross_norm = LayerNorm(cfg.norm_eps, self.scope("cross_norm"), ck)
            self.cross_ffn = ReluFFN(self.weights, ck, prefix="cross_")
            self.cross_ffn_norm = LayerNorm(cfg.norm_eps, self.scope("cross_ffn_norm"), ck)

    def __call__(self, x, context, bias):
        x = self.attn_norm(ttnn.add(x, self.attn(x)))
        x = self.ffn_norm(ttnn.add(x, self.ffn(x)))
        if self.cross is not None:
            x = self.cross_norm(ttnn.add(x, self.cross(x, context, bias)))
            x = self.cross_ffn_norm(ttnn.add(x, self.cross_ffn(x)))
        return x


class Decoder(Module):
    """A24-A26. One body; the last step is `mlp` (Mini) or `tied` (Ultra), per Table A2."""

    def __init__(self, cfg: XCellConfig, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.head = cfg.output_head
        self.slope = cfg.leaky_slope
        self.scale = 1.0 / math.sqrt(cfg.d_model)
        self.fc1_w = self.torch_to_tt("fc1.weight", dtype=_dtype())
        self.fc1_b = self.torch_to_tt("fc1.bias", transform=_ROW)
        self.fc2_w = self.torch_to_tt("fc2.weight", dtype=_dtype())
        self.fc2_b = self.torch_to_tt("fc2.bias", transform=_ROW)
        self.fc3_w = self.torch_to_tt("fc3.weight", dtype=_dtype())
        self.fc3_b = self.torch_to_tt("fc3.bias", transform=_ROW)
        if self.head == "mlp":
            self.out_w = self.torch_to_tt("out.weight")
            self.out_b = self.torch_to_tt("out.bias", transform=_ROW)
        else:
            self.gene_bias = self.torch_to_tt("gene_bias.weight", transform=_ASIS)

    def __call__(self, h_genes, h_cls, raw_gene, tokens):
        # A24: gene output first, CLS second. h_cls is [N,1,d]; broadcast it over the gene axis.
        cls = ttnn.repeat(h_cls, (1, h_genes.shape[1], 1))
        d = ttnn.concat([h_genes, cls], dim=-1)
        ttnn.deallocate(cls)
        h = ttnn.leaky_relu(self._lin(d, self.fc1_w, bias=self.fc1_b), self.slope)
        ttnn.deallocate(d)
        h2 = ttnn.leaky_relu(self._lin(h, self.fc2_w, bias=self.fc2_b), self.slope)
        ttnn.deallocate(h)
        h3 = self._lin(h2, self.fc3_w, bias=self.fc3_b)   # A25, no activation after W3
        ttnn.deallocate(h2)
        if self.head == "mlp":
            out = self._lin(h3, self.out_w, bias=self.out_b)
            ttnn.deallocate(h3)
            return out                                     # [N,G,1]
        # A26: tied projection through the RAW gene embeddings, then the per-gene bias.
        prod = ttnn.multiply(h3, raw_gene)
        ttnn.deallocate(h3)
        dot = ttnn.sum(prod, dim=-1, keepdim=True)
        ttnn.deallocate(prod)
        dot = ttnn.multiply(dot, self.scale)
        b = ttnn.embedding(tokens, self.gene_bias, layout=ttnn.TILE_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.add(dot, b)
        ttnn.deallocate(dot); ttnn.deallocate(b)
        return out                                         # [N,G,1]


class XCellModel(Module):
    """embed -> L blocks (cross-attention at Table A2's indices) -> decoder."""

    def __init__(self, cfg: XCellConfig, state_dict: Weights, ck):
        super().__init__(state_dict, ck)
        self.cfg = cfg
        self.embed = InputEmbedding(cfg, self.scope("embed"), ck)
        self.priors = {
            name: PriorProjection(cfg, self.scope(f"priors.{name}"), ck)
            for name, _dim, _src in PRIOR_SOURCES
        }
        block_cls = LegacyBlock if cfg.block == "legacy" else ModernBlock
        cross = set(cfg.cross_attn_layers)
        self.blocks = [
            block_cls(cfg, cross=(i in cross), state_dict=self.scope(f"blocks.{i}"), ck=ck)
            for i in range(cfg.n_layers)
        ]
        self.final_norm = (RMSNorm(cfg.norm_eps, self.scope("final_norm"), ck)
                           if cfg.block == "modern" else None)
        self.decoder = Decoder(cfg, self.scope("decoder"), ck)

    def context(self, priors: dict[str, ttnn.Tensor], pert_token: ttnn.Tensor) -> ttnn.Tensor:
        """A6-A11 -> [N, CONTEXT_TILE, d], the six tokens then tile padding."""
        toks = [self.priors[name](priors[name]) for name, _d, _s in PRIOR_SOURCES]
        # A11: stop-gradient gene embedding of the perturbed gene, LayerNormed as in A2.
        # Autograd is not in play here, so the stop-gradient is a no-op on device.
        gene = self.embed.gene_norm(self.embed.raw_gene(pert_token))
        toks.append(gene)
        c = ttnn.concat(toks, dim=1)
        for t in toks:
            ttnn.deallocate(t)
        pad = CONTEXT_TILE - c.shape[1]
        if pad:
            c = ttnn.pad(c, [(0, 0), (0, pad), (0, 0)], value=0.0)
        return c

    def __call__(self, values, tokens, pert_mask, priors, pert_token, cross_bias):
        h, raw = self.embed(values, tokens, pert_mask)
        ctx = self.context(priors, pert_token)
        for block in self.blocks:
            h = block(h, ctx, cross_bias)
        ttnn.deallocate(ctx)
        if self.final_norm is not None:
            h = self.final_norm(h)
        genes = ttnn.slice(h, [0, 1, 0], [h.shape[0], h.shape[1], h.shape[2]])
        cls = ttnn.slice(h, [0, 0, 0], [h.shape[0], 1, h.shape[2]])
        ttnn.deallocate(h)
        out = self.decoder(genes, cls, raw, tokens)
        ttnn.deallocate(genes); ttnn.deallocate(cls); ttnn.deallocate(raw)
        return out


def cross_bias(n_rows: int, n_query: int, prior_missing: torch.Tensor | None,
               device=None) -> torch.Tensor:
    """The additive cross-attention bias, built on every call rather than when something is absent.

    Shape `[N, 1, Q, CONTEXT_TILE]`: `mask_shape[2] == q_shape[2]` is enforced by the SDPA op, so
    the cheap `[N,1,1,C]` key-only broadcast is not available. Columns past the six real sources
    are tile padding and get -1e9; a source flagged absent gets -1e9 too, which is the
    `key_padding_mask` of the appendix. Padding and absence are the same operation here, so they
    take the same code path and there is no "did anything go missing" branch to forget.
    """
    n_ctx = len(PRIOR_SOURCES) + 1
    bias = torch.zeros(n_rows, 1, n_query, CONTEXT_TILE, dtype=torch.float32)
    bias[:, :, :, n_ctx:] = NEG
    if prior_missing is not None:
        m = prior_missing.to(torch.bool)
        if m.shape[-1] != n_ctx:
            raise ValueError(f"prior_missing must have {n_ctx} columns, got {tuple(m.shape)}")
        bias[:, 0, :, :n_ctx] = torch.where(m[:, None, :], torch.tensor(NEG), bias[:, 0, :, :n_ctx])
    return bias


class XCell(TorchWrapper):
    """Top-level X-Cell: torch in, torch out.

    Weight keys are `tt_bio.xcell_reference.XCell`'s, so the same `state_dict` loads into the
    reference and into this, which is what makes the per-component PCC comparison meaningful.
    """

    def __init__(self, cfg: XCellConfig = XCELL_MINI, state_dict=None, *, fast: bool = False):
        super().__init__()
        self.cfg = cfg
        self.device = get_device()
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2 if not fast else ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.model = XCellModel(cfg, state_dict, self.compute_kernel_config)

    def _up(self, t: torch.Tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, layout=layout, device=self.device, dtype=dtype)

    def _ids(self, t: torch.Tensor):
        return ttnn.from_torch(t.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.device, dtype=ttnn.uint32)

    def forward(self, values, tokens, pert_mask, priors, pert_token, prior_missing=None):
        """One forward pass. Shapes match `xcell_reference.XCell.forward`."""
        n, g = values.shape
        bias = cross_bias(n, g + 1, prior_missing)
        out = self.model(
            self._up(values.reshape(n, g, 1)),
            self._ids(tokens),
            self._ids(pert_mask),
            {name: self._up(priors[name].reshape(n, 1, dim))
             for name, dim, _s in PRIOR_SOURCES},
            self._ids(pert_token.reshape(n, 1)),
            self._up(bias),
        )
        return ttnn.to_torch(out).float().reshape(n, g)


@torch.no_grad()
def predict(model: XCell, control, tokens, priors, pert_token, prior_missing=None,
            n_steps: int = 4, generator: torch.Generator | None = None):
    """The 4-step cumulative refinement of A.4, identical in structure to the reference's.

    The rank comparison and the scatter run on the host between steps, which is what the trace
    two-gate rule cares about: there IS a host-only numerical op between device replays, so a
    single trace spanning all four steps is not viable as written. Per-step tracing is, and the
    ranks are drawn once before the loop rather than per step, so moving the update on-device to
    recover a full-loop trace is a live option for pass 4 rather than a dead end.
    """
    if n_steps < 1 or n_steps > len(REVEAL_FRACTIONS):
        raise ValueError(f"n_steps must be 1..{len(REVEAL_FRACTIONS)}, got {n_steps}")
    n, g = control.shape
    x = control.clone()
    mask = torch.zeros_like(control, dtype=torch.long)
    ranks = torch.rand(n, g, generator=generator)
    prev, pred = 0.0, None
    for alpha in REVEAL_FRACTIONS[:n_steps]:
        pred = model.forward(x, tokens, mask, priors, pert_token, prior_missing)
        newly = (ranks <= alpha) & (ranks > prev)
        x = torch.where(newly, pred.to(x.dtype), x)
        mask = torch.where(newly, torch.ones_like(mask), mask)
        prev = alpha
    return pred
