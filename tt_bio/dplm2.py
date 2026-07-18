"""DPLM-2 ESM backbone on Tenstorrent (ttnn).

DPLM-2 (ByteDance, github.com/bytedance/dplm) is a discrete-diffusion protein
language model that co-generates sequence and structure. Its backbone is a
HuggingFace ESM-2 transformer (facebook/esm2_t30_150M_UR50D family) with two
modifications from byprot.models.dplm2.modules.dplm2_modeling_esm:

  1. ModifiedRotaryEmbedding — when the input mixes both modalities (struct
     tokens first half, aa tokens second half), the RoPE table is built for
     L/2 and the SAME phases apply to both halves.
  2. ModifiedEsmSelfAttention — query pre-scaled by head_dim**-0.5 and SDPA
     called with scale=1.0. We fold that into SDPA(scale=head_dim**-0.5)
     applied to rotary(q)/rotary(k), which is bit-equivalent (rotation is
     linear) and avoids a bf16 pre-scale.

This pass (p1) implements the backbone forward only — embed -> 30 pre-norm
ESM-2 blocks (qk-bias attention + exact-erf GELU FFN) -> emb_layer_norm_after
-> EsmLMHead — at bf16, parity-checked (PCC >= 0.999) against the PyTorch
reference in tests/dplm2_reference.py (itself validated against the official
byprot EsmForDPLM2). The discrete-diffusion refinement loop, the LFQ 3D
structure tokenizer/detokenizer, and the CLI/job-path wiring are pass 2; see
docs/dplm2-port.md.

We reuse the tt-bio framework primitives (Module / WeightScope / torch_to_tt /
_lin / _split_heads / _merge_heads / SDPA program config) and the ESMC RoPE
helpers (rope_tables / apply_rotary). The ESM-2 blocks themselves differ from
ESMC (GELU FFN vs SwiGLU, separate q/k/v biases vs fused qkv, pre-norm with
output.dense+residual vs ESMC's residual scaling) so they are implemented
here rather than reused from tt_bio.esmc.
"""

from __future__ import annotations

import torch
import ttnn

from tt_bio.tenstorrent import (
    Module, TorchWrapper, WeightScope, Weights, _dtype, get_device,
    _sdpa_program_config_for_lengths,
)
from tt_bio.esmc import rope_tables, apply_rotary  # reuse the ESMC RoPE helpers
from tt_bio.dplm2_sampler import DPLM2Sampler, DPLM2Tokenizer  # diffusion loop + vocab

# airkingbd/dplm2_150m config.json (verified against the downloaded checkpoint).
DPLM2_150M = dict(
    hidden_size=640, num_attention_heads=20, num_hidden_layers=30,
    intermediate_size=2560, vocab_size=8229, pad_token_id=1, mask_token_id=32,
    layer_norm_eps=1e-5, max_position_embeddings=1026, token_dropout=True,
)
AA_VOCAB_BOUND = 33  # aa tokens id < 33; struct tokens id >= 33
MASK_RATIO_TRAIN = 0.15 * 0.8  # 0.12, hardcoded in HF EsmEmbeddings


def modality_type(input_ids: torch.Tensor, pad_id: int = 1) -> torch.Tensor:
    """0=struct, 1=aa, 2=pad — matches DPLM2.get_modality_type."""
    mask = input_ids.ne(pad_id)
    m = ((input_ids < AA_VOCAB_BOUND) & mask).int()
    m[~mask] = 2
    return m


class Embedding(Module):
    """Word embeddings + ESM token-dropout (config.token_dropout=True).

    No position embeddings (rotary); no pre-LN (emb_layer_norm_before=False).
    Padded positions are zeroed. token-dropout zeros mask-token rows and
    rescales by (1-0.12)/(1-mask_ratio_observed) — done on device via two
    multiplies (keep-mask + per-batch scale).
    """

    def __init__(self, state_dict: Weights, compute_kernel_config, cfg):
        super().__init__(state_dict, compute_kernel_config)
        # Keep the fp32 embedding table on host: ttnn.embedding requires bf16
        # weights, and bf16-quantizing the table injects a ~0.5% error into the
        # initial residual that DPLM-2's unscaled stream amplifies ~250x over 30
        # layers. A one-time host fp32 lookup keeps the residual stream exact.
        self._weight = self.weights["word_embeddings.weight"].float().contiguous()
        self._token_dropout = cfg["token_dropout"]
        self._mask_id = cfg["mask_token_id"]
        self._pad_id = cfg["pad_token_id"]

    def __call__(self, tokens_tt, input_ids, input_mask):
        emb = torch.nn.functional.embedding(input_ids, self._weight)  # [B,L,D] fp32 host
        if self._token_dropout:
            keep = (input_ids != self._mask_id).float().unsqueeze(-1)  # [B,L,1]
            src = input_mask.sum(-1).clamp(min=1)
            obs = (input_ids == self._mask_id).sum(-1).float() / src
            scale = (1.0 - MASK_RATIO_TRAIN) / (1.0 - obs)  # [B]
            emb = emb * keep * scale.view(-1, 1, 1)
        emb = emb * input_mask.float().unsqueeze(-1)  # zero padded positions
        return ttnn.from_torch(
            emb, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)


class Attention(Module):
    """Pre-norm ESM-2 self-attention with q/k/v biases + RoPE.

    h = LayerNorm(x); q,k,v = Linear(h) (bias); RoPE(q),RoPE(k);
    o = SDPA(q,k,v, attn_mask, scale=head_dim**-0.5); out = Linear_o(o) + x.

    Run in fp32: DPLM-2 has no ESMC-style residual scaling, so the residual
    stream reaches magnitude ~1e3 on real proteins and bf16 matmul/SDPA error
    compounds to ~0.995 PCC over 30 layers. fp32 clears 0.999 (parity bar).
    This is a gated precision deviation from the bf16 default — see docs/dplm2-port.md.
    """

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.ln_w = self.torch_to_tt("LayerNorm.weight")
        self.ln_b = self.torch_to_tt("LayerNorm.bias")
        row = lambda x: x.reshape(1, -1)
        # q/k/v in fp32 (then cast to bf16 for ttnn SDPA, which is bf16-only but
        # internally high-precision); output projection fp32 for the residual add.
        # LN stays bf16: ttnn's bf16 layernorm kernel is more accurate than its
        # fp32 one (measured), and LN normalizes x so the bf16 ULP on the ~1e3
        # residual doesn't leak unbounded.
        self.q_w = self.torch_to_tt("self.query.weight", dtype=ttnn.float32)
        self.q_b = self.torch_to_tt("self.query.bias", transform=row, dtype=ttnn.float32)
        self.k_w = self.torch_to_tt("self.key.weight", dtype=ttnn.float32)
        self.k_b = self.torch_to_tt("self.key.bias", transform=row, dtype=ttnn.float32)
        self.v_w = self.torch_to_tt("self.value.weight", dtype=ttnn.float32)
        self.v_b = self.torch_to_tt("self.value.bias", transform=row, dtype=ttnn.float32)
        self.o_w = self.torch_to_tt("output.dense.weight", dtype=ttnn.float32)
        self.o_b = self.torch_to_tt("output.dense.bias", transform=row, dtype=ttnn.float32)

    def __call__(self, x, cos, sin, attn_mask, joint):
        ck = self.compute_kernel_config
        d_model = x.shape[-1]
        head_dim = d_model // self.n_heads
        x_bf = ttnn.typecast(x, ttnn.bfloat16)
        h = ttnn.layer_norm(x_bf, weight=self.ln_w, bias=self.ln_b, epsilon=1e-5, compute_kernel_config=ck)
        h_f = ttnn.typecast(h, ttnn.float32)
        ttnn.deallocate(h)
        q = self._lin(h_f, self.q_w, bias=self.q_b, dtype=ttnn.float32)
        k = self._lin(h_f, self.k_w, bias=self.k_b, dtype=ttnn.float32)
        v = self._lin(h_f, self.v_w, bias=self.v_b, dtype=ttnn.float32)
        ttnn.deallocate(h_f)
        q = ttnn.typecast(q, ttnn.bfloat16)
        k = ttnn.typecast(k, ttnn.bfloat16)
        v = ttnn.typecast(v, ttnn.bfloat16)
        qkv = ttnn.concat([q, k, v], dim=-1)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        q, k, v = self._split_heads(qkv, self.n_heads)  # [B,H,L,d_h]
        if joint:
            half = q.shape[2] // 2
            cos_h, sin_h = rope_tables(half, head_dim, device=self.device)
            q1, q2 = ttnn.chunk(q, 2, dim=2)
            k1, k2 = ttnn.chunk(k, 2, dim=2)
            q1 = apply_rotary(q1, cos_h, sin_h); q2 = apply_rotary(q2, cos_h, sin_h)
            k1 = apply_rotary(k1, cos_h, sin_h); k2 = apply_rotary(k2, cos_h, sin_h)
            q = ttnn.concat([q1, q2], dim=2); k = ttnn.concat([k1, k2], dim=2)
            ttnn.deallocate(q1); ttnn.deallocate(q2)
            ttnn.deallocate(k1); ttnn.deallocate(k2)
        else:
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, scale=head_dim ** -0.5,
            program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]),
        )
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        o = self._merge_heads(o)
        o = ttnn.typecast(o, ttnn.float32)
        out = self._lin(o, self.o_w, bias=self.o_b, dtype=ttnn.float32)
        ttnn.deallocate(o)
        ttnn.deallocate(x_bf)
        return ttnn.add(x, out)


class FFN(Module):
    """Pre-norm ESM-2 feed-forward: h=LayerNorm(x); GELU(Linear_inter(h)); Linear_o + x."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.ln_w = self.torch_to_tt("LayerNorm.weight")
        self.ln_b = self.torch_to_tt("LayerNorm.bias")
        row = lambda x: x.reshape(1, -1)
        self.i_w = self.torch_to_tt("intermediate.dense.weight", dtype=ttnn.float32)
        self.i_b = self.torch_to_tt("intermediate.dense.bias", transform=row, dtype=ttnn.float32)
        self.o_w = self.torch_to_tt("output.dense.weight", dtype=ttnn.float32)
        self.o_b = self.torch_to_tt("output.dense.bias", transform=row, dtype=ttnn.float32)

    def __call__(self, x):
        ck = self.compute_kernel_config
        x_bf = ttnn.typecast(x, ttnn.bfloat16)
        h = ttnn.layer_norm(x_bf, weight=self.ln_w, bias=self.ln_b, epsilon=1e-5, compute_kernel_config=ck)
        h_f = ttnn.typecast(h, ttnn.float32)
        ttnn.deallocate(h)
        inter = self._lin(h_f, self.i_w, bias=self.i_b, dtype=ttnn.float32)
        ttnn.deallocate(h_f)
        inter = ttnn.gelu(inter)  # default approximate="none" -> erf, matches ESM gelu
        out = self._lin(inter, self.o_w, bias=self.o_b, dtype=ttnn.float32)
        ttnn.deallocate(inter)
        ttnn.deallocate(x_bf)
        return ttnn.add(x, out)


class Layer(Module):
    """One ESM-2 block: attention sublayer then FFN sublayer (both pre-norm)."""

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.attn = Attention(n_heads, self.scope("attention"), compute_kernel_config)
        self.ffn = FFN(self.scope(""), compute_kernel_config)

    def __call__(self, x, cos, sin, attn_mask, joint):
        x = self.attn(x, cos, sin, attn_mask, joint)
        return self.ffn(x)


class EsmLMHead(Module):
    """EsmLMHead: dense -> gelu -> LayerNorm -> decoder + bias."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        row = lambda x: x.reshape(1, -1)
        # Run the whole head in fp32: it is cheap (640x640 + 640x8229) and the
        # 8229-wide decoder matmul in bf16 drops logit PCC just under 0.999.
        self.d_w = self.torch_to_tt("dense.weight", dtype=ttnn.float32)
        self.d_b = self.torch_to_tt("dense.bias", transform=row, dtype=ttnn.float32)
        self.ln_w = self.torch_to_tt("layer_norm.weight", dtype=ttnn.float32)
        self.ln_b = self.torch_to_tt("layer_norm.bias", dtype=ttnn.float32)
        self.dec_w = self.torch_to_tt("decoder.weight", dtype=ttnn.float32)
        self.dec_b = self.torch_to_tt("bias", transform=row, dtype=ttnn.float32)

    def __call__(self, x):
        ck = self.compute_kernel_config
        a = ttnn.typecast(x, ttnn.float32)
        a = self._lin(a, self.d_w, bias=self.d_b, dtype=ttnn.float32)
        a = ttnn.gelu(a)
        a = ttnn.layer_norm(a, weight=self.ln_w, bias=self.ln_b, epsilon=1e-5, compute_kernel_config=ck)
        logits = self._lin(a, self.dec_w, bias=self.dec_b, dtype=ttnn.float32)
        ttnn.deallocate(a)
        return logits


class DPLM2Model(Module):
    """Full DPLM-2 ESM backbone: embed -> N layers -> emb_layer_norm_after -> lm_head.

    __call__(tokens_tt, input_ids, input_mask, joint) -> (logits[B,L,vocab], hidden[B,L,d]).
    """

    def __init__(self, cfg, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.cfg = cfg
        self.n_heads = cfg["num_attention_heads"]
        n_layers = cfg["num_hidden_layers"]
        head_dim = cfg["hidden_size"] // self.n_heads
        self.head_dim = head_dim
        self.embed = Embedding(self.scope("esm.embeddings"), compute_kernel_config, cfg)
        self.layers = [
            Layer(self.n_heads, self.scope(f"esm.encoder.layer.{i}"), compute_kernel_config)
            for i in range(n_layers)
        ]
        self.final_ln_w = self.torch_to_tt("esm.encoder.emb_layer_norm_after.weight", dtype=ttnn.float32)
        self.final_ln_b = self.torch_to_tt("esm.encoder.emb_layer_norm_after.bias", dtype=ttnn.float32)
        self.head = EsmLMHead(self.scope("lm_head"), compute_kernel_config)

    def __call__(self, tokens_tt, input_ids, input_mask, joint):
        seq_len = tokens_tt.shape[-1]
        rope_len = seq_len // 2 if joint else seq_len
        cos, sin = rope_tables(rope_len, self.head_dim, device=self.device)

        x = self.embed(tokens_tt, input_ids, input_mask)  # fp32
        # Extended key-padding mask: [B,1,1,L], padded keys -> -inf so no token
        # attends to them. ESM-2 (and DPLM-2) mask keys only; pad query positions
        # still produce outputs but are ignored downstream. Without this, padded
        # positions corrupt the non-pad logits.
        ext = (1.0 - input_mask.float()).view(1, 1, 1, -1) * torch.finfo(torch.float32).min
        ext = ext.expand(1, 1, seq_len, seq_len).contiguous()
        attn_mask = ttnn.from_torch(
            ext, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask, joint)
        x = ttnn.layer_norm(
            x, weight=self.final_ln_w, bias=self.final_ln_b, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        logits = self.head(x)
        return logits, x


class DPLM2(TorchWrapper):
    """Top-level DPLM-2 backbone (torch in / torch out), mirrors EsmForDPLM2.

    forward(input_ids[B,L], joint=None) -> (logits[B,L,vocab], last_hidden[B,L,d]).
    `joint` auto-detected from type_ids if None (both aa and struct present).
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = dict(DPLM2_150M if cfg is None else cfg)

    @classmethod
    def from_pretrained(cls, repo_id: str = "airkingbd/dplm2_150m") -> "DPLM2":
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id, "pytorch_model.bin")
        sd = torch.load(p, map_location="cpu")
        keep = {k: v for k, v in sd.items()
                if not k.startswith("esm.contact_head")
                and not k.endswith("rotary_embeddings.inv_freq")
                and not k.startswith("esm.embeddings.position_embeddings")}
        m = cls(DPLM2_150M)
        m.load_state_dict(keep, strict=False)
        return m

    def _create_module(self, weights: WeightScope) -> DPLM2Model:
        return DPLM2Model(self.cfg, weights, self.compute_kernel_config)

    def forward(self, input_ids: torch.Tensor, joint=None):
        pad_id = self.cfg["pad_token_id"]
        input_mask = input_ids.ne(pad_id)
        if joint is None:
            tids = modality_type(input_ids, pad_id)
            joint = bool((tids == 0).any() and (tids == 1).any())
        tokens_tt = ttnn.from_torch(
            input_ids.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        logits, hidden = self.module(tokens_tt, input_ids, input_mask, joint)
        return self._to_torch(logits), self._to_torch(hidden)


class DPLM2Generator:
    """Discrete-diffusion generation on top of the ttnn DPLM-2 backbone.

    Thin wrapper: the diffusion loop lives in tt_bio.dplm2_sampler (pure torch,
    shared with the PyTorch reference) and only the per-step backbone forward
    runs on device. `backbone` is a `DPLM2` instance (ttnn); its forward returns
    (logits, hidden) and we feed logits to the sampler.

    Tasks (mirroring generate_dplm2.py):
      - "sequence_generation": aa-only, all positions start masked.
      - "co_generation" / "backbone_generation": joint struct+aa, all masked.
      - "folding": aa given, struct masked (partial_mask = aa positions).
      - "inverse_folding": struct given, aa masked (partial_mask = struct).
    """

    def __init__(self, backbone: "DPLM2", tok: DPLM2Tokenizer,
                 num_diffusion_timesteps: int = 500):
        self.backbone = backbone
        self.tok = tok
        def backbone_fn(input_ids):
            logits, _ = self.backbone(input_ids)
            return logits
        self.sampler = DPLM2Sampler(tok, backbone_fn, num_diffusion_timesteps)

    @classmethod
    def from_pretrained(cls, repo_id: str = "airkingbd/dplm2_150m",
                        num_diffusion_timesteps: int = 500) -> "DPLM2Generator":
        backbone = DPLM2.from_pretrained(repo_id)
        tok = DPLM2Tokenizer.from_pretrained(repo_id)
        return cls(backbone, tok, num_diffusion_timesteps)

    def generate(self, input_tokens, max_iter=None, temperature=1.0,
                 partial_masks=None, unmasking_strategy="stochastic1.0",
                 sampling_strategy="annealing@2.0:0.1"):
        return self.sampler.generate(
            input_tokens, max_iter=max_iter, temperature=temperature,
            partial_masks=partial_masks,
            unmasking_strategy=unmasking_strategy,
            sampling_strategy=sampling_strategy)
