"""SaProt structure-aware protein language model on Tenstorrent (ttnn).

SaProt (westlake-repl) is an ESM-2 masked-LM encoder over a fused
structure-aware vocabulary: 20 amino acids x 21 Foldseek 3Di states plus 5
special tokens = 446. Architecturally it is an ESM-2 transformer (pre-LN,
rotary embeddings, GELU feed-forward, attention projections with bias, a
final ``emb_layer_norm_after``), distinct from ESMC (QK-LayerNorm + SwiGLU +
residual scaling). The rotary embedding, the ttnn framework
(``Module``/``WeightScope``/``get_device``), the token-embedding lookup, the
length bucketing and the multi-card fanout are reused from the ESMC port; the
ESM-2 block and the 446-token embedding table are the new work.

Reference: HuggingFace ``EsmForMaskedLM`` (westlake-repl/SaProt_*_AF2),
``transformers.models.esm.modeling_esm``. The 3Di tokenization front-end runs
on host (Foldseek) and is not a ttnn concern -- structure in, fused tokens out.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import ttnn

from tt_bio.tenstorrent import (
    Module,
    TorchWrapper,
    Weights,
    WeightScope,
    _dtype,
    _sdpa_program_config_for_lengths,
    get_device,
)
from tt_bio.esmc import rope_tables, _rope, _batch_tokens, BUCKET

ROPE_BASE = 10000.0
# token_dropout compensation baked into embeddings at inference (no mask tokens):
# embeddings *= (1 - 0.15*0.8) / (1 - 0) = 0.88. See EsmEmbeddings.forward.
_EMBED_SCALE = 0.88

# Foldseek vocabularies (utils.constants): 20 AA + "#" pad, 21 3Di states + "#".
FOLDSEEK_SEQ_VOCAB = "ACDEFGHIKLMNPQRSTVWY#"
FOLDSEEK_STRUC_VOCAB = "pynwrqhgdlvtmfsaeikc#"
_SPECIALS = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
VOCAB = _SPECIALS + [a + b for a, b in itertools.product(FOLDSEEK_SEQ_VOCAB, FOLDSEEK_STRUC_VOCAB)]
assert len(VOCAB) == 446, len(VOCAB)
_TOK_TO_IDX = {t: i for i, t in enumerate(VOCAB)}
CLS, PAD, EOS, UNK, MASK = 0, 1, 2, 3, 4

# name -> (arch config, HF repo). ESM-2 variants; vocab=446 for all.
CONFIGS = {
    "saprot-35m": (
        dict(hidden=480, n_heads=20, n_layers=12, intermediate=1920),
        "westlake-repl/SaProt_35M_AF2",
    ),
    "saprot-650m": (
        dict(hidden=1280, n_heads=20, n_layers=33, intermediate=5120),
        "westlake-repl/SaProt_650M_AF2",
    ),
    "saprot-1.3b": (
        dict(hidden=2560, n_heads=40, n_layers=40, intermediate=10240),
        "westlake-repl/SaProt_1.3B_AF2",
    ),
}
MODELS = tuple(CONFIGS)


def tokenize(aa_seq: str, struc_seq: str) -> "torch.Tensor":
    """Pair an amino-acid sequence with a Foldseek 3Di sequence into fused
    SaProt tokens: ``[<cls>, a0+s0, a1+s1, ..., <eos>]`` -> [1, L+2] long."""
    aa_seq = aa_seq.upper()
    if len(struc_seq) < len(aa_seq):  # right-pad 3Di with the unknown-structure "#"
        struc_seq = struc_seq + "#" * (len(aa_seq) - len(struc_seq))
    ids = [CLS]
    for a, s in zip(aa_seq, struc_seq):
        ids.append(_TOK_TO_IDX.get(a + s, UNK))
    ids.append(EOS)
    return torch.tensor([ids], dtype=torch.long)


class SaprotEmbeddingLayer(Module):
    """Token embedding + the ESM-2 token-dropout compensation (0.88 at inference).

    Rotary position embeddings -> no learned position embeddings are added.
    Weight key: ``embed.weight`` of shape [446, hidden].
    """

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.weight = self.torch_to_tt("weight", transform=lambda x: x)

    def __call__(self, tokens: ttnn.Tensor, embed_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x = ttnn.embedding(
            tokens, self.weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.multiply(x, _EMBED_SCALE)
        if embed_mask is not None:
            x = ttnn.multiply(x, embed_mask)
        return x


class ESM2Attention(Module):
    """ESM-2 self-attention: pre-LN, fused q/k/v (with bias), per-head RoPE, SDPA,
    output projection (with bias). No QK-LayerNorm (that is ESMC-specific)."""

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.qkv_weight = self.torch_to_tt("qkv.weight", dtype=_dtype())
        self.qkv_bias = self.torch_to_tt("qkv.bias")
        self.out_weight = self.torch_to_tt("out.weight", dtype=_dtype())
        self.out_bias = self.torch_to_tt("out.bias")

    def __call__(self, x, cos, sin, attn_mask=None, key_valid=None):
        ck = self.compute_kernel_config
        d_model = x.shape[-1]
        head_dim = d_model // self.n_heads
        x_norm = ttnn.layer_norm(
            x, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5, compute_kernel_config=ck,
        )
        qkv = self._lin(x_norm, self.qkv_weight, bias=self.qkv_bias)
        ttnn.deallocate(x_norm)
        q, k, v = self._split_heads(qkv, self.n_heads)
        q, k = _rope(q, k, cos, sin)
        if key_valid is not None:
            k = ttnn.multiply(k, key_valid)
            v = ttnn.multiply(v, key_valid)
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, scale=head_dim ** -0.5,
            program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]),
        )
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        o = self._merge_heads(o)
        out = self._lin(o, self.out_weight, bias=self.out_bias)
        ttnn.deallocate(o)
        return out


class ESM2FFN(Module):
    """ESM-2 feed-forward: pre-LN, Linear(intermediate)+GELU, Linear(hidden), residual."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.fc1_weight = self.torch_to_tt("fc1.weight", dtype=_dtype())
        self.fc1_bias = self.torch_to_tt("fc1.bias")
        self.fc2_weight = self.torch_to_tt("fc2.weight", dtype=_dtype())
        self.fc2_bias = self.torch_to_tt("fc2.bias")

    def __call__(self, x):
        ck = self.compute_kernel_config
        x_norm = ttnn.layer_norm(
            x, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5, compute_kernel_config=ck,
        )
        h = self._lin(x_norm, self.fc1_weight, bias=self.fc1_bias)
        ttnn.deallocate(x_norm)
        h = ttnn.gelu(h)
        out = self._lin(h, self.fc2_weight, bias=self.fc2_bias)
        ttnn.deallocate(h)
        return out


class ESM2Block(Module):
    """ESM-2 pre-LN block: x = x + attn(x); x = x + ffn(x). No residual scaling."""

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.attn = ESM2Attention(n_heads, self.scope("attn"), compute_kernel_config)
        self.ffn = ESM2FFN(self.scope("ffn"), compute_kernel_config)

    def __call__(self, x, cos, sin, attn_mask=None, key_valid=None):
        r = self.attn(x, cos, sin, attn_mask, key_valid)
        x = ttnn.add(x, r)
        ttnn.deallocate(r)
        r = self.ffn(x)
        x = ttnn.add(x, r)
        ttnn.deallocate(r)
        return x


class SaprotLMHead(Module):
    """ESM-2 MLM head: Linear -> GELU -> LayerNorm -> decoder (tied to the token
    embedding) + bias. The decoder weight is the embedding table, passed in at
    call time (weight tying)."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.dense_weight = self.torch_to_tt("dense.weight")
        self.dense_bias = self.torch_to_tt("dense.bias")
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.decoder_weight = self.torch_to_tt("decoder_weight")
        self.decoder_bias = self.torch_to_tt("decoder_bias")

    def __call__(self, x):
        ck = self.compute_kernel_config
        x = self._lin(x, self.dense_weight, bias=self.dense_bias)
        x = ttnn.gelu(x)
        x = ttnn.layer_norm(x, weight=self.norm_weight, bias=self.norm_bias,
                            epsilon=1e-5, compute_kernel_config=ck)
        logits = self._lin(x, self.decoder_weight, bias=self.decoder_bias)
        ttnn.deallocate(x)
        return logits


class SaprotModel(Module):
    """Full SaProt stack: embed -> N ESM-2 blocks -> emb_layer_norm_after -> LM head.

    __call__ returns (logits[B,L,446], embeddings[B,L,hidden]); embeddings are the
    post-``emb_layer_norm_after`` hidden states (the structure-aware per-residue
    representation)."""

    def __init__(self, n_heads, n_layers, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.embed = SaprotEmbeddingLayer(self.scope("embed"), compute_kernel_config)
        self.blocks = [
            ESM2Block(n_heads, self.scope(f"blocks.{i}"), compute_kernel_config)
            for i in range(n_layers)
        ]
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.head = SaprotLMHead(self.scope("head"), compute_kernel_config)

    def __call__(self, tokens, attn_mask=None, key_valid=None, embed_mask=None):
        seq_len = tokens.shape[-1]
        head_dim = self.norm_weight.shape[-1] // self.n_heads
        cos, sin = rope_tables(seq_len, head_dim, device=self.device)
        x = self.embed(tokens, embed_mask)
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask, key_valid)
        emb = ttnn.layer_norm(x, weight=self.norm_weight, bias=self.norm_bias,
                              epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(x)
        logits = self.head(emb)
        return logits, emb


def _remap_state_dict(sd: dict, n_layers: int) -> dict:
    """Remap HuggingFace EsmForMaskedLM keys to the SaprotModel scheme."""
    out = {}
    out["embed.weight"] = sd["esm.embeddings.word_embeddings.weight"]
    for i in range(n_layers):
        p = f"esm.encoder.layer.{i}."
        out[f"blocks.{i}.attn.norm.weight"] = sd[p + "attention.LayerNorm.weight"]
        out[f"blocks.{i}.attn.norm.bias"] = sd[p + "attention.LayerNorm.bias"]
        out[f"blocks.{i}.attn.qkv.weight"] = torch.cat(
            [sd[p + "attention.self.query.weight"], sd[p + "attention.self.key.weight"],
             sd[p + "attention.self.value.weight"]], dim=0).contiguous()
        out[f"blocks.{i}.attn.qkv.bias"] = torch.cat(
            [sd[p + "attention.self.query.bias"], sd[p + "attention.self.key.bias"],
             sd[p + "attention.self.value.bias"]], dim=0).contiguous()
        out[f"blocks.{i}.attn.out.weight"] = sd[p + "attention.output.dense.weight"]
        out[f"blocks.{i}.attn.out.bias"] = sd[p + "attention.output.dense.bias"]
        out[f"blocks.{i}.ffn.norm.weight"] = sd[p + "LayerNorm.weight"]
        out[f"blocks.{i}.ffn.norm.bias"] = sd[p + "LayerNorm.bias"]
        out[f"blocks.{i}.ffn.fc1.weight"] = sd[p + "intermediate.dense.weight"]
        out[f"blocks.{i}.ffn.fc1.bias"] = sd[p + "intermediate.dense.bias"]
        out[f"blocks.{i}.ffn.fc2.weight"] = sd[p + "output.dense.weight"]
        out[f"blocks.{i}.ffn.fc2.bias"] = sd[p + "output.dense.bias"]
    out["norm.weight"] = sd["esm.encoder.emb_layer_norm_after.weight"]
    out["norm.bias"] = sd["esm.encoder.emb_layer_norm_after.bias"]
    out["head.dense.weight"] = sd["lm_head.dense.weight"]
    out["head.dense.bias"] = sd["lm_head.dense.bias"]
    out["head.norm.weight"] = sd["lm_head.layer_norm.weight"]
    out["head.norm.bias"] = sd["lm_head.layer_norm.bias"]
    out["head.decoder_bias"] = sd["lm_head.bias"]
    # lm_head.decoder.weight is tied to the token embedding; store a dedicated
    # linear-weight copy (transposed for ttnn.linear) rather than reusing the
    # embedding-table tensor in the decoder matmul.
    out["head.decoder_weight"] = sd["esm.embeddings.word_embeddings.weight"]
    return out


class Saprot(TorchWrapper):
    """Top-level SaProt model (torch in / torch out). Mirrors EsmForMaskedLM.

    forward(tokens[B,L]) -> (logits[B,L,446], emb[B,L,hidden]). Optional padding
    masks (built by ``esmc._batch_tokens``) let a batch of unequal-length sequences
    share one padded, bucketed forward.
    """

    def __init__(self, hidden: int, n_heads: int, n_layers: int, intermediate: int):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.intermediate = intermediate

    @classmethod
    def from_pretrained(cls, name: str = "saprot-650m") -> "Saprot":
        from huggingface_hub import snapshot_download
        config, repo = CONFIGS[name]
        p = snapshot_download(repo)
        sd = torch.load(os.path.join(p, "pytorch_model.bin"), map_location="cpu")
        remapped = _remap_state_dict(sd, config["n_layers"])
        model = cls(**config)
        model.load_state_dict(remapped, strict=False)
        return model

    def _create_module(self, weights: WeightScope) -> SaprotModel:
        return SaprotModel(self.n_heads, self.n_layers, weights, self.compute_kernel_config)

    def forward(self, tokens, attn_mask=None, key_valid=None, embed_mask=None):
        tokens_tt = ttnn.from_torch(
            tokens.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        mask_tt = None if attn_mask is None else ttnn.from_torch(
            attn_mask.unsqueeze(1).to(torch.bfloat16), device=self.tt_device,
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        )
        kv_tt = None if key_valid is None else ttnn.from_torch(
            key_valid.to(torch.bfloat16), device=self.tt_device,
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        )
        em_tt = None if embed_mask is None else ttnn.from_torch(
            embed_mask.to(torch.bfloat16), device=self.tt_device,
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        )
        logits, emb = self.module(tokens_tt, mask_tt, kv_tt, em_tt)
        return self._to_torch(logits), self._to_torch(emb)


def load_saprot(name: str = "saprot-650m", *, fast: bool = False):
    from tt_bio import tenstorrent
    tenstorrent.set_fast_mode(fast)
    if name not in CONFIGS:
        raise ValueError(f"unknown SaProt model {name!r}; choose from {list(MODELS)}")
    return Saprot.from_pretrained(name)


@dataclass
class SaprotEmbedding:
    """One sequence's structure-aware embeddings from the SaProt trunk.

    ``per_residue`` has one row per residue (the <cls>/<eos> special tokens are
    stripped). ``pooled`` is a single fixed-size vector. ``logits`` are the
    per-residue MLM logits ([L, 446]) when requested.
    """

    id: str
    sequence: str
    per_residue: np.ndarray
    pooled: np.ndarray
    logits: Optional[np.ndarray]


_POOLERS = {"mean": lambda e: e.mean(axis=0), "max": lambda e: e.max(axis=0),
            "cls": None}


def embed_sequences(model, sequences, *, return_logits=False, pool="mean", batch_size=8):
    """Embed each {id: (aa, struc)} pair with an already-loaded SaProt ``model``.

    ``sequences`` maps id -> (aa_seq, struc_seq) or id -> aa_seq (3Di taken as
    all-"#", i.e. sequence-only). Sequences are length-bucketed and run through a
    single padded, bucketed device forward per batch.
    """
    if pool not in _POOLERS:
        raise ValueError(f"unknown pool {pool!r}; choose from {sorted(_POOLERS)}")
    norm = {}
    for sid, v in sequences.items():
        if isinstance(v, (list, tuple)):
            aa, struc = v[0], (v[1] if len(v) > 1 else "")
        else:
            aa, struc = v, ""
        if not aa:
            raise ValueError(f"sequence {sid!r} is empty")
        norm[sid] = (aa.upper(), struc)

    items = list(norm.items())
    order = sorted(range(len(items)), key=lambda i: len(items[i][1][0]))
    budget = batch_size * 512
    batches, cur, cur_max = [], [], 0
    for i in order:
        tok = len(items[i][1][0]) + 2
        nxt_max = max(cur_max, ((tok + BUCKET - 1) // BUCKET) * BUCKET)
        if cur and (len(cur) >= batch_size or (len(cur) + 1) * nxt_max > budget):
            batches.append(cur); cur, cur_max = [], 0
        cur.append(i); cur_max = max(cur_max, ((tok + BUCKET - 1) // BUCKET) * BUCKET)
    if cur:
        batches.append(cur)

    by_id = {}
    for idx in batches:
        batch = [items[i] for i in idx]
        toks = [tokenize(aa, struc) for _, (aa, struc) in batch]
        input_ids, lens, attn_mask, key_valid = _batch_tokens([t[0] for t in toks])
        embed_mask = None if key_valid is None else key_valid.squeeze(1)
        logits_b, emb_b = model(input_ids, attn_mask, key_valid, embed_mask)
        for row, (sid, (aa, struc)) in enumerate(batch):
            li = lens[row]
            emb = emb_b[row, :li].numpy().astype(np.float32)
            per_residue, cls = emb[1:-1], emb[0]
            lg = None
            if return_logits:
                lg = logits_b[row, 1:-1].numpy().astype(np.float32)
            pooled = cls if pool == "cls" else _POOLERS[pool](per_residue)
            by_id[sid] = SaprotEmbedding(sid, aa, per_residue, pooled.astype(np.float32), lg)
    return [by_id[k] for k in sequences]


def embed(sequences, model: str = "saprot-650m", *, fast=False, return_logits=False,
          pool="mean", batch_size=8):
    """One-shot embedding: load ``model`` and embed ``sequences``.

    ``sequences`` may be a single (aa, struc) pair, a dict of {id: (aa, struc)},
    or {id: aa} (sequence-only, 3Di = "#"). Returns one SaprotEmbedding per input.
    """
    if isinstance(sequences, str):
        sequences = {"seq0": sequences}
    elif isinstance(sequences, (list, tuple)) and len(sequences) == 2 and isinstance(sequences[0], str):
        sequences = {"seq0": sequences}
    m = load_saprot(model, fast=fast)
    return embed_sequences(m, sequences, return_logits=return_logits, pool=pool, batch_size=batch_size)

