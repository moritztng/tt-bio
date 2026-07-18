"""Faithful PyTorch reference for the DPLM-2 ESM backbone.

DPLM-2 (ByteDance, github.com/bytedance/dplm) is a discrete-diffusion protein
language model whose backbone is a HuggingFace ESM-2 transformer
(facebook/esm2_t30_150M_UR50D family) with two modifications from
byprot.models.dplm2.modules.dplm2_modeling_esm:

  1. ModifiedRotaryEmbedding — when the input mixes both modalities (struct
     tokens in the first half, aa tokens in the second), the RoPE table is
     built for L/2 and the SAME phases apply to both halves (struct pos i and
     aa pos i share rotary phase i).
  2. ModifiedEsmSelfAttention — query pre-scaled by head_dim**-0.5 and
     F.scaled_dot_product_attention called with scale=1.0.

The rest is stock HF ESM-2: pre-norm blocks (attention.LayerNorm -> q/k/v
(bias) -> rotary -> SDPA -> output.dense + residual; LayerNorm ->
intermediate.dense -> exact-erf GELU -> output.dense + residual),
emb_layer_norm_after, and EsmLMHead (dense -> gelu -> layer_norm -> decoder
+ bias).

We re-implement the forward here (byprot pins transformers==4.39.2 which
conflicts with tt-bio's 4.57.6) using the REAL weight names from
airkingbd/dplm2_150m, so the same state_dict loads into this reference and
into the ttnn port. Golden reference for parity. Pass 1 = backbone forward
only; the diffusion loop, LFQ struct tokenizer, and CLI wiring are pass 2.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DPLM2_150M = dict(
    hidden_size=640, num_attention_heads=20, num_hidden_layers=30,
    intermediate_size=2560, vocab_size=8229, pad_token_id=1, mask_token_id=32,
    layer_norm_eps=1e-5, max_position_embeddings=1026, token_dropout=True,
)
ROPE_BASE = 10000.0
AA_VOCAB_BOUND = 33  # aa tokens id < 33; struct tokens id >= 33


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=ROPE_BASE):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def _cos_sin(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)

    def forward(self, q, k, joint):
        if joint:
            half = q.shape[2] // 2
            cos, sin = self._cos_sin(half, q.device, q.dtype)
            q1, q2 = q.chunk(2, dim=2); k1, k2 = k.chunk(2, dim=2)
            q1 = apply_rotary_pos_emb(q1, cos, sin); q2 = apply_rotary_pos_emb(q2, cos, sin)
            k1 = apply_rotary_pos_emb(k1, cos, sin); k2 = apply_rotary_pos_emb(k2, cos, sin)
            return torch.cat((q1, q2), dim=2), torch.cat((k1, k2), dim=2)
        cos, sin = self._cos_sin(q.shape[2], q.device, q.dtype)
        return apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)


class EsmEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], padding_idx=cfg["pad_token_id"])
        self.token_dropout = cfg["token_dropout"]
        self.mask_token_id = cfg["mask_token_id"]

    def forward(self, input_ids, attention_mask):
        emb = self.word_embeddings(input_ids)
        if self.token_dropout:
            emb = emb.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8
            src_lengths = attention_mask.sum(-1)
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).float() / src_lengths
            emb = (emb * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]).to(emb.dtype)
        emb = emb * attention_mask.unsqueeze(-1).to(emb.dtype)
        return emb


class EsmSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["hidden_size"] // self.n_heads
        d = cfg["hidden_size"]
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x, attn_mask, joint):
        B, L, _ = x.shape
        q = self.query(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = q * (self.head_dim ** -0.5)
        q, k = self.rotary(q, k, joint)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, -1)
        return ctx


class _Dense(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.dense = nn.Linear(w[0], w[1])


class EsmAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_size"]
        self.self = EsmSelfAttention(cfg)
        self.output = _Dense((d, d))  # EsmSelfOutput.dense
        self.LayerNorm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])

    def forward(self, x, attn_mask, joint):
        h = self.LayerNorm(x)
        ctx = self.self(h, attn_mask, joint)
        return self.output.dense(ctx) + x


class EsmLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_size"]
        self.attention = EsmAttention(cfg)
        self.LayerNorm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])
        self.intermediate = _Dense((d, cfg["intermediate_size"]))
        self.output = _Dense((cfg["intermediate_size"], d))

    def forward(self, x, attn_mask, joint):
        a = self.attention(x, attn_mask, joint)
        return self.ffn_only(a)

    def ffn_only(self, x):
        h = self.LayerNorm(x)
        inter = gelu(self.intermediate.dense(h))
        return self.output.dense(inter) + x


class EsmEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.ModuleList([EsmLayer(cfg) for _ in range(cfg["num_hidden_layers"])])
        self.emb_layer_norm_after = nn.LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])

    def forward(self, x, attn_mask, joint):
        for layer in self.layer:
            x = layer(x, attn_mask, joint)
        return self.emb_layer_norm_after(x)


class EsmLMHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_size"]
        self.dense = nn.Linear(d, d)
        self.layer_norm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])
        self.decoder = nn.Linear(d, cfg["vocab_size"], bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg["vocab_size"]))

    def forward(self, x):
        x = gelu(self.dense(x))
        x = self.layer_norm(x)
        return self.decoder(x) + self.bias


class EsmModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = EsmEmbeddings(cfg)
        self.encoder = EsmEncoder(cfg)

    def forward(self, input_ids, input_mask_bool, joint):
        x = self.embeddings(input_ids, input_mask_bool)
        attn_mask = (1.0 - input_mask_bool[:, None, None, :].to(torch.float32)) * torch.finfo(torch.float32).min
        return self.encoder(x, attn_mask, joint)


class DPLM2Reference(nn.Module):
    """Mirrors byprot EsmForDPLM2.forward (backbone only).

    forward(input_ids, joint=None) -> (logits[B,L,vocab], last_hidden[B,L,d]).
    `joint` auto-detected from type_ids if None (both aa and struct present).
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = dict(DPLM2_150M if cfg is None else cfg)
        self.cfg = cfg
        self.pad_id = cfg["pad_token_id"]
        self.esm = EsmModel(cfg)
        self.lm_head = EsmLMHead(cfg)

    def get_modality_type(self, input_ids):
        input_mask = input_ids.ne(self.pad_id)
        modality = ((input_ids < AA_VOCAB_BOUND) & input_mask).int()
        modality[~input_mask] = 2
        return modality

    def _extended_mask(self, input_mask):
        # HF get_extended_attention_mask: valid -> 0, pad -> finfo.min (additive).
        m = input_mask[:, None, None, :].to(torch.float32)
        return (1.0 - m) * torch.finfo(torch.float32).min

    def forward(self, input_ids, joint=None):
        input_mask = input_ids.ne(self.pad_id)
        type_ids = self.get_modality_type(input_ids)
        if joint is None:
            joint = bool((type_ids == 0).any() and (type_ids == 1).any())
        x = self.esm(input_ids, input_mask, joint)
        return self.lm_head(x), x


def make_dplm2_150m(seed: int = 0) -> DPLM2Reference:
    torch.manual_seed(seed)
    return DPLM2Reference(DPLM2_150M).eval()


def load_dplm2_150m() -> DPLM2Reference:
    """Load the real airkingbd/dplm2_150m checkpoint into the reference."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("airkingbd/dplm2_150m", "pytorch_model.bin")
    sd = torch.load(p, map_location="cpu")
    keep = {}
    for k, v in sd.items():
        if k.startswith("esm.contact_head") or k.endswith("rotary_embeddings.inv_freq"):
            continue
        if k.startswith("esm.embeddings.position_embeddings"):
            continue  # rotary, not used
        keep[k] = v
    m = DPLM2Reference(DPLM2_150M).eval()
    missing, unexpected = m.load_state_dict(keep, strict=False)
    # rotary inv_freq is a non-persistent buffer; position_embeddings dropped.
    missing = [k for k in missing if "rotary" not in k]
    assert not missing, f"missing: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    return m
