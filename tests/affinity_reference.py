"""Faithful PyTorch reference for the PLAPT standalone affinity head.

Two portable components, both reimplemented from scratch (no ``transformers``
dependency, which is broken in the dev env via a tokenizers mismatch):

  1. ``ChemBERTaReference`` — a 6-layer RoBERTa encoder + pooler, matching
     ``seyonec/ChemBERTa-zinc-base-v1`` (hidden 768, 12 heads, intermediate
     3072, gelu, LayerNorm eps 1e-5, vocab 767, max_pos 514, type_vocab 1,
     pad_token_id 1). ``pooler_output = tanh(dense(hidden[:, 0]))``.
  2. ``FusionHead`` — the PLAPT affinity MLP, weights extracted from
     ``models/affinity_predictor.onnx`` (MIT, Bindwell 2024) into
     ``tt_bio/_vendor/plapt/head_weights.npz``. Architecture (from the ONNX
     graph): two parallel branches slice the 1792-d concat
     (prot_pooler[1024] || mol_pooler[768]) -> Linear(->512)+ReLU each, concat
     to 1024 -> BatchNorm -> Linear(1024->512)+ReLU -> Linear(512->64)+ReLU ->
     Linear(64->1). Output is a normalized affinity; PLAPT rescales it to
     neg_log10_affinity_M = out * 1.5614094578916633 + 6.51286529169358.

This is the golden reference the ttnn port (``tt_bio.affinity``) is tested
against: identical weights into both, compare per component (the tt-bio idiom).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_VENDOR = Path(__file__).resolve().parent.parent / "tt_bio" / "_vendor" / "plapt"
CHEMBERTA_CFG_PATH = _VENDOR / "chemberta" / "config.json"
HEAD_WEIGHTS_PATH = _VENDOR / "head_weights.npz"

# PLAPT output normalization (from plapt.py:PredictionModule).
AFFINITY_MEAN = 6.51286529169358
AFFINITY_SCALE = 1.5614094578916633


def load_chemberta_config() -> dict:
    return json.loads(CHEMBERTA_CFG_PATH.read_text())


def _position_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    # HF RobertaEmbeddings.create_position_ids_from_input_ids: cumsum of the
    # non-pad mask, times the mask, plus padding_idx. For an unpadded sequence
    # this yields [pad+1, pad+2, ..., pad+L].
    mask = input_ids.ne(pad_token_id).int()
    return (torch.cumsum(mask, dim=1) * mask + pad_token_id).long()


class RobertaEmbeddings(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        H = cfg["hidden_size"]
        self.word_embeddings = nn.Embedding(cfg["vocab_size"], H, padding_idx=cfg["pad_token_id"])
        self.position_embeddings = nn.Embedding(cfg["max_position_embeddings"], H)
        self.token_type_embeddings = nn.Embedding(cfg["type_vocab_size"], H)
        self.LayerNorm = nn.LayerNorm(H, eps=cfg["layer_norm_eps"])
        self.pad_token_id = cfg["pad_token_id"]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pos = _position_ids(input_ids, self.pad_token_id)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(pos)
            + self.token_type_embeddings(torch.zeros_like(input_ids))
        )
        return self.LayerNorm(x)


class RobertaSelfAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        H = cfg["hidden_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.head_dim = H // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(H, H)
        self.key = nn.Linear(H, H)
        self.value = nn.Linear(H, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, H = x.shape
        q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        probs = F.softmax(scores, dim=-1)
        ctx = probs @ v  # [B, H, L, head_dim]
        return ctx.transpose(1, 2).reshape(B, L, H)


class RobertaLayer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        H = cfg["hidden_size"]
        self.self = RobertaSelfAttention(cfg)
        self.att_dense = nn.Linear(H, H)
        self.att_LN = nn.LayerNorm(H, eps=cfg["layer_norm_eps"])
        self.inter_dense = nn.Linear(H, cfg["intermediate_size"])
        self.out_dense = nn.Linear(cfg["intermediate_size"], H)
        self.out_LN = nn.LayerNorm(H, eps=cfg["layer_norm_eps"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.self(x)
        a = self.att_dense(a)
        a = self.att_LN(a + x)
        h = F.gelu(self.inter_dense(a))
        o = self.out_dense(h)
        return self.out_LN(o + a)


class RobertaPooler(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.dense = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dense(hidden[:, 0]))


class ChemBERTaReference(nn.Module):
    """RoBERTa encoder + pooler for ChemBERTa-zinc-base-v1 (inference, no LM head)."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or load_chemberta_config()
        self.cfg = cfg
        self.embeddings = RobertaEmbeddings(cfg)
        self.encoder = nn.ModuleList([RobertaLayer(cfg) for _ in range(cfg["num_hidden_layers"])])
        self.pooler = RobertaPooler(cfg)

    def forward(self, input_ids: torch.Tensor):
        x = self.embeddings(input_ids)
        for layer in self.encoder:
            x = layer(x)
        return self.pooler(x), x  # (pooler_output, last_hidden_state)


def _gemm(x: torch.Tensor, w, b) -> torch.Tensor:
    # ONNX Gemm: y = x @ W^T + b, with W stored as [out, in]. Buffers are tensors.
    wt = w if isinstance(w, torch.Tensor) else torch.from_numpy(w)
    bt = b if isinstance(b, torch.Tensor) else torch.from_numpy(b)
    return x @ wt.t() + bt


class FusionHead(nn.Module):
    """PLAPT affinity MLP (weights from the ONNX export, MIT)."""

    BN_EPS = 1e-5  # ONNX BatchNormalization default epsilon.

    def __init__(self, weights_path: Path | str | None = None):
        super().__init__()
        if weights_path is None:
            weights_path = HEAD_WEIGHTS_PATH
        npz = np.load(weights_path)
        self.register_buffer("prot_w", torch.from_numpy(npz["ProtLinear_Weights"]))
        self.register_buffer("prot_b", torch.from_numpy(npz["ProtLinear_Biases"]))
        self.register_buffer("mol_w", torch.from_numpy(npz["MolLinear_Weights"]))
        self.register_buffer("mol_b", torch.from_numpy(npz["MolLinear_Biases"]))
        self.register_buffer("bn_scale", torch.from_numpy(npz["Normalize_Scaling"]))
        self.register_buffer("bn_bias", torch.from_numpy(npz["Normalize_Biases"]))
        self.register_buffer("bn_mean", torch.from_numpy(npz["Normalize_MovingMean"]))
        self.register_buffer("bn_var", torch.from_numpy(npz["Normalize_MovingVariance"]))
        self.register_buffer("l1_w", torch.from_numpy(npz["Linear1_Weights"]))
        self.register_buffer("l1_b", torch.from_numpy(npz["Linear1_Biases"]))
        self.register_buffer("l2_w", torch.from_numpy(npz["Linear2_Weights"]))
        self.register_buffer("l2_b", torch.from_numpy(npz["Linear2_Biases"]))
        self.register_buffer("fl_w", torch.from_numpy(npz["FinalLinear_Weights"]))
        self.register_buffer("fl_b", torch.from_numpy(npz["FinalLinear_Biases"]))

    def forward(self, prot_pooler: torch.Tensor, mol_pooler: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([prot_pooler, mol_pooler], dim=-1)  # [B, 1792]
        prot = F.relu(_gemm(feat[..., :1024], self.prot_w, self.prot_b))
        mol = F.relu(_gemm(feat[..., 1024:], self.mol_w, self.mol_b))
        x = torch.cat([prot, mol], dim=-1)  # [B, 1024]
        x = F.batch_norm(
            x.unsqueeze(-1).unsqueeze(-1), self.bn_mean, self.bn_var, self.bn_scale, self.bn_bias,
            training=False, eps=self.BN_EPS,
        ).squeeze(-1).squeeze(-1)
        x = F.relu(_gemm(x, self.l1_w, self.l1_b))
        x = F.relu(_gemm(x, self.l2_w, self.l2_b))
        return _gemm(x, self.fl_w, self.fl_b)  # [B, 1]

    def to_affinity(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * AFFINITY_SCALE + AFFINITY_MEAN


def make_chemberta(seed: int = 0) -> ChemBERTaReference:
    torch.manual_seed(seed)
    return ChemBERTaReference().eval()


def fusion_head() -> FusionHead:
    return FusionHead().eval()
