"""Standalone sequence-based affinity head (PLAPT-style) on Tenstorrent.

Predicts protein-ligand binding affinity from SEQUENCE + SMILES only (no
structure/folding): a frozen protein PLM + a small ligand encoder + a light
fusion MLP head. Reference: PLAPT (Bindwell, MIT) - ProtBERT + ChemBERTa-zinc-
base-v1 pooler outputs concatenated, fed to a small branching MLP that emits a
normalized affinity rescaled to neg_log10_affinity_M.

Pass 1 ports the two portable components to ttnn and verifies per-component
PCC vs the from-scratch PyTorch reference in tests/affinity_reference.py:

  * ChemBERTa - the 6-layer RoBERTa ligand encoder + pooler
    (seyonec/ChemBERTa-zinc-base-v1, ~43M params, hidden 768).
  * AffinityHead - the fusion MLP (weights extracted from PLAPT MIT
    affinity_predictor.onnx into tt_bio/_vendor/plapt/head_weights.npz).

The protein tower (ProtBERT) is intentionally NOT ported this pass: the PLAPT
head is ProtBERT-specific (1024-d pooler input, trained on ProtBERT semantics)
and is dimensionally incompatible with tt-bio ESMC (960-d). See the pass-1
notes for the reuse decision. End-to-end wiring (host ProtBERT tower + SMILES
tokenization + benchmark) is deferred.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import ttnn

from tt_bio.tenstorrent import (
    Module,
    TorchWrapper,
    WeightScope,
    Weights,
    _dtype,
    _sdpa_program_config_for_lengths,
    get_device,
)

_VENDOR = Path(__file__).resolve().parent / "_vendor" / "plapt"
HEAD_WEIGHTS_PATH = _VENDOR / "head_weights.npz"

AFFINITY_MEAN = 6.51286529169358
AFFINITY_SCALE = 1.5614094578916633
BN_EPS = 1e-5


def _position_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """HF RobertaEmbeddings.create_position_ids_from_input_ids."""
    mask = input_ids.ne(pad_token_id).int()
    return (torch.cumsum(mask, dim=1) * mask + pad_token_id).long()


class _Embedding(Module):
    """BERT/RoBERTa embeddings: word + position + token_type(=0) -> LayerNorm.

    Position-agnostic: the caller supplies position_ids (RoBERTa cumsum or BERT
    arange). token_type is always id 0, so we slice row 0 of the table into a
    [1,1,H] broadcast bias (works for type_vocab_size 1 or 2).
    """

    def __init__(self, state_dict: Weights, compute_kernel_config, cfg: dict):
        super().__init__(state_dict, compute_kernel_config)
        self.pad_token_id = cfg["pad_token_id"]
        self.eps = cfg.get("layer_norm_eps", BN_EPS)
        self.word = self.torch_to_tt("word_embeddings.weight", transform=lambda x: x)
        self.pos = self.torch_to_tt("position_embeddings.weight", transform=lambda x: x)
        self.ttype = self.torch_to_tt(
            "token_type_embeddings.weight", transform=lambda x: x[0].reshape(1, 1, -1)
        )
        self.ln_w = self.torch_to_tt("LayerNorm.weight")
        self.ln_b = self.torch_to_tt("LayerNorm.bias")

    def __call__(self, tokens: ttnn.Tensor, position_ids: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.embedding(tokens, self.word, layout=ttnn.TILE_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        p = ttnn.embedding(position_ids, self.pos, layout=ttnn.TILE_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.add(x, p)
        ttnn.deallocate(p)
        x = ttnn.add(x, self.ttype)
        x = ttnn.layer_norm(x, weight=self.ln_w, bias=self.ln_b, epsilon=self.eps,
                            compute_kernel_config=self.compute_kernel_config)
        return x


class _Layer(Module):
    """One post-LN RoBERTa encoder layer (separate Q/K/V, biases, gelu FFN)."""

    def __init__(self, state_dict: Weights, compute_kernel_config, cfg: dict):
        super().__init__(state_dict, compute_kernel_config)
        H = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.head_dim = H // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.eps = cfg.get("layer_norm_eps", BN_EPS)
        self.q_w = self.torch_to_tt("self.query.weight", dtype=_dtype())
        self.q_b = self.torch_to_tt("self.query.bias", transform=lambda x: x.reshape(1, 1, -1))
        self.k_w = self.torch_to_tt("self.key.weight", dtype=_dtype())
        self.k_b = self.torch_to_tt("self.key.bias", transform=lambda x: x.reshape(1, 1, -1))
        self.v_w = self.torch_to_tt("self.value.weight", dtype=_dtype())
        self.v_b = self.torch_to_tt("self.value.bias", transform=lambda x: x.reshape(1, 1, -1))
        self.o_w = self.torch_to_tt("att_dense.weight", dtype=_dtype())
        self.o_b = self.torch_to_tt("att_dense.bias", transform=lambda x: x.reshape(1, 1, -1))
        self.att_ln_w = self.torch_to_tt("att_LN.weight")
        self.att_ln_b = self.torch_to_tt("att_LN.bias")
        self.i_w = self.torch_to_tt("inter_dense.weight", dtype=_dtype())
        self.i_b = self.torch_to_tt("inter_dense.bias", transform=lambda x: x.reshape(1, 1, -1))
        self.f_w = self.torch_to_tt("out_dense.weight", dtype=_dtype())
        self.f_b = self.torch_to_tt("out_dense.bias", transform=lambda x: x.reshape(1, 1, -1))
        self.out_ln_w = self.torch_to_tt("out_LN.weight")
        self.out_ln_b = self.torch_to_tt("out_LN.bias")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        q = self._lin(x, self.q_w, bias=self.q_b)
        k = self._lin(x, self.k_w, bias=self.k_b)
        v = self._lin(x, self.v_w, bias=self.v_b)
        qkv = ttnn.concat([q, k, v], dim=-1)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        q, k, v = self._split_heads(qkv, self.n_heads)
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=self.scale,
            program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]),
        )
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        o = self._merge_heads(o)
        o = self._lin(o, self.o_w, bias=self.o_b)
        a = ttnn.layer_norm(ttnn.add(o, x), weight=self.att_ln_w, bias=self.att_ln_b,
                            epsilon=self.eps, compute_kernel_config=ck)
        ttnn.deallocate(o)
        h = ttnn.gelu(self._lin(a, self.i_w, bias=self.i_b))
        out = self._lin(h, self.f_w, bias=self.f_b)
        ttnn.deallocate(h)
        y = ttnn.layer_norm(ttnn.add(out, a), weight=self.out_ln_w, bias=self.out_ln_b,
                            epsilon=self.eps, compute_kernel_config=ck)
        ttnn.deallocate(out)
        return y


class _Pooler(Module):
    """RobertaPooler: tanh(dense(hidden[:, 0]))."""

    def __init__(self, state_dict: Weights, compute_kernel_config, cfg: dict):
        super().__init__(state_dict, compute_kernel_config)
        self.w = self.torch_to_tt("dense.weight", dtype=_dtype())
        self.b = self.torch_to_tt("dense.bias", transform=lambda x: x.reshape(1, 1, -1))

    def __call__(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        # hidden [B, L, H] (rank 3) -> take token 0 -> [B, 1, H] -> dense -> tanh.
        # ttnn.slice requires ROW_MAJOR layout (see tt_bio/protenix.py).
        shp = hidden.shape
        cls = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
        cls = ttnn.slice(cls, [0, 0, 0], [shp[0], 1, shp[2]],
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cls = ttnn.to_layout(cls, ttnn.TILE_LAYOUT)
        z = self._lin(cls, self.w, bias=self.b)
        return ttnn.tanh(z)


class _ChemBERTaModel(Module):
    """Full ChemBERTa ligand encoder: embeddings -> 6 layers -> pooler."""

    def __init__(self, state_dict: Weights, compute_kernel_config, cfg: dict):
        super().__init__(state_dict, compute_kernel_config)
        self.cfg = cfg
        self.embed = _Embedding(self.scope("embeddings"), compute_kernel_config, cfg)
        self.layers = [
            _Layer(self.scope(f"layer.{i}"), compute_kernel_config, cfg)
            for i in range(cfg["num_hidden_layers"])
        ]
        self.pooler = _Pooler(self.scope("pooler"), compute_kernel_config, cfg)

    def __call__(self, tokens: ttnn.Tensor, position_ids: ttnn.Tensor):
        x = self.embed(tokens, position_ids)
        for layer in self.layers:
            x = layer(x)
        return self.pooler(x), x


_CHEMBERTA_KEYMAP = {
    "roberta.embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
    "roberta.embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
    "roberta.embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
    "roberta.embeddings.LayerNorm.weight": "embeddings.LayerNorm.weight",
    "roberta.embeddings.LayerNorm.bias": "embeddings.LayerNorm.bias",
    "roberta.pooler.dense.weight": "pooler.dense.weight",
    "roberta.pooler.dense.bias": "pooler.dense.bias",
}


def _chemberta_layer_keymap(i: int) -> dict:
    p = f"roberta.encoder.layer.{i}."
    q = f"layer.{i}."
    return {
        p + "attention.self.query.weight": q + "self.query.weight",
        p + "attention.self.query.bias": q + "self.query.bias",
        p + "attention.self.key.weight": q + "self.key.weight",
        p + "attention.self.key.bias": q + "self.key.bias",
        p + "attention.self.value.weight": q + "self.value.weight",
        p + "attention.self.value.bias": q + "self.value.bias",
        p + "attention.output.dense.weight": q + "att_dense.weight",
        p + "attention.output.dense.bias": q + "att_dense.bias",
        p + "attention.output.LayerNorm.weight": q + "att_LN.weight",
        p + "attention.output.LayerNorm.bias": q + "att_LN.bias",
        p + "intermediate.dense.weight": q + "inter_dense.weight",
        p + "intermediate.dense.bias": q + "inter_dense.bias",
        p + "output.dense.weight": q + "out_dense.weight",
        p + "output.dense.bias": q + "out_dense.bias",
        p + "output.LayerNorm.weight": q + "out_LN.weight",
        p + "output.LayerNorm.bias": q + "out_LN.bias",
    }


def remap_chemberta_state_dict(sd: dict, n_layers: int = 6) -> dict:
    """Map HuggingFace ChemBERTa keys to the tt_bio.affinity layout."""
    import collections
    out = collections.OrderedDict()
    for k, v in sd.items():
        if k in _CHEMBERTA_KEYMAP:
            out[_CHEMBERTA_KEYMAP[k]] = v
    for i in range(n_layers):
        for hf, ours in _chemberta_layer_keymap(i).items():
            if hf in sd:
                out[ours] = sd[hf]
    return out


class ChemBERTa(TorchWrapper):
    """ChemBERTa-zinc-base-v1 ligand encoder on device (torch in / torch out).

    forward(input_ids[int B,L]) -> (pooler_output[B,768], last_hidden[B,L,768]).
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        import json
        if cfg is None:
            cfg = json.loads((_VENDOR / "chemberta" / "config.json").read_text())
        self.cfg = cfg

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> "ChemBERTa":
        """Load real ChemBERTa-zinc-base-v1 weights (HF pytorch_model.bin)."""
        if weights_path is None:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download("seyonec/ChemBERTa-zinc-base-v1", "pytorch_model.bin")
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)
        m = cls()
        remapped = remap_chemberta_state_dict(sd)
        m.load_state_dict(remapped, strict=False)
        return m

    def _create_module(self, weights: WeightScope) -> _ChemBERTaModel:
        return _ChemBERTaModel(weights, self.compute_kernel_config, self.cfg)

    def _encode_tt(self, input_ids: torch.Tensor):
        """Run the encoder, returning (pooler, last_hidden) as on-device ttnn
        tensors (bf16). Used by the device-resident `--fast` path."""
        pos = _position_ids(input_ids, self.cfg["pad_token_id"])
        tokens_tt = ttnn.from_torch(
            input_ids.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        pos_tt = ttnn.from_torch(
            pos.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        return self.module(tokens_tt, pos_tt)

    def forward_tt(self, input_ids: torch.Tensor):
        """On-device encode: returns (pooler, last_hidden) as ttnn tensors."""
        return self._encode_tt(input_ids)

    def forward(self, input_ids: torch.Tensor):
        pool, hidden = self._encode_tt(input_ids)
        return self._to_torch(pool), self._to_torch(hidden)


class _FusionHeadModel(Module):
    """PLAPT affinity MLP on device (weights from the MIT ONNX export).

    Architecture (per the ONNX graph): two parallel branches slice the
    [prot_pooler(1024) || mol_pooler(768)] concat, Linear->512 + ReLU each,
    concat to 1024, BatchNorm(inf), Linear 1024->512 + ReLU, Linear 512->64 +
    ReLU, Linear 64->1. BatchNorm in inference is a per-channel affine; we
    fold it to y = x * w + b with w = scale/sqrt(var+eps), b = bias - mean*w.
    """

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.prot_w = self.torch_to_tt("prot_w", dtype=_dtype())
        self.prot_b = self.torch_to_tt("prot_b", transform=lambda x: x.reshape(1, 1, -1))
        self.mol_w = self.torch_to_tt("mol_w", dtype=_dtype())
        self.mol_b = self.torch_to_tt("mol_b", transform=lambda x: x.reshape(1, 1, -1))
        # Folded BatchNorm affine (per-channel, broadcast over batch).
        scale = self.weights["bn_scale"]
        bias = self.weights["bn_bias"]
        mean = self.weights["bn_mean"]
        var = self.weights["bn_var"]
        w = scale / torch.sqrt(var + BN_EPS)
        b = bias - mean * w
        self.bn_w = ttnn.from_torch(w.reshape(1, 1, -1), layout=ttnn.TILE_LAYOUT,
                                    device=self.device, dtype=_dtype())
        self.bn_b = ttnn.from_torch(b.reshape(1, 1, -1), layout=ttnn.TILE_LAYOUT,
                                    device=self.device, dtype=_dtype())
        self.l1_w = self.torch_to_tt("l1_w", dtype=_dtype())
        self.l1_b = self.torch_to_tt("l1_b", transform=lambda x: x.reshape(1, 1, -1))
        self.l2_w = self.torch_to_tt("l2_w", dtype=_dtype())
        self.l2_b = self.torch_to_tt("l2_b", transform=lambda x: x.reshape(1, 1, -1))
        self.fl_w = self.torch_to_tt("fl_w", dtype=_dtype())
        self.fl_b = self.torch_to_tt("fl_b", transform=lambda x: x.reshape(1, 1, -1))

    def __call__(self, prot_pooler: ttnn.Tensor, mol_pooler: ttnn.Tensor) -> ttnn.Tensor:
        prot = ttnn.relu(self._lin(prot_pooler, self.prot_w, bias=self.prot_b))
        mol = ttnn.relu(self._lin(mol_pooler, self.mol_w, bias=self.mol_b))
        x = ttnn.concat([prot, mol], dim=-1)
        ttnn.deallocate(prot); ttnn.deallocate(mol)
        x = ttnn.add(ttnn.multiply(x, self.bn_w), self.bn_b)
        x = ttnn.relu(self._lin(x, self.l1_w, bias=self.l1_b))
        x = ttnn.relu(self._lin(x, self.l2_w, bias=self.l2_b))
        return self._lin(x, self.fl_w, bias=self.fl_b)


def _fusion_head_state_dict(npz_path: Path | str | None = None) -> dict:
    """Load the ONNX-extracted head weights into a tt_bio.affinity-keyed dict."""
    if npz_path is None:
        npz_path = HEAD_WEIGHTS_PATH
    npz = np.load(npz_path)
    import collections
    out = collections.OrderedDict()
    out["prot_w"] = torch.from_numpy(npz["ProtLinear_Weights"])
    out["prot_b"] = torch.from_numpy(npz["ProtLinear_Biases"])
    out["mol_w"] = torch.from_numpy(npz["MolLinear_Weights"])
    out["mol_b"] = torch.from_numpy(npz["MolLinear_Biases"])
    out["bn_scale"] = torch.from_numpy(npz["Normalize_Scaling"])
    out["bn_bias"] = torch.from_numpy(npz["Normalize_Biases"])
    out["bn_mean"] = torch.from_numpy(npz["Normalize_MovingMean"])
    out["bn_var"] = torch.from_numpy(npz["Normalize_MovingVariance"])
    out["l1_w"] = torch.from_numpy(npz["Linear1_Weights"])
    out["l1_b"] = torch.from_numpy(npz["Linear1_Biases"])
    out["l2_w"] = torch.from_numpy(npz["Linear2_Weights"])
    out["l2_b"] = torch.from_numpy(npz["Linear2_Biases"])
    out["fl_w"] = torch.from_numpy(npz["FinalLinear_Weights"])
    out["fl_b"] = torch.from_numpy(npz["FinalLinear_Biases"])
    return out


class AffinityHead(TorchWrapper):
    """PLAPT fusion MLP on device (torch in / torch out).

    forward(prot_pooler[B,1024], mol_pooler[B,768]) -> normalized affinity[B,1].
    Use ``to_affinity`` to rescale to neg_log10_affinity_M.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls, npz_path: str | None = None) -> "AffinityHead":
        m = cls()
        m.load_state_dict(_fusion_head_state_dict(npz_path), strict=False)
        return m

    def _create_module(self, weights: WeightScope) -> _FusionHeadModel:
        return _FusionHeadModel(weights, self.compute_kernel_config)

    def forward(self, prot_pooler: torch.Tensor, mol_pooler: torch.Tensor) -> torch.Tensor:
        bsz = prot_pooler.shape[0]
        prot_tt = self._from_torch(prot_pooler.to(torch.float32))
        mol_tt = self._from_torch(mol_pooler.to(torch.float32))
        out = self.module(prot_tt, mol_tt)
        return self._to_torch(out).reshape(bsz, 1)

    @staticmethod
    def to_affinity(normalized: torch.Tensor) -> torch.Tensor:
        return normalized * AFFINITY_SCALE + AFFINITY_MEAN

# ---------------------------------------------------------------------------
# ProtBERT (Rostlab/prot_bert, BERT-large post-LN) — pass 2
# ---------------------------------------------------------------------------

PROTBERT_CFG = dict(
    vocab_size=30, hidden_size=1024, num_hidden_layers=30,
    num_attention_heads=16, intermediate_size=4096, max_position_embeddings=40000,
    type_vocab_size=2, pad_token_id=0, layer_norm_eps=1e-12,
)
PROTBERT_MAX_LEN = 3200

# ProtBERT BERT tokenizer vocab (Rostlab/prot_bert). Single-char amino-acid
# tokens + 4 BERT specials + the "." (period) token. Pure-python so the port
# has no runtime transformers dependency.
PROTBERT_VOCAB = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, ".": 4,
    "L": 5, "A": 6, "G": 7, "V": 8, "E": 9, "S": 10, "I": 11, "K": 12,
    "R": 13, "D": 14, "T": 15, "P": 16, "N": 17, "Q": 18, "F": 19, "Y": 20,
    "M": 21, "H": 22, "C": 23, "W": 24, "X": 25, "U": 26, "B": 27, "Z": 28,
    "O": 29,
}
_PROT_CLS, _PROT_SEP, _PROT_PAD, _PROT_UNK = 2, 3, 0, 1


def preprocess_protein(seq: str) -> str:
    """PLAPT preprocessing: U/Z/O/B -> X, then space-separate residues."""
    import re
    return " ".join(re.sub(r"[UZOB]", "X", seq))


def tokenize_protein(seq: str, max_length: int = PROTBERT_MAX_LEN) -> torch.Tensor:
    """Protein sequence -> ProtBERT token ids [1, L] (CLS ... SEP), truncated to
    max_length. Matches BertTokenizer(preprocess(seq), max_length, truncation)."""
    pre = preprocess_protein(seq)
    ids = [_PROT_CLS]
    for ch in pre.split():
        ids.append(PROTBERT_VOCAB.get(ch, _PROT_UNK))
        if len(ids) >= max_length - 1:
            break
    ids.append(_PROT_SEP)
    return torch.tensor([ids], dtype=torch.long)


class _ProtBertModel(Module):
    """ProtBERT encoder + pooler: embeddings -> 30 post-LN layers -> pooler.

    Reuses _Embedding/_Layer/_Pooler (same subkey layout as ChemBERTa); only the
    cfg differs (eps 1e-12, hidden 1024, 30 layers, 16 heads) and position_ids
    are absolute arange (BERT) instead of RoBERTa cumsum.
    """

    def __init__(self, state_dict: Weights, compute_kernel_config, cfg: dict):
        super().__init__(state_dict, compute_kernel_config)
        self.cfg = cfg
        self.embed = _Embedding(self.scope("embeddings"), compute_kernel_config, cfg)
        self.layers = [
            _Layer(self.scope(f"layer.{i}"), compute_kernel_config, cfg)
            for i in range(cfg["num_hidden_layers"])
        ]
        self.pooler = _Pooler(self.scope("pooler"), compute_kernel_config, cfg)

    def __call__(self, tokens: ttnn.Tensor, position_ids: ttnn.Tensor):
        x = self.embed(tokens, position_ids)
        for layer in self.layers:
            x = layer(x)
        return self.pooler(x), x


_PROTBERT_KEYMAP = {
    "bert.embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
    "bert.embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
    "bert.embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
    "bert.embeddings.LayerNorm.weight": "embeddings.LayerNorm.weight",
    "bert.embeddings.LayerNorm.bias": "embeddings.LayerNorm.bias",
    "bert.pooler.dense.weight": "pooler.dense.weight",
    "bert.pooler.dense.bias": "pooler.dense.bias",
}


def _protbert_layer_keymap(i: int) -> dict:
    p = f"bert.encoder.layer.{i}."
    q = f"layer.{i}."
    return {
        p + "attention.self.query.weight": q + "self.query.weight",
        p + "attention.self.query.bias": q + "self.query.bias",
        p + "attention.self.key.weight": q + "self.key.weight",
        p + "attention.self.key.bias": q + "self.key.bias",
        p + "attention.self.value.weight": q + "self.value.weight",
        p + "attention.self.value.bias": q + "self.value.bias",
        p + "attention.output.dense.weight": q + "att_dense.weight",
        p + "attention.output.dense.bias": q + "att_dense.bias",
        p + "attention.output.LayerNorm.weight": q + "att_LN.weight",
        p + "attention.output.LayerNorm.bias": q + "att_LN.bias",
        p + "intermediate.dense.weight": q + "inter_dense.weight",
        p + "intermediate.dense.bias": q + "inter_dense.bias",
        p + "output.dense.weight": q + "out_dense.weight",
        p + "output.dense.bias": q + "out_dense.bias",
        p + "output.LayerNorm.weight": q + "out_LN.weight",
        p + "output.LayerNorm.bias": q + "out_LN.bias",
    }


def remap_protbert_state_dict(sd: dict, n_layers: int = 30) -> dict:
    """Map HuggingFace ProtBERT (BertForMaskedLM) keys to the tt_bio.affinity
    layout. Accepts both ``bert.*`` (BertForMaskedLM) and bare (BertModel) keys."""
    import collections
    out = collections.OrderedDict()
    bare = {
        "embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
        "embeddings.LayerNorm.weight": "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias": "embeddings.LayerNorm.bias",
        "pooler.dense.weight": "pooler.dense.weight",
        "pooler.dense.bias": "pooler.dense.bias",
    }
    for k, v in sd.items():
        if k in _PROTBERT_KEYMAP:
            out[_PROTBERT_KEYMAP[k]] = v
        elif k in bare:
            out[bare[k]] = v
    for i in range(n_layers):
        km = _protbert_layer_keymap(i)
        for hf, ours in km.items():
            if hf in sd:
                out[ours] = sd[hf]
            else:
                bare_hf = hf.replace("bert.encoder", "encoder")
                if bare_hf in sd:
                    out[ours] = sd[bare_hf]
    return out


class ProtBERT(TorchWrapper):
    """ProtBERT (Rostlab/prot_bert) protein encoder on device (torch in/out).

    forward(input_ids[int B,L]) -> (pooler_output[B,1024], last_hidden[B,L,1024]).
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        import json
        if cfg is None:
            cfg = json.loads((_VENDOR / "protbert" / "config.json").read_text())
        self.cfg = cfg

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> "ProtBERT":
        """Load real Rostlab/prot_bert weights (HF pytorch_model.bin)."""
        if weights_path is None:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download("Rostlab/prot_bert", "pytorch_model.bin")
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)
        m = cls()
        m.load_state_dict(remap_protbert_state_dict(sd), strict=False)
        return m

    def _create_module(self, weights: WeightScope) -> _ProtBertModel:
        return _ProtBertModel(weights, self.compute_kernel_config, self.cfg)

    def _encode_tt(self, input_ids: torch.Tensor):
        """Run the encoder, returning (pooler, last_hidden) as on-device ttnn
        tensors (bf16). Used by the device-resident `--fast` path so the pooler
        feeds the fusion head without a host round-trip."""
        L = input_ids.shape[1]
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0).expand_as(input_ids)
        tokens_tt = ttnn.from_torch(
            input_ids.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        pos_tt = ttnn.from_torch(
            pos.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        return self.module(tokens_tt, pos_tt)

    def forward_tt(self, input_ids: torch.Tensor):
        """On-device encode: returns (pooler, last_hidden) as ttnn tensors."""
        return self._encode_tt(input_ids)

    def forward(self, input_ids: torch.Tensor):
        pool, hidden = self._encode_tt(input_ids)
        return self._to_torch(pool), self._to_torch(hidden)


# ---------------------------------------------------------------------------
# SMILES tokenizer (ChemBERTa RoBERTa BPE, pure-python) + end-to-end pipeline
# ---------------------------------------------------------------------------

import json as _json
import regex as _re

_CHEMBERTA_TOK_DIR = _VENDOR / "chemberta"
_SMILES_BPE_PATTERN = _re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+"""
)
_smiles_vocab: dict | None = None
_smiles_merges: dict | None = None


def _load_smiles_bpe() -> tuple[dict, dict]:
    """Lazily load the vendored ChemBERTa BPE vocab + merge ranks (no network)."""
    global _smiles_vocab, _smiles_merges
    if _smiles_vocab is None:
        _smiles_vocab = _json.loads((_CHEMBERTA_TOK_DIR / "vocab.json").read_text())
        lines = (_CHEMBERTA_TOK_DIR / "merges.txt").read_text().split("\n")
        m = {}
        for i, line in enumerate(lines[1:]):  # skip "#version" header
            if not line or " " not in line:
                continue
            a, b = line.split(" ", 1)
            m[(a, b)] = i
        _smiles_merges = m
    return _smiles_vocab, _smiles_merges


def _bpe(word: str, merges: dict) -> list[str]:
    """Apply BPE merges to a pre-token (string of chars) -> list of subwords."""
    w = list(word)
    if len(w) < 2:
        return w
    while True:
        best = None
        best_rank = None
        for i in range(len(w) - 1):
            r = merges.get((w[i], w[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best = (w[i], w[i + 1])
        if best is None:
            break
        new = []
        i = 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                new.append(w[i] + w[i + 1])
                i += 2
            else:
                new.append(w[i])
                i += 1
        w = new
    return w


def tokenize_smiles(smiles: str, max_length: int = 278) -> torch.Tensor:
    """SMILES string -> ChemBERTa token ids [1, L] (``<s>`` ... ``</s>``), truncated
    to max_length. Pure-python BPE over the vendored vocab/merges — verified
    bit-exact vs ``RobertaTokenizer`` (see tests/test_affinity.py)."""
    vocab, merges = _load_smiles_bpe()
    ids = [0]  # <s>
    for pre in _SMILES_BPE_PATTERN.findall(smiles):
        for tok in _bpe(pre, merges):
            ids.append(vocab.get(tok, 3))  # <unk>=3
            if len(ids) >= max_length - 1:
                break
        if len(ids) >= max_length - 1:
            break
    ids.append(2)  # </s>
    return torch.tensor([ids], dtype=torch.long)


class Affinity:
    """End-to-end PLAPT affinity pipeline on device: protein seq + SMILES -> pKd.

    protein -> ProtBERT pooler (1024) ; SMILES -> ChemBERTa pooler (768) ;
    fusion MLP -> normalized affinity -> rescaled to neg_log10_affinity_M (pKd).

    The pipeline is loaded once and kept resident across every ``predict`` /
    ``predict_many`` call (weights stay on-device; the only per-call host traffic
    is the tiny input-id tensors). ``fast=True`` additionally keeps the pooler
    activations on-device between the encoders and the fusion head — no
    bf16->fp32->bf16 host round-trip on the pooler outputs. That round-trip is
    bit-exact for bf16 (bf16->fp32 is zero-extension, fp32->bf16 of an
    already-bf16 value is identity), so ``--fast`` is parity-identical to the
    default path and just skips two host syncs per pair.
    """

    def __init__(self, prot: "ProtBERT", mol: "ChemBERTa", head: "AffinityHead"):
        self.prot = prot
        self.mol = mol
        self.head = head

    @classmethod
    def from_pretrained(cls) -> "Affinity":
        return cls(ProtBERT.from_pretrained(), ChemBERTa.from_pretrained(),
                   AffinityHead.from_pretrained())

    def _predict_tt(self, protein_seq: str, smiles: str) -> torch.Tensor:
        """Device-resident forward: encode both towers, feed on-device bf16
        poolers straight into the fusion head, return the normalized affinity
        as a host fp32 tensor [B,1] (B=1 here)."""
        prot_ids = tokenize_protein(protein_seq)
        mol_ids = tokenize_smiles(smiles)
        prot_pool, _ = self.prot._encode_tt(prot_ids)
        mol_pool, _ = self.mol._encode_tt(mol_ids)
        norm_tt = self.head.module(prot_pool, mol_pool)
        return self.head._to_torch(norm_tt)

    def predict(self, protein_seq: str, smiles: str, *, fast: bool = False) -> dict:
        if fast:
            norm = self._predict_tt(protein_seq, smiles)
            bsz = norm.shape[0]
            norm = norm.reshape(bsz, 1)
        else:
            prot_ids = tokenize_protein(protein_seq)
            mol_ids = tokenize_smiles(smiles)
            prot_pool, _ = self.prot(prot_ids)
            mol_pool, _ = self.mol(mol_ids)
            norm = self.head(prot_pool, mol_pool)
        pkd = AffinityHead.to_affinity(norm)
        pkd_v = float(pkd.reshape(-1)[0].item())
        return {
            "neg_log10_affinity_M": pkd_v,
            "affinity_uM": float((10 ** 6) * (10 ** (-pkd_v))),
        }

    def predict_many(
        self, pairs: list[tuple[str, str]], *, fast: bool = False,
    ) -> list[dict]:
        """Score a list of (protein, smiles) pairs with one resident pipeline.

        Reuses the loaded encoders + head across the whole batch (the ~30s
        weight load is paid once, then each pair is a forward pass). ``fast``
        toggles the device-resident pooler handoff (parity-identical)."""
        out = []
        for prot, mol in pairs:
            if not prot or not mol:
                out.append(None)
                continue
            out.append(self.predict(prot, mol, fast=fast))
        return out


# ---------------------------------------------------------------------------
# Multi-card data-parallel fanout (mirrors tt_bio.esmc.embed_multicard)
# ---------------------------------------------------------------------------

import os as _os
import sys as _sys
import pickle as _pickle
import shutil as _shutil
import subprocess as _subprocess
import tempfile as _tempfile


def _shard_pairs(pairs: list[tuple[str, str]], n: int) -> list[list[tuple[int, tuple[str, str]]]]:
    """Split pairs into ``n`` length-balanced shards for data-parallel scoring.

    Returns shards of ``(original_index, (protein, smiles))`` so the gather can
    restore input order without ``list.index`` (which is O(n^2) and ambiguous on
    duplicate pairs). Protein length dominates per-pair cost (ProtBERT is
    O(L^2) attention over 30 layers), so pairs are length-sorted and striped
    round-robin across shards so every card gets a similar length distribution."""
    shards: list[list[tuple[int, tuple[str, str]]]] = [[] for _ in range(n)]
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))
    for rank, i in enumerate(order):
        shards[rank % n].append((i, pairs[i]))
    return shards


def _run_affinity_shard(in_path: str, out_path: str) -> None:
    """Subprocess entry: score one shard on the pinned card, pickle results.

    Runs in a fresh interpreter with ``TT_VISIBLE_DEVICES`` set so the assigned
    physical chip is logical device 0. Reads a pickled ``{pairs, fast}`` request
    and writes ``[(index, result_dict), ...]`` (indices into the original list).
    """
    with open(in_path, "rb") as f:
        req = _pickle.load(f)
    model = Affinity.from_pretrained()
    results = []
    for idx, (prot, mol) in req["pairs"]:
        if not prot or not mol:
            results.append((idx, None))
            continue
        res = model.predict(prot, mol, fast=req["fast"])
        res["protein"] = prot
        res["smiles"] = mol
        results.append((idx, res))
    with open(out_path, "wb") as f:
        _pickle.dump(results, f)


def _thread_cap_env(n_workers: int) -> dict:
    cap = max(1, (_os.cpu_count() or 1) // max(1, n_workers))
    return {var: str(cap) for var in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            if var not in _os.environ}


def _spawn_affinity_shard(idx: int, device: int, shard: list[tuple[int, tuple[str, str]]],
                          workdir: str, *, fast: bool, thread_cap_env: dict | None = None):
    """Launch a pinned subprocess scoring ``shard`` on physical card ``device``.

    ``shard`` is a list of ``(original_index, (protein, smiles))``. Returns
    ``(proc, out_path, device, log_path, logf)``."""
    in_path = _os.path.join(workdir, f"shard{idx}.in.pkl")
    out_path = _os.path.join(workdir, f"shard{idx}.out.pkl")
    log_path = _os.path.join(workdir, f"shard{idx}.log")
    with open(in_path, "wb") as f:
        _pickle.dump(dict(pairs=shard, fast=fast), f)
    env = {**_os.environ, **(thread_cap_env or {}),
           "TT_VISIBLE_DEVICES": str(device), "TT_BIO_LOGICAL_DEVICE_ID": "0"}
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if device in _detect_p300_devices() and not env.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            env["TT_MESH_GRAPH_DESC_PATH"] = mgd
    logf = open(log_path, "w")
    proc = _subprocess.Popen(
        [_sys.executable, "-c",
         "import sys; from tt_bio.affinity import _run_affinity_shard; "
         "_run_affinity_shard(sys.argv[1], sys.argv[2])",
         in_path, out_path],
        env=env, stdout=logf, stderr=_subprocess.STDOUT)
    return proc, out_path, device, log_path, logf


def _read_log_tail(path: str, n: int) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def predict_multicard(pairs: list[tuple[str, str]], *, devices: list[int],
                      fast: bool = False) -> list[dict]:
    """Data-parallel affinity scoring across multiple physical TT cards.

    Shards ``pairs`` across ``devices`` (one pinned subprocess per card, each
    loads its own resident pipeline), runs :meth:`Affinity.predict_many` in each,
    then gathers results and restores the original input order. Embarrassingly
    parallel: each pair is independent (no cross-pair state), so a pair's pKd is
    identical to running it on one card — sharding changes only which chip
    scores which pair. More cards than pairs is harmless (extra cards idle)."""
    devices = list(devices)[:max(1, len(pairs))]
    shards = _shard_pairs(pairs, len(devices))
    workdir = _tempfile.mkdtemp(prefix="tt-bio-affinity-fanout-")
    thread_cap_env = _thread_cap_env(len(devices))
    try:
        handles = [
            _spawn_affinity_shard(idx, dev, shard, workdir,
                                  fast=fast, thread_cap_env=thread_cap_env)
            for idx, (dev, shard) in enumerate(zip(devices, shards)) if shard
        ]
        gathered: list[tuple[int, dict]] = []
        for proc, out_path, device, log_path, logf in handles:
            proc.wait()
            logf.close()
            if proc.returncode != 0:
                raise RuntimeError(f"affinity shard on device {device} failed "
                                   f"(exit {proc.returncode}):\n{_read_log_tail(log_path, 25)}")
            with open(out_path, "rb") as f:
                gathered.extend(_pickle.load(f))
    finally:
        _shutil.rmtree(workdir, ignore_errors=True)
    gathered.sort(key=lambda r: r[0])
    return [r[1] for r in gathered]


def predict(pairs: list[tuple[str, str]], *, devices: list[int] | None = None,
            fast: bool = False) -> list[dict]:
    """Score (protein, smiles) pairs, optionally fanned across multiple cards.

    With 0/1 device the pipeline is loaded in-process on this card; with >1 the
    pairs are sharded across the cards (data-parallel, one resident pipeline per
    card) and results are returned in the original input order.
    """
    if devices and len(devices) > 1:
        return predict_multicard(pairs, devices=devices, fast=fast)
    model = Affinity.from_pretrained()
    results = []
    for prot, mol in pairs:
        if not prot or not mol:
            results.append(None)
            continue
        res = model.predict(prot, mol, fast=fast)
        res["protein"] = prot
        res["smiles"] = mol
        results.append(res)
    return results
