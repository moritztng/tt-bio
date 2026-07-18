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
    """RoBERTa embeddings: word + position + token_type(=0) -> LayerNorm."""

    def __init__(self, state_dict: Weights, compute_kernel_config, cfg: dict):
        super().__init__(state_dict, compute_kernel_config)
        self.pad_token_id = cfg["pad_token_id"]
        self.word = self.torch_to_tt("word_embeddings.weight", transform=lambda x: x)
        self.pos = self.torch_to_tt("position_embeddings.weight", transform=lambda x: x)
        # token_type is always id 0 -> a single [1,1,H] broadcast bias.
        self.ttype = self.torch_to_tt(
            "token_type_embeddings.weight", transform=lambda x: x.reshape(1, 1, -1)
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
        x = ttnn.layer_norm(x, weight=self.ln_w, bias=self.ln_b, epsilon=BN_EPS,
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
                            epsilon=BN_EPS, compute_kernel_config=ck)
        ttnn.deallocate(o)
        h = ttnn.gelu(self._lin(a, self.i_w, bias=self.i_b))
        out = self._lin(h, self.f_w, bias=self.f_b)
        ttnn.deallocate(h)
        y = ttnn.layer_norm(ttnn.add(out, a), weight=self.out_ln_w, bias=self.out_ln_b,
                            epsilon=BN_EPS, compute_kernel_config=ck)
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

    def forward(self, input_ids: torch.Tensor):
        pos = _position_ids(input_ids, self.cfg["pad_token_id"])
        tokens_tt = ttnn.from_torch(
            input_ids.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        pos_tt = ttnn.from_torch(
            pos.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        pool, hidden = self.module(tokens_tt, pos_tt)
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
