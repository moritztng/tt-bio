"""RFD4-Proteina autoencoder: checkpoint folding, host featurisation and a torch forward.

Pure torch, no ttnn, so the fold is checked against upstream on CPU in float64
(`tests/test_rfd4_ae_fold.py`) and the device port (`tt_bio.rfd4.autoencoder`) reads the same
folded tensors.

The released autoencoder (`nn_ae/nn_130m_unified`) has no conditioning features
(`feats_cond_seq: null`) and no pair track (`use_attn_pair_bias: false`). Its conditioning vector
is therefore exactly zero: the two cond transitions have no bias, so they map zero to zero. Every
adaLN then reads `LayerNorm(0) = norm_cond.bias`, and every output gate reads
`sigmoid(W 0 + b) = sigmoid(b)`. Both are per-layer constants, folded here:

    adaLN(x)           = LN(x) * sigmoid(Wg nb + bg) + Wb nb         -> LN(x) * g + b
    out-gate(o)        = o * sigmoid(b_gate)                         -> into to_out_node / linear_out

and each block is a plain pre-norm transformer layer with qk-LayerNorm and a sigmoid query gate:

    h = LN_node(LN(x) * g1 + b1);  q, k, v = h Wqkv;  q, k = LN_q(q), LN_k(k)
    x = mask * (x + (sigmoid(h Wg) * MHA(q, k, v)) Wo')
    t = LN(x) * g2 + b2;           a, gate = t W1
    x = mask * (x + (silu(gate) * a) W2')

The output heads (encoder: mean / log-scale; decoder: 147 residue logits, 37x3 coordinates, 37 atom
presence logits) each start with their own affine LayerNorm. The affine part is folded into the
head weight, so the decoder runs one parameter-free LN and one [768, 295] projection.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

D = 768
HEADS = 12
DH = 64
LAYERS = 12
LATENT = 32
N_RES = 147
N_TOKEN_TYPES = 7
TOKEN_TYPE_ATOMIZED = 3
LN_EPS = 1e-5
# Upstream's additive mask value (pair_bias_attn.NEG_INF).
NEG_INF = -1e4
# Batch key of the C_SEQ condition (atomworks `C_SEQ.full_name`).
C_SEQ = "condition_sequence_1_residue"
# Decoder head layout in the fused projection.
HEAD_SLICES = {"logits": (0, N_RES), "coords": (N_RES, N_RES + 111), "presence": (N_RES + 111, N_RES + 148)}

_TABLES = Path(__file__).with_name("ae_tables.json")


def tables() -> dict:
    """Residue vocabulary tables the decoder needs, exported from upstream at the pinned sha.

    `allowed` [7, 147]: residue classes legal per token type (`get_allowed_residue_type_mask`).
    `atom_mask` [147, 37]: legal atom37 slots per residue class (`_build_unified_atom37_mask`).
    `mask_token`: index of `<M>`, the unconditioned entry of C_SEQ.
    """
    d = json.loads(_TABLES.read_text())
    assert d["c_seq_key"] == C_SEQ and d["n_tokens"] == N_RES, d["c_seq_key"]
    return {
        "allowed": torch.tensor(d["allowed"], dtype=torch.bool),
        "atom_mask": torch.tensor(d["atom_mask"], dtype=torch.bool),
        "mask_token": int(d["mask_token"]),
    }


def _sub(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _adaln_const(sd: dict, p: str) -> tuple[torch.Tensor, torch.Tensor]:
    nb = sd[p + "norm_cond.bias"]
    g = torch.sigmoid(sd[p + "to_gamma.0.weight"] @ nb + sd[p + "to_gamma.0.bias"])
    b = sd[p + "to_beta.weight"] @ nb
    return g, b


def _fold_ln_into_linear(w_ln, b_ln, w):
    """`(LN0(x) * w_ln + b_ln) @ w.T` -> `LN0(x) @ w2.T + b2`."""
    return w * w_ln[None, :], w @ b_ln


def fold_layer(sd: dict, p: str, dtype=torch.float64) -> dict:
    sd = {k: v.to(dtype) for k, v in _sub(sd, p).items()}
    m = "mhba."
    g1, b1 = _adaln_const(sd, m + "adaln.")
    s_attn = torch.sigmoid(sd[m + "scale_output.to_adaln_zero_gamma.0.bias"])
    t = "transition."
    g2, b2 = _adaln_const(sd, t + "adaln.")
    s_tr = torch.sigmoid(sd[t + "scale_output.to_adaln_zero_gamma.0.bias"])
    a = m + "mha."
    return {
        "ln1_w": g1, "ln1_b": b1,
        "node_w": sd[a + "node_norm.weight"], "node_b": sd[a + "node_norm.bias"],
        "qkv_w": sd[a + "to_qkv.weight"], "qkv_b": sd[a + "to_qkv.bias"],
        "q_ln_w": sd[a + "q_layer_norm.weight"], "q_ln_b": sd[a + "q_layer_norm.bias"],
        "k_ln_w": sd[a + "k_layer_norm.weight"], "k_ln_b": sd[a + "k_layer_norm.bias"],
        "g_w": sd[a + "to_g.weight"], "g_b": sd[a + "to_g.bias"],
        "o_w": sd[a + "to_out_node.weight"] * s_attn[:, None], "o_b": sd[a + "to_out_node.bias"] * s_attn,
        "ln2_w": g2, "ln2_b": b2,
        "w1": sd[t + "transition.swish_linear.0.weight"],
        "w2": sd[t + "transition.linear_out.weight"] * s_tr[:, None],
    }


def _check_zero_cond(sd: dict, part: str) -> None:
    """The fold is exact only for a checkpoint whose conditioning is empty: refuse any other."""
    extra = [k for k in sd if k.startswith(f"{part}.cond_factory.") or k.startswith(f"{part}.pair_rep_factory.")]
    if extra:
        raise ValueError(f"RFD4 AE {part} carries conditioning or pair weights ({extra[:3]}); the zero-cond fold does not apply")


def fold(sd: dict, part: str, dtype=torch.float64) -> dict:
    """Fold `encoder` or `decoder` of the AE state dict into plain-transformer tensors."""
    assert part in ("encoder", "decoder"), part
    _check_zero_cond(sd, part)
    out = {"layers": [fold_layer(sd, f"{part}.transformer_layers.{i}.", dtype) for i in range(LAYERS)]}
    out["in_w"] = sd[f"{part}.init_repr_factory.linear_out.weight"].to(dtype)
    emb = _sub(sd, f"{part}.init_repr_factory.feat_creators.")
    out["emb"] = {k: v.to(dtype) for k, v in emb.items()}
    heads = ["latent_decoder_mean_n_log_scale"] if part == "encoder" else ["logit_linear", "struct_linear", "atom_presence_linear"]
    ws, bs = [], []
    for h in heads:
        w, b = _fold_ln_into_linear(sd[f"{part}.{h}.0.weight"].to(dtype), sd[f"{part}.{h}.0.bias"].to(dtype),
                                    sd[f"{part}.{h}.1.weight"].to(dtype))
        ws.append(w)
        bs.append(b)
    out["head_w"], out["head_b"] = torch.cat(ws), torch.cat(bs)
    return out


# ----------------------------------------------------------------------------------- features
#
# The sequence features exactly as upstream's FeatureFactory builds them (nn/feature_factory.py at the
# pinned sha), concatenated in config order and masked by padding. Embedding tables come from the
# checkpoint (`feat_creators.<i>.*`). The linear projection `in_w` runs on the device.

def _charge_index(charges: torch.Tensor) -> torch.Tensor:
    return (charges.long() + 4).clamp(0, 10)


def encoder_features(batch: dict, emb: dict) -> torch.Tensor:
    """[b, n, 1587]: token_type, x1_a37coors_nm, x1_a37coors_nm_rel, chain_idx_seq,
    incident_bond_type_counts, atom37_element_charge_embedding."""
    dt = emb["0.embedding.weight"].dtype
    mask37 = batch["atom37_mask"]
    coors = batch["coords_nm"].to(dt) * mask37[..., None]
    rel = (coors - coors[:, :, 1:2, :]) * mask37[..., None]
    m = mask37.to(dt)
    f_abs = torch.cat([coors.flatten(-2), m], -1)
    f_rel = torch.cat([rel.flatten(-2), m], -1)
    chain = batch["chain_iid"].unsqueeze(-1).to(dt)
    bonds = (batch["incident_bond_type_counts"].to(dt) * m.unsqueeze(-1)).flatten(-2)
    mb = mask37.bool()
    el = batch["ground_truth_element"].long().masked_fill(~mb, 0)
    ch = batch["charge"].long().masked_fill(~mb, 0)
    ec = torch.cat([emb["5.element_embedding.weight"][el], emb["5.charge_embedding.weight"][_charge_index(ch)]], -1)
    ec = (ec * mb.unsqueeze(-1)).flatten(-2)
    tt = emb["0.embedding.weight"][batch["token_type"].long()]
    f = torch.cat([tt, f_abs, f_rel, chain, bonds, ec], -1)
    return f * batch["padding_mask"][..., None]


def decoder_features(batch: dict, emb: dict, mask_token: int) -> torch.Tensor:
    """[b, n, 214]: ca_coors_nm, z_latent_seq, token_type, cond_residue_type."""
    dt = emb["2.embedding.weight"].dtype
    ca = batch["ca_coors_nm"].to(dt)
    z = batch["z_latent"].to(dt)
    tt = emb["2.embedding.weight"][batch["token_type"].long()]
    cseq = batch.get(C_SEQ)
    if cseq is None:
        cr = torch.zeros(*ca.shape[:2], N_RES, dtype=dt)
    else:
        cond = cseq != mask_token
        cr = emb["3.embedding.weight"][(cseq * cond).long()] * cond[..., None]
    f = torch.cat([ca, z, tt, cr], -1)
    return f * batch["padding_mask"][..., None]


def ae_masks(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """`model_masks.compute_ae_masks`: (attend, emit)."""
    attend = batch["padding_mask"] & ~batch["is_in_context_conditioning"]
    return attend, attend & batch["supports_local_latent"]


# ------------------------------------------------------------------------------- torch forward

def _ln(x, w=None, b=None):
    return F.layer_norm(x, (x.shape[-1],), w, b, LN_EPS)


def layer_forward(L: dict, x: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """One folded block. `bias` [b, 1, n, n] additive key mask, `mask` [b, n, 1]."""
    b_, n, _ = x.shape
    h = _ln(_ln(x) * L["ln1_w"] + L["ln1_b"], L["node_w"], L["node_b"])
    q, k, v = (h @ L["qkv_w"].T + L["qkv_b"]).chunk(3, -1)
    q, k = _ln(q, L["q_ln_w"], L["q_ln_b"]), _ln(k, L["k_ln_w"], L["k_ln_b"])
    q, k, v = (t.reshape(b_, n, HEADS, DH).transpose(1, 2) for t in (q, k, v))
    a = torch.softmax(q @ k.transpose(-1, -2) * DH ** -0.5 + bias, -1) @ v
    a = a.transpose(1, 2).reshape(b_, n, D)
    o = torch.sigmoid(h @ L["g_w"].T + L["g_b"]) * a
    x = (x + o @ L["o_w"].T + L["o_b"]) * mask
    t = _ln(x) * L["ln2_w"] + L["ln2_b"]
    u, gate = (t @ L["w1"].T).chunk(2, -1)
    return (x + (F.silu(gate) * u) @ L["w2"].T) * mask


def key_bias(attend: torch.Tensor, dtype) -> torch.Tensor:
    """[b, 1, n, n] additive mask: NEG_INF on masked keys, as upstream's PairBiasAttention."""
    pair = attend[:, :, None] & attend[:, None, :]
    bias = torch.zeros(pair.shape, dtype=dtype).masked_fill(~pair, NEG_INF)
    bias = bias.masked_fill(~pair.any(-1, keepdim=True), 0.0)
    return bias[:, None]


def trunk_forward(f: dict, feats: torch.Tensor, attend: torch.Tensor) -> torch.Tensor:
    """Feature projection and the 12 blocks; returns the final token stream [b, n, 768]."""
    mask = attend[..., None].to(feats.dtype)
    x = (feats @ f["in_w"].T) * mask
    bias = key_bias(attend, feats.dtype)
    for L in f["layers"]:
        x = layer_forward(L, x, bias, mask)
    return x


def heads_forward(f: dict, x: torch.Tensor) -> torch.Tensor:
    return _ln(x) @ f["head_w"].T + f["head_b"]


def encode(f: dict, batch: dict, noise: torch.Tensor | None = None) -> dict:
    """Encoder forward. `noise` is the standard normal draw of `z = mean + noise * exp(log_scale)`;
    upstream draws it with `torch.randn_like`, the caller supplies it so both sides share one draw."""
    attend, emit = ae_masks(batch)
    x = trunk_forward(f, encoder_features(batch, f["emb"]), attend)
    out = heads_forward(f, x) * emit[..., None]
    mean, log_scale = out.chunk(2, -1)
    em = emit[..., None]
    z = mean if noise is None else mean + noise.to(mean.dtype) * torch.exp(log_scale)
    return {"mean": mean, "log_scale": log_scale, "z_latent": z * em, "emit_mask": emit}


def decode_heads(raw: torch.Tensor, batch: dict, emit: torch.Tensor, tab: dict) -> dict:
    """The decoder's head post-processing, shared by the torch and device paths: residue masking per
    token type, argmax, coordinates relative to CA, atom presence and legal-slot masks."""
    s = HEAD_SLICES
    em = emit[..., None]
    logits = raw[..., s["logits"][0]:s["logits"][1]] * em
    coords = raw[..., s["coords"][0]:s["coords"][1]] * em
    presence = raw[..., s["presence"][0]:s["presence"][1]]
    if "token_type" in batch:
        allowed = tab["allowed"][batch["token_type"].long()] | ~em
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    b, n = emit.shape
    c = coords.reshape(b, n, 37, 3).clone()
    c[..., 1, :] = 0.0
    c = (c + batch["ca_coors_nm"].to(c.dtype)[:, :, None, :]) * emit[..., None, None]
    aatype = torch.argmax(logits, -1) * emit
    return {"coors_nm": c, "seq_logits": logits, "emit_mask": emit, "aatype_max": aatype,
            "atom_presence_logits": presence}


def finish_decode(dec: dict, batch: dict, tab: dict, required_atom_mask=None) -> dict:
    """`AutoEncoder.decode`: C_SEQ overrides conditioned positions, then the atom mask."""
    emit = dec["emit_mask"]
    rt = dec["aatype_max"]
    if C_SEQ in batch:
        cs = batch[C_SEQ]
        rt = torch.where((cs != tab["mask_token"]) & emit, cs, rt)
    pred = dec["atom_presence_logits"] > 0
    pred[..., 1] = True
    if required_atom_mask is not None:
        pred = pred | required_atom_mask
    atom_mask = tab["atom_mask"][rt] & pred & emit[..., None]
    return {"coors_nm": dec["coors_nm"] * emit[..., None, None] * atom_mask[..., None],
            "residue_type": rt * emit, "emit_mask": emit, "atom_mask": atom_mask}


def decode(f: dict, batch: dict, tab: dict) -> dict:
    attend, emit = ae_masks(batch)
    x = trunk_forward(f, decoder_features(batch, f["emb"], tab["mask_token"]), attend)
    return decode_heads(heads_forward(f, x), batch, emit, tab)
