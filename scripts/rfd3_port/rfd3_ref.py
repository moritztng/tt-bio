"""Standalone torch reference of RFD3 TokenInitializer, faithful to the upstream
source (RosettaCommons/foundry models/rfd3, production). Dependencies on
`foundry`/`atomworks`/`opt_einsum`/`jaxtyping` are stubbed so this runs with just
torch + einops (+ numpy). Used as the PCC reference for the ttnn port.

Source files mirrored: layer_utils.py, pairformer_layers.py, blocks.py,
block_utils.py, attention.py, encoders.py (TokenInitializer only; atom_transformer
is None for the TokenInitializer config since n_blocks=0).
"""
import functools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.functional import silu


# --- foundry stubs ----------------------------------------------------------
def exists(x):
    return x is not None


def activation_checkpointing(fn):
    @functools.wraps(fn)
    def _wrap(*args, **kw):
        return fn(*args, **kw)
    return _wrap


def scatter_mean(out, dim, idx, src):
    """Mimic foundry.utils.torch.scatter_mean(out, dim, idx, src) -> out (in-place-ish).
    out: pre-zeroed target; idx: index along dim; src: values. Returns out with means."""
    counts = torch.zeros_like(out)
    counts.scatter_add_(dim, idx, torch.ones_like(src))
    out.scatter_add_(dim, idx, src)
    out = out / counts.clamp(min=1)
    return out


# opt_einsum.contract -> torch.einsum (small shapes; path opt irrelevant here)
einsum = torch.einsum


# --- layer_utils.py ---------------------------------------------------------
def RMSNorm(*args, **kwargs):
    if "bias" in kwargs:
        kwargs.pop("bias")
    return nn.RMSNorm(*args, **kwargs)


linearNoBias = functools.partial(torch.nn.Linear, bias=False)


class EmbeddingLayer(nn.Linear):
    def __init__(self, this_in_features, total_embedding_features, out_features, device=None, dtype=None):
        self.total_embedding_features = total_embedding_features
        self.out_features = out_features
        super().__init__(this_in_features, out_features, bias=False, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        super().reset_parameters()
        a = math.sqrt(6.0 / float(self.total_embedding_features + self.out_features))
        nn.init._no_grad_uniform_(self.weight, -a, a)


def collapse(x, L):
    return x.reshape((L, x.numel() // L))


class MultiDimLinear(nn.Linear):
    def __init__(self, in_features, out_shape, norm=False, **kwargs):
        self.out_shape = out_shape
        out_features = int(np.prod(out_shape))
        super().__init__(in_features, out_features, **kwargs)
        if norm:
            self.ln = RMSNorm((out_features,))
            self.use_ln = True
        else:
            self.use_ln = False
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        super().reset_parameters()
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        out = super().forward(x)
        if self.use_ln:
            out = self.ln(out)
        return out.reshape(x.shape[:-1] + self.out_shape)


class LinearBiasInit(nn.Linear):
    def __init__(self, *args, biasinit, **kwargs):
        assert biasinit == -2.0
        self.biasinit = biasinit
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        self.bias.data.fill_(self.biasinit)


class Transition(nn.Module):
    def __init__(self, n, c):
        super().__init__()
        self.layer_norm_1 = RMSNorm(c)
        self.linear_1 = linearNoBias(c, n * c)
        self.linear_2 = linearNoBias(c, n * c)
        self.linear_3 = linearNoBias(n * c, c)

    @activation_checkpointing
    def forward(self, X):
        X = self.layer_norm_1(X)
        A = self.linear_1(X)
        B = self.linear_2(X)
        X = self.linear_3(silu(A) * B)
        return X


class AdaLN(nn.Module):
    def __init__(self, c_a, c_s, n=2):
        super().__init__()
        self.ln_a = RMSNorm(normalized_shape=(c_a,), elementwise_affine=False)
        self.ln_s = RMSNorm(normalized_shape=(c_s,), bias=False)
        self.to_gain = nn.Sequential(nn.Linear(c_s, c_a), nn.Sigmoid())
        self.to_bias = linearNoBias(c_s, c_a)

    def forward(self, Ai, Si):
        Ai = self.ln_a(Ai)
        Si = self.ln_s(Si)
        return self.to_gain(Si) * Ai + self.to_bias(Si)


# --- block_utils.py (only the TokenInitializer-relevant helpers) ------------
def build_valid_mask(tok_idx, n_atoms_per_tok_max=None):
    tokens, counts = torch.unique(tok_idx, return_counts=True)
    A = int(counts.max()) if n_atoms_per_tok_max is None else int(n_atoms_per_tok_max)
    atom_idx_grid = torch.arange(A, device=tok_idx.device)[None, :]
    valid_mask = atom_idx_grid < counts[:, None]
    return valid_mask


def _atom_flat_idx(valid_mask):
    return valid_mask.flatten().nonzero(as_tuple=False).squeeze(1)


def ungroup_atoms(Q_L, valid_mask):
    B, n_atoms, c = Q_L.shape
    n_tokens, A = valid_mask.shape
    flat_idx = _atom_flat_idx(valid_mask)
    idx = flat_idx.view(1, -1, 1).expand(B, -1, c)
    Q_IA = torch.zeros(B, n_tokens * A, c, dtype=Q_L.dtype, device=Q_L.device)
    Q_IA = Q_IA.scatter(1, idx, Q_L)
    return Q_IA.reshape(B, n_tokens, A, c)


def group_atoms(Q_IA, valid_mask):
    B, n_tok, A, c = Q_IA.shape
    flat_idx = _atom_flat_idx(valid_mask)
    Q_L = Q_IA.reshape(B, n_tok * A, c)[:, flat_idx, :]
    return Q_L.contiguous()


def pairwise_mean_pool(pairwise_atom_features, atom_to_token_map, I, dtype):
    B = pairwise_atom_features.shape[0]
    atom_to_token_onehot = F.one_hot(atom_to_token_map.long(), num_classes=I).to(dtype)
    temp = torch.einsum("ia,bacd->bicd", atom_to_token_onehot.T, pairwise_atom_features)
    del pairwise_atom_features
    token_features_sum = torch.einsum("cj,bicd->bijd", atom_to_token_onehot, temp)
    del temp
    atom_counts_per_token = atom_to_token_onehot.sum(dim=0)
    token_pair_counts = torch.outer(atom_counts_per_token, atom_counts_per_token)
    token_pair_counts = token_pair_counts.unsqueeze(0).expand(B, -1, -1)
    token_pair_counts = torch.clamp(token_pair_counts, min=1)
    return token_features_sum / token_pair_counts.unsqueeze(-1)


# --- attention.py (GatedCrossAttention only; the Pairformer uses its own attn) --
class GatedCrossAttention(nn.Module):
    def __init__(self, c_query, c_kv, c_pair=None, c_model=128, n_head=4, kq_norm=True, dropout=0.0, **_):
        super().__init__()
        self.n_head = n_head
        self.scale = 1 / math.sqrt(c_model // n_head)
        assert c_model % n_head == 0
        self.ln_q = RMSNorm(c_query)
        self.ln_kv = RMSNorm(c_kv)
        self.to_q = linearNoBias(c_query, c_model)
        self.to_k = linearNoBias(c_kv, c_model)
        self.to_v = linearNoBias(c_kv, c_model)
        self.to_g = nn.Sequential(linearNoBias(c_query, c_model), nn.Sigmoid())
        self.to_out = nn.Sequential(nn.Linear(c_model, c_query), nn.Dropout(dropout))
        self.kq_norm = kq_norm
        if self.kq_norm:
            self.k_norm = RMSNorm(c_model)
            self.q_norm = RMSNorm(c_model)
        self.c_pair = c_pair
        if c_pair is not None:
            self.to_b = nn.Sequential(RMSNorm(c_pair), linearNoBias(c_pair, n_head))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_g[0].weight)
        nn.init.xavier_uniform_(self.to_out[0].weight)

    def forward(self, q, kv, attn_mask=None, pair_bias=None):
        q = self.ln_q(q)
        kv = self.ln_kv(kv)
        q, k, v, g = self.to_q(q), self.to_k(kv), self.to_v(kv), self.to_g(q)
        if self.kq_norm:
            k = self.k_norm(k)
            q = self.q_norm(q)
        q, k, v, g = map(lambda t: rearrange(t, "b t n (h c) -> b h t n c", h=self.n_head), (q, k, v, g))
        attn = einsum("bhtqc,bhtkc->bhtqk", q, k) * self.scale
        if pair_bias is not None:
            b = self.to_b(pair_bias)
            b = rearrange(b, "b t q k (h) -> b (h) t q k", h=self.n_head)
            attn = attn + b
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask[None, None], float("-inf"))
            invalid_queries = torch.logical_not(torch.any(attn_mask, dim=-1, keepdim=False))
            attn[:, :, invalid_queries, :] = 0.0
        attn = F.softmax(attn, dim=-1)
        attn_out = einsum("bhtqk,bhtkd->bhtqd", attn, v)
        attn_out = attn_out * g
        attn_out = rearrange(attn_out, "b h t n c -> b t n (h c)")
        attn_out = self.to_out(attn_out)
        return attn_out


# --- pairformer_layers.py ---------------------------------------------------
class AttentionPairBiasPairformerDeepspeed(nn.Module):
    def __init__(self, c_a, c_s, c_pair, n_head, kq_norm=False):
        super().__init__()
        self.n_head = n_head
        self.c_a = c_a
        self.c_pair = c_pair
        self.c = c_a // n_head
        self.to_q = MultiDimLinear(c_a, (n_head, self.c))
        self.to_k = MultiDimLinear(c_a, (n_head, self.c), bias=False, norm=kq_norm)
        self.to_v = MultiDimLinear(c_a, (n_head, self.c), bias=False, norm=kq_norm)
        self.to_b = linearNoBias(c_pair, n_head)
        self.to_g = nn.Sequential(MultiDimLinear(c_a, (n_head, self.c), bias=False), nn.Sigmoid())
        self.to_a = linearNoBias(c_a, c_a)
        self.ln_0 = RMSNorm((c_pair,))
        self.ln_1 = RMSNorm((c_a,))
        self.use_deepspeed_evo = False
        self.force_bfloat16 = True

    def forward(self, A_I, S_I, Z_II, Beta_II=None):
        assert S_I is None
        A_I = self.ln_1(A_I)
        if (self.use_deepspeed_evo or self.force_bfloat16) and A_I.device.type != "mps":
            A_I = A_I.to(torch.bfloat16)
        Q_IH = self.to_q(A_I)
        K_IH = self.to_k(A_I)
        V_IH = self.to_v(A_I)
        B_IIH = self.to_b(self.ln_0(Z_II)) + Beta_II[..., None]
        G_IH = self.to_g(A_I)
        B, L = B_IIH.shape[:2]
        if not self.use_deepspeed_evo or L <= 24:
            Q_IH = Q_IH / torch.sqrt(torch.tensor(self.c).to(Q_IH.device, Q_IH.dtype))
            A_IIH = torch.softmax(torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2)
            A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)
            A_I = G_IH * A_I
            A_I = A_I.flatten(start_dim=-2)
        else:
            raise NotImplementedError
        A_I = self.to_a(A_I)
        return A_I


class PairformerBlock(nn.Module):
    def __init__(self, c_s, c_z, attention_pair_bias, p_drop=0.1, triangle_multiplication=None,
                 triangle_attention=None, n_transition=4, use_deepspeed_evo=True,
                 use_triangle_mult=False, use_triangle_attn=False):
        super().__init__()
        self.z_transition = Transition(c=c_z, n=n_transition)
        if c_s > 0:
            self.s_transition = Transition(c=c_s, n=n_transition)
            self.attention_pair_bias = AttentionPairBiasPairformerDeepspeed(
                c_a=c_s, c_s=0, c_pair=c_z, **attention_pair_bias)

    @activation_checkpointing
    def forward(self, S_I, Z_II):
        _device = Z_II.device
        _use_autocast = _device.type != "mps"
        with torch.amp.autocast(device_type=_device.type, enabled=_use_autocast, dtype=torch.bfloat16):
            Z_II = Z_II + self.z_transition(Z_II)
            if S_I is not None:
                S_I = S_I + self.attention_pair_bias(
                    S_I, None, Z_II, Beta_II=torch.tensor([0.0], device=Z_II.device))
                S_I = S_I + self.s_transition(S_I)
        return S_I, Z_II


# --- blocks.py (TokenInitializer-relevant subset) --------------------------
class PositionPairDistEmbedder(nn.Module):
    def __init__(self, c_atompair, embed_frame=True):
        super().__init__()
        self.embed_frame = embed_frame
        if embed_frame:
            self.process_d = linearNoBias(3, c_atompair)
        self.process_inverse_dist = linearNoBias(1, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def forward_af3(self, D_LL, V_LL):
        P_LL = self.process_d(D_LL) * V_LL
        if self.training:
            P_LL = (P_LL + self.process_inverse_dist(1 / (1 + torch.linalg.norm(D_LL, dim=-1, keepdim=True) ** 2)) * V_LL)
            P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL
        else:
            P_LL[V_LL[..., 0]] += self.process_inverse_dist(1 / (1 + torch.linalg.norm(D_LL[V_LL[..., 0]], dim=-1, keepdim=True) ** 2))
            P_LL[V_LL[..., 0]] += self.process_valid_mask(V_LL[V_LL[..., 0]].to(P_LL.dtype))
        return P_LL

    def forward(self, ref_pos, valid_mask):
        D_LL = ref_pos.unsqueeze(-2) - ref_pos.unsqueeze(-3)
        V_LL = valid_mask
        if self.embed_frame:
            return self.forward_af3(D_LL, V_LL)
        norm = torch.linalg.norm(D_LL, dim=-1, keepdim=True) ** 2
        norm = torch.clamp(norm, min=1e-6)
        inv_dist = 1 / (1 + norm)
        P_LL = self.process_inverse_dist(inv_dist) * V_LL
        P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL
        return P_LL


class OneDFeatureEmbedder(nn.Module):
    def __init__(self, features, output_channels):
        super().__init__()
        self.features = {k: v for k, v in features.items() if exists(v)}
        total = sum(self.features.values())
        self.embedders = nn.ModuleDict({
            feature: EmbeddingLayer(n_channels, total, output_channels)
            for feature, n_channels in self.features.items()
        })

    def forward(self, f, collapse_length):
        return sum(
            self.embedders[feature](collapse(f[feature].float(), collapse_length))
            for feature, n_channels in self.features.items() if exists(n_channels)
        )


class SinusoidalDistEmbed(nn.Module):
    def __init__(self, c_atompair, n_freqs=32):
        super().__init__()
        assert c_atompair % 2 == 0
        self.n_freqs = n_freqs
        self.c_atompair = c_atompair
        self.output_proj = linearNoBias(2 * n_freqs, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def forward(self, pos, valid_mask):
        D_LL = pos.unsqueeze(-2) - pos.unsqueeze(-3)
        dist_matrix = torch.linalg.norm(D_LL, dim=-1)
        half_dim = self.n_freqs
        freq = torch.exp(-math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(dist_matrix.device)
        angles = dist_matrix.unsqueeze(-1) * freq
        sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        P_LL = self.output_proj(sincos)
        P_LL = P_LL * valid_mask
        P_LL = P_LL + self.process_valid_mask(valid_mask.to(P_LL.dtype)) * valid_mask
        return P_LL


class RelativePositionEncodingWithIndexRemoval(nn.Module):
    def __init__(self, r_max, s_max, c_z):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.num_tok_pos_bins = (2 * self.r_max + 2) + 1
        self.linear = linearNoBias(2 * self.num_tok_pos_bins + (2 * self.s_max + 2) + 1, c_z)

    def forward(self, f):
        b_samechain_II = f["asym_id"].unsqueeze(-1) == f["asym_id"].unsqueeze(-2)
        b_same_entity_II = f["entity_id"].unsqueeze(-1) == f["entity_id"].unsqueeze(-2)
        d_residue_II = torch.where(
            b_samechain_II,
            torch.clip(f["residue_index"].unsqueeze(-1) - f["residue_index"].unsqueeze(-2) + self.r_max, 0, 2 * self.r_max),
            2 * self.r_max + 1)
        b_sameresidue_II = f["residue_index"].unsqueeze(-1) == f["residue_index"].unsqueeze(-2)
        tok_distance = f["token_index"].unsqueeze(-1) - f["token_index"].unsqueeze(-2) + self.r_max
        d_token_II = torch.where(
            b_samechain_II * b_sameresidue_II,
            torch.clip(tok_distance, 0, 2 * self.r_max),
            2 * self.r_max + 1)
        d_chain_II = torch.where(
            b_same_entity_II,
            torch.clip(f["sym_id"].unsqueeze(-1) - f["sym_id"].unsqueeze(-2) + self.s_max, 0, 2 * self.s_max),
            2 * self.s_max + 1)
        A_relchain_II = F.one_hot(d_chain_II.long(), 2 * self.s_max + 2)
        unindexing_pair_mask = f["unindexing_pair_mask"]
        d_token_II[unindexing_pair_mask] = self.num_tok_pos_bins - 1
        d_residue_II[unindexing_pair_mask] = self.num_tok_pos_bins - 1
        A_relpos_II = F.one_hot(d_residue_II.long(), self.num_tok_pos_bins)
        A_reltoken_II = F.one_hot(d_token_II, self.num_tok_pos_bins)
        return self.linear(torch.cat([A_relpos_II, A_reltoken_II, b_same_entity_II.unsqueeze(-1), A_relchain_II], dim=-1).to(torch.float))


class Downcast(nn.Module):
    def __init__(self, c_atom, c_token, c_s=None, method="mean", cross_attention_block=None):
        super().__init__()
        self.method = method
        self.c_token = c_token
        self.c_atom = c_atom
        if c_s is not None:
            self.process_s = nn.Sequential(RMSNorm((c_s,)), linearNoBias(c_s, c_token))
        else:
            self.process_s = None
        if self.method == "mean":
            self.project = linearNoBias(c_atom, c_token)
        elif self.method == "cross_attention":
            self.gca = GatedCrossAttention(c_query=c_token, c_kv=c_atom, **cross_attention_block)
        else:
            raise ValueError(f"Unknown downcast method: {self.method}")

    def forward_(self, Q_IA, A_I, S_I=None, valid_mask=None):
        if self.method == "mean":
            A_I_update = self.project(Q_IA).sum(-2) / valid_mask.sum(-1, keepdim=True)
        elif self.method == "cross_attention":
            assert exists(A_I) and exists(valid_mask)
            attn_mask = valid_mask[..., None, :]
            A_I_update = self.gca(q=A_I[..., None, :], kv=Q_IA, attn_mask=attn_mask).squeeze(-2)
        A_I = A_I + A_I_update if exists(A_I) else A_I_update
        if self.process_s is not None:
            A_I = A_I + self.process_s(S_I)
        return A_I

    def forward(self, Q_L, A_I, S_I=None, tok_idx=None):
        valid_mask = build_valid_mask(tok_idx)
        if Q_L.ndim == 2:
            squeeze = True
            Q_L = Q_L.unsqueeze(0)
        else:
            squeeze = False
        A_I = A_I.unsqueeze(0) if exists(A_I) and A_I.ndim == 2 else A_I
        S_I = S_I.unsqueeze(0) if exists(S_I) and S_I.ndim == 2 else S_I
        Q_IA = ungroup_atoms(Q_L, valid_mask)
        A_I = self.forward_(Q_IA, A_I, S_I, valid_mask=valid_mask)
        if squeeze:
            A_I = A_I.squeeze(0)
        return A_I


# --- encoders.py (TokenInitializer) ----------------------------------------
class TokenInitializer(nn.Module):
    def __init__(self, c_s, c_z, c_atom, c_atompair, relative_position_encoding,
                 n_pairformer_blocks, pairformer_block, downcast, token_1d_features,
                 atom_1d_features, atom_transformer, use_chunked_pll=False):
        super().__init__()
        self.use_chunked_pll = use_chunked_pll
        self.atom_1d_embedder_1 = OneDFeatureEmbedder(atom_1d_features, c_s)
        self.atom_1d_embedder_2 = OneDFeatureEmbedder(atom_1d_features, c_atom)
        self.token_1d_embedder = OneDFeatureEmbedder(token_1d_features, c_s)
        self.downcast_atom = Downcast(c_atom=c_s, c_token=c_s, c_s=None, **downcast)
        self.transition_post_token = Transition(c=c_s, n=2)
        self.transition_post_atom = Transition(c=c_s, n=2)
        self.process_s_init = nn.Sequential(RMSNorm(c_s), linearNoBias(c_s, c_s))
        self.to_z_init_i = linearNoBias(c_s, c_z)
        self.to_z_init_j = linearNoBias(c_s, c_z)
        self.relative_position_encoding = RelativePositionEncodingWithIndexRemoval(c_z=c_z, **relative_position_encoding)
        self.relative_position_encoding2 = RelativePositionEncodingWithIndexRemoval(c_z=c_z, **relative_position_encoding)
        self.process_token_bonds = linearNoBias(1, c_z)
        self.process_z_init = nn.Sequential(RMSNorm(c_z * 2), linearNoBias(c_z * 2, c_z))
        self.transition_1 = nn.ModuleList([Transition(c=c_z, n=2), Transition(c=c_z, n=2)])
        self.ref_pos_embedder_tok = PositionPairDistEmbedder(c_z, embed_frame=False)
        self.transformer_stack = nn.ModuleList([
            PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block) for _ in range(n_pairformer_blocks)])
        self.process_s_trunk = nn.Sequential(RMSNorm(c_s), linearNoBias(c_s, c_atom))
        self.process_single_l = nn.Sequential(nn.ReLU(), linearNoBias(c_atom, c_atompair))
        self.process_single_m = nn.Sequential(nn.ReLU(), linearNoBias(c_atom, c_atompair))
        self.process_z = nn.Sequential(RMSNorm(c_z), linearNoBias(c_z, c_atompair))
        self.motif_pos_embedder = SinusoidalDistEmbed(c_atompair=c_atompair)
        self.ref_pos_embedder = PositionPairDistEmbedder(c_atompair, embed_frame=False)
        self.pair_mlp = nn.Sequential(nn.ReLU(), linearNoBias(c_atompair, c_atompair), nn.ReLU(),
                                      linearNoBias(c_atompair, c_atompair), nn.ReLU(), linearNoBias(c_atompair, c_atompair))
        self.process_pll = linearNoBias(c_atompair, c_atompair)
        self.project_pll = linearNoBias(c_atompair, c_z)
        if atom_transformer["n_blocks"] > 0:
            raise NotImplementedError("atom_transformer n_blocks>0 not vendored (TokenInitializer cfg uses 0)")
        self.atom_transformer = None

    def forward(self, f):
        tok_idx = f["atom_to_token_map"]
        L = len(tok_idx)
        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)
        I = len(f["restype"])

        def init_tokens():
            S_I = self.token_1d_embedder(f, I)
            S_I = S_I + self.transition_post_token(S_I)
            S_I = self.downcast_atom(Q_L=self.atom_1d_embedder_1(f, L), A_I=S_I, tok_idx=tok_idx)
            S_I = S_I + self.transition_post_atom(S_I)
            S_I = self.process_s_init(S_I)
            Z_init_II = self.to_z_init_i(S_I).unsqueeze(-3) + self.to_z_init_j(S_I).unsqueeze(-2)
            Z_init_II = Z_init_II + self.relative_position_encoding(f)
            Z_init_II = Z_init_II + self.process_token_bonds(f["token_bonds"].unsqueeze(-1).float())
            token_id = f["ref_space_uid"][f["is_ca"]]
            valid_mask = (token_id.unsqueeze(-1) == token_id.unsqueeze(-2)).unsqueeze(-1)
            Z_init_II = Z_init_II + self.ref_pos_embedder_tok(f["ref_pos"][f["is_ca"]], valid_mask)
            for block in self.transformer_stack:
                S_I, Z_init_II = block(S_I, Z_init_II)
            Z_init_II = torch.cat([Z_init_II, self.relative_position_encoding2(f)], dim=-1)
            Z_init_II = self.process_z_init(Z_init_II)
            for b in range(2):
                Z_init_II = Z_init_II + self.transition_1[b](Z_init_II)
            return {"S_init_I": S_I, "Z_init_II": Z_init_II}

        def init_atoms(S_init_I, Z_init_II):
            Q_L_init = self.atom_1d_embedder_2(f, L)
            C_L = Q_L_init + self.process_s_trunk(S_init_I)[..., tok_idx, :]
            valid_mask = (f["is_motif_atom_with_fixed_coord"].unsqueeze(-1) & f["is_motif_atom_with_fixed_coord"].unsqueeze(-2)).unsqueeze(-1)
            P_LL = self.motif_pos_embedder(f["motif_pos"], valid_mask)
            atoms_in_same_token = (f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)).unsqueeze(-1)
            atoms_has_seq = (f["is_motif_atom_with_fixed_seq"].unsqueeze(-1) & f["is_motif_atom_with_fixed_seq"].unsqueeze(-2)).unsqueeze(-1)
            valid_mask = atoms_in_same_token & atoms_has_seq
            P_LL = P_LL + self.ref_pos_embedder(f["ref_pos"], valid_mask)
            P_LL = P_LL + (self.process_single_l(C_L).unsqueeze(-2) + self.process_single_m(C_L).unsqueeze(-3))
            P_LL = P_LL + self.process_z(Z_init_II)[..., tok_idx, :, :][..., tok_idx, :]
            P_LL = P_LL + self.pair_mlp(P_LL)
            P_LL = P_LL.contiguous()
            pooled = pairwise_mean_pool(self.process_pll(P_LL).unsqueeze(0), atom_to_token_map=tok_idx,
                                        I=int(tok_idx.max().item()) + 1, dtype=P_LL.dtype).squeeze(0)
            Z_init_II = Z_init_II + self.project_pll(pooled)
            return {"Q_L_init": Q_L_init, "C_L": C_L, "P_LL": P_LL, "S_I": S_init_I, "Z_II": Z_init_II}

        tokens = init_tokens()
        return init_atoms(**tokens)


# --- config (from rfd3_net.yaml, verified) ----------------------------------
TOKEN_INITIALIZER_CONFIG = dict(
    c_s=384, c_z=128, c_atom=128, c_atompair=16,
    relative_position_encoding=dict(r_max=32, s_max=2),
    n_pairformer_blocks=2,
    pairformer_block=dict(use_triangle_attn=False, use_triangle_mult=False,
                          attention_pair_bias=dict(n_head=16, kq_norm=True)),
    downcast=dict(method="cross_attention", cross_attention_block=dict(n_head=4, c_model=128, kq_norm=True, dropout=0.0)),
    token_1d_features=dict(ref_motif_token_type=3, restype=32, ref_plddt=1, is_non_loopy=1),
    atom_1d_features=dict(ref_atom_name_chars=256, ref_element=128, ref_charge=1, ref_mask=1,
                          ref_is_motif_atom_with_fixed_coord=1, ref_is_motif_atom_unindexed=1,
                          has_zero_occupancy=1, ref_pos=3, ref_atomwise_rasa=3, active_donor=1,
                          active_acceptor=1, is_atom_level_hotspot=1),
    atom_transformer=dict(n_blocks=0),
)


def build_token_initializer(config=None):
    cfg = dict(config) if config else dict(TOKEN_INITIALIZER_CONFIG)
    return TokenInitializer(**cfg)
