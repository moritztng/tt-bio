"""Standalone torch reference of the RFD3 DiffusionModule + EDM sampler, faithful to
upstream (RosettaCommons/foundry models/rfd3, production). Reuses the shared
building blocks vendored in rfd3_ref.py (RMSNorm, Transition, AdaLN, GatedCrossAttention,
PairformerBlock, LinearBiasInit, scatter_mean, etc.). Deps on foundry/atomworks/
opt_einsum/jaxtyping are stubbed. Used as the shared-draws PCC bridge for the ttnn port
(device-vs-this-reference with identical RNG draws is the VALID trajectory metric, per
memory diffusion-port-parity-shared-draws; no vast.ai CUDA golden needed).
"""
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfd3_ref import (
    RMSNorm,
    Transition,
    AdaLN,
    GatedCrossAttention,
    PairformerBlock,
    LinearBiasInit,
    Downcast,
    scatter_mean,
    linearNoBias,
    EmbeddingLayer,
    collapse,
    exists,
    build_valid_mask,
    ungroup_atoms,
    group_atoms,
    DiTBlockRef,
    _indices_to_mask,
)


# --- FourierEmbedding (foundry.model.layers.blocks) ---
class FourierEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.register_buffer("w", torch.zeros(c, dtype=torch.float32))
        self.register_buffer("b", torch.zeros(c, dtype=torch.float32))
        nn.init.normal_(self.w)
        nn.init.normal_(self.b)

    def forward(self, t):
        return torch.cos(2 * math.pi * (t[..., None] * self.w + self.b))


# --- block_utils ---
def bucketize_scaled_distogram(R_L, min_dist=1.0, max_dist=30.0, sigma_data=16.0, n_bins=65):
    D_LL = torch.linalg.norm(R_L.unsqueeze(-2) - R_L.unsqueeze(-3), dim=-1)
    lo, hi = min_dist / sigma_data, max_dist / sigma_data
    bins = torch.linspace(lo, hi, n_bins - 1, device=D_LL.device)
    return F.one_hot(torch.bucketize(D_LL, bins), num_classes=n_bins).float()


def build_index_mask(tok_idx, n_sequence_neighbours, k_max, chain_id=None, base_mask=None):
    device = tok_idx.device
    L = tok_idx.shape[0]
    k_max = min(k_max, L)
    I = int(tok_idx.max().item()) + 1
    n_atoms_per_token = torch.zeros(I, device=device).float()
    n_atoms_per_token.scatter_add_(0, tok_idx.long(), torch.ones_like(tok_idx).float())
    token_indices = torch.arange(I, device=device)
    token_diff = (token_indices[:, None] - token_indices[None, :]).abs()
    atom_indices = torch.arange(L, device=device)
    atom_diff = (atom_indices[:, None] - atom_indices[None, :]).abs()
    token_mask = token_diff <= n_sequence_neighbours
    token_i = tok_idx[:, None]
    token_j = tok_idx[None, :]
    mask = token_mask[token_i, token_j]
    mask = mask & (atom_diff <= (k_max // 2))
    n_query_per_token = torch.zeros((L, I), device=device).float()
    n_query_per_token.scatter_add_(1, tok_idx.long()[None, :].expand(L, -1).contiguous(), mask.float())
    fully_included = n_query_per_token == n_atoms_per_token[None, :]
    n_atoms_fully_included = torch.zeros((I, I), device=device)
    n_atoms_fully_included.index_add_(0, tok_idx.long(), fully_included.float())
    full_token_mask = n_atoms_fully_included == n_atoms_per_token[:, None]
    full_token_mask = full_token_mask[token_i, token_j]
    mask &= full_token_mask
    if chain_id is not None:
        same_chain = chain_id.unsqueeze(-1) == chain_id.unsqueeze(-2)
        mask = mask & same_chain
    if base_mask is not None:
        mask = mask & base_mask
    return mask


def extend_index_mask_with_neighbours(mask, D_LL, k):
    if D_LL.ndim == 2:
        D_LL = D_LL.unsqueeze(0)
    B, L, _ = D_LL.shape
    k = min(k, L)
    device = D_LL.device
    inf = torch.tensor(float("inf"), dtype=D_LL.dtype, device=device)
    all_idx_row = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
    indices = torch.where(mask.contiguous(), all_idx_row, inf)
    indices = indices.sort(dim=1)[0][:, :k]
    D_LL = torch.where(mask.contiguous(), inf, D_LL)
    filler_idx = torch.topk(D_LL, k, dim=-1, largest=False).indices
    filler_idx = filler_idx.flip(dims=[-1])
    to_fill = indices == inf
    to_fill = to_fill.expand_as(filler_idx).contiguous()
    indices = indices.expand_as(filler_idx).contiguous()
    indices = torch.where(to_fill, filler_idx, indices)
    return indices.long()


def get_sparse_attention_indices(res_idx, D_LL, n_seq_neighbours, k_max, chain_id=None, base_mask=None):
    mask = build_index_mask(res_idx, n_seq_neighbours, k_max, chain_id=chain_id, base_mask=base_mask)
    indices = extend_index_mask_with_neighbours(mask, D_LL, k_max)
    indices, _ = torch.sort(indices, dim=-1)
    return indices.detach()


def create_attention_indices(f, n_attn_keys, n_attn_seq_neighbours, X_L=None, tok_idx=None):
    tok_idx = f["atom_to_token_map"] if tok_idx is None else tok_idx
    device = X_L.device if X_L is not None else tok_idx.device
    L = len(tok_idx)
    if X_L is None:
        X_L = torch.randn((1, L, 3), device=device, dtype=torch.float)
    D_LL = torch.cdist(X_L, X_L, p=2)
    base_mask = ~f["unindexing_pair_mask"][tok_idx[None, :], tok_idx[:, None]]
    k_actual = min(n_attn_keys, L)
    chain_ids = f["asym_id"][tok_idx] if "asym_id" in f else None
    if chain_ids is not None and len(torch.unique(chain_ids)) > 3:
        k_inter = max(32, k_actual // 4)
        k_intra = k_actual - k_inter
        intra = get_sparse_attention_indices(tok_idx, D_LL, n_attn_seq_neighbours, k_intra, chain_ids, base_mask)
        inter = torch.zeros(D_LL.shape[0], L, k_inter, dtype=torch.long, device=device)
        for b in range(D_LL.shape[0]):
            for c in torch.unique(chain_ids):
                ci = chain_ids[c]
                other = (chain_ids != ci) & base_mask[c, :]
                oi = torch.where(other)[0]
                n_sel = min(k_inter, len(oi))
                if n_sel > 0:
                    _, closest = torch.topk(D_LL[b, c, oi], n_sel, largest=False)
                    inter[b, c, :n_sel] = oi[closest]
                if n_sel < k_inter:
                    inter[b, c, n_sel:] = torch.randint(0, L, (k_inter - n_sel,), device=device)
        return torch.cat([intra, inter], dim=-1)
    return get_sparse_attention_indices(tok_idx, D_LL, n_attn_seq_neighbours, k_actual, chain_ids, base_mask)


# --- Upcast (decoder) ---
class Upcast(nn.Module):
    def __init__(self, c_atom, c_token, n_split=3, cross_attention_block=None):
        super().__init__()
        self.n_split = n_split
        self.c_atom, self.c_token = c_atom, c_token
        self.gca = GatedCrossAttention(c_query=c_atom, c_kv=c_token // n_split,
                                       **cross_attention_block)

    def forward_(self, Q_IA, A_I, valid_mask=None):
        A_I = A_I.reshape(*A_I.shape[:-1], self.n_split, self.c_token // self.n_split)
        n_tokens, n_atom_per_tok = Q_IA.shape[1], Q_IA.shape[2]
        attn_mask = torch.ones((n_tokens, n_atom_per_tok, self.n_split), dtype=torch.bool, device=Q_IA.device)
        if valid_mask is not None:
            attn_mask[~valid_mask, :] = False
        return Q_IA + self.gca(q=Q_IA, kv=A_I, attn_mask=attn_mask)

    def forward(self, Q_L, A_I, tok_idx=None):
        valid_mask = build_valid_mask(tok_idx)
        if Q_L.ndim == 2:
            Q_L = Q_L.unsqueeze(0)
        Q_IA = ungroup_atoms(Q_L, valid_mask)
        Q_IA = self.forward_(Q_IA, A_I, valid_mask)
        return group_atoms(Q_IA, valid_mask)


# --- atom-level + token-level transformer stacks (reuse DiTBlockRef at any dims) ---
class LocalAtomTransformerRef(nn.Module):
    def __init__(self, c_atom=128, c_s=128, c_pair=16, n_head=4, n_block=3):
        super().__init__()
        self.blocks = nn.ModuleList([DiTBlockRef(c_token=c_atom, c_s=c_s, c_pair=c_pair, n_head=n_head)
                                     for _ in range(n_block)])

    def forward(self, Q_L, C_L, P_LL, valid_mask):
        for b in self.blocks:
            Q_L = b(Q_L, C_L, P_LL, valid_mask=valid_mask)
        return Q_L


class CompactStreamingDecoderRef(nn.Module):
    """Decoder atom blocks condition on C_L (c_s=c_atom=128); downcast process_s on S_I (c_s=384)."""
    def __init__(self, c_atom=128, c_atompair=16, c_token=768, c_s=384, n_head=4, n_block=3, n_split=3):
        super().__init__()
        cab = dict(n_head=n_head, c_model=c_atom)
        self.upcast = nn.ModuleList([Upcast(c_atom, c_token, n_split=n_split, cross_attention_block=cab)
                                    for _ in range(n_block)])
        self.atom_transformer = nn.ModuleList([DiTBlockRef(c_token=c_atom, c_s=c_atom, c_pair=c_atompair, n_head=n_head)
                                    for _ in range(n_block)])
        self.downcast = Downcast(c_atom=c_atom, c_token=c_token, c_s=c_s, method="cross_attention",
                                 cross_attention_block=cab)

    def forward(self, A_I, S_I, Z_II, Q_L, C_L, P_LL, valid_mask, tok_idx):
        for up, blk in zip(self.upcast, self.atom_transformer):
            Q_L = up(Q_L, A_I, tok_idx=tok_idx)
            Q_L = blk(Q_L, C_L, P_LL, valid_mask=valid_mask)
        A_I = self.downcast(Q_L, A_I, S_I, tok_idx=tok_idx)
        return A_I, Q_L, {}


# --- LinearEmbedWithPool (process_a) ---
class LinearEmbedWithPoolRef(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.linear = linearNoBias(c_in, c_out)

    def forward(self, R_L, tok_idx):
        # R_L: [B, L, 3] -> linear -> [B, L, c_out] -> scatter_mean over tokens -> [B, I, c_out]
        emb = self.linear(R_L)
        I = int(tok_idx.max().item()) + 1
        idx = tok_idx.long().view(1, -1, 1).expand_as(emb).contiguous()
        out = scatter_mean(torch.zeros(emb.shape[0], I, emb.shape[-1], dtype=emb.dtype, device=emb.device),
                           -2, idx, emb)
        return out


# --- LinearSequenceHead ---
class LinearSequenceHeadRef(nn.Module):
    def __init__(self, c_token=768, n_token=32):
        super().__init__()
        self.linear = nn.Linear(c_token, n_token)
        self.register_buffer("valid_out_mask", torch.ones(n_token, dtype=torch.bool))

    def forward(self, A_I):
        return self.linear(A_I)


# --- DiffusionTokenEncoder (encoders.py) ---
class DiffusionTokenEncoderRef(nn.Module):
    def __init__(self, c_s=384, c_z=128, sigma_data=16.0, n_pairformer_blocks=2, n_head=16,
                 use_distogram=True, use_self=True):
        super().__init__()
        self.c_s, self.c_z = c_s, c_z
        self.sigma_data = sigma_data
        self.use_distogram, self.use_self = use_distogram, use_self
        self.n_bins = 65
        self.transition_1 = nn.ModuleList([Transition(c=c_s, n=2), Transition(c=c_s, n=2)])
        cat_c_z = c_z + int(use_distogram) * self.n_bins + int(use_self) * self.n_bins
        self.process_z = nn.Sequential(RMSNorm(cat_c_z), linearNoBias(cat_c_z, c_z))
        self.transition_2 = nn.ModuleList([Transition(c=c_z, n=2), Transition(c=c_z, n=2)])
        self.pairformer_stack = nn.ModuleList([
            PairformerBlock(c_s=c_s, c_z=c_z, use_triangle_attn=False, use_triangle_mult=False,
                            attention_pair_bias=dict(n_head=n_head, kq_norm=True))
            for _ in range(n_pairformer_blocks)])

    def forward(self, f, R_L, S_init_I, Z_init_II, C_L, P_LL, D_II_self=None):
        B = R_L.shape[0]
        S_I = S_init_I
        for tr in self.transition_1:
            S_I = S_I + tr(S_I)
        Z_II = Z_init_II.unsqueeze(0).expand(B, -1, -1, -1)
        Z_list = [Z_II]
        if self.use_distogram:
            D_LL = bucketize_scaled_distogram(R_L[..., f["is_ca"], :], min_dist=1, max_dist=30,
                                              sigma_data=self.sigma_data, n_bins=self.n_bins)
            Z_list.append(D_LL)
        if self.use_self:
            if D_II_self is None:
                D_II_self = torch.zeros(Z_II.shape[:-1] + (self.n_bins,), device=Z_II.device, dtype=Z_II.dtype)
            Z_list.append(D_II_self)
        Z_II = torch.cat(Z_list, dim=-1)
        Z_II = self.process_z(Z_II)
        for tr in self.transition_2:
            Z_II = Z_II + tr(Z_II)
        for blk in self.pairformer_stack:
            S_I, Z_II = blk(S_I, Z_II)
        return S_I, Z_II


# --- LocalTokenTransformer (the 18-block DiT) ---
class LocalTokenTransformerRef(nn.Module):
    def __init__(self, c_token=768, c_tokenpair=128, c_s=384, n_block=18, n_head=16,
                 n_local_tokens=8, n_keys=32):
        super().__init__()
        self.n_local_tokens, self.n_keys = n_local_tokens, n_keys
        self.blocks = nn.ModuleList([DiTBlockRef(c_token=c_token, c_s=c_s, c_pair=c_tokenpair, n_head=n_head)
                                    for _ in range(n_block)])

    def forward(self, A_I, S_I, Z_II, f, X_L, full=True):
        indices = create_attention_indices(f, n_attn_keys=self.n_keys,
                                           n_attn_seq_neighbours=self.n_local_tokens,
                                           X_L=X_L, tok_idx=torch.arange(A_I.shape[1], device=A_I.device))
        mask = _indices_to_mask(indices)
        for blk in self.blocks:
            A_I = blk(A_I, S_I, Z_II, valid_mask=mask)
        return A_I


# --- RFD3DiffusionModule (RFD3_diffusion_module.py) ---
class RFD3DiffusionModuleRef(nn.Module):
    def __init__(self, c_atom=128, c_atompair=16, c_token=768, c_s=384, c_z=128, c_t_embed=256,
                 sigma_data=16.0, f_pred="edm", n_attn_seq_neighbours=2, n_attn_keys=128,
                 n_recycle=2, n_head=4, n_enc_blocks=3, n_dec_blocks=3, n_dit_blocks=18,
                 n_split=3, n_pairformer=2, n_head_pf=16):
        super().__init__()
        self.c_atom, self.c_token, self.c_s = c_atom, c_token, c_s
        self.sigma_data, self.f_pred = sigma_data, f_pred
        self.n_attn_seq_neighbours, self.n_attn_keys = n_attn_seq_neighbours, n_attn_keys
        self.n_recycle = n_recycle
        self.process_r = linearNoBias(3, c_atom)
        self.to_r_update = nn.Sequential(RMSNorm(c_atom), linearNoBias(c_atom, 3))
        self.sequence_head = LinearSequenceHeadRef(c_token=c_token)
        self.fourier_embedding = nn.ModuleList([FourierEmbedding(c_t_embed), FourierEmbedding(c_t_embed)])
        self.process_n = nn.ModuleList([
            nn.Sequential(RMSNorm(c_t_embed), linearNoBias(c_t_embed, c_atom)),
            nn.Sequential(RMSNorm(c_t_embed), linearNoBias(c_t_embed, c_s))])
        cab = dict(n_head=n_head, c_model=c_atom, kq_norm=True, dropout=0.0)
        self.downcast_c = Downcast(c_atom=c_atom, c_token=c_s, c_s=None, method="cross_attention", cross_attention_block=cab)
        self.downcast_q = Downcast(c_atom=c_atom, c_token=c_token, c_s=c_s, method="cross_attention", cross_attention_block=cab)
        self.process_a = LinearEmbedWithPoolRef(c_in=3, c_out=c_token)
        self.process_c = nn.Sequential(RMSNorm(c_atom), linearNoBias(c_atom, c_atom))
        self.diffusion_token_encoder = DiffusionTokenEncoderRef(
            c_s=c_s, c_z=c_z, sigma_data=sigma_data, n_pairformer_blocks=n_pairformer, n_head=n_head_pf)
        self.diffusion_transformer = LocalTokenTransformerRef(
            c_token=c_token, c_tokenpair=c_z, c_s=c_s, n_block=n_dit_blocks, n_head=n_head_pf)
        self.encoder = LocalAtomTransformerRef(c_atom=c_atom, c_s=c_atom, c_pair=c_atompair, n_head=n_head, n_block=n_enc_blocks)
        self.decoder = CompactStreamingDecoderRef(c_atom=c_atom, c_atompair=c_atompair, c_token=c_token,
                                                  c_s=c_s, n_head=n_head, n_block=n_dec_blocks, n_split=n_split)
        self.bucketize_fn = functools.partial(bucketize_scaled_distogram, min_dist=1, max_dist=30,
                                              sigma_data=sigma_data, n_bins=65)

    def scale_positions_in(self, X_noisy_L, t):
        if t.ndim == 1:
            t = t[..., None, None]
        elif t.ndim == 2:
            t = t[..., None]
        return X_noisy_L / torch.sqrt(t ** 2 + self.sigma_data ** 2)

    def scale_positions_out(self, R_update_L, X_noisy_L, t):
        if t.ndim == 1:
            t = t[..., None, None]
        elif t.ndim == 2:
            t = t[..., None]
        return (self.sigma_data ** 2 / (self.sigma_data ** 2 + t ** 2)) * X_noisy_L + \
               (self.sigma_data * t / (self.sigma_data ** 2 + t ** 2) ** 0.5) * R_update_L

    def process_time_(self, t_L, i):
        C = self.process_n[i](self.fourier_embedding[i](
            0.25 * torch.log(torch.clamp(t_L, min=1e-20) / self.sigma_data)))
        return C * (t_L > 0).float()[..., None]

    def forward(self, X_noisy_L, t, f, Q_L_init, C_L, P_LL, S_I, Z_II, n_recycle=None, **_):
        tok_idx = f["atom_to_token_map"]
        L = len(tok_idx); I = int(tok_idx.max().item()) + 1
        f = dict(f)
        f["attn_indices"] = create_attention_indices(f, n_attn_keys=self.n_attn_keys,
                                                      n_attn_seq_neighbours=self.n_attn_seq_neighbours, X_L=X_noisy_L)
        atom_mask = _indices_to_mask(f["attn_indices"])
        t_L = t.unsqueeze(-1).expand(-1, L) * (~f["is_motif_atom_with_fixed_coord"]).float().unsqueeze(0)
        t_I = t.unsqueeze(-1).expand(-1, I) * (~f["is_motif_token_with_fully_fixed_coord"]).float().unsqueeze(0)
        R_L_uniform = self.scale_positions_in(X_noisy_L, t)
        R_noisy_L = self.scale_positions_in(X_noisy_L, t_L)
        A_I = self.process_a(R_noisy_L, tok_idx=tok_idx)
        S_I = self.downcast_c(C_L, S_I, tok_idx=tok_idx)
        Q_L = Q_L_init.unsqueeze(0) + self.process_r(R_noisy_L)
        C_L = C_L.unsqueeze(0) + self.process_time_(t_L, i=0)
        S_I = S_I.unsqueeze(0) + self.process_time_(t_I, i=1)
        C_L = C_L + self.process_c(C_L)
        Q_L = self.encoder(Q_L, C_L, P_LL, valid_mask=atom_mask)
        A_I = self.downcast_q(Q_L, A_I=A_I, S_I=S_I, tok_idx=tok_idx)
        recycled = self.forward_with_recycle(
            n_recycle, X_noisy_L=X_noisy_L, R_L_uniform=R_L_uniform, t_L=t_L, f=f, Q_L=Q_L,
            C_L=C_L, P_LL=P_LL, A_I=A_I, S_I=S_I, Z_II=Z_II)
        return {"X_L": recycled["X_L"], "sequence_logits_I": recycled["sequence_logits_I"]}

    def forward_with_recycle(self, n_recycle, **kwargs):
        n_recycle = n_recycle if n_recycle is not None else self.n_recycle
        recycled = {}
        for i in range(n_recycle):
            with torch.no_grad() if i < n_recycle - 1 else _nullctx():
                recycled = self.process_(D_II_self=recycled.get("D_II_self"),
                                         X_L_self=recycled.get("X_L"), **kwargs)
        return recycled

    def process_(self, D_II_self, X_L_self, *, R_L_uniform, X_noisy_L, t_L, f, Q_L, C_L, P_LL, A_I, S_I, Z_II):
        S_I, Z_II = self.diffusion_token_encoder(f, R_L_uniform, S_init_I=S_I, Z_init_II=Z_II,
                                                  C_L=C_L, P_LL=P_LL, D_II_self=D_II_self)
        X_L_ca = X_noisy_L[..., f["is_ca"], :] if X_L_self is None else X_L_self[..., f["is_ca"], :]
        A_I = self.diffusion_transformer(A_I, S_I, Z_II, f, X_L=X_L_ca, full=True)
        atom_mask = _indices_to_mask(f["attn_indices"])
        A_I, Q_L, _ = self.decoder(A_I, S_I, Z_II, Q_L, C_L, P_LL, valid_mask=atom_mask,
                                   tok_idx=f["atom_to_token_map"])
        R_update_L = self.to_r_update(Q_L)
        X_out_L = self.scale_positions_out(R_update_L, X_noisy_L, t_L)
        seq_logits = self.sequence_head(A_I)
        D_II_self = self.bucketize_fn(X_out_L[..., f["is_ca"], :].detach())
        return {"X_L": X_out_L, "D_II_self": D_II_self, "sequence_logits_I": seq_logits}


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _nullctx(): return _NullCtx()


# --- F5 symmetry: rfd3.inference.symmetry.symmetry_utils.apply_symmetry_to_xyz_atomwise
# (partial_diffusion=False case only -- no partial_t in this port's F5 scope). Recenter
# the non-fixed-motif atoms at their mean, then reconstruct every symmetric replica's
# coordinates from the ASU's own (now-centered) atoms via that replica's rigid transform
# -- every step "snaps" the design back to exact symmetry from the ASU alone, rather than
# relying on the network to keep it symmetric. FIXED_ENTITY_ID=-1 (atoms outside any
# symmetric group) is unreachable in this port's current F5 scope (no ligand/motif
# combined with symmetry yet -- see tt_bio.rfd3_featurize's F5 grounding).
_FIXED_ENTITY_ID = -1


def apply_symmetry_atomwise(X_L, sym_transform, sym_transform_id, sym_entity_id, is_sym_asu):
    fixed_mask = sym_entity_id == _FIXED_ENTITY_ID
    non_fixed = ~fixed_mask
    X_L = X_L.clone()
    X_L[:, non_fixed, :] = X_L[:, non_fixed, :] - X_L[:, non_fixed, :].mean(dim=1, keepdim=True)
    out = X_L.clone()
    for entity_id in torch.unique(sym_entity_id).tolist():
        if entity_id == _FIXED_ENTITY_ID:
            continue
        entity_mask = sym_entity_id == entity_id
        asu_mask = is_sym_asu & entity_mask
        if int(asu_mask.sum()) == 0:
            continue
        asu_xyz = X_L[:, asu_mask, :]
        for tid in torch.unique(sym_transform_id[entity_mask]).tolist():
            subunit = entity_mask & (sym_transform_id == tid)
            R, t = sym_transform[str(tid)]
            out[:, subunit, :] = torch.einsum("blc,cd->bld", asu_xyz, R.to(asu_xyz.dtype)) + t.to(asu_xyz.dtype)
    return out


# --- EDM sampler (inference_sampler.py, default solver) ---
class EDMSamplerRef:
    def __init__(self, num_timesteps=10, sigma_data=16.0, s_min=4e-4, s_max=160.0, p=7,
                 gamma_0=0.6, gamma_min=1.0, noise_scale=1.003, step_scale=1.5):
        self.num_timesteps = num_timesteps
        self.sigma_data, self.s_min, self.s_max, self.p = sigma_data, s_min, s_max, p
        self.gamma_0, self.gamma_min = gamma_0, gamma_min
        self.noise_scale, self.step_scale = noise_scale, step_scale

    def noise_schedule(self, device):
        t = torch.linspace(0, 1, self.num_timesteps, device=device)
        return self.sigma_data * (self.s_max ** (1 / self.p) + t * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))) ** self.p

    def sample(self, diffusion_module, D, L, coord_atom_lvl_to_be_noised, f, initializer_outputs,
               draws, is_motif_fixed, sym_feats=None, sym_step_frac=0.9):
        sched = self.noise_schedule(coord_atom_lvl_to_be_noised.device)
        c0 = sched[0]
        X_L = draws.initial(c0, D, L, coord_atom_lvl_to_be_noised, is_motif_fixed)
        traj = []
        # F5: symmetrize the denoised output while c_t > gamma_min_sym (the last
        # ~10% of steps run unconstrained, per upstream's SampleDiffusionWithSymmetry
        # default sym_step_frac=0.9 -- see tt_bio.rfd3_featurize's F5 grounding).
        gamma_min_sym = sched[min(int(len(sched) * sym_step_frac), len(sched) - 1)] if sym_feats else None
        for step, (c_tm1, c_t) in enumerate(zip(sched, sched[1:])):
            gamma = self.gamma_0 if c_t > self.gamma_min else 0.0
            t_hat = c_tm1 * (gamma + 1)
            eps = self.noise_scale * torch.sqrt(torch.square(t_hat) - torch.square(c_tm1)) * draws.step(step, (D, L, 3))
            eps[..., is_motif_fixed, :] = 0
            X_noisy_L = X_L + eps
            outs = diffusion_module(X_noisy_L=X_noisy_L, t=t_hat.tile(D), f=f, n_recycle=None, **initializer_outputs)
            X_denoised_L = outs["X_L"]
            if sym_feats is not None and c_t > gamma_min_sym:
                X_denoised_L = apply_symmetry_atomwise(
                    X_denoised_L, sym_feats["sym_transform"], sym_feats["sym_transform_id"],
                    sym_feats["sym_entity_id"], sym_feats["is_sym_asu"])
            delta_L = (X_noisy_L - X_denoised_L) / t_hat
            d_t = c_t - t_hat
            X_L = X_noisy_L + self.step_scale * d_t * delta_L
            traj.append({"X_noisy_L": X_noisy_L, "X_denoised_L": X_denoised_L, "t_hat": t_hat, "X_L": X_L})
        return X_L, traj


class SharedDraws:
    """Deterministic CPU MT19937 draws shared between device port and host reference."""
    def __init__(self, seed=42):
        self.g = torch.Generator().manual_seed(seed)
        self._steps = []

    def initial(self, c0, D, L, coord, is_motif_fixed):
        noise = c0 * torch.randn((D, L, 3), generator=self.g)
        noise[..., is_motif_fixed, :] = 0
        return noise + coord

    def step(self, i, shape):
        while len(self._steps) <= i:
            self._steps.append(None)
        if self._steps[i] is None:
            self._steps[i] = torch.randn(shape, generator=self.g)
        return self._steps[i]
