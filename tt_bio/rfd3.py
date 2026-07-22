"""RFD3 (RFdiffusion3) TokenInitializer — ttnn on-device port.

Mirrors the upstream `TokenInitializer` (RosettaCommons/foundry models/rfd3,
production): relative_position_encoding (r_max=32, s_max=2) + 2-block no-triangle
Pairformer (AttentionPairBias + z/s Transitions, NO triangle_mult / NO
triangle_attn) + 1D/atom-1D feature embedders, MSA-free. Produces
{Q_L_init, C_L, P_LL, S_I, Z_II}.

Design (per p1 §4 / state §2c.3): the index/one-hot/scatter/gather feature
engineering runs on HOST (pure torch, cheap, index-heavy — no matmul); the heavy
linears / RMSNorm / Transition / pair-bias attention / Downcast cross-attention run
on the TT device via ttnn. TokenInitializer is step-invariant (computed once per
fold), so host<->device round-trips at the feature boundaries are free here.

Weight remap is a trivial prefix-strip: the 118 `model.token_initializer.*` ckpt keys
are canonical and load 1:1 (verified 0 missing / 0 extra vs the faithful reference).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

import ttnn

from .tenstorrent import Module, get_device, CORE_GRID_MAIN, _sdpa_program_config_for_lengths


# --- host-side feature helpers (pure torch; mirror upstream, deps stubbed) ----
def _collapse(x, L):
    return x.reshape((L, x.numel() // L))


def _build_relpos_onehot(f, r_max, s_max):
    """Host: build the [I,I, 2*(2*r_max+3)+(2*s_max+2)+1] one-hot cat for
    RelativePositionEncodingWithIndexRemoval. Returns float32 [I,I,C_in] for the
    device linear."""
    b_samechain = f["asym_id"].unsqueeze(-1) == f["asym_id"].unsqueeze(-2)
    b_same_entity = f["entity_id"].unsqueeze(-1) == f["entity_id"].unsqueeze(-2)
    num_tok_pos_bins = (2 * r_max + 2) + 1
    d_residue = torch.where(
        b_samechain,
        torch.clip(f["residue_index"].unsqueeze(-1) - f["residue_index"].unsqueeze(-2) + r_max, 0, 2 * r_max),
        2 * r_max + 1)
    b_sameresidue = f["residue_index"].unsqueeze(-1) == f["residue_index"].unsqueeze(-2)
    tok_distance = f["token_index"].unsqueeze(-1) - f["token_index"].unsqueeze(-2) + r_max
    d_token = torch.where(
        b_samechain & b_sameresidue,
        torch.clip(tok_distance, 0, 2 * r_max),
        2 * r_max + 1)
    d_chain = torch.where(
        b_same_entity,
        torch.clip(f["sym_id"].unsqueeze(-1) - f["sym_id"].unsqueeze(-2) + s_max, 0, 2 * s_max),
        2 * s_max + 1)
    A_relchain = F.one_hot(d_chain.long(), 2 * s_max + 2)
    unindexing = f["unindexing_pair_mask"]
    d_token[unindexing] = num_tok_pos_bins - 1
    d_residue[unindexing] = num_tok_pos_bins - 1
    A_relpos = F.one_hot(d_residue.long(), num_tok_pos_bins)
    A_reltoken = F.one_hot(d_token, num_tok_pos_bins)
    return torch.cat([A_relpos, A_reltoken, b_same_entity.unsqueeze(-1), A_relchain], dim=-1).to(torch.float32)


def _pairwise_inv_dist(ref_pos, valid_mask):
    """Host: PositionPairDistEmbedder (no-frame) inputs -> (D_LL [L,L,3], inv_dist
    [L,L,1], V_LL [L,L,1]) for the device linears."""
    D_LL = ref_pos.unsqueeze(-2) - ref_pos.unsqueeze(-3)
    norm = torch.linalg.norm(D_LL, dim=-1, keepdim=True) ** 2
    norm = torch.clamp(norm, min=1e-6)
    inv_dist = 1 / (1 + norm)
    return D_LL, inv_dist.unsqueeze(-1), valid_mask.to(torch.float32)


def _sinusoidal_embed(pos, valid_mask, n_freqs=32):
    """Host: SinusoidalDistEmbed inputs -> (sincos [L,L,2*n_freqs], V_LL [L,L,1])."""
    D = pos.unsqueeze(-2) - pos.unsqueeze(-3)
    dist = torch.linalg.norm(D, dim=-1)
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, n_freqs, dtype=torch.float32) / n_freqs)
    angles = dist.unsqueeze(-1) * freq
    sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(torch.float32)
    return sincos, valid_mask.to(torch.float32)


def _build_valid_mask(tok_idx):
    tokens, counts = torch.unique(tok_idx, return_counts=True)
    A = int(counts.max())
    return torch.arange(A, device=tok_idx.device)[None, :] < counts[:, None]


def _scatter_mean_pool(pairwise_atom, tok_idx, I):
    """Host: mean-pool [L,L,c] -> [I,I,c] (pairwise_mean_pool)."""
    onehot = F.one_hot(tok_idx.long(), num_classes=I).to(torch.float32)
    temp = torch.einsum("ia,bacd->bicd", onehot.T, pairwise_atom.unsqueeze(0))
    sums = torch.einsum("cj,bicd->bijd", onehot, temp)
    counts = onehot.sum(0)
    pc = torch.outer(counts, counts).clamp(min=1).unsqueeze(0)
    return (sums / pc.unsqueeze(-1)).squeeze(0)


# --- ttnn helpers ----------------------------------------------------------
def _tt(x, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)


def _pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d.clamp(min=1e-12)) if d > 0 else 0.0


# --- ttnn Transition (RFD3: RMSNorm + silu-gated SwiGLU, keys layer_norm_1/linear_1-3) --
class Transition(Module):
    def __init__(self, state_dict, ckc, c, n, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.norm_w = self.torch_to_tt("layer_norm_1.weight", dtype=self.dtype)
        self.fc1_w = self.torch_to_tt("linear_1.weight", dtype=self.dtype)
        self.fc2_w = self.torch_to_tt("linear_2.weight", dtype=self.dtype)
        self.fc3_w = self.torch_to_tt("linear_3.weight", dtype=self.dtype)

    def __call__(self, x):
        x = ttnn.rms_norm(x, weight=self.norm_w, epsilon=1e-6,
                            compute_kernel_config=self.compute_kernel_config)
        a = ttnn.linear(x, self.fc1_w, activation="silu",
                         compute_kernel_config=self.compute_kernel_config,
                         dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        b = ttnn.linear(x, self.fc2_w, compute_kernel_config=self.compute_kernel_config,
                         dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(x)
        m = ttnn.multiply(a, b)
        ttnn.deallocate(b)
        out = ttnn.linear(m, self.fc3_w, compute_kernel_config=self.compute_kernel_config,
                           dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(m)
        return out


# --- Pairformer attention (AttentionPairBiasPairformerDeepspeed): unconditioned MHA,
# per-head kq_norm, pair bias from RMSNorm(Z)+0, gate, output linear. NO mask (full I×I). -
class PairformerAttention(Module):
    def __init__(self, state_dict, ckc, c_a=384, c_z=128, n_head=16, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.n_head = n_head
        self.head_dim = c_a // n_head  # 24
        self.ln_1_w = self.torch_to_tt("ln_1.weight", dtype=self.dtype)
        self.to_q_w = self.torch_to_tt("to_q.weight", dtype=self.dtype)
        self.to_q_b = self.torch_to_tt("to_q.bias", dtype=self.dtype)
        self.to_k_w = self.torch_to_tt("to_k.weight", dtype=self.dtype)
        self.to_k_ln = self.torch_to_tt("to_k.ln.weight", dtype=self.dtype)
        self.to_v_w = self.torch_to_tt("to_v.weight", dtype=self.dtype)
        self.to_v_ln = self.torch_to_tt("to_v.ln.weight", dtype=self.dtype)
        self.ln_0_w = self.torch_to_tt("ln_0.weight", dtype=self.dtype)
        self.to_b_w = self.torch_to_tt("to_b.weight", dtype=self.dtype)
        self.to_g_w = self.torch_to_tt("to_g.0.weight", dtype=self.dtype)
        self.to_a_w = self.torch_to_tt("to_a.weight", dtype=self.dtype)

    def __call__(self, s, z):
        # s: [1,I,384], z: [1,I,I,128]
        ckc = self.compute_kernel_config
        a = ttnn.rms_norm(s, weight=self.ln_1_w, epsilon=1e-6,
                          compute_kernel_config=ckc)
        q = ttnn.linear(a, self.to_q_w, bias=self.to_q_b,
                          compute_kernel_config=self.compute_kernel_config,
                          dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        k = ttnn.linear(a, self.to_k_w, compute_kernel_config=self.compute_kernel_config,
                          dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        k = ttnn.rms_norm(k, weight=self.to_k_ln, epsilon=1e-6,
                            compute_kernel_config=self.compute_kernel_config)
        v = ttnn.linear(a, self.to_v_w, compute_kernel_config=self.compute_kernel_config,
                           dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        v = ttnn.rms_norm(v, weight=self.to_v_ln, epsilon=1e-6,
                            compute_kernel_config=self.compute_kernel_config)
        B, I = s.shape[0], s.shape[1]
        # split heads: [1,I,384] -> [1,I,16,24] -> [1,16,I,24]
        q = ttnn.reshape(q, (B, I, self.n_head, self.head_dim))
        k = ttnn.reshape(k, (B, I, self.n_head, self.head_dim))
        v = ttnn.reshape(v, (B, I, self.n_head, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        # pair bias: [1,I,I,128] -> rms_norm -> linear -> [1,I,I,16] -> [1,16,I,I]
        z = ttnn.rms_norm(z, weight=self.ln_0_w, epsilon=1e-6,
                           compute_kernel_config=self.compute_kernel_config)
        bias = ttnn.linear(z, self.to_b_w, compute_kernel_config=self.compute_kernel_config,
                            dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        bias = ttnn.permute(bias, (0, 3, 1, 2))  # [1,16,I,I]
        # Manual attention (SDPA forbids head_dim=24 padding); bf16 softmax matches the
        # reference (autocast bf16). softmax over keys (dim=-1).
        kt = ttnn.permute(k, (0, 1, 3, 2))                 # [1,16,24,I]
        sc = ttnn.matmul(q, kt, compute_kernel_config=ckc)  # [1,16,I,I]
        ttnn.deallocate(kt)
        sc = ttnn.typecast(sc, ttnn.float32, memory_config=sc.memory_config())
        sc = ttnn.multiply(sc, self.head_dim ** -0.5)
        bias_f = ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config())
        sc = ttnn.add(sc, bias_f)
        ttnn.deallocate(bias_f)
        attn = ttnn.softmax(sc, dim=-1)                    # fp32 softmax reduction
        attn_bf = ttnn.typecast(attn, self.dtype, memory_config=attn.memory_config())
        ttnn.deallocate(attn)
        o = ttnn.matmul(attn_bf, v, compute_kernel_config=ckc, dtype=self.dtype)  # [1,16,I,24]
        ttnn.deallocate(attn_bf)
        # gate
        g = ttnn.linear(a, self.to_g_w, compute_kernel_config=self.compute_kernel_config,
                         dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        g = ttnn.reshape(g, (B, I, self.n_head, self.head_dim))
        g = ttnn.permute(g, (0, 2, 1, 3))
        g = ttnn.sigmoid(g)
        o = ttnn.multiply(o, g)
        # merge heads: [1,16,I,24] -> [1,I,384]
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (B, I, self.n_head * self.head_dim))
        out = ttnn.linear(o, self.to_a_w, compute_kernel_config=self.compute_kernel_config,
                            dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        return out


class PairformerBlock(Module):
    def __init__(self, state_dict, ckc, c_s=384, c_z=128, n_head=16, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.z_transition = Transition(self.scope("z_transition"), ckc, c_z, n=4, dtype=self.dtype)
        self.s_transition = Transition(self.scope("s_transition"), ckc, c_s, n=4, dtype=self.dtype)
        self.attn = PairformerAttention(self.scope("attention_pair_bias"), ckc, c_s, c_z, n_head, dtype=self.dtype)

    def __call__(self, s, z):
        z = ttnn.add(z, self.z_transition(z))
        s = ttnn.add(s, self.attn(s, z))
        s = ttnn.add(s, self.s_transition(s))
        return s, z


class TokenInitializer(Module):
    """ttnn on-device port of RFD3 TokenInitializer. forward(f) takes the host `f`
    dict (43 keys, as captured) and returns {Q_L_init, C_L, P_LL, S_I, Z_II} on host."""

    C_S, C_Z, C_ATOM, C_ATOMPAIR = 384, 128, 128, 16
    N_PAIRFORMER, N_HEAD = 2, 16
    R_MAX, S_MAX = 32, 2

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        dev = self.device

        # OneD embedder weights (each feature -> linear to its channel). nn.Linear (out,in);
        # torch_to_tt transposes to (in,out) for ttnn.linear.
        def _embedder_weights(prefix):
            return {feat: self.torch_to_tt(f"{prefix}.embedders.{feat}.weight", dtype=self.dtype)
                    for feat in self._feat_keys(prefix)}
        self.w_tok1d = _embedder_weights("token_1d_embedder")
        self.w_atom1d_1 = _embedder_weights("atom_1d_embedder_1")
        self.w_atom1d_2 = _embedder_weights("atom_1d_embedder_2")

        self.downcast_gca = self.scope("downcast_atom.gca")
        # GatedCrossAttention weights (device port; c_query=c_kv=c_s=384, c_model=128, n_head=4, hd=32)
        g = "downcast_atom.gca."
        self.gca_ln_q = self.torch_to_tt(g + "ln_q.weight", dtype=self.dtype)
        self.gca_ln_kv = self.torch_to_tt(g + "ln_kv.weight", dtype=self.dtype)
        self.gca_to_q = self.torch_to_tt(g + "to_q.weight", dtype=self.dtype)
        self.gca_to_k = self.torch_to_tt(g + "to_k.weight", dtype=self.dtype)
        self.gca_to_v = self.torch_to_tt(g + "to_v.weight", dtype=self.dtype)
        self.gca_to_g = self.torch_to_tt(g + "to_g.0.weight", dtype=self.dtype)
        self.gca_k_norm = self.torch_to_tt(g + "k_norm.weight", dtype=self.dtype)
        self.gca_q_norm = self.torch_to_tt(g + "q_norm.weight", dtype=self.dtype)
        self.gca_to_out_w = self.torch_to_tt(g + "to_out.0.weight", dtype=self.dtype)
        self.gca_to_out_b = self.torch_to_tt(g + "to_out.0.bias", dtype=self.dtype)
        self.tr_post_tok = Transition(self.scope("transition_post_token"), ckc, self.C_S, n=2, dtype=self.dtype)
        self.tr_post_atom = Transition(self.scope("transition_post_atom"), ckc, self.C_S, n=2, dtype=self.dtype)
        self.process_s_init_n = self.torch_to_tt("process_s_init.0.weight", dtype=self.dtype)
        self.process_s_init_w = self.torch_to_tt("process_s_init.1.weight", dtype=self.dtype)
        self.to_z_init_i = self.torch_to_tt("to_z_init_i.weight", dtype=self.dtype)
        self.to_z_init_j = self.torch_to_tt("to_z_init_j.weight", dtype=self.dtype)
        self.relpos_lin = self.torch_to_tt("relative_position_encoding.linear.weight", dtype=self.dtype)
        self.relpos2_lin = self.torch_to_tt("relative_position_encoding2.linear.weight", dtype=self.dtype)
        self.proc_token_bonds = self.torch_to_tt("process_token_bonds.weight", dtype=self.dtype)
        self.refpos_tok_invd = self.torch_to_tt("ref_pos_embedder_tok.process_inverse_dist.weight", dtype=self.dtype)
        self.refpos_tok_vm = self.torch_to_tt("ref_pos_embedder_tok.process_valid_mask.weight", dtype=self.dtype)
        self.proc_z_init_n = self.torch_to_tt("process_z_init.0.weight", dtype=self.dtype)
        self.proc_z_init_w = self.torch_to_tt("process_z_init.1.weight", dtype=self.dtype)
        self.tr1_0 = Transition(self.scope("transition_1.0"), ckc, self.C_Z, n=2, dtype=self.dtype)
        self.tr1_1 = Transition(self.scope("transition_1.1"), ckc, self.C_Z, n=2, dtype=self.dtype)
        self.blocks = [PairformerBlock(self.scope(f"transformer_stack.{i}"), ckc,
                                       self.C_S, self.C_Z, self.N_HEAD, dtype=self.dtype)
                        for i in range(self.N_PAIRFORMER)]
        self.proc_s_trunk_n = self.torch_to_tt("process_s_trunk.0.weight", dtype=self.dtype)
        self.proc_s_trunk_w = self.torch_to_tt("process_s_trunk.1.weight", dtype=self.dtype)
        self.proc_single_l_w = self.torch_to_tt("process_single_l.1.weight", dtype=self.dtype)
        self.proc_single_m_w = self.torch_to_tt("process_single_m.1.weight", dtype=self.dtype)
        self.proc_z_n = self.torch_to_tt("process_z.0.weight", dtype=self.dtype)
        self.proc_z_w = self.torch_to_tt("process_z.1.weight", dtype=self.dtype)
        self.motif_pos_proj = self.torch_to_tt("motif_pos_embedder.output_proj.weight", dtype=self.dtype)
        self.motif_pos_vm = self.torch_to_tt("motif_pos_embedder.process_valid_mask.weight", dtype=self.dtype)
        self.refpos_invd = self.torch_to_tt("ref_pos_embedder.process_inverse_dist.weight", dtype=self.dtype)
        self.refpos_vm = self.torch_to_tt("ref_pos_embedder.process_valid_mask.weight", dtype=self.dtype)
        self.pair_mlp_w = [self.torch_to_tt(f"pair_mlp.{i}.weight", dtype=self.dtype) for i in (1, 3, 5)]
        self.proc_pll_w = self.torch_to_tt("process_pll.weight", dtype=self.dtype)
        self.project_pll_w = self.torch_to_tt("project_pll.weight", dtype=self.dtype)

    @staticmethod
    def _feat_keys(prefix):
        if prefix == "token_1d_embedder":
            return ["ref_motif_token_type", "restype", "ref_plddt", "is_non_loopy"]
        return ["ref_atom_name_chars", "ref_element", "ref_charge", "ref_mask",
                "ref_is_motif_atom_with_fixed_coord", "ref_is_motif_atom_unindexed",
                "has_zero_occupancy", "ref_pos", "ref_atomwise_rasa", "active_donor",
                "active_acceptor", "is_atom_level_hotspot"]

    def _embed1d(self, f, weights, collapse_len, keys):
        """Sum of per-feature device linears on collapsed features -> [collapse_len, C]."""
        acc = None
        for feat in keys:
            x = _collapse(f[feat].float(), collapse_len)
            xt = _tt(x, self.device, self.dtype)
            y = ttnn.linear(xt, weights[feat], compute_kernel_config=self.compute_kernel_config,
                              dtype=self.dtype, core_grid=CORE_GRID_MAIN)
            acc = y if acc is None else ttnn.add(acc, y)
        return acc

    # --- host-side GatedCrossAttention (Downcast cross-attn; device port deferred to p4) ---
    def _host_gca(self, s_h, ql_h, vm):
        """Mirror Downcast(cross_attention) + GatedCrossAttention(kq_norm=True) on host.
        s_h [I, C_S], ql_h [L, C_S], vm [I, A]. Returns the per-token update [I, C_S]."""
        W = self.downcast_gca
        c_s, c_model, n_head = self.C_S, 128, 4
        I, A = s_h.shape[0], vm.shape[1]
        hd = c_model // n_head
        # ungroup atoms: Q_L [L,384] -> Q_IA [1, I, A, 384]
        flat_idx = vm.flatten().nonzero(as_tuple=False).squeeze(1)
        idx = flat_idx.view(1, -1, 1).expand(1, -1, c_s)
        Q_IA = torch.zeros(1, I * A, c_s, dtype=ql_h.dtype)
        Q_IA = Q_IA.scatter(1, idx, ql_h.unsqueeze(0)).reshape(1, I, A, c_s)
        q = s_h.unsqueeze(0).unsqueeze(2)          # [1, I, 1, C_S]
        kv = Q_IA                                   # [1, I, A, C_S]
        attn_mask = vm.unsqueeze(1)                 # [I, 1, A]
        q = F.rms_norm(q, (c_s,), W["ln_q.weight"], 1e-6)
        kv = F.rms_norm(kv, (c_s,), W["ln_kv.weight"], 1e-6)
        qq = F.linear(q, W["to_q.weight"]); kk = F.linear(kv, W["to_k.weight"]); vv = F.linear(kv, W["to_v.weight"])
        gg = torch.sigmoid(F.linear(q, W["to_g.0.weight"]))
        kk = F.rms_norm(kk, (c_model,), W["k_norm.weight"], 1e-6)
        qq = F.rms_norm(qq, (c_model,), W["q_norm.weight"], 1e-6)

        def heads(t):
            b, t_, n, _ = t.shape
            return t.reshape(b, t_, n, n_head, hd).permute(0, 3, 1, 2, 4)  # [b,h,t,n,c]

        qh, kh, vh, gh = heads(qq), heads(kk), heads(vv), heads(gg)
        scale = 1.0 / math.sqrt(hd)
        attn = torch.einsum("bhtqc,bhtkc->bhtqk", qh, kh) * scale   # [1,4,I,1,A]
        attn = attn.masked_fill(~attn_mask[None, None], float("-inf"))
        invalid = ~torch.any(attn_mask, dim=-1)                    # [I]
        if invalid.any():
            attn[:, :, invalid, :, :] = 0.0
        attn = F.softmax(attn, dim=-1)
        o = torch.einsum("bhtqk,bhtkd->bhtqd", attn, vh) * gh       # [1,4,I,1,hd]
        o = o.permute(0, 2, 3, 1, 4).reshape(1, I, 1, c_model)       # [1,I,1,128]
        o = F.linear(o, W["to_out.0.weight"], W["to_out.0.bias"])  # [1,I,1,C_S]
        return o.squeeze(0).squeeze(1)                            # [I, C_S]

    def _device_gca(self, s_h, ql_h, vm):
        """On-device GatedCrossAttention (Downcast). s_h [I, C_S], ql_h [L, C_S],
        vm [I, A]. Returns the per-token update [I, C_S] on host. head_dim=32 (tile-aligned)
        so manual matmul-softmax attention is clean (same recipe as PairformerAttention)."""
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        c_s, c_model, n_head = self.C_S, 128, 4
        hd = c_model // n_head  # 32
        I, A = s_h.shape[0], vm.shape[1]
        # ungroup atoms on host: Q_L [L,384] -> Q_IA [1, I, A, 384]
        flat_idx = vm.flatten().nonzero(as_tuple=False).squeeze(1)
        idx = flat_idx.view(1, -1, 1).expand(1, -1, c_s)
        Q_IA = torch.zeros(1, I * A, c_s, dtype=ql_h.dtype).scatter(1, idx, ql_h.unsqueeze(0))
        Q_IA = Q_IA.reshape(1, I, A, c_s)
        q = _tt(s_h.unsqueeze(0), dev, dt)
        kv = _tt(Q_IA, dev, dt)
        q = ttnn.rms_norm(q, weight=self.gca_ln_q, epsilon=1e-6, compute_kernel_config=ckc)
        kv = ttnn.rms_norm(kv, weight=self.gca_ln_kv, epsilon=1e-6, compute_kernel_config=ckc)
        qq = ttnn.linear(q, self.gca_to_q, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        kk = ttnn.linear(kv, self.gca_to_k, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        vv = ttnn.linear(kv, self.gca_to_v, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        gg = ttnn.linear(q, self.gca_to_g, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        qq = ttnn.rms_norm(qq, weight=self.gca_q_norm, epsilon=1e-6, compute_kernel_config=ckc)
        kk = ttnn.rms_norm(kk, weight=self.gca_k_norm, epsilon=1e-6, compute_kernel_config=ckc)
        # batch over tokens: q [1,I,128]->[I,4,1,32]; k/v [1,I,A,128]->[I,4,A,32]
        qq = ttnn.permute(ttnn.reshape(qq, (1, I, n_head, hd)), (0, 2, 1, 3))            # [1,4,I,32]
        qq = ttnn.reshape(qq, (I, n_head, 1, hd))                                    # [I,4,1,32]
        gg = ttnn.reshape(ttnn.permute(ttnn.reshape(gg, (1, I, n_head, hd)), (0, 2, 1, 3)), (I, n_head, 1, hd))
        kk = ttnn.permute(ttnn.reshape(kk, (1, I, A, n_head, hd)), (0, 3, 1, 2, 4))   # [1,4,I,A,32]
        vv = ttnn.permute(ttnn.reshape(vv, (1, I, A, n_head, hd)), (0, 3, 1, 2, 4))
        kk = ttnn.reshape(kk, (I, n_head, A, hd))                                    # [I,4,A,32]
        vv = ttnn.reshape(vv, (I, n_head, A, hd))
        kt = ttnn.permute(kk, (0, 1, 3, 2))                                        # [I,4,32,A]
        sc = ttnn.matmul(qq, kt, compute_kernel_config=ckc)                          # [I,4,1,A]
        ttnn.deallocate(qq); ttnn.deallocate(kt)
        sc = ttnn.typecast(sc, ttnn.float32, memory_config=sc.memory_config())
        sc = ttnn.multiply(sc, hd ** -0.5)
        mask = torch.where(vm, 0.0, -1e4).to(torch.float32).unsqueeze(1).unsqueeze(1)  # [I,1,1,A]
        mask = _tt(mask, dev, ttnn.float32)
        sc = ttnn.add(sc, mask)
        ttnn.deallocate(mask)
        attn = ttnn.softmax(sc, dim=-1)
        attn = ttnn.typecast(attn, dt, memory_config=attn.memory_config())
        o = ttnn.matmul(attn, vv, compute_kernel_config=ckc, dtype=dt)                       # [I,4,1,32]
        ttnn.deallocate(attn); ttnn.deallocate(vv)
        o = ttnn.multiply(o, ttnn.sigmoid(gg))
        ttnn.deallocate(gg)
        o = ttnn.reshape(o, (1, I, c_model))
        o = ttnn.linear(o, self.gca_to_out_w, bias=self.gca_to_out_b,
                          compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        return ttnn.to_torch(o).float().squeeze(0)                                  # [I, C_S]

    def __call__(self, f):
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        tok_idx = f["atom_to_token_map"].long()
        L = len(tok_idx)
        f = dict(f)  # shallow copy (we mutate ref_atom_name_chars)
        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)
        I = len(f["restype"])

        # ===== init_tokens =====
        # token_1d embedder (device linears, summed)
        s = self._embed1d(f, self.w_tok1d, I, list(self.w_tok1d.keys()))
        s = ttnn.add(s, self.tr_post_tok(s))
        # atom_1d embedder_1 (device) -> Q_L [L, C_S]
        ql = self._embed1d(f, self.w_atom1d_1, L, list(self.w_atom1d_1.keys()))
        # downcast_atom (host GCA this pass): S_I += gca(S_I, Q_L, tok_idx)
        s_h = ttnn.to_torch(s).float().squeeze(0)            # [I, C_S]
        ql_h = ttnn.to_torch(ql).float().squeeze(0)         # [L, C_S]
        vm = _build_valid_mask(tok_idx)                     # [I, A]
        s_h = s_h + self._host_gca(s_h, ql_h, vm)            # [I, C_S]  (device GCA port WIP -> p5)
        s = _tt(s_h.unsqueeze(0), dev, dt)                 # back to device [1,I,C_S]
        s = ttnn.add(s, self.tr_post_atom(s))
        # process_s_init: RMSNorm + linear
        s = ttnn.rms_norm(s, weight=self.process_s_init_n, epsilon=1e-6, compute_kernel_config=ckc)
        s = ttnn.linear(s, self.process_s_init_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        s_h = ttnn.to_torch(s).float().squeeze(0)           # [I, C_S] host (for outer-sum + later gathers)
        # Z_init = outer(to_z_init_i(S), to_z_init_j(S)) [1,I,I,C_Z]
        zi = ttnn.linear(s, self.to_z_init_i, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        zj = ttnn.linear(s, self.to_z_init_j, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        zi = ttnn.reshape(zi, (1, I, 1, self.C_Z))
        zj = ttnn.reshape(zj, (1, 1, I, self.C_Z))
        z = ttnn.add(zi, zj)                              # [1,I,I,128]
        # + relative_position_encoding (host one-hot -> device linear)
        rph = _tt(_build_relpos_onehot(f, self.R_MAX, self.S_MAX).unsqueeze(0), dev, dt)
        z = ttnn.add(z, ttnn.linear(rph, self.relpos_lin, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN))
        # + process_token_bonds
        tb = _tt(f["token_bonds"].unsqueeze(-1).float().unsqueeze(0), dev, dt)  # [1,I,I,1]
        z = ttnn.add(z, ttnn.linear(tb, self.proc_token_bonds, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN))
        # + ref_pos_embedder_tok (no-frame; token-level, I×I)
        is_ca = f["is_ca"]
        rpos_ca = f["ref_pos"][is_ca].float()              # [I, 3]
        tid = f["ref_space_uid"][is_ca].long()            # [I]
        vm_tok = (tid.unsqueeze(-1) == tid.unsqueeze(-2)).unsqueeze(-1).float()  # [I,I,1]
        invd = 1.0 / (1.0 + (rpos_ca.unsqueeze(-2) - rpos_ca.unsqueeze(-3)).pow(2).sum(-1, keepdim=True))
        invd = _tt(invd.unsqueeze(0), dev, dt); vm_tok = _tt(vm_tok.unsqueeze(0), dev, dt)
        rp = ttnn.multiply(ttnn.linear(invd, self.refpos_tok_invd, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vm_tok)
        rp = ttnn.add(rp, ttnn.multiply(ttnn.linear(vm_tok, self.refpos_tok_vm, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vm_tok))
        z = ttnn.add(z, rp)
        # 2 Pairformer blocks
        for blk in self.blocks:
            s_dev, z = blk(_tt(s_h.unsqueeze(0), dev, dt), z)
            s_h = ttnn.to_torch(s_dev).float().squeeze(0)
        # cat([Z, relpos2]) -> process_z_init (RMSNorm(2*C_Z) + linear) -> 2x transition_1
        rph2 = _tt(_build_relpos_onehot(f, self.R_MAX, self.S_MAX).unsqueeze(0), dev, dt)
        z2 = ttnn.linear(rph2, self.relpos2_lin, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        z = ttnn.concat([z, z2], dim=-1)               # [1,I,I,256]
        z = ttnn.rms_norm(z, weight=self.proc_z_init_n, epsilon=1e-6, compute_kernel_config=ckc)
        z = ttnn.linear(z, self.proc_z_init_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        z = ttnn.add(z, self.tr1_0(z))
        z = ttnn.add(z, self.tr1_1(z))
        S_init_I = s_h                                          # [I, C_S] host
        Z_init_II = ttnn.to_torch(z).float().squeeze(0)        # [I, I, C_Z] host
        return self._init_atoms(f, S_init_I, Z_init_II, tok_idx, L, I)

    def _init_atoms(self, f, S_init_I, Z_init_II, tok_idx, L, I):
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        # Q_L_init = atom_1d_embedder_2 (device linears) [L, C_ATOM]
        ql_init = self._embed1d(f, self.w_atom1d_2, L, list(self.w_atom1d_2.keys()))
        # process_s_trunk(S_init_I): RMSNorm + linear -> [I, C_ATOM]; gather to atoms via tok_idx
        s_tr = _tt(S_init_I.unsqueeze(0), dev, dt)
        s_tr = ttnn.rms_norm(s_tr, weight=self.proc_s_trunk_n, epsilon=1e-6, compute_kernel_config=ckc)
        s_tr = ttnn.linear(s_tr, self.proc_s_trunk_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        s_tr_h = ttnn.to_torch(s_tr).float().squeeze(0)        # [I, C_ATOM]
        c_l_h = s_tr_h[tok_idx]                                # [L, C_ATOM] (gather)
        c_l = ttnn.add(ql_init, _tt(c_l_h.unsqueeze(0), dev, dt))  # C_L [1,L,C_ATOM]

        # ---- P_LL [L, L, C_ATOMPAIR=16] ----
        # motif_pos_embedder (SinusoidalDistEmbed): host sincos -> device output_proj + valid_mask linears
        mp = f["motif_pos"].float()
        vm_mp = (f["is_motif_atom_with_fixed_coord"].unsqueeze(-1) & f["is_motif_atom_with_fixed_coord"].unsqueeze(-2)).unsqueeze(-1).float()
        sc, vsc = _sinusoidal_embed(mp, vm_mp)                  # [L,L,64], [L,L,1]
        sc = _tt(sc.unsqueeze(0), dev, dt); vsc = _tt(vsc.unsqueeze(0), dev, dt)
        p = ttnn.multiply(ttnn.linear(sc, self.motif_pos_proj, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vsc)
        p = ttnn.add(p, ttnn.multiply(ttnn.linear(vsc, self.motif_pos_vm, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vsc))
        # ref_pos_embedder (no-frame): host inv_dist -> device linears
        rp = f["ref_pos"].float()
        same_tok = (f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)).unsqueeze(-1).float()
        has_seq = (f["is_motif_atom_with_fixed_seq"].unsqueeze(-1) & f["is_motif_atom_with_fixed_seq"].unsqueeze(-2)).unsqueeze(-1).float()
        vm_rp = same_tok * has_seq
        D = rp.unsqueeze(-2) - rp.unsqueeze(-3)
        invd = 1.0 / (1.0 + D.pow(2).sum(-1, keepdim=True).clamp(min=1e-6))
        invd = _tt(invd.unsqueeze(0), dev, dt); vm_rp = _tt(vm_rp.unsqueeze(0), dev, dt)
        p = ttnn.add(p, ttnn.multiply(ttnn.linear(invd, self.refpos_invd, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vm_rp))
        p = ttnn.add(p, ttnn.multiply(ttnn.linear(vm_rp, self.refpos_vm, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vm_rp))
        # process_single_l/m (ReLU + linear on C_L)
        c_l_h = ttnn.to_torch(c_l).float().squeeze(0)        # [L, C_ATOM]
        c_l_dev = _tt(c_l_h.unsqueeze(0), dev, dt)
        sl = ttnn.relu(c_l_dev)
        sl = ttnn.linear(sl, self.proc_single_l_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)  # [1,L,16]
        sm = ttnn.relu(_tt(c_l_h.unsqueeze(0), dev, dt))
        sm = ttnn.linear(sm, self.proc_single_m_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        p = ttnn.add(p, ttnn.unsqueeze(sl, -2))             # [1,L,1,16] + [1,L,L,16] -> [1,L,L,16]
        p = ttnn.add(p, ttnn.unsqueeze(sm, -3))
        # process_z(Z_init_II): RMSNorm + linear -> [I,I,16]; gather to atoms [L,L,16]
        z_dev = _tt(Z_init_II.unsqueeze(0), dev, dt)
        z_dev = ttnn.rms_norm(z_dev, weight=self.proc_z_n, epsilon=1e-6, compute_kernel_config=ckc)
        pz = ttnn.linear(z_dev, self.proc_z_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)  # [1,I,I,16]
        pz_h = ttnn.to_torch(pz).float().squeeze(0)          # [I,I,16]
        pz_h = pz_h[tok_idx][:, tok_idx, :]                   # [L,L,16] (gather both axes)
        p = ttnn.add(p, _tt(pz_h.unsqueeze(0), dev, dt))
        # pair_mlp (ReLU + linear x3) residual
        m = p
        for w in self.pair_mlp_w:
            m = ttnn.relu(m)
            m = ttnn.linear(m, w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        p = ttnn.add(p, m)
        p_h = ttnn.to_torch(p).float().squeeze(0)            # [L,L,16]
        # pooled = scatter_mean_pool(process_pll(P_LL)) -> project_pll -> add to Z
        pll = ttnn.linear(p, self.proc_pll_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        pll_h = ttnn.to_torch(pll).float().squeeze(0)       # [L,L,16]
        pooled = _scatter_mean_pool(pll_h, tok_idx, I)        # [I,I,16]
        pooled = _tt(pooled.unsqueeze(0), dev, dt)
        zupd = ttnn.linear(pooled, self.project_pll_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)  # [1,I,I,128]
        z_dev = _tt(Z_init_II.unsqueeze(0), dev, dt)
        z_dev = ttnn.add(z_dev, zupd)
        Z_II = ttnn.to_torch(z_dev).float().squeeze(0)       # [I,I,128]
        Q_L_init = ttnn.to_torch(ql_init).float().squeeze(0)  # [L,128]
        C_L = ttnn.to_torch(c_l).float().squeeze(0)          # [L,128]
        P_LL = p_h                                          # [L,L,16]
        return {"Q_L_init": Q_L_init, "C_L": C_L, "P_LL": P_LL, "S_I": S_init_I, "Z_II": Z_II}


def build_token_initializer(state_dict, compute_kernel_config=None, dtype=None):
    """Construct the ttnn TokenInitializer from a flat `token_initializer.*` state dict
    (prefix already stripped) + a compute_kernel_config. Mirrors the construction order
    used by the torch reference so weight keys line up 1:1."""
    if compute_kernel_config is None:
        dev = get_device()
        kernel_cls = (
            ttnn.types.WormholeComputeKernelConfig
            if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig
        )
        compute_kernel_config = kernel_cls(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
    return TokenInitializer(state_dict, compute_kernel_config, dtype=dtype)
