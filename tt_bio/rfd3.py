"""RFD3 (RFdiffusion3) ttnn component ports.

Includes the TokenInitializer, dense-mask LocalAtomTransformer encoder,
CompactStreamingDecoder (device Upcast/Downcast cross-attention), and
LinearSequenceHead. The atom attention mask is mathematically equivalent to
upstream's gather-sparse path.

Design (per p1 §4 / state §2c.3): the index/one-hot/scatter/gather feature
engineering runs on HOST (pure torch, cheap, index-heavy — no matmul); the heavy
linears / RMSNorm / Transition / pair-bias attention / Downcast cross-attention run
on the TT device via ttnn. Decoder atom grouping uses device gathers, keeping the
three-block decoder resident.

Weight remap is a trivial prefix-strip: the 118 `model.token_initializer.*` ckpt keys
are canonical and load 1:1 (verified 0 missing / 0 extra vs the faithful reference).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import ttnn

from .tenstorrent import Module, get_device, CORE_GRID_MAIN


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

    # --- host-side GatedCrossAttention reference (kept for parity isolation) ---
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
        # Batch over tokens. Keep token before head until after flattening the token
        # batch; moving head before token here scrambles both axes.
        qq = ttnn.permute(ttnn.reshape(qq, (1, I, 1, n_head, hd)), (0, 1, 3, 2, 4))
        qq = ttnn.reshape(qq, (I, n_head, 1, hd))                                    # [I,4,1,32]
        gg = ttnn.permute(ttnn.reshape(gg, (1, I, 1, n_head, hd)), (0, 1, 3, 2, 4))
        gg = ttnn.reshape(gg, (I, n_head, 1, hd))
        kk = ttnn.permute(ttnn.reshape(kk, (1, I, A, n_head, hd)), (0, 1, 3, 2, 4))
        vv = ttnn.permute(ttnn.reshape(vv, (1, I, A, n_head, hd)), (0, 1, 3, 2, 4))
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
        s_h = s_h + self._device_gca(s_h, ql_h, vm)          # [I, C_S]
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


def _dense_attention_mask(indices):
    """Convert [B,L,K] neighbour indices to the equivalent dense additive mask."""
    indices = indices.long()
    batch, length, _ = indices.shape
    keep = torch.zeros(batch, length, length, dtype=torch.bool)
    keep.scatter_(2, indices.cpu(), True)
    return torch.where(keep, 0.0, -1e4).unsqueeze(1)


class GatedCrossAttention(Module):
    """RFD3 GatedCrossAttention on device; token grouping stays host-side."""

    def __init__(
        self,
        state_dict,
        ckc,
        c_query,
        c_kv,
        c_model=128,
        n_head=4,
        dtype=None,
    ):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.c_query = c_query
        self.c_kv = c_kv
        self.c_model = c_model
        self.n_head = n_head
        self.head_dim = c_model // n_head
        self.ln_q = self.torch_to_tt("ln_q.weight", dtype=self.dtype)
        self.ln_kv = self.torch_to_tt("ln_kv.weight", dtype=self.dtype)
        self.to_q = self.torch_to_tt("to_q.weight", dtype=self.dtype)
        self.to_k = self.torch_to_tt("to_k.weight", dtype=self.dtype)
        self.to_v = self.torch_to_tt("to_v.weight", dtype=self.dtype)
        self.to_g = self.torch_to_tt("to_g.0.weight", dtype=self.dtype)
        self.k_norm = self.torch_to_tt("k_norm.weight", dtype=self.dtype)
        self.q_norm = self.torch_to_tt("q_norm.weight", dtype=self.dtype)
        self.to_out_w = self.torch_to_tt("to_out.0.weight", dtype=self.dtype)
        self.to_out_b = self.torch_to_tt("to_out.0.bias", dtype=self.dtype)

    def run_device(self, q, kv, attn_mask=None):
        """q [B,T,Q,Cq], kv [B,T,K,Ckv]; return device [B,T,Q,Cq]."""
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        batch, tokens, n_query, _ = q.shape
        n_key = kv.shape[2]
        q = ttnn.rms_norm(q, weight=self.ln_q, epsilon=1e-6, compute_kernel_config=ckc)
        kv = ttnn.rms_norm(kv, weight=self.ln_kv, epsilon=1e-6, compute_kernel_config=ckc)
        qq = ttnn.linear(q, self.to_q, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        kk = ttnn.linear(kv, self.to_k, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        vv = ttnn.linear(kv, self.to_v, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        gg = ttnn.linear(q, self.to_g, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        qq = ttnn.rms_norm(qq, weight=self.q_norm, epsilon=1e-6, compute_kernel_config=ckc)
        kk = ttnn.rms_norm(kk, weight=self.k_norm, epsilon=1e-6, compute_kernel_config=ckc)

        def split(x, count):
            x = ttnn.reshape(
                x, (batch, tokens, count, self.n_head, self.head_dim)
            )
            x = ttnn.permute(x, (0, 1, 3, 2, 4))
            return ttnn.reshape(
                x, (batch * tokens, self.n_head, count, self.head_dim)
            )

        qq = split(qq, n_query)
        kk = split(kk, n_key)
        vv = split(vv, n_key)
        gg = split(gg, n_query)
        scores = ttnn.matmul(
            qq, ttnn.permute(kk, (0, 1, 3, 2)), compute_kernel_config=ckc
        )
        scores = ttnn.multiply(scores, self.head_dim**-0.5)
        if attn_mask is not None:
            mask = attn_mask
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            mask = torch.where(mask, 0.0, -1e4).to(torch.float32)
            mask = mask.expand(batch, -1, -1, -1).reshape(
                batch * tokens, 1, n_query, n_key
            )
            scores = ttnn.add(scores, _tt(mask, dev, dt))
        attention = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attention, vv, compute_kernel_config=ckc, dtype=dt)
        out = ttnn.multiply(out, ttnn.sigmoid(gg))
        out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(
            out, (batch, tokens, n_query, self.c_model)
        )
        out = ttnn.linear(
            out,
            self.to_out_w,
            bias=self.to_out_b,
            compute_kernel_config=ckc,
            dtype=dt,
            core_grid=CORE_GRID_MAIN,
        )
        return out

    def __call__(self, q_host, kv_host, attn_mask=None):
        """Host-boundary wrapper used by isolated component tests."""
        q = _tt(q_host, self.device, self.dtype)
        kv = _tt(kv_host, self.device, self.dtype)
        return ttnn.to_torch(self.run_device(q, kv, attn_mask)).float()


class RFD3AtomBlock(Module):
    """One dense-mask RFD3 structure-local transformer block.

    Parameterized by dims so the same block serves the atom encoder/decoder
    (c_a=128, c_s=128, c_pair=16, n_head=4, head_dim=32) and the 18-block token
    DiT (c_a=768, c_s=384, c_pair=128, n_head=16, head_dim=48). Weight shapes
    encode c_a/c_s/c_pair; n_head is the only structural knob that is not."""

    def __init__(self, state_dict, ckc, c_a=128, c_s=128, c_pair=16, n_head=4, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        dt = self.dtype
        self.n_head = n_head
        self.head_dim = c_a // n_head
        a = "attention_pair_bias."
        self.a_ln_s = self.torch_to_tt(a + "ada_ln_1.ln_s.weight", dtype=dt)
        self.a_gain_w = self.torch_to_tt(a + "ada_ln_1.to_gain.0.weight", dtype=dt)
        self.a_gain_b = self.torch_to_tt(a + "ada_ln_1.to_gain.0.bias", dtype=dt)
        self.a_bias_w = self.torch_to_tt(a + "ada_ln_1.to_bias.weight", dtype=dt)
        self.q_w = self.torch_to_tt(a + "to_q.weight", dtype=dt)
        self.k_w = self.torch_to_tt(a + "to_k.weight", dtype=dt)
        self.v_w = self.torch_to_tt(a + "to_v.weight", dtype=dt)
        self.b_w = self.torch_to_tt(a + "to_b.weight", dtype=dt)
        self.g_w = self.torch_to_tt(a + "to_g.0.weight", dtype=dt)
        self.q_ln = self.torch_to_tt(a + "ln_q.weight", dtype=dt)
        self.k_ln = self.torch_to_tt(a + "ln_k.weight", dtype=dt)
        self.o_w = self.torch_to_tt(a + "to_o.weight", dtype=dt)
        self.a_out_w = self.torch_to_tt(a + "linear_output_project.0.weight", dtype=dt)
        self.a_out_b = self.torch_to_tt(a + "linear_output_project.0.bias", dtype=dt)

        t = "transition_block."
        self.t_ln_s = self.torch_to_tt(t + "ada_ln.ln_s.weight", dtype=dt)
        self.t_gain_w = self.torch_to_tt(t + "ada_ln.to_gain.0.weight", dtype=dt)
        self.t_gain_b = self.torch_to_tt(t + "ada_ln.to_gain.0.bias", dtype=dt)
        self.t_bias_w = self.torch_to_tt(t + "ada_ln.to_bias.weight", dtype=dt)
        self.t_fc1 = self.torch_to_tt(t + "linear_1.weight", dtype=dt)
        self.t_fc2 = self.torch_to_tt(t + "linear_2.weight", dtype=dt)
        self.t_fc3 = self.torch_to_tt(t + "linear_3.weight", dtype=dt)
        self.t_out_w = self.torch_to_tt(t + "linear_output_project.0.weight", dtype=dt)
        self.t_out_b = self.torch_to_tt(t + "linear_output_project.0.bias", dtype=dt)

    def _adaln(self, a, s, ln_s, gain_w, gain_b, bias_w):
        ckc, dt = self.compute_kernel_config, self.dtype
        a = ttnn.rms_norm(a, epsilon=1e-6, compute_kernel_config=ckc)
        s = ttnn.rms_norm(s, weight=ln_s, epsilon=1e-6, compute_kernel_config=ckc)
        gain = ttnn.linear(
            s, gain_w, bias=gain_b, compute_kernel_config=ckc,
            dtype=dt, core_grid=CORE_GRID_MAIN,
        )
        bias = ttnn.linear(
            s, bias_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN
        )
        return ttnn.add(ttnn.multiply(a, ttnn.sigmoid(gain)), bias)

    def __call__(self, q, c, p, additive_mask):
        ckc, dt = self.compute_kernel_config, self.dtype
        batch, length = q.shape[0], q.shape[1]
        norm = self._adaln(
            q, c, self.a_ln_s, self.a_gain_w, self.a_gain_b, self.a_bias_w
        )
        qq = ttnn.linear(norm, self.q_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        kk = ttnn.linear(norm, self.k_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        vv = ttnn.linear(norm, self.v_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        gg = ttnn.linear(norm, self.g_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        qq = ttnn.rms_norm(qq, weight=self.q_ln, epsilon=1e-6, compute_kernel_config=ckc)
        kk = ttnn.rms_norm(kk, weight=self.k_ln, epsilon=1e-6, compute_kernel_config=ckc)

        def heads(x):
            x = ttnn.reshape(
                x, (batch, length, self.n_head, self.head_dim)
            )
            return ttnn.permute(x, (0, 2, 1, 3))

        qq, kk, vv, gg = map(heads, (qq, kk, vv, gg))
        pair_bias = ttnn.linear(
            p, self.b_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN
        )
        pair_bias = ttnn.permute(pair_bias, (0, 3, 1, 2))
        bias = ttnn.add(pair_bias, additive_mask)
        # The reference's softmax reduction is fp32 even under bf16 autocast.
        # Keep q/k/v and output storage bf16, but match that reduction boundary.
        scores = ttnn.matmul(
            qq, ttnn.permute(kk, (0, 1, 3, 2)), compute_kernel_config=ckc
        )
        scores = ttnn.typecast(
            scores, ttnn.float32, memory_config=scores.memory_config()
        )
        scores = ttnn.multiply(scores, self.head_dim**-0.5)
        bias_f = ttnn.typecast(
            bias, ttnn.float32, memory_config=bias.memory_config()
        )
        scores = ttnn.add(scores, bias_f)
        attention = ttnn.softmax(scores, dim=-1)
        attention = ttnn.typecast(
            attention, dt, memory_config=attention.memory_config()
        )
        out = ttnn.matmul(attention, vv, compute_kernel_config=ckc, dtype=dt)
        out = ttnn.multiply(out, ttnn.sigmoid(gg))
        out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(
            out, (batch, length, self.n_head * self.head_dim)
        )
        out = ttnn.linear(
            out, self.o_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN
        )
        gate = ttnn.linear(
            c, self.a_out_w, bias=self.a_out_b, compute_kernel_config=ckc,
            dtype=dt, core_grid=CORE_GRID_MAIN,
        )
        q = ttnn.add(q, ttnn.multiply(out, ttnn.sigmoid(gate)))

        norm = self._adaln(
            q, c, self.t_ln_s, self.t_gain_w, self.t_gain_b, self.t_bias_w
        )
        left = ttnn.linear(
            norm, self.t_fc1, activation="silu", compute_kernel_config=ckc,
            dtype=dt, core_grid=CORE_GRID_MAIN,
        )
        right = ttnn.linear(
            norm, self.t_fc2, compute_kernel_config=ckc,
            dtype=dt, core_grid=CORE_GRID_MAIN,
        )
        update = ttnn.linear(
            ttnn.multiply(left, right), self.t_fc3, compute_kernel_config=ckc,
            dtype=dt, core_grid=CORE_GRID_MAIN,
        )
        gate = ttnn.linear(
            c, self.t_out_w, bias=self.t_out_b, compute_kernel_config=ckc,
            dtype=dt, core_grid=CORE_GRID_MAIN,
        )
        return ttnn.add(q, ttnn.multiply(update, ttnn.sigmoid(gate)))


class LocalAtomTransformer(Module):
    """Three-block RFD3 atom encoder using dense additive-mask attention."""

    def __init__(self, state_dict, ckc, n_blocks=3, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.blocks = [
            RFD3AtomBlock(self.scope(f"blocks.{i}"), ckc, dtype=self.dtype)
            for i in range(n_blocks)
        ]

    def run_device(self, q, c, p, additive_mask):
        for block in self.blocks:
            q = block(q, c, p, additive_mask)
        return q

    def __call__(self, q_host, c_host, p_host, indices):
        dt, dev = self.dtype, self.device
        q = _tt(q_host, dev, dt)
        c = _tt(c_host, dev, dt)
        p = _tt(p_host.unsqueeze(0) if p_host.ndim == 3 else p_host, dev, dt)
        mask = _tt(_dense_attention_mask(indices), dev, dt)
        return ttnn.to_torch(self.run_device(q, c, p, mask)).float()


class CompactStreamingDecoder(Module):
    """RFD3 decoder: three device Upcast/atom blocks plus device Downcast."""

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.upcast = [
            GatedCrossAttention(
                self.scope(f"upcast.{i}.gca"), ckc,
                c_query=128, c_kv=256, dtype=self.dtype,
            )
            for i in range(3)
        ]
        self.atom_blocks = [
            RFD3AtomBlock(
                self.scope(f"atom_transformer.{i}"), ckc, dtype=self.dtype
            )
            for i in range(3)
        ]
        self.downcast = GatedCrossAttention(
            self.scope("downcast.gca"), ckc,
            c_query=768, c_kv=128, dtype=self.dtype,
        )
        self.process_s_n = self.torch_to_tt(
            "downcast.process_s.0.weight", dtype=self.dtype
        )
        self.process_s_w = self.torch_to_tt(
            "downcast.process_s.1.weight", dtype=self.dtype
        )

    def _grouping_indices(self, tok_idx, batch):
        valid = _build_valid_mask(tok_idx)
        length = tok_idx.numel()
        padded = torch.full(valid.shape, length, dtype=torch.int64)
        padded[valid] = torch.arange(length)
        pack = torch.cat(
            [padded.reshape(-1) + b * (length + 1) for b in range(batch)]
        )
        flat_valid = valid.flatten().nonzero(as_tuple=False).squeeze(1)
        unpack = torch.cat(
            [flat_valid + b * valid.numel() for b in range(batch)]
        )
        return valid, pack, unpack

    def _pack_atoms_device(self, q, pack_indices, valid):
        batch, length, channels = q.shape
        q = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
        q = ttnn.pad(q, [[0, 0], [0, 1], [0, 0]], 0.0)
        q = ttnn.reshape(q, (batch * (length + 1), channels))
        idx = ttnn.from_torch(
            pack_indices.to(torch.int32).reshape(1, -1),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            dtype=ttnn.uint32,
        )
        packed = ttnn.embedding(
            idx, q, layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        packed = ttnn.reshape(
            packed, (batch, valid.shape[0], valid.shape[1], channels)
        )
        return ttnn.to_layout(packed, ttnn.TILE_LAYOUT)

    def _unpack_atoms_device(self, q, unpack_indices, length):
        batch, tokens, atoms, channels = q.shape
        q = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
        q = ttnn.reshape(q, (batch * tokens * atoms, channels))
        idx = ttnn.from_torch(
            unpack_indices.to(torch.int32).reshape(1, -1),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            dtype=ttnn.uint32,
        )
        unpacked = ttnn.embedding(
            idx, q, layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        unpacked = ttnn.reshape(unpacked, (batch, length, channels))
        return ttnn.to_layout(unpacked, ttnn.TILE_LAYOUT)

    def __call__(self, a_host, s_host, q_host, c_host, p_host, tok_idx, indices):
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        batch, length, _ = q_host.shape
        valid, pack_indices, unpack_indices = self._grouping_indices(tok_idx, batch)
        a = _tt(a_host, dev, dt)
        a_split = ttnn.reshape(a, (a_host.shape[0], a_host.shape[1], 3, 256))
        q = _tt(q_host, dev, dt)
        c = _tt(c_host, dev, dt)
        p = _tt(p_host.unsqueeze(0) if p_host.ndim == 3 else p_host, dev, dt)
        mask = _tt(_dense_attention_mask(indices), dev, dt)
        for upcast, atom_block in zip(self.upcast, self.atom_blocks):
            q_grouped = self._pack_atoms_device(q, pack_indices, valid)
            valid_q = valid.unsqueeze(-1).expand(-1, -1, 3)
            q_grouped = ttnn.add(
                q_grouped, upcast.run_device(q_grouped, a_split, valid_q)
            )
            q = self._unpack_atoms_device(q_grouped, unpack_indices, length)
            q = atom_block(q, c, p, mask)

        q_grouped = self._pack_atoms_device(q, pack_indices, valid)
        query = ttnn.unsqueeze(a, 2)
        down_mask = valid.unsqueeze(1)
        a_update = ttnn.squeeze(
            self.downcast.run_device(query, q_grouped, down_mask), 2
        )
        s = _tt(s_host, dev, dt)
        s = ttnn.rms_norm(
            s, weight=self.process_s_n, epsilon=1e-6, compute_kernel_config=ckc
        )
        s = ttnn.linear(
            s, self.process_s_w, compute_kernel_config=ckc,
            dtype=dt, core_grid=CORE_GRID_MAIN,
        )
        a_out = ttnn.add(ttnn.add(a, a_update), s)
        return ttnn.to_torch(a_out).float(), ttnn.to_torch(q).float()


class LinearSequenceHead(Module):
    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.weight = self.torch_to_tt("linear.weight", dtype=self.dtype)
        self.bias = self.torch_to_tt("linear.bias", dtype=self.dtype)
        self.valid_out_mask = self.weights["valid_out_mask"].bool()

    def __call__(self, a_host):
        logits = ttnn.linear(
            _tt(a_host, self.device, self.dtype),
            self.weight,
            bias=self.bias,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
            core_grid=CORE_GRID_MAIN,
        )
        logits = ttnn.to_torch(logits).float()
        masked = logits.masked_fill(~self.valid_out_mask.view(1, 1, -1), float("-inf"))
        return logits, masked.argmax(dim=-1)


def _bucketize_scaled_distogram(R_L, min_dist=1.0, max_dist=30.0, sigma_data=16.0, n_bins=65):
    """Host port of block_utils.bucketize_scaled_distogram. R_L: [B, N, 3] -> one-hot [B, N, N, n_bins]."""
    D_LL = torch.linalg.norm(R_L.unsqueeze(-2) - R_L.unsqueeze(-3), dim=-1)  # [B, N, N]
    lo, hi = min_dist / sigma_data, max_dist / sigma_data
    bins = torch.linspace(lo, hi, n_bins - 1, device=R_L.device)
    bin_idxs = torch.bucketize(D_LL, bins)
    return torch.nn.functional.one_hot(bin_idxs, num_classes=n_bins).float()


class DiffusionTokenEncoder(Module):
    """RFD3 DiffusionTokenEncoder: self-conditioning distogram + noise distogram -> 2-block
    no-triangle Pairformer. Reuses the verified PairformerBlock (c_s=384, c_z=128, n_head=16)."""

    C_S, C_Z, N_HEAD = 384, 128, 16
    N_BINS, N_PAIRFORMER = 65, 2

    def __init__(self, state_dict, ckc, sigma_data=16.0, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.sigma_data = sigma_data
        self.transition_1 = [Transition(self.scope(f"transition_1.{i}"), ckc, self.C_S, n=2, dtype=self.dtype)
                             for i in range(2)]
        cat_c_z = self.C_Z + self.N_BINS + self.N_BINS  # 128 + 65 (distogram) + 65 (self)
        self.process_z_n = self.torch_to_tt("process_z.0.weight", dtype=self.dtype)
        self.process_z_w = self.torch_to_tt("process_z.1.weight", dtype=self.dtype)
        self.transition_2 = [Transition(self.scope(f"transition_2.{i}"), ckc, self.C_Z, n=2, dtype=self.dtype)
                             for i in range(2)]
        self.pairformer_stack = [PairformerBlock(self.scope(f"pairformer_stack.{i}"), ckc,
                                 self.C_S, self.C_Z, self.N_HEAD, dtype=self.dtype)
                                for i in range(self.N_PAIRFORMER)]

    def __call__(self, R_L_ca, S_init_I, Z_init_II, D_II_self=None):
        """R_L_ca: [B, I, 3] (scaled C-alpha positions), S_init_I: [B, I, c_s],
        Z_init_II: [I, I, c_z] (expanded over batch), D_II_self: [B, I, I, 65] or None.
        Returns (S_I [B,I,c_s], Z_II [B,I,I,c_z]) on host."""
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        B, I = R_L_ca.shape[0], R_L_ca.shape[1]
        s = _tt(S_init_I if S_init_I.ndim == 3 else S_init_I.unsqueeze(0).expand(B, -1, -1).contiguous(), dev, dt)
        for tr in self.transition_1:
            s = ttnn.add(s, tr(s))
        D_LL = _bucketize_scaled_distogram(R_L_ca, sigma_data=self.sigma_data, n_bins=self.N_BINS)
        if D_II_self is None:
            D_II_self = torch.zeros(B, I, I, self.N_BINS, dtype=D_LL.dtype, device=D_LL.device)
        z = _tt(Z_init_II.unsqueeze(0).expand(B, -1, -1, -1).contiguous(), dev, dt)
        zcat = ttnn.concat([z, _tt(D_LL, dev, dt), _tt(D_II_self, dev, dt)], dim=-1)  # [B,I,I,258]
        z = ttnn.rms_norm(zcat, weight=self.process_z_n, epsilon=1e-6, compute_kernel_config=ckc)
        z = ttnn.linear(z, self.process_z_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(zcat)
        for tr in self.transition_2:
            z = ttnn.add(z, tr(z))
        for blk in self.pairformer_stack:
            s, z = blk(s, z)
        return ttnn.to_torch(s).float(), ttnn.to_torch(z).float()


class LocalTokenTransformer(Module):
    """RFD3 18-block token DiT. Each block is the dense-additive-mask
    StructureLocalAtomTransformerBlock (conditioned AttentionPairBias + ConditionedTransition)
    at c_token=768, c_s=384, c_tokenpair=128, n_head=16, head_dim=48."""

    C_TOKEN, C_S, C_PAIR, N_HEAD, N_BLOCK = 768, 384, 128, 16, 18

    def __init__(self, state_dict, ckc, n_block=18, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.blocks = [RFD3AtomBlock(self.scope(f"blocks.{i}"), ckc,
                        c_a=self.C_TOKEN, c_s=self.C_S, c_pair=self.C_PAIR, n_head=self.N_HEAD, dtype=self.dtype)
                       for i in range(n_block)]

    def run_device(self, a, s, z, additive_mask):
        for block in self.blocks:
            a = block(a, s, z, additive_mask)
        return a

    def __call__(self, a_host, s_host, z_host, indices):
        dev, dt = self.device, self.dtype
        a = _tt(a_host, dev, dt)
        s = _tt(s_host, dev, dt)
        z = _tt(z_host.unsqueeze(0) if z_host.ndim == 3 else z_host, dev, dt)
        mask = _tt(_dense_attention_mask(indices), dev, dt)
        return ttnn.to_torch(self.run_device(a, s, z, mask)).float()


def _default_compute_kernel_config():
    dev = get_device()
    kernel_cls = (
        ttnn.types.WormholeComputeKernelConfig
        if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig
    )
    return kernel_cls(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def build_token_initializer(state_dict, compute_kernel_config=None, dtype=None):
    """Construct the ttnn TokenInitializer from a flat `token_initializer.*` state dict
    (prefix already stripped) + a compute_kernel_config. Mirrors the construction order
    used by the torch reference so weight keys line up 1:1."""
    if compute_kernel_config is None:
        compute_kernel_config = _default_compute_kernel_config()
    return TokenInitializer(state_dict, compute_kernel_config, dtype=dtype)


def build_atom_encoder(state_dict, compute_kernel_config=None, dtype=None):
    compute_kernel_config = compute_kernel_config or _default_compute_kernel_config()
    return LocalAtomTransformer(
        state_dict, compute_kernel_config, n_blocks=3, dtype=dtype
    )


def build_decoder(state_dict, compute_kernel_config=None, dtype=None):
    compute_kernel_config = compute_kernel_config or _default_compute_kernel_config()
    return CompactStreamingDecoder(state_dict, compute_kernel_config, dtype=dtype)


def build_sequence_head(state_dict, compute_kernel_config=None, dtype=None):
    compute_kernel_config = compute_kernel_config or _default_compute_kernel_config()
    return LinearSequenceHead(state_dict, compute_kernel_config, dtype=dtype)


def build_diffusion_token_encoder(state_dict, compute_kernel_config=None, dtype=None, sigma_data=16.0):
    compute_kernel_config = compute_kernel_config or _default_compute_kernel_config()
    return DiffusionTokenEncoder(state_dict, compute_kernel_config, sigma_data=sigma_data, dtype=dtype)


def build_dit(state_dict, compute_kernel_config=None, dtype=None, n_block=18):
    compute_kernel_config = compute_kernel_config or _default_compute_kernel_config()
    return LocalTokenTransformer(state_dict, compute_kernel_config, n_block=n_block, dtype=dtype)
