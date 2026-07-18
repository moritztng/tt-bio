"""Energy head (VinaScoreHead) + InterformerEnergyBackbone for the reference
oracle. The energy checkpoint (v0.2_energy_model) has edge_output_layer=None, so
the released model BYPASSES edge_output_layer: pair_emb = inter_edge[:,1:,1:,:];
the 3 LayerNorms + outer-product are dead weights in that ckpt. We port the full
head (cfg-gated) but the real-ckpt parity + G.pkl regeneration use the bypass
path to match the released model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from interformer_reference import (ComplexEncoder, EncoderLayer,
    PositionwiseFeedForward, NUM_ATOM_TYPES)


class VinaScoreHead(nn.Module):
    """Reference (fp32) energy head, weight-compatible with the upstream
    Interformer.VinaScoreHead state_dict keys."""

    def __init__(self, hidden_dim, dropout=0.2, edge_output_layer=True):
        super().__init__()
        self.num_atom_types = 29
        self.train_dist_cut_off = 4.0
        self.pair_ln = nn.LayerNorm(hidden_dim)
        self.node_ln = nn.LayerNorm(hidden_dim)
        self.final_pair_ln = nn.LayerNorm(hidden_dim)
        self.meanHead = PositionwiseFeedForward(hidden_dim, hidden_dim * 2, d_out=4, dropout=dropout)
        self.sigmaHead = PositionwiseFeedForward(hidden_dim, hidden_dim * 2, d_out=4, dropout=dropout)
        self.WeightHead = PositionwiseFeedForward(hidden_dim, hidden_dim * 2, d_out=4, dropout=dropout)
        self.use_edge_output_layer = edge_output_layer
        es = self.num_atom_types ** 2
        self.vdw_table = nn.Embedding(es, 1)
        self.hydro_table = nn.Embedding(es, 1)
        self.hbond_table = nn.Embedding(es, 1)
        for t in (self.vdw_table, self.hydro_table, self.hbond_table):
            t.weight.requires_grad = False

    def atom_type2pair_type(self, x):
        a = torch.where(x >= self.num_atom_types, x - self.num_atom_types, x)
        return a[:, :, None, :] * self.num_atom_types + a[:, None, :]

    def edge_output_layer_fn(self, node_feats, pair_feats):
        pf = pair_feats[:, 1:, 1:, :]
        pf = self.pair_ln(pf)
        pf = (pf + pf.transpose(1, 2)) * 0.5
        nf = self.node_ln(node_feats[:, 1:])
        nf = torch.einsum('b i d, b j d -> b i j d', nf, nf)
        return self.final_pair_ln(nf + pf)

    def forward(self, inter_node, inter_edge, x, D, pair_mask=None):
        if self.use_edge_output_layer:
            pair_emb = self.edge_output_layer_fn(inter_node, inter_edge)
        else:
            pair_emb = inter_edge[:, 1:, 1:, :]
        pt = self.atom_type2pair_type(x).squeeze(-1)
        vdw_pair = self.vdw_table(pt)
        d = D.squeeze(-1) - vdw_pair.squeeze(-1)
        hydro_pair = self.hydro_table(pt).bool()
        hbond_pair = self.hbond_table(pt).bool()
        mean = F.elu(self.meanHead(pair_emb))
        sigma = F.elu(self.sigmaHead(pair_emb)) + 1.0 + 1e-5
        pi = self.WeightHead(pair_emb)
        hydro_soft = torch.where(hydro_pair, 0.0, float('-inf'))
        hbond_soft = torch.where(hbond_pair, 0.0, float('-inf'))
        zero = torch.zeros_like(hydro_soft)
        mask = torch.cat([zero, zero, hydro_soft, hbond_soft], dim=-1)
        pi_soft = torch.softmax(pi + mask, dim=-1) + 1e-9
        return {'mean': mean, 'sigma': sigma, 'pi': pi_soft, 'd': d.unsqueeze(-1),
                'vdw_pair': vdw_pair, 'hydro_pair': hydro_pair, 'hbond_pair': hbond_pair}


class InterformerEnergyBackbone(nn.Module):
    """Energy-mode backbone: ComplexEncoder (edge_feat_size=0) + intra/inter
    EncoderLayers + final_ln + affinity_proj + VinaScoreHead. In energy_mode the
    inter encoder starts from intra_edge_feats (NOT a fresh edge_feat(D))."""

    def __init__(self, hidden_dim=128, num_heads=8, n_layers=6, ffn_scale=4,
                 node_feat_size=1, K=128, rbf_cutoff=10.0, dropout_rate=0.2,
                 attention_dropout_rate=0.2, input_dropout_rate=0.0,
                 edge_output_layer=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        ffn_dim = hidden_dim * ffn_scale
        self.complex_feat_layer = ComplexEncoder(
            node_feat_size, 0, hidden_dim, num_heads, input_dropout_rate,
            K=K, rbf_cutoff=rbf_cutoff)
        self.intra_encoder = nn.ModuleList([
            EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, i)
            for i in range(n_layers)])
        self.inter_encoder = nn.ModuleList([
            EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, i)
            for i in range(n_layers // 2)])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.affinity_proj = PositionwiseFeedForward(hidden_dim, ffn_dim, d_out=1, dropout=dropout_rate)
        self.VinaScoreHead = VinaScoreHead(hidden_dim, dropout_rate, edge_output_layer=edge_output_layer)

    def forward_backbone_energy(self, node_feats, intra_edge_feats, intra_attn_bias,
                                inter_attn_bias, x, D, pair_mask=None):
        x_node = node_feats
        for layer in self.intra_encoder:
            x_node, _ = layer(x_node, intra_edge_feats, intra_attn_bias)
        inter_node = x_node
        inter_edge = intra_edge_feats
        for layer in self.inter_encoder:
            inter_node, inter_edge = layer(inter_node, inter_edge, inter_attn_bias)
        g = self.VinaScoreHead(inter_node, inter_edge, x, D, pair_mask)
        return g, inter_node, inter_edge


def random_energy_batched_data(b=1, n=64, node_feat_size=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 512 * node_feat_size, (b, n, node_feat_size), generator=g).long()
    x[:, :, 0:1] = torch.randint(1, 28, (b, n, 1), generator=g).long()
    intra_D = torch.rand(b, n, n, generator=g) * 9.0 + 0.5
    D = torch.rand(b, n, n, generator=g) * 9.0 + 0.5
    intra_D = intra_D * (1 - torch.eye(n).unsqueeze(0))
    D = D * (1 - torch.eye(n).unsqueeze(0))
    attn_bias = torch.zeros(b, n + 1, n + 1)
    pair_mask = torch.ones(b, n, n, 1, dtype=torch.bool)
    ligand_len = torch.tensor([n // 2] * b).long()
    pocket_len = torch.tensor([n - n // 2] * b).long()
    return {'x': x, 'intra_D': intra_D, 'D': D, 'attn_bias': attn_bias,
            'pair_mask': pair_mask, 'edata': None,
            'ligand_len': ligand_len, 'pocket_len': pocket_len}
