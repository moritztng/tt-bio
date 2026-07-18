"""From-scratch PyTorch reference of the Interformer transformer backbone +
affinity readout (Tencent AI4S, Apache-2.0).

Re-implemented from github.com/tencent-ailab/Interformer
(interformer/model/transformer/graphormer/*) so the dense on-device port in
tt_bio/interformer.py has a parity oracle WITHOUT pulling in pytorch_lightning /
torchmetrics / obabel (none of which are in the dev env). Submodule names mirror
the source so a real released checkpoint loads with load_state_dict (strict=False)
for real-weight parity.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ATOM_TYPES = 29


def softplus_inverse(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return x + torch.log(-torch.expm1(-x))


class RBFLayer(nn.Module):
    def __init__(self, K=128, cutoff=10.0, edge_types=NUM_ATOM_TYPES ** 2, dtype=torch.float32):
        super().__init__()
        self.cutoff = cutoff
        centers = softplus_inverse(torch.linspace(1.0, math.exp(-cutoff), K)).to(dtype)
        self.centers = nn.Parameter(F.softplus(centers))
        widths = softplus_inverse(torch.full((K,), 0.5 / ((1.0 - math.exp(-cutoff) / K) ** 2))).to(dtype)
        self.widths = nn.Parameter(F.softplus(widths))
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def cutoff_fn(self, D):
        x = D / self.cutoff
        x3, x4, x5 = torch.pow(x, 3.0), torch.pow(x, 4.0), torch.pow(x, 5.0)
        return torch.where(x < 1, 1 - 6 * x5 + 15 * x4 - 10 * x3, torch.zeros_like(x))

    def forward(self, D, edge_types=None):
        D = D.unsqueeze(-1)
        rbf_D = self.cutoff_fn(D) * torch.exp(-self.widths * torch.pow((torch.exp(-D) - self.centers), 2))
        return rbf_D


class ComplexEncoder(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_dim, num_heads,
                 input_dropout_rate, K=128, rbf_cutoff=10.0, num_atom_types=NUM_ATOM_TYPES):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_atom_types = num_atom_types
        self.atom_encoder = nn.Embedding(512 * node_feat_size + 1, hidden_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(input_dropout_rate)
        self.graph_token = nn.Embedding(1, hidden_dim)
        if edge_feat_size > 0:
            self.edge_encoder = nn.Embedding(512 * edge_feat_size + 1, hidden_dim, padding_idx=0)
        self.rbf = RBFLayer(K, cutoff=rbf_cutoff, edge_types=num_atom_types ** 2)
        self.rel_pos_3d_proj = nn.Linear(K, hidden_dim)

    def atom_type2pair_type(self, x):
        atoms_type_x = x % self.num_atom_types
        pair_type = atoms_type_x[:, :, None, :] * self.num_atom_types + atoms_type_x[:, None, :]
        return pair_type

    def wrap_bias(self, attn_bias):
        return attn_bias.clone().unsqueeze(1).repeat(1, self.num_heads, 1, 1)

    def edge_feat(self, D, edata=None):
        # D: [b, n, n] (already squeezed); returns [b, n+1, n+1, hidden]
        D = D.float()
        rbf = self.rbf(D)                       # [b, n, n, K]
        edge_feats = self.rel_pos_3d_proj(rbf)  # [b, n, n, hidden]
        if hasattr(self, 'edge_encoder') and edata is not None:
            edge_feats = edge_feats + self.edge_encoder(edata).sum(dim=-2)
        b, n, _, h = edge_feats.shape
        graph_edge = torch.zeros(b, n + 1, n + 1, h, device=edge_feats.device, dtype=edge_feats.dtype)
        graph_edge[:, 1:, 1:] = edge_feats
        return graph_edge

    def forward(self, x):
        # x: [b, n, k] int -> node feats [b, n+1, hidden]
        node_feature = self.atom_encoder(x).sum(dim=-2)
        n_graph = x.size(0)
        graph_token = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node = torch.cat([graph_token, node_feature], dim=1)
        graph_node = self.input_dropout(graph_node)
        return graph_node


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, num_layer):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)
        self.linear_e = nn.Linear(hidden_size, num_heads * self.att_size)
        self.e_output_layer = nn.Linear(num_heads * self.att_size, hidden_size)

    def forward(self, q, k, v, e, attn_bias=None):
        b = q.size(0)
        d = self.att_size
        h = self.num_heads
        q = self.linear_q(q).view(b, -1, h, d).transpose(1, 2)   # [b,h,n,d]
        k = self.linear_k(k).view(b, -1, h, d).transpose(1, 2)
        v = self.linear_v(v).view(b, -1, h, d).transpose(1, 2)
        e = self.linear_e(e).reshape(b, e.size(1), e.size(2), h, d).permute(0, 3, 1, 2, 4)  # [b,h,i,j,d]
        q = q * self.scale
        qk_e = q.unsqueeze(3) * k.unsqueeze(2) * e              # [b,h,i,j,d]
        w = qk_e.sum(dim=-1)                                    # [b,h,i,j]
        if attn_bias is not None:
            w = w + attn_bias
        w = torch.softmax(w, dim=-1)
        x = w.matmul(v)                                         # [b,h,n,d]
        x = x.transpose(1, 2).contiguous().view(b, -1, h * d)
        x = self.output_layer(x)
        e_out = self.e_output_layer(qk_e.permute(0, 2, 3, 1, 4).reshape(b, e.size(2), e.size(3), h * d))
        return x, e_out


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, num_layer):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads, num_layer)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.edge_attn_norm = nn.LayerNorm(hidden_size)
        self.edge_ffn_norm = nn.LayerNorm(hidden_size)
        self.edge_ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)

    def forward(self, x, e, attn_bias=None):
        y = self.self_attention_norm(x)
        e_hat = self.edge_attn_norm(e)
        y, e_hat = self.self_attention(y, y, y, e_hat, attn_bias)
        x = x + y
        y = self.ffn(self.ffn_norm(x))
        x = x + y
        e = e + e_hat
        e = e + self.edge_ffn(self.edge_ffn_norm(e))
        return x, e


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, d_out=None, dropout=0.1):
        super().__init__()
        d_out = d_model if d_out is None else d_out
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.act_func = nn.PReLU()

    def forward(self, x):
        return self.W_2(self.dropout(self.act_func(self.W_1(x))))


class InterformerBackbone(nn.Module):
    """Affinity-mode backbone: ComplexEncoder + intra (n_layers) + inter
    (n_layers//2) EncoderLayers + final_ln + affinity_proj. This is the dense
    transformer that is ported to ttnn. The MDN/VinaScoreHead (energy mode) and
    the Monte-Carlo docking sampler stay on host (see docs/interformer-port.md)."""

    def __init__(self, hidden_dim=128, num_heads=8, n_layers=6, ffn_scale=4,
                 node_feat_size=2, edge_feat_size=1, K=128, rbf_cutoff=10.0,
                 dropout_rate=0.2, attention_dropout_rate=0.2, input_dropout_rate=0.0,
                 pose_sel_mode=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pose_sel_mode = pose_sel_mode
        ffn_dim = hidden_dim * ffn_scale
        self.complex_feat_layer = ComplexEncoder(
            node_feat_size, edge_feat_size, hidden_dim, num_heads,
            input_dropout_rate, K=K, rbf_cutoff=rbf_cutoff)
        self.intra_encoder = nn.ModuleList([
            EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, i)
            for i in range(n_layers)])
        self.inter_encoder = nn.ModuleList([
            EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, i)
            for i in range(n_layers // 2)])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.affinity_proj = PositionwiseFeedForward(hidden_dim, ffn_dim, d_out=1, dropout=dropout_rate)
        if pose_sel_mode:
            self.out_pose_sel_proj = PositionwiseFeedForward(hidden_dim, ffn_dim, d_out=1, dropout=dropout_rate)

    def forward(self, batched_data):
        x = batched_data['x']
        node_feats = self.complex_feat_layer(x)
        intra_edge_feats = self.complex_feat_layer.edge_feat(batched_data['intra_D'], batched_data.get('edata'))
        inter_attn_bias = self.complex_feat_layer.wrap_bias(batched_data['attn_bias'])
        inter_edge = self.complex_feat_layer.edge_feat(batched_data['D'], batched_data.get('edata'))
        intra_attn_bias = inter_attn_bias.clone()
        intra_attn_bias[:, :, 1:, 1:] = intra_attn_bias[:, :, 1:, 1:].masked_fill(
            batched_data['pair_mask'].permute(0, 3, 1, 2), float('-inf'))
        intra_node = node_feats
        for layer in self.intra_encoder:
            intra_node, _ = layer(intra_node, intra_edge_feats, intra_attn_bias)
        inter_node = intra_node
        for layer in self.inter_encoder:
            inter_node, inter_edge = layer(inter_node, inter_edge, inter_attn_bias)
        vn = self.final_ln(inter_node[:, 0, :])
        affinity = self.affinity_proj(vn)
        return affinity, inter_node, vn

    def forward_backbone(self, node_feats, intra_edge_feats, inter_edge_feats, intra_attn_bias, inter_attn_bias):
        """Run ONLY the on-device portion (encoder stack + readout) on host-prepared
        inputs -- mirrors tt_bio.interformer.InterformerModule.__call__ for parity."""
        x = node_feats
        for layer in self.intra_encoder:
            x, _ = layer(x, intra_edge_feats, intra_attn_bias)
        e_inter = inter_edge_feats
        for layer in self.inter_encoder:
            x, e_inter = layer(x, e_inter, inter_attn_bias)
        vn = self.final_ln(x[:, 0, :])
        affinity = self.affinity_proj(vn)
        return affinity, x, vn



def random_batched_data(b=1, n=64, node_feat_size=2, edge_feat_size=1, dtype=torch.float32, seed=0):
    """Build a random but well-formed batched_data dict (host glue output)."""
    g = torch.Generator().manual_seed(seed)
    # atom-type channels in [0, 512*node_feat_size] (embedding range), with padding 0 allowed
    x = torch.randint(0, 512 * node_feat_size, (b, n, node_feat_size), generator=g).long()
    x[:, :, 0:1] = torch.randint(1, 28, (b, n, 1), generator=g).long()  # first channel is the 29-type atom type
    edata = torch.randint(0, 512 * edge_feat_size, (b, n, n, edge_feat_size), generator=g).long() if edge_feat_size > 0 else None
    # distances in (0, rbf_cutoff]
    intra_D = torch.rand(b, n, n, generator=g) * 9.0 + 0.5
    D = torch.rand(b, n, n, generator=g) * 9.0 + 0.5
    intra_D = intra_D * (1 - torch.eye(n).unsqueeze(0)) + torch.eye(n).unsqueeze(0) * 0.0
    D = D * (1 - torch.eye(n).unsqueeze(0))
    attn_bias = torch.zeros(b, n + 1, n + 1)  # [b, N+1, N+1] incl. virtual node
    pair_mask = torch.ones(b, n, n, 1, dtype=torch.bool)  # [b, N, N, 1] (no VN)
    ligand_mask = torch.zeros(b, n, 1)
    ligand_mask[:, : n // 2, :] = 1.0
    return {'x': x, 'intra_D': intra_D, 'D': D, 'attn_bias': attn_bias,
            'pair_mask': pair_mask, 'ligand_mask': ligand_mask, 'edata': edata}
