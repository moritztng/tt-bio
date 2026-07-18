"""Interformer (Tencent AI4S, Apache-2.0) on-device port -- the dense transformer
backbone + affinity readout.

HYBRID SPLIT (the key design decision -- see docs/interformer-port.md):
  * ON DEVICE (this file): the interaction-aware (edge-biased) transformer
    encoder (intra x n_layers + inter x n_layers//2 EncoderLayers) and the
    affinity readout (final_ln + PReLU FFN). Pure dense math: matmul, layernorm,
    softmax, FFN. This is the accuracy-relevant, TT-friendly trunk.
  * ON HOST (NOT ported -- irregular glue): graph construction (RDKit/obabel
    atom typing, pocket extraction), the distance matrix D, RBF expansion, the
    learned atom/edge embeddings (gather), padding / attention-bias masks, the
    MDN / VinaScoreHead (energy-mode gaussian heads + vdw/hydro/hbond tables +
    shelve energy-file output), and the Monte-Carlo docking / energy-min sampler
    (docking/reconstruct_ligands, compiled C++).

The forward here takes host-prepared node_feats + intra/inter edge_feats +
intra/inter attn_bias and returns the affinity scalar -- the same split the
reference (tests/interformer_reference.py) exposes for component parity.
"""
from __future__ import annotations
import torch
from tt_bio.tenstorrent import TorchWrapper, Module, WeightScope, _dtype


def _ident(x):
    return x


class InterformerModule(Module):
    """On-device transformer encoder + affinity readout. Built from a
    weight-compatible state_dict (source checkpoint keys load directly)."""

    def __init__(self, weights, compute_kernel_config, cfg):
        super().__init__(WeightScope.wrap(weights), compute_kernel_config)
        import ttnn
        self.ttnn = ttnn
        self.cfg = cfg
        H = cfg['hidden_dim']
        nh = cfg['num_heads']
        self.nh = nh
        self.dk = H // nh
        self.scale = self.dk ** -0.5
        self.ffn_dim = H * cfg['ffn_scale']
        # rel_pos_3d_proj (host RBF [b,n,n,K] -> [b,n,n,H]) -- dense Linear on device.
        self.rel_pos_w = self.torch_to_tt('complex_feat_layer.rel_pos_3d_proj.weight')
        self.rel_pos_b = self.torch_to_tt('complex_feat_layer.rel_pos_3d_proj.bias')
        self.intra = [self._build_layer(i, 'intra_encoder') for i in range(cfg['n_layers'])]
        self.inter = [self._build_layer(i, 'inter_encoder') for i in range(cfg['n_layers'] // 2)]
        self.final_ln_w = self.torch_to_tt('final_ln.weight', transform=_ident)
        self.final_ln_b = self.torch_to_tt('final_ln.bias', transform=_ident)
        self.aff_w1 = self.torch_to_tt('affinity_proj.W_1.weight')
        self.aff_b1 = self.torch_to_tt('affinity_proj.W_1.bias', transform=_ident)
        self.aff_w2 = self.torch_to_tt('affinity_proj.W_2.weight')
        self.aff_b2 = self.torch_to_tt('affinity_proj.W_2.bias', transform=_ident)
        self.aff_prelu = self.torch_to_tt('affinity_proj.act_func.weight', transform=_ident)

    def _build_layer(self, i, scope):
        s = self.scope(scope).child(str(i))
        return _EncoderLayerWeights(s, self.compute_kernel_config, self.ttnn)

    # ---- helpers ----
    def _ln(self, x, w, b, eps=1e-5):
        return self.ttnn.layer_norm(
            x, weight=w, bias=b, epsilon=eps, compute_kernel_config=self.compute_kernel_config)

    def rel_pos_proj(self, rbf_tt):
        """Host RBF [b,n,n,K] -> device Linear -> [b,n,n,H]."""
        return self._lin(rbf_tt, self.rel_pos_w, self.rel_pos_b)

    def _encoder_layer(self, layer, x, e, attn_bias):
        ttnn = self.ttnn
        b = x.shape[0]
        N = x.shape[1]
        H = self.cfg['hidden_dim']
        nh = self.nh
        dk = self.dk
        # --- self-attention (edge-biased): logit_ij = sum_d q_i k_j e_ij ---
        y = self._ln(x, layer.sa_ln_w, layer.sa_ln_b)
        e_n = self._ln(e, layer.ea_ln_w, layer.ea_ln_b)
        q = self._lin(y, layer.lq, layer.lq_b)
        q = ttnn.reshape(q, (b, N, nh, dk)); q = ttnn.permute(q, (0, 2, 1, 3)); q = ttnn.reshape(q, (b * nh, N, dk))
        k = self._lin(y, layer.lk, layer.lk_b)
        k = ttnn.reshape(k, (b, N, nh, dk)); k = ttnn.permute(k, (0, 2, 1, 3)); k = ttnn.reshape(k, (b * nh, N, dk))
        v = self._lin(y, layer.lv, layer.lv_b)
        v = ttnn.reshape(v, (b, N, nh, dk)); v = ttnn.permute(v, (0, 2, 1, 3)); v = ttnn.reshape(v, (b * nh, N, dk))
        e_h = self._lin(e_n, layer.le, layer.le_b)
        e_h = ttnn.reshape(e_h, (b, N, N, nh, dk)); e_h = ttnn.permute(e_h, (0, 3, 1, 2, 4)); e_h = ttnn.reshape(e_h, (b * nh, N, N, dk))
        q = ttnn.multiply(q, self.scale)
        qk = ttnn.multiply(ttnn.unsqueeze(q, 2), ttnn.unsqueeze(k, 1))        # [b*nh, N, N, dk]
        qk_e = ttnn.multiply(qk, e_h)                                          # [b*nh, N, N, dk]
        w = ttnn.sum(qk_e, dim=-1)                                              # [b*nh, N, N]
        ab = ttnn.reshape(attn_bias, (b * nh, N, N))
        w = ttnn.add(w, ab)
        w = ttnn.softmax(w, dim=-1)
        o = ttnn.matmul(w, v)                                                  # [b*nh, N, dk]
        o = ttnn.reshape(o, (b, nh, N, dk)); o = ttnn.permute(o, (0, 2, 1, 3)); o = ttnn.reshape(o, (b, N, H))
        o = self._lin(o, layer.out_w, layer.out_b)
        e_out = ttnn.reshape(qk_e, (b, nh, N, N, dk)); e_out = ttnn.permute(e_out, (0, 2, 3, 1, 4)); e_out = ttnn.reshape(e_out, (b, N, N, H))
        e_out = self._lin(e_out, layer.eo_w, layer.eo_b)
        # --- residuals + node FFN ---
        x = ttnn.add(x, o)
        y = self._ln(x, layer.ffn_ln_w, layer.ffn_ln_b)
        y = ttnn.relu(self._lin(y, layer.ff1_w, layer.ff1_b))
        y = self._lin(y, layer.ff2_w, layer.ff2_b)
        x = ttnn.add(x, y)
        # --- residuals + edge FFN ---
        e = ttnn.add(e, e_out)
        y = self._ln(e, layer.ef_ln_w, layer.ef_ln_b)
        y = ttnn.relu(self._lin(y, layer.ef1_w, layer.ef1_b))
        y = self._lin(y, layer.ef2_w, layer.ef2_b)
        e = ttnn.add(e, y)
        return x, e

    def readout(self, inter_node):
        ttnn = self.ttnn
        b = inter_node.shape[0]
        H = self.cfg['hidden_dim']
        ln_all = self._ln(inter_node, self.final_ln_w, self.final_ln_b)   # [b, N, H]
        vn = ttnn.slice(ln_all, [0, 0, 0], [b, 1, H])                     # [b, 1, H]
        vn = ttnn.reshape(vn, (b, H))
        # PReLU FFN: prelu(x) = relu(x) + a*(x - relu(x))
        h = self._lin(vn, self.aff_w1, self.aff_b1)
        r = ttnn.relu(h)
        h = ttnn.add(r, ttnn.multiply(ttnn.subtract(h, r), self.aff_prelu))
        h = self._lin(h, self.aff_w2, self.aff_b2)
        return h                                                        # [b, 1]

    def __call__(self, node_feats, intra_edge, inter_edge, intra_bias, inter_bias):
        x = node_feats
        e_intra = intra_edge
        for layer in self.intra:
            x, _ = self._encoder_layer(layer, x, e_intra, intra_bias)
        e_inter = inter_edge
        for layer in self.inter:
            x, e_inter = self._encoder_layer(layer, x, e_inter, inter_bias)
        return self.readout(x), x


class _EncoderLayerWeights(Module):
    """Holds the tiled device weights for one EncoderLayer."""

    def __init__(self, weights, compute_kernel_config, ttnn):
        super().__init__(weights, compute_kernel_config)
        self.ttnn = ttnn
        self.sa_ln_w = self.torch_to_tt('self_attention_norm.weight', transform=_ident)
        self.sa_ln_b = self.torch_to_tt('self_attention_norm.bias', transform=_ident)
        self.lq = self.torch_to_tt('self_attention.linear_q.weight')
        self.lq_b = self.torch_to_tt('self_attention.linear_q.bias', transform=_ident)
        self.lk = self.torch_to_tt('self_attention.linear_k.weight')
        self.lk_b = self.torch_to_tt('self_attention.linear_k.bias', transform=_ident)
        self.lv = self.torch_to_tt('self_attention.linear_v.weight')
        self.lv_b = self.torch_to_tt('self_attention.linear_v.bias', transform=_ident)
        self.out_w = self.torch_to_tt('self_attention.output_layer.weight')
        self.out_b = self.torch_to_tt('self_attention.output_layer.bias', transform=_ident)
        self.le = self.torch_to_tt('self_attention.linear_e.weight')
        self.le_b = self.torch_to_tt('self_attention.linear_e.bias', transform=_ident)
        self.eo_w = self.torch_to_tt('self_attention.e_output_layer.weight')
        self.eo_b = self.torch_to_tt('self_attention.e_output_layer.bias', transform=_ident)
        self.ffn_ln_w = self.torch_to_tt('ffn_norm.weight', transform=_ident)
        self.ffn_ln_b = self.torch_to_tt('ffn_norm.bias', transform=_ident)
        self.ff1_w = self.torch_to_tt('ffn.layer1.weight')
        self.ff1_b = self.torch_to_tt('ffn.layer1.bias', transform=_ident)
        self.ff2_w = self.torch_to_tt('ffn.layer2.weight')
        self.ff2_b = self.torch_to_tt('ffn.layer2.bias', transform=_ident)
        self.ea_ln_w = self.torch_to_tt('edge_attn_norm.weight', transform=_ident)
        self.ea_ln_b = self.torch_to_tt('edge_attn_norm.bias', transform=_ident)
        self.ef_ln_w = self.torch_to_tt('edge_ffn_norm.weight', transform=_ident)
        self.ef_ln_b = self.torch_to_tt('edge_ffn_norm.bias', transform=_ident)
        self.ef1_w = self.torch_to_tt('edge_ffn.layer1.weight')
        self.ef1_b = self.torch_to_tt('edge_ffn.layer1.bias', transform=_ident)
        self.ef2_w = self.torch_to_tt('edge_ffn.layer2.weight')
        self.ef2_b = self.torch_to_tt('edge_ffn.layer2.bias', transform=_ident)


class InterformerBackbone(TorchWrapper):
    """tt-bio wrapper for the Interformer on-device backbone.

    Construct with a weight-compatible state_dict (source checkpoint keys load
    directly) OR with random weights (state_dict=None builds a fresh reference
    and ports its weights) for parity testing. The forward runs host-prepared
    node_feats + intra/inter edge_feats + intra/inter attn_bias through the
    on-device transformer and returns (affinity, inter_node).
    """

    DEFAULT_CFG = dict(hidden_dim=128, num_heads=8, n_layers=6, ffn_scale=4,
                       K=128, node_feat_size=2, edge_feat_size=1)

    def __init__(self, state_dict=None, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = dict(self.DEFAULT_CFG)
        self.cfg = cfg
        if state_dict is None:
            from tests.interformer_reference import InterformerBackbone as RefBackbone
            ref = RefBackbone(hidden_dim=cfg['hidden_dim'], num_heads=cfg['num_heads'],
                              n_layers=cfg['n_layers'], ffn_scale=cfg['ffn_scale'],
                              node_feat_size=cfg['node_feat_size'],
                              edge_feat_size=cfg['edge_feat_size'])
            ref.eval()
            state_dict = ref.state_dict()
            self._ref = ref
        self.module = self._create_module(WeightScope.wrap(state_dict))

    def _create_module(self, weights):
        return InterformerModule(weights, self.compute_kernel_config, self.cfg)

    def __call__(self, node_feats, intra_edge, inter_edge, intra_bias, inter_bias):
        affinity_tt, inter_node_tt = self.module(node_feats, intra_edge, inter_edge,
                                                  intra_bias, inter_bias)
        import ttnn
        affinity = torch.Tensor(ttnn.to_torch(affinity_tt)).float()
        inter_node = torch.Tensor(ttnn.to_torch(inter_node_tt)).float()
        return affinity, inter_node

    def rel_pos_proj(self, rbf_tt):
        return self.module.rel_pos_proj(rbf_tt)

    def encoder_layer(self, idx, scope, x, e, attn_bias):
        layers = self.module.intra if scope == 'intra' else self.module.inter
        return self.module._encoder_layer(layers[idx], x, e, attn_bias)

    def readout(self, inter_node_tt):
        return self.module.readout(inter_node_tt)
