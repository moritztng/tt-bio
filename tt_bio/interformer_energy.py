"""Interformer energy head (VinaScoreHead) on-device port -- the dense learned
parts of the energy model that drives pose selection.

HYBRID SPLIT (extends tt_bio/interformer.py):
  * ON DEVICE (this file): the 3 Gaussian heads (meanHead/sigmaHead/WeightHead,
    each a PositionwiseFeedForward = Linear(H,2H)+PReLU+Linear(2H,4)) + elu, and
    (cfg-gated) the edge_output_layer (3 LayerNorms + outer-product + symmetrize).
  * ON HOST (the wrapper, InterformerEnergyBackbone): the pair-type gather, the
    VdW / hydrophobic / H-bond pair-type lookup tables, d = D - vdw_pair, the
    hydro/hbond soft mask, the softmax(+mask)+1e-9, and the shelve G.db output --
    all irregular glue coupled to atom-type embeddings, matching the hybrid
    design in docs/interformer-port.md.

The released energy checkpoint (v0.2_energy_model) has edge_output_layer=None,
so the released model BYPASSES edge_output_layer: pair_emb = inter_edge (the
last inter-encoder edge output). The 3 LayerNorms + outer-product are dead
weights in that ckpt. We port the edge_output_layer path too (cfg-gated) so the
head is complete for any ckpt config; the real-ckpt parity + G.pkl regeneration
use the bypass path to match the released model.

The heads are point-wise over the last dim, so for the bypass path we run them
on the FULL [b, n+1, n+1, H] inter_edge (no device slice) and the host slices
[:, 1:, 1:, :] -> [b, n, n, 4] afterwards; the [b, n, n, 4] part (indices 1..n)
is bit-identical to the reference (which strips VN first), modulo bf16.
"""
from __future__ import annotations
import torch
from tt_bio.tenstorrent import TorchWrapper, Module, WeightScope, _dtype


def _ident(x):
    return x


class InterformerEnergyHeadModule(Module):
    """On-device energy head: 3 Gaussian PFFNs + elu (+ cfg-gated
    edge_output_layer). Built from a weight-compatible state_dict."""

    def __init__(self, weights, compute_kernel_config, cfg):
        super().__init__(WeightScope.wrap(weights), compute_kernel_config)
        import ttnn
        self.ttnn = ttnn
        self.cfg = cfg
        self.use_eol = cfg.get('edge_output_layer', None)  # None/False -> bypass
        # narrow to the VinaScoreHead subtree
        self.weights = self.scope('VinaScoreHead')
        if self.use_eol:
            self.pair_ln_w = self.torch_to_tt('pair_ln.weight', transform=_ident)
            self.pair_ln_b = self.torch_to_tt('pair_ln.bias', transform=_ident)
            self.node_ln_w = self.torch_to_tt('node_ln.weight', transform=_ident)
            self.node_ln_b = self.torch_to_tt('node_ln.bias', transform=_ident)
            self.final_pair_ln_w = self.torch_to_tt('final_pair_ln.weight', transform=_ident)
            self.final_pair_ln_b = self.torch_to_tt('final_pair_ln.bias', transform=_ident)
        head_keys = {'mean': 'meanHead', 'sigma': 'sigmaHead', 'weight': 'WeightHead'}
        for p, cap in head_keys.items():
            setattr(self, f'{p}_w1', self.torch_to_tt(f'{cap}.W_1.weight'))
            setattr(self, f'{p}_b1', self.torch_to_tt(f'{cap}.W_1.bias', transform=_ident))
            setattr(self, f'{p}_w2', self.torch_to_tt(f'{cap}.W_2.weight'))
            setattr(self, f'{p}_b2', self.torch_to_tt(f'{cap}.W_2.bias', transform=_ident))
            setattr(self, f'{p}_prelu', self.torch_to_tt(f'{cap}.act_func.weight', transform=_ident))

    def _ln(self, x, w, b, eps=1e-5):
        return self.ttnn.layer_norm(
            x, weight=w, bias=b, epsilon=eps, compute_kernel_config=self.compute_kernel_config)

    def _pffn(self, x, w1, b1, w2, b2, prelu):
        ttnn = self.ttnn
        h = self._lin(x, w1, b1)
        r = ttnn.relu(h)
        h = ttnn.add(r, ttnn.multiply(ttnn.subtract(h, r), prelu))
        return self._lin(h, w2, b2)

    def _edge_output_layer(self, inter_node_tt, inter_edge_tt):
        # pair_feats[:, 1:, 1:, :] -> pair_ln -> symmetrize; node[:, 1:] -> node_ln
        # -> outer product; add; final_pair_ln. (Only exercised when use_eol.)
        ttnn = self.ttnn
        b = inter_edge_tt.shape[0]
        N1 = inter_edge_tt.shape[1]
        H = self.cfg['hidden_dim']
        N = N1 - 1
        pf = ttnn.slice(inter_edge_tt, [0, 1, 1, 0], [b, N1, N1, H])     # [b, N, N, H]
        pf = self._ln(pf, self.pair_ln_w, self.pair_ln_b)
        pf_t = ttnn.permute(pf, (0, 2, 1, 3))
        pf = ttnn.multiply(ttnn.add(pf, pf_t), 0.5)
        nf = ttnn.slice(inter_node_tt, [0, 1, 0], [b, N1, H])             # [b, N, H]
        nf = self._ln(nf, self.node_ln_w, self.node_ln_b)
        nf_i = ttnn.unsqueeze(nf, 1)                                      # [b, 1, N, H]
        nf_j = ttnn.unsqueeze(nf, 2)                                      # [b, N, 1, H]
        op = ttnn.multiply(nf_i, nf_j)                                    # [b, N, N, H]
        out = ttnn.add(op, pf)
        return self._ln(out, self.final_pair_ln_w, self.final_pair_ln_b)

    def __call__(self, inter_node_tt, inter_edge_tt):
        ttnn = self.ttnn
        if self.use_eol:
            pair_emb = self._edge_output_layer(inter_node_tt, inter_edge_tt)
        else:
            pair_emb = inter_edge_tt  # full [b, n+1, n+1, H]; host slices later
        # Scalar offsets (+1+1e-5 for sigma, +1e-9 for pi-soft) are applied on
        # host (fp32) -- 1e-5 rounds to 0 in bf16, and the offsets couple to the
        # host-computed hydro/hbond mask + softmax, so they belong to host glue.
        mean = ttnn.elu(self._pffn(pair_emb, self.mean_w1, self.mean_b1,
                                  self.mean_w2, self.mean_b2, self.mean_prelu), alpha=1.0)
        sigma = ttnn.elu(self._pffn(pair_emb, self.sigma_w1, self.sigma_b1,
                                    self.sigma_w2, self.sigma_b2, self.sigma_prelu), alpha=1.0)
        pi = self._pffn(pair_emb, self.weight_w1, self.weight_b1,
                       self.weight_w2, self.weight_b2, self.weight_prelu)
        return mean, sigma, pi


class InterformerEnergyBackbone(TorchWrapper):
    """tt-bio wrapper: device trunk + device energy head + host glue (pair-type
    lookups, mask, softmax, shelve G.db output). Construct with the real energy
    checkpoint state_dict (source keys load directly)."""

    def __init__(self, state_dict, cfg):
        super().__init__()
        self.cfg = cfg
        H = cfg['hidden_dim']
        self.num_atom_types = 29
        self.use_eol = cfg.get('edge_output_layer', None)
        # fixed pair-type tables (host-side lookup, loaded from the ckpt)
        sd = state_dict
        self.vdw_table = sd['VinaScoreHead.vdw_table.weight'].detach().float()
        self.hydro_table = sd['VinaScoreHead.hydro_table.weight'].detach().float()
        self.hbond_table = sd['VinaScoreHead.hbond_table.weight'].detach().float()
        from tt_bio.interformer import InterformerModule
        self.trunk = InterformerModule(WeightScope.wrap(sd), self.compute_kernel_config, cfg)
        self.head = InterformerEnergyHeadModule(WeightScope.wrap(sd), self.compute_kernel_config, cfg)

    def _atom_type2pair_type(self, x):
        a = torch.where(x >= self.num_atom_types, x - self.num_atom_types, x)
        return a[:, :, None, :] * self.num_atom_types + a[:, None, :]

    def __call__(self, node_feats, intra_edge, inter_edge, intra_bias, inter_bias, x, D):
        """Run the device trunk (energy_mode: inter_edge starts = intra_edge) +
        device energy head, then host glue (slice, mask, softmax, tables).

        node_feats [b, n+1, H], intra_edge [b, n+1, n+1, H], inter_edge = intra_edge
        for energy mode, x [b, n, k] atom types, D [b, n, n, 1] ground-truth dist.
        Returns the G.pkl dict (numpy, batch-squeezed): mean/sigma/pi [n, n, 4],
        d/vdw_pair [n, n, 1], hydro_pair/hbond_pair [n, n, 1] bool, ligand_len,
        pocket_len.
        """
        import ttnn
        dev = self.tt_device
        bf16 = ttnn.bfloat16

        def up(t):
            return ttnn.from_torch(t.contiguous().to(torch.float32), device=dev,
                                   layout=ttnn.TILE_LAYOUT, dtype=bf16)

        def down(t):
            return torch.Tensor(ttnn.to_torch(t)).float()

        # energy mode: inter encoder starts from intra_edge (NOT a fresh edge_feat(D))
        aff_tt, inter_node_tt, inter_edge_tt = self.trunk(
            up(node_feats), up(intra_edge), up(intra_edge), up(intra_bias), up(inter_bias))
        mean_tt, sigma_tt, pi_tt = self.head(inter_node_tt, inter_edge_tt)
        mean = down(mean_tt)[:, 1:, 1:, :]            # [b, n, n, 4]
        sigma = down(sigma_tt)[:, 1:, 1:, :] + 1.0 + 1e-5   # fp32 offset (bf16 loses 1e-5)
        pi = down(pi_tt)[:, 1:, 1:, :]
        # host glue: pair-type lookups + d + mask + softmax
        x = x.long()
        pair_type = self._atom_type2pair_type(x).squeeze(-1)            # [b, n, n]
        vdw_pair = self.vdw_table[pair_type]                            # [b, n, n, 1]
        d = D.squeeze(-1) - vdw_pair.squeeze(-1)                        # [b, n, n]
        hydro_pair = self.hydro_table[pair_type].bool()
        hbond_pair = self.hbond_table[pair_type].bool()
        hydro_soft = torch.where(hydro_pair, 0.0, float('-inf'))
        hbond_soft = torch.where(hbond_pair, 0.0, float('-inf'))
        zero = torch.zeros_like(hydro_soft)
        mask = torch.cat([zero, zero, hydro_soft, hbond_soft], dim=-1)   # [b, n, n, 4]
        pi_soft = torch.softmax(pi + mask, dim=-1) + 1e-9
        return {
            'mean': mean.squeeze(0).numpy(), 'sigma': sigma.squeeze(0).numpy(),
            'pi': pi_soft.squeeze(0).numpy(), 'd': d.unsqueeze(-1).squeeze(0).numpy(),
            'vdw_pair': vdw_pair.squeeze(0).numpy(),
            'hydro_pair': hydro_pair.squeeze(0).numpy(),
            'hbond_pair': hbond_pair.squeeze(0).numpy(),
        }
