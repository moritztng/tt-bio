"""Interformer on-device backbone parity vs the from-scratch PyTorch reference.

Run:  TT_VISIBLE_DEVICES=1 python3 tests/test_interformer.py

Verifies (bf16, HiFi4) PCC >= 0.999 for each dense on-device component:
  C1 rel_pos_3d_proj  (host RBF -> device Linear)
  C2 one EncoderLayer (edge-biased attention + node FFN + edge FFN)
  C3 full backbone    (intra x6 + inter x3)
  C4 affinity readout (final_ln + PReLU FFN)
The host glue (embeddings, RBF, masks, padding, distances) is computed by the
reference and fed IDENTICALLY to both sides, so the comparison isolates the
on-device dense math.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import torch
import ttnn
from interformer_reference import InterformerBackbone as RefBackbone, random_batched_data


def pcc(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a * b).sum() / denom)

def absdiff(a, b):
    return float((a.float().reshape(-1) - b.float().reshape(-1)).abs().max())


def main():
    torch.manual_seed(0)
    cfg = dict(hidden_dim=128, num_heads=8, n_layers=6, ffn_scale=4,
               K=128, node_feat_size=2, edge_feat_size=1)
    ref = RefBackbone(hidden_dim=cfg['hidden_dim'], num_heads=cfg['num_heads'],
                      n_layers=cfg['n_layers'], ffn_scale=cfg['ffn_scale'],
                      node_feat_size=cfg['node_feat_size'], edge_feat_size=cfg['edge_feat_size'])
    ref.eval()
    bd = random_batched_data(b=1, n=64, node_feat_size=cfg['node_feat_size'],
                             edge_feat_size=cfg['edge_feat_size'], seed=1)

    # ---- host glue (computed by the reference; identical inputs to both sides) ----
    with torch.no_grad():
        node_feats = ref.complex_feat_layer(bd['x'])                       # [b, N+1, H]
        intra_edge = ref.complex_feat_layer.edge_feat(bd['intra_D'], bd.get('edata'))
        inter_edge = ref.complex_feat_layer.edge_feat(bd['D'], bd.get('edata'))
        inter_bias = ref.complex_feat_layer.wrap_bias(bd['attn_bias'])     # [b, h, N+1, N+1]
        intra_bias = inter_bias.clone()
        intra_bias[:, :, 1:, 1:] = intra_bias[:, :, 1:, 1:].masked_fill(
            bd['pair_mask'].permute(0, 3, 1, 2), float('-inf'))
        rbf = ref.complex_feat_layer.rbf(bd['intra_D'])                    # [b, n, n, K]

    # ---- build on-device port from the SAME weights ----
    from tt_bio.interformer import InterformerBackbone as TTPort
    port = TTPort(state_dict=ref.state_dict(), cfg=cfg)
    dev = port.tt_device
    bf16 = ttnn.bfloat16

    def up(x):
        return ttnn.from_torch(x.contiguous().to(torch.float32), device=dev,
                               layout=ttnn.TILE_LAYOUT, dtype=bf16)
    def down(x):
        return torch.Tensor(ttnn.to_torch(x)).float()

    results = {}

    # C1: rel_pos_3d_proj
    with torch.no_grad():
        ref_c1 = ref.complex_feat_layer.rel_pos_3d_proj(rbf)
    tt_c1 = down(port.rel_pos_proj(up(rbf)))
    results['C1 rel_pos_3d_proj'] = pcc(ref_c1, tt_c1)

    # C2: one EncoderLayer (intra[0])
    with torch.no_grad():
        ref_x, ref_e = ref.intra_encoder[0](node_feats, intra_edge, intra_bias)
    tt_x, tt_e = port.encoder_layer(0, 'intra', up(node_feats), up(intra_edge), up(intra_bias))
    tt_x = down(tt_x); tt_e = down(tt_e)
    results['C2 EncoderLayer node'] = pcc(ref_x, tt_x)
    results['C2 EncoderLayer edge'] = pcc(ref_e, tt_e)

    # C3: full backbone
    with torch.no_grad():
        ref_aff, ref_inter, ref_vn = ref.forward_backbone(node_feats, intra_edge, inter_edge,
                                                          intra_bias, inter_bias)
    tt_aff, tt_inter = port(up(node_feats), up(intra_edge), up(inter_edge),
                            up(intra_bias), up(inter_bias))  # __call__ returns torch tensors
    results['C3 backbone inter_node'] = pcc(ref_inter, tt_inter)
    results['C3 backbone affinity (absdiff)'] = absdiff(ref_aff, tt_aff)

    # C4: affinity readout in isolation (feed ref inter_node)
    with torch.no_grad():
        ref_aff2 = ref.affinity_proj(ref.final_ln(ref_inter[:, 0, :]))
    tt_aff2 = down(port.module.readout(up(ref_inter)))
    results['C4 readout affinity (absdiff)'] = absdiff(ref_aff2, tt_aff2)

    # ---- report ----
    print("\n=== Interformer backbone parity (bf16, HiFi4) vs PyTorch reference ===")
    ok = True
    for k, v in results.items():
        if 'absdiff' in k:
            status = "PASS" if v < 0.05 else "FAIL"
            if v >= 0.05: ok = False
            print(f"  {status}  {k:32s} absdiff={v:.5f}")
        else:
            status = "PASS" if v >= 0.999 else "FAIL"
            if v < 0.999: ok = False
            print(f"  {status}  {k:32s} PCC={v:.5f}")
    print("ALL PASS" if ok else "SOME FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
