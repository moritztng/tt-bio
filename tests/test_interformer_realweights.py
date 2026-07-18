"""Interformer on-device backbone REAL-WEIGHT parity vs the from-scratch
PyTorch reference, both loaded from the released Zenodo affinity checkpoint.

Run:  TT_VISIBLE_DEVICES=0 PYTHONPATH=.:tests python3 tests/test_interformer_realweights.py

Pass 1 verified component parity with RANDOM weights (wiring only). Random
weights do not prove correctness -- only that the port is wired like the
reference. This test loads the real 1.1 GB Zenodo affinity checkpoint
(checkpoints/v0.2_affinity_model/model0) into BOTH the reference and the ttnn
port (strict load, no missing/unexpected keys) and re-runs the same component
parity (C1-C4) with the released weights. PCC >= 0.999 (absdiff < 0.05 for the
scalar affinity) under real weights is the real correctness gate.

The host glue (embeddings, RBF, masks, padding) is computed by the reference
from a well-formed random batched_data and fed IDENTICALLY to both sides, so
the comparison isolates the on-device dense math -- same isolation as the
random-weight test, now over the trained weight distribution.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import ttnn
from interformer_reference import InterformerBackbone as RefBackbone, random_batched_data
from interformer_checkpoint import load_affinity_checkpoint


def pcc(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))

def absdiff(a, b):
    return float((a.float().reshape(-1) - b.float().reshape(-1)).abs().max())


def main():
    sd, cfg = load_affinity_checkpoint()
    print(f"# real ckpt cfg: {cfg}")
    ref = RefBackbone(
        hidden_dim=cfg['hidden_dim'], num_heads=cfg['num_heads'],
        n_layers=cfg['n_layers'], ffn_scale=cfg['ffn_scale'],
        node_feat_size=cfg['node_feat_size'], edge_feat_size=cfg['edge_feat_size'],
        K=cfg['K'], rbf_cutoff=cfg['rbf_cutoff'], pose_sel_mode=cfg['pose_sel_mode'])
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"# WARN missing={len(missing)} unexpected={len(unexpected)}")
        print("  missing[:5]:", missing[:5])
        print("  unexpected[:5]:", unexpected[:5])
    ref.eval()

    bd = random_batched_data(b=1, n=64, node_feat_size=cfg['node_feat_size'],
                             edge_feat_size=cfg['edge_feat_size'], seed=1)

    with torch.no_grad():
        node_feats = ref.complex_feat_layer(bd['x'])
        intra_edge = ref.complex_feat_layer.edge_feat(bd['intra_D'], bd.get('edata'))
        inter_edge = ref.complex_feat_layer.edge_feat(bd['D'], bd.get('edata'))
        inter_bias = ref.complex_feat_layer.wrap_bias(bd['attn_bias'])
        intra_bias = inter_bias.clone()
        intra_bias[:, :, 1:, 1:] = intra_bias[:, :, 1:, 1:].masked_fill(
            bd['pair_mask'].permute(0, 3, 1, 2), float('-inf'))
        rbf = ref.complex_feat_layer.rbf(bd['intra_D'])

    from tt_bio.interformer import InterformerBackbone as TTPort
    port = TTPort(state_dict=sd, cfg=cfg)
    dev = port.tt_device
    bf16 = ttnn.bfloat16

    def up(x):
        return ttnn.from_torch(x.contiguous().to(torch.float32), device=dev,
                               layout=ttnn.TILE_LAYOUT, dtype=bf16)
    def down(x):
        return torch.Tensor(ttnn.to_torch(x)).float()

    results = {}

    with torch.no_grad():
        ref_c1 = ref.complex_feat_layer.rel_pos_3d_proj(rbf)
    tt_c1 = down(port.rel_pos_proj(up(rbf)))
    results['C1 rel_pos_3d_proj'] = pcc(ref_c1, tt_c1)

    with torch.no_grad():
        ref_x, ref_e = ref.intra_encoder[0](node_feats, intra_edge, intra_bias)
    tt_x, tt_e = port.encoder_layer(0, 'intra', up(node_feats), up(intra_edge), up(intra_bias))
    tt_x = down(tt_x); tt_e = down(tt_e)
    results['C2 EncoderLayer node'] = pcc(ref_x, tt_x)
    results['C2 EncoderLayer edge'] = pcc(ref_e, tt_e)

    with torch.no_grad():
        ref_aff, ref_inter, ref_vn = ref.forward_backbone(node_feats, intra_edge, inter_edge,
                                                          intra_bias, inter_bias)
    tt_aff, tt_inter, _ = port(up(node_feats), up(intra_edge), up(inter_edge),
                           up(intra_bias), up(inter_bias))
    results['C3 backbone inter_node'] = pcc(ref_inter, tt_inter)
    results['C3 backbone affinity (absdiff)'] = absdiff(ref_aff, tt_aff)

    with torch.no_grad():
        ref_aff2 = ref.affinity_proj(ref.final_ln(ref_inter[:, 0, :]))
    tt_aff2 = down(port.module.readout(up(ref_inter)))
    results['C4 readout affinity (absdiff)'] = absdiff(ref_aff2, tt_aff2)

    print("\n=== Interformer backbone REAL-WEIGHT parity (bf16, HiFi4) vs PyTorch reference ===")
    print(f"# ref affinity (real weights): {float(ref_aff):.5f}   port affinity: {float(tt_aff):.5f}")
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
