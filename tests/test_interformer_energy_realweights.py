"""Interformer ENERGY head REAL-WEIGHT parity vs the from-scratch PyTorch
reference, both loaded from the released Zenodo energy checkpoint
(v0.2_energy_model). The trunk parity is established (pass 2); this gates the
ENERGY head (3 Gaussian PFFNs + elu + edge_output_layer cfg-gated) on top.

Run:  TT_VISIBLE_DEVICES=2 PYTHONPATH=.:tests python3 tests/test_interformer_energy_realweights.py

Gate: mean/sigma/pi PCC >= 0.999 (bf16 HiFi4 device vs fp32 reference). The
released energy ckpt has edge_output_layer=None, so the head uses the bypass
path (pair_emb = inter_edge[:, 1:, 1:, :]); the 3 LayerNorms + outer-product are
dead weights in this ckpt and are exercised by the random-weight component test.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from interformer_reference_energy import (
    InterformerEnergyBackbone as RefBackbone, random_energy_batched_data)
from interformer_checkpoint import load_energy_checkpoint


def pcc(a, b):
    import numpy as _np
    if isinstance(a, _np.ndarray): a = torch.from_numpy(a)
    if isinstance(b, _np.ndarray): b = torch.from_numpy(b)
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))


def main():
    sd, cfg = load_energy_checkpoint()
    print(f"# energy ckpt cfg: {cfg}")
    ref = RefBackbone(
        hidden_dim=cfg['hidden_dim'], num_heads=cfg['num_heads'],
        n_layers=cfg['n_layers'], ffn_scale=cfg['ffn_scale'],
        node_feat_size=cfg['node_feat_size'], K=cfg['K'], rbf_cutoff=cfg['rbf_cutoff'],
        edge_output_layer=cfg['edge_output_layer'])
    m, u = ref.load_state_dict(sd, strict=False)
    print(f"# ref load missing={len(m)} unexpected={len(u)} (unexpected[:3]={u[:3]})")
    ref.eval()

    bd = random_energy_batched_data(b=1, n=64, node_feat_size=cfg['node_feat_size'], seed=3)
    with torch.no_grad():
        node_feats = ref.complex_feat_layer(bd['x'])
        intra_edge = ref.complex_feat_layer.edge_feat(bd['intra_D'], None)
        inter_bias = ref.complex_feat_layer.wrap_bias(bd['attn_bias'])
        intra_bias = inter_bias.clone()
        intra_bias[:, :, 1:, 1:] = intra_bias[:, :, 1:, 1:].masked_fill(
            bd['pair_mask'].permute(0, 3, 1, 2), float('-inf'))
        g_ref, _, _ = ref.forward_backbone_energy(
            node_feats, intra_edge, intra_bias, inter_bias, bd['x'], bd['D'], bd['pair_mask'])

    from tt_bio.interformer_energy import InterformerEnergyBackbone as TTPort
    port = TTPort(sd, cfg)
    with torch.no_grad():
        g_port = port(node_feats, intra_edge, intra_edge, intra_bias, inter_bias,
                      bd['x'], bd['D'])

    results = {}
    for k in ('mean', 'sigma', 'pi'):
        results[k] = pcc(g_ref[k], g_port[k])
    # also check the host-glue outputs match (they are deterministic from x, D)
    for k in ('vdw_pair', 'd'):
        results[k] = pcc(g_ref[k], g_port[k])

    print("\n=== Interformer ENERGY head REAL-WEIGHT parity (bf16 HiFi4) vs fp32 ref ===")
    ok = True
    for k, v in results.items():
        status = "PASS" if v >= 0.999 else "FAIL"
        if v < 0.999: ok = False
        print(f"  {status}  {k:10s} PCC={v:.5f}")
    # sanity: pi sums to ~1, sigma >= 1e-5
    print(f"# pi sum (port) = {float(g_port['pi'].sum(axis=-1).mean()):.4f} (expect ~1)")
    print(f"# sigma min (port) = {float(g_port['sigma'].min()):.6f} (expect >= 1e-5)")
    print(f"# vdw_pair max (port) = {float(g_port['vdw_pair'].max()):.4f}")
    print("ALL PASS" if ok else "SOME FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
