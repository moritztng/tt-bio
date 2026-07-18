"""Parity test for the port's edge_output_layer=True device path (the 3 LayerNorms
+ outer-product), vs the fp32 reference with edge_output_layer=True, on random data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import torch
from interformer_reference_energy import (InterformerEnergyBackbone as RefBackbone,
    random_energy_batched_data)
from interformer_checkpoint import load_energy_checkpoint

def pcc(a, b):
    import numpy as np
    if isinstance(a, np.ndarray): a = torch.from_numpy(a)
    if isinstance(b, np.ndarray): b = torch.from_numpy(b)
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))

sd, cfg = load_energy_checkpoint()
cfg['edge_output_layer'] = True  # force the eol path
ref = RefBackbone(hidden_dim=128, num_heads=8, n_layers=6, ffn_scale=4,
                  node_feat_size=1, K=128, rbf_cutoff=10.0, edge_output_layer=True)
ref.load_state_dict(sd, strict=False); ref.eval()
bd = random_energy_batched_data(b=1, n=64, node_feat_size=1, seed=5)
with torch.no_grad():
    nf = ref.complex_feat_layer(bd['x'])
    ie = ref.complex_feat_layer.edge_feat(bd['intra_D'], None)
    ib_ = ref.complex_feat_layer.wrap_bias(bd['attn_bias'])
    ia_ = ib_.clone()
    ia_[:, :, 1:, 1:] = ia_[:, :, 1:, 1:].masked_fill(bd['pair_mask'].permute(0, 3, 1, 2), float('-inf'))
    g_ref, _, _ = ref.forward_backbone_energy(nf, ie, ia_, ib_, bd['x'], bd['D'], bd['pair_mask'])
from tt_bio.interformer_energy import InterformerEnergyBackbone as TTPort
port = TTPort(sd, cfg)
with torch.no_grad():
    g_port = port(nf, ie, ie, ia_, ib_, bd['x'], bd['D'])
print("# edge_output_layer=True port vs ref (random data):")
ok = True
for k in ('mean', 'sigma', 'pi'):
    v = pcc(g_ref[k], g_port[k]); st = "PASS" if v >= 0.999 else "FAIL"
    if v < 0.999: ok = False
    print(f"#   {k}: {v:.5f} {st}")
print("EOL_PARITY_OK" if ok else "EOL_PARITY_FAIL")
