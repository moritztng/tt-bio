"""Grade ``tt_bio.diffdock.heads.CenterHead`` (tr/rot) with the RELEASED weights.

Run in the reference environment, from the upstream clone (``utils/so3.py`` loads its cached
tables from the working directory):

    cd ~/scratch/dfd/upstream && PYTHONPATH=.:/home/moritz/.coworker/wt/dfd-parity \\
      ~/scratch/dfd/venv-ref/bin/python \\
      /home/moritz/.coworker/wt/dfd-parity/scripts/diffdock_port/heads_parity.py

The reference is built from upstream's own modules (``TensorProductConvLayer``,
``GaussianSmearing``, ``o3.spherical_harmonics``, ``utils.so3.score_norm``) loaded with the
checkpoint's ``center_*``, ``final_conv.*``, ``tr_final_layer.*`` and ``rot_final_layer.*``
tensors; the glue between them is ``CGModel.forward`` lines 368-395 and
``build_center_conv_graph``, transcribed. Inputs are random ligand features and positions
across several graphs and noise levels: this grades the head with real weights, not the
whole network on a real complex -- that is the goldens' job.

Also graded on its own: ``tt_bio.diffdock.harmonics.spherical_harmonics`` vs e3nn.
"""

import json
import sys

import torch
from e3nn import o3
from torch import nn

sys.path.insert(0, "/home/moritz/.coworker/wt/dfd-parity")
from models.layers import GaussianSmearing as RefSmearing  # noqa: E402
from models.tensor_layers import TensorProductConvLayer as RefConv  # noqa: E402
from utils import so3  # noqa: E402

from tt_bio.diffdock.harmonics import spherical_harmonics  # noqa: E402
from tt_bio.diffdock.heads import CenterHead  # noqa: E402

CKPT = "/home/moritz/scratch/dfd/ckpt/extracted/score_model/best_ema_inference_epoch_model.pt"
NS, SIGMA_DIM, DIST_DIM = 48, 64, 64
IN_IRREPS = "48x0e + 10x1o + 10x1e + 10x0o"
SH = "1x0e + 1x1o + 1x2e"


def _cmp(a, b):
    a, b = a.double(), b.double()
    rel = ((a - b).abs().max() / b.abs().max()).item()
    pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
    return rel, pcc


def grade_sh():
    torch.manual_seed(0)
    v = torch.randn(4096, 3) * 7
    v[0] = 0.0                                            # the zero-length edge case
    out = {}
    for irreps, ls in ((SH, (0, 1, 2)), ("2e", (2,))):
        a = spherical_harmonics(v, ls)
        b = o3.spherical_harmonics(irreps, v, normalize=True, normalization="component")
        out[irreps] = _cmp(a, b)
    return out


def ref_head(m, node_attr, pos, batch, num_graphs, node_sigma_emb, graph_sigma_emb,
             tr_sigma, rot_sigma):
    # build_center_conv_graph, verbatim but for `data`
    edge_index = torch.cat([batch.unsqueeze(0), torch.arange(len(batch)).unsqueeze(0)], dim=0)
    center_pos, count = torch.zeros((num_graphs, 3)), torch.zeros((num_graphs, 3))
    center_pos.index_add_(0, index=batch, source=pos)
    center_pos = center_pos / torch.bincount(batch).unsqueeze(1)
    edge_vec = pos[edge_index[1]] - center_pos[edge_index[0]]
    edge_attr = m.center_distance_expansion(edge_vec.norm(dim=-1))
    edge_sigma_emb = node_sigma_emb[edge_index[1].long()]
    edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
    edge_sh = o3.spherical_harmonics(SH, edge_vec, normalize=True, normalization="component")
    # CGModel.forward 368-395, fixed_center_conv=True, odd_parity=False, scale_by_sigma=True
    center_edge_attr = m.center_edge_embedding(edge_attr)
    center_edge_attr = torch.cat([center_edge_attr, node_attr[edge_index[1], :NS]], -1)
    g = m.final_conv(node_attr, edge_index, center_edge_attr, edge_sh, out_nodes=num_graphs)
    tr_pred = g[:, :3] + g[:, 6:9]
    rot_pred = g[:, 3:6] + g[:, 9:]
    tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
    tr_pred = tr_pred / tr_norm * m.tr_final_layer(torch.cat([tr_norm, graph_sigma_emb], dim=1))
    rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
    rot_pred = rot_pred / rot_norm * m.rot_final_layer(torch.cat([rot_norm, graph_sigma_emb], dim=1))
    tr_pred = tr_pred / tr_sigma.unsqueeze(1)
    rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1)
    return tr_pred, rot_pred


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    ref = nn.Module()
    ref.center_distance_expansion = RefSmearing(0.0, 30, DIST_DIM)
    ref.center_edge_embedding = nn.Sequential(nn.Linear(DIST_DIM + SIGMA_DIM, NS), nn.ReLU(),
                                              nn.Dropout(0.0), nn.Linear(NS, NS))
    ref.final_conv = RefConv(IN_IRREPS, SH, "2x1o + 2x1e", 2 * NS, residual=False,
                             dropout=0.0, batch_norm=True)
    ref.tr_final_layer = nn.Sequential(nn.Linear(1 + SIGMA_DIM, NS), nn.Dropout(0.0), nn.ReLU(),
                                       nn.Linear(NS, 1))
    ref.rot_final_layer = nn.Sequential(nn.Linear(1 + SIGMA_DIM, NS), nn.Dropout(0.0), nn.ReLU(),
                                        nn.Linear(NS, 1))
    prefixes = ("center_distance_expansion.", "center_edge_embedding.", "final_conv.",
                "tr_final_layer.", "rot_final_layer.")
    sub = {k: v for k, v in sd.items() if k.startswith(prefixes)}
    ref.load_state_dict(sub, strict=True)
    ref.eval()

    ours = CenterHead(in_irreps=IN_IRREPS, sh_irreps=SH, ns=NS, distance_embed_dim=DIST_DIM,
                      sigma_embed_dim=SIGMA_DIM).eval()
    skip = ("tp.weight", "tp.output_mask", "_w3j")
    missing, unexpected = ours.load_state_dict(
        {k: v for k, v in sub.items() if not any(s in k for s in skip)}, strict=False)
    assert not unexpected and not missing, (missing, unexpected)

    torch.manual_seed(1)
    sizes = [1, 7, 23, 40, 64]            # includes a one-atom ligand: zero-length centre edge
    batch = torch.cat([torch.full((s,), i) for i, s in enumerate(sizes)])
    n, g = batch.numel(), len(sizes)
    pos = torch.randn(n, 3) * 4 + torch.randn(g, 3)[batch] * 20
    node_attr = torch.randn(n, o3.Irreps(IN_IRREPS).dim)
    t = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.05])
    tr_sigma = 0.1 ** (1 - t) * 19.0 ** t
    rot_sigma = 0.03 ** (1 - t) * 1.55 ** t
    from tt_bio.diffdock.features import timestep_embedding
    graph_emb = timestep_embedding(t, SIGMA_DIM, 1000)
    node_emb = graph_emb[batch]

    with torch.no_grad():
        a_tr, a_rot = ours(node_attr, pos, batch, g, node_emb, graph_emb, tr_sigma, rot_sigma)
        b_tr, b_rot = ref_head(ref, node_attr, pos, batch, g, node_emb, graph_emb, tr_sigma,
                               rot_sigma)
    res = {"sh": grade_sh(), "tr": _cmp(a_tr, b_tr), "rot": _cmp(a_rot, b_rot)}
    for k, v in res.items():
        print(k, v)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
