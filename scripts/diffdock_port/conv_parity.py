"""Grade ``tt_bio.diffdock.conv`` against upstream's own ``TensorProductConvLayer``.

Run in the reference environment, which has e3nn, torch_scatter and the upstream clone:

    cd ~/scratch/dfd/upstream && \
      PYTHONPATH=.:/home/moritz/.coworker/wt/dfd-parity \
      ~/scratch/dfd/venv-ref/bin/python \
      /home/moritz/.coworker/wt/dfd-parity/scripts/diffdock_port/conv_parity.py

The reference is upstream's module itself, not a re-implementation of it, so the grade covers
the FC block layout, the tensor product, the sum-then-divide scatter, e3nn's irrep batch norm
and the padded residual as one unit. Upstream's state dict loads into ours unchanged: the
dropout slots are :class:`torch.nn.Identity` here and carry no parameters, so the module
indices and therefore every key line up.

The batch-norm running statistics are randomised away from their initial (0, 1): at the
checkpoint's values a wrong normalisation convention is nearly invisible.
"""

import json
import sys

import torch
from e3nn import o3

sys.path.insert(0, "/home/moritz/.coworker/wt/dfd-parity")
from models.tensor_layers import TensorProductConvLayer as RefLayer  # noqa: E402

from tt_bio.diffdock.conv import TensorProductConvLayer as OurLayer  # noqa: E402
from tt_bio.diffdock import config as dfd_config  # noqa: E402

SH = "1x0e + 1x1o + 1x2e"


def randomise_bn(ref):
    if ref.batch_norm is None:
        return
    with torch.no_grad():
        ref.batch_norm.running_mean.normal_(0.0, 0.5)
        ref.batch_norm.running_var.uniform_(0.3, 2.0)
        ref.batch_norm.weight.normal_(1.0, 0.2)
        ref.batch_norm.bias.normal_(0.0, 0.2)


def grade(in_irreps, out_irreps, n_edge_features, hidden, edge_groups, residual,
          n_nodes=96, n_edges=640, seed=0):
    torch.manual_seed(seed)
    ref = RefLayer(in_irreps, SH, out_irreps, n_edge_features, residual=residual,
                   batch_norm=True, dropout=0.0, hidden_features=hidden,
                   edge_groups=edge_groups).eval()
    randomise_bn(ref)
    ours = OurLayer(in_irreps, SH, out_irreps, n_edge_features, residual=residual,
                    batch_norm=True, hidden_features=hidden, edge_groups=edge_groups).eval()
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=False)
    # e3nn's TensorProduct carries a zero-length `weight` (shared_weights=False) and an
    # `output_mask` buffer; neither is a
    # parameter of the computation, and nor are the 3j constants its compiled kernel exports.
    unexpected = [k for k in unexpected
                  if k not in ("tp.weight", "tp.output_mask") and "_w3j" not in k]
    assert not unexpected, unexpected
    assert not [k for k in missing if "w3j" not in k], missing

    node_attr = torch.randn(n_nodes, o3.Irreps(in_irreps).dim)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_sh = o3.spherical_harmonics(SH, torch.randn(n_edges, 3), normalize=True,
                                     normalization="component")
    sizes = [n_edges // edge_groups] * edge_groups
    sizes[-1] += n_edges - sum(sizes)
    groups = [torch.randn(s, n_edge_features) for s in sizes]
    ref_attr = groups if edge_groups > 1 else groups[0]

    with torch.no_grad():
        a = ours(node_attr, edge_index, groups, edge_sh)
        b = ref(node_attr, edge_index, ref_attr, edge_sh)
    rel = ((a - b).abs().max() / b.abs().max()).item()
    pcc = torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()
    return rel, pcc


def main():
    cfg = dfd_config.load("/home/moritz/scratch/dfd/ckpt/extracted/score_model",
                              old_model=False, name="score")
    seen, rows = set(), []
    for n, layer in enumerate(cfg.layer_plan()):
        key = (layer["in_irreps"], layer["out_irreps"], layer["n_edge_features"],
               layer["hidden_features"], layer["edge_groups"], layer["residual"])
        if key in seen:
            continue
        seen.add(key)
        rel, pcc = grade(*key, seed=n)
        rows.append({"layer": n, "in": key[0], "out": key[1], "edge_groups": key[4],
                     "residual": key[5], "max_rel_err": rel, "pcc": pcc})
        print(f"[{n:2d}] groups={key[4]} residual={int(key[5])} "
              f"{key[0]:34s} -> {key[1]:34s}  rel={rel:.3e}  pcc={pcc:.15f}")
    worst_rel = max(r["max_rel_err"] for r in rows)
    worst_pcc = min(r["pcc"] for r in rows)
    print(f"\nworst max_rel_err {worst_rel:.3e}   worst pcc {worst_pcc:.15f}")
    print(json.dumps({"signatures": len(rows), "worst_max_rel_err": worst_rel,
                      "worst_pcc": worst_pcc, "rows": rows}))


if __name__ == "__main__":
    main()
