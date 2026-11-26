"""Grade tt_bio.diffdock.tensor_product against e3nn, in the reference environment.

    PYTHONPATH=<worktree> /tmp/dfdvenv/bin/python scripts/diffdock_port/tp_parity.py

Prints one line per irrep signature: PCC and max relative error against
``o3.FullyConnectedTensorProduct`` with the same random weights. Signatures are the ones the
released DiffDock-L actually builds, taken from the checkpoint via config.layer_plan().
"""
import json
import sys

import torch
from e3nn import o3

from tt_bio.diffdock.config import load as load_config
from tt_bio.diffdock.irreps import spherical_harmonics_irreps
from tt_bio.diffdock.tensor_product import FullyConnectedTensorProduct

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/moritz/scratch/dfd/ckpt/extracted"


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def grade(in_irreps, sh_irreps, out_irreps, n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    ours = FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps)
    ref = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)
    assert ours.weight_numel == ref.weight_numel, (ours.weight_numel, ref.weight_numel)
    x1 = torch.randn(n, ours.in_irreps.dim, generator=g)
    x2 = torch.randn(n, ours.sh_irreps.dim, generator=g)
    w = torch.randn(n, ours.weight_numel, generator=g)
    a, b = ours(x1, x2, w), ref(x1, x2, w)
    rel = ((a - b).abs().max() / b.abs().max()).item()
    return pcc(a, b), rel, ours.weight_numel


def main():
    cfg = load_config(f"{CKPT}/score_model", old_model=False)
    sh = str(spherical_harmonics_irreps(cfg.sh_lmax))
    rows, worst_pcc, worst_rel = [], 1.0, 0.0
    seen = set()
    for entry in cfg.layer_plan():
        sig = (entry["in_irreps"], entry.get("sh_irreps", sh), entry["out_irreps"])
        if sig in seen:
            continue
        seen.add(sig)
        p, rel, numel = grade(*sig)
        worst_pcc, worst_rel = min(worst_pcc, p), max(worst_rel, rel)
        rows.append(dict(name=entry["name"], in_irreps=sig[0], sh_irreps=sig[1],
                         out_irreps=sig[2], weight_numel=numel, pcc=p, max_rel_err=rel))
        print(f"{entry['name']:>20}  {sig[0]} x {sig[1]} -> {sig[2]}")
        print(f"{'':>20}  numel {numel:>5}  PCC {p:.10f}  max_rel_err {rel:.3e}")
    print(f"\nworst PCC {worst_pcc:.10f}  worst max_rel_err {worst_rel:.3e}  "
          f"over {len(rows)} distinct signatures")
    print(json.dumps(dict(signatures=rows, worst_pcc=worst_pcc, worst_max_rel_err=worst_rel)))


if __name__ == "__main__":
    main()
