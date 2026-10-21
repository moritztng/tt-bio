"""Dump the upstream RFD4 feature dict for one annotated CIF, as the parity golden.

Runs ONLY in the upstream reference env (/home/moritz/rfd4-ref-env/venv), never in tt-bio's:
it imports rfproteina and atomworks 3.0.4. tt-bio's own featurizer is compared against the
.npz this writes, field by field.

    rfd4-ref-env/venv/bin/python -I dump_upstream_features.py <in.cif[.zst]> <out.npz>
"""
import sys
from pathlib import Path

import numpy as np
import torch


def dump(cif: Path, out: Path) -> dict:
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.inference_utils.datasets import DesignProblemDataset

    pipeline = build_design_validation_pipeline()
    ds = DesignProblemDataset([cif], transform=pipeline)
    example = ds[0]
    flat = {}

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}/{k}" if prefix else str(k), v)
        elif isinstance(obj, torch.Tensor):
            flat[prefix] = obj.detach().cpu().numpy()
        elif isinstance(obj, np.ndarray):
            flat[prefix] = obj
        elif isinstance(obj, (int, float, bool, str)):
            flat[prefix] = np.array(obj)

    walk("", example)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **flat)
    return flat


if __name__ == "__main__":
    cif, out = Path(sys.argv[1]), Path(sys.argv[2])
    flat = dump(cif, out)
    print(f"{len(flat)} arrays -> {out}")
    for k in sorted(flat)[:400]:
        a = flat[k]
        print(f"  {k}: {a.shape} {a.dtype}")
