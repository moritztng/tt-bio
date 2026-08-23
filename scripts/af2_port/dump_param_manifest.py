"""Dump the `params_model_1_ptm.npz` key/shape tree and the remap's output tree.

Both manifests are committed under `scripts/af2_port/parity_artifacts/`, which lets
`tests/test_af2_weights.py` gate the whole remap without the 373 MB checkpoint and without a
card: it rebuilds a zero-filled source from the checkpoint manifest, runs the remap, and
compares against the expected manifest key by key.

Run once per checkpoint, not per port change:

    PYTHONPATH=. env/bin/python3 scripts/af2_port/dump_param_manifest.py \
        --npz ~/pxd_tool_weights/af2/params_model_1_ptm.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tt_bio.af2_weights import remap_af2_params

ARTIFACTS = Path(__file__).resolve().parent / "parity_artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=str(ARTIFACTS))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.npz, allow_pickle=False) as npz:
        source = {k: npz[k] for k in npz.files}

    checkpoint = {k: list(v.shape) for k, v in sorted(source.items())}
    remapped = {k: list(v.shape) for k, v in sorted(remap_af2_params(source).items())}

    npz_params = sum(int(np.prod(s)) if s else 1 for s in checkpoint.values())
    remap_params = sum(int(np.prod(s)) if s else 1 for s in remapped.values())

    (out / "params_model_1_ptm_shapes.json").write_text(
        json.dumps(checkpoint, indent=0, sort_keys=True) + "\n"
    )
    (out / "params_model_1_ptm_remapped_shapes.json").write_text(
        json.dumps(remapped, indent=0, sort_keys=True) + "\n"
    )
    print(f"checkpoint arrays {len(checkpoint)} params {npz_params}")
    print(f"remapped keys     {len(remapped)} params {remap_params}")
    print(f"unconsumed params {npz_params - remap_params}")


if __name__ == "__main__":
    main()
