#!/usr/bin/env python3
"""Which bin grid is ESMFold-2's distogram output on? Settled on a real fold.

ESMFold-2's inference code never states its output grid. Two candidates:
  conditioning  linspace(2, 22, 65) boundaries -> breaks linspace(2.3125, 21.6875, 63)
                (the grid its distogram-conditioning input bins with; AF3/OF3's grid)
  boltz-style   breaks linspace(2, 22, 63)
For every token pair whose distogram is confident (modal bin probability >= 0.5) this checks
whether the predicted Cb distance (Ca for glycine) in the folded structure falls inside the
modal bin under each grid, and how far the distogram's expected distance is from it. The right
grid puts the distance in the modal bin far more often and with a smaller mean error.

    TT_VISIBLE_DEVICES=<card> python perf/fdx_confidence/esmfold2_grid_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
from tt_bio.esmfold2_runtime import fold_complex, load_ttnn_esmfold2  # noqa: E402

GRIDS = {"conditioning": np.linspace(2.3125, 21.6875, 63), "boltz-style": np.linspace(2, 22, 63)}


def main():
    doc = yaml.safe_load((ROOT / "examples/9bk6.yaml").read_text())
    chains = [(s["protein"]["id"], s["protein"]["sequence"], None, "protein", [])
              for s in doc["sequences"]]
    model = load_ttnn_esmfold2()
    res = fold_complex(model, chains, seed=0)
    logits = res.distogram.float()                                   # [L, L, 64]
    pc = res.complex.to_protein_complex()
    xyz = np.asarray(pc.atom37_positions, dtype=np.float64)          # [L, 37, 3]
    seq = "".join(c[1] for c in chains)
    rep = np.where(np.array([a == "G" for a in seq])[:, None], xyz[:, 1], xyz[:, 3])
    d = np.linalg.norm(rep[:, None] - rep[None], axis=-1)
    p = torch.softmax(logits, -1).numpy()
    mode = p.argmax(-1)
    sure = (p.max(-1) >= 0.5) & ~np.eye(len(d), dtype=bool)
    out = {"n_tokens": int(len(d)), "n_confident_pairs": int(sure.sum())}
    for name, br in GRIDS.items():
        lo = np.concatenate([[-np.inf], br])[mode]
        hi = np.concatenate([br, [np.inf]])[mode]
        inside = (d > lo) & (d <= hi)
        centers = np.concatenate([[br[0] - (br[1] - br[0]) / 2], (br[1:] + br[:-1]) / 2,
                                  [br[-1] + (br[1] - br[0]) / 2]])
        expect = (p * centers).sum(-1)
        near = sure & (d < 21)
        out[name] = {"modal_bin_contains_distance": round(float(inside[sure].mean()), 4),
                     "mean_abs_expected_minus_distance_A": round(
                         float(np.abs(expect - d)[near].mean()), 4)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
