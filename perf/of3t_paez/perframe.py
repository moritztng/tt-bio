#!/usr/bin/env python3
"""of3t-paez: where each side's pae label flips sit, per frame token, and its per-token residual."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_fullstep64"))
import ref_step  # noqa: E402
from tt_bio.train import losses  # noqa: E402

BATCH = Path("/home/ttuser/of3t-campaign-refs/bundle_min_043/batch_step003.pt")
preds = torch.load(sys.argv[1], weights_only=False)
lab = ref_step.labels(BATCH)
fai = np.asarray(lab["frame_atom_index"])
m = np.asarray(lab["coord_mask"], bool)
tf = np.asarray(lab["true_xyz"], np.float64)
pm = (m[fai].sum(-1) >= 3)[:, None] & m[None, :]


def bins(px):
    px = px.numpy()
    sq = ((losses.express_in_frame(px, px[..., fai, :])
           - losses.express_in_frame(tf, tf[..., fai, :])) ** 2).sum(-1)
    return np.where(pm, losses.bin_index(sq * pm, *losses.PAE_GRID, n_edges=65, square=True), -1)


b0 = bins(preds["f64"])
real = np.nonzero(m)[0]
rec = {"real": real.tolist(), "frame_atom_index_real": fai[real].tolist(), "sides": {}}
for k in ("dev", "bf16", "f64_at_dev_inputs"):
    b = bins(preds[k])
    fl = (b != b0) & pm
    per_frame = fl.sum(1)[real]
    signed = np.where(pm, b - b0, 0).sum(1)[real]
    res = (preds[k] - preds["f64"]).norm(dim=-1).numpy()[real]
    top = np.argsort(-per_frame)[:8]
    rec["sides"][k] = {"flips": int(fl.sum()),
                       "top8_frames": [[int(real[i]), int(per_frame[i]), int(signed[i])] for i in top],
                       "top8_share": float(per_frame[top].sum() / max(fl.sum(), 1)),
                       "per_token_resid_A_top8": sorted([[int(real[i]), round(float(res[i]), 4)]
                                                         for i in np.argsort(-res)[:8]], key=lambda x: -x[1])}
print(json.dumps(rec["sides"], indent=1))
print("frames of real tokens (first 10):", rec["frame_atom_index_real"][:10])
json.dump(rec, open(sys.argv[2], "w"), indent=1)
