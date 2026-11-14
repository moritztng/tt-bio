"""Upstream reference for tt_bio.confidence_export.contact_probs.

Draws distogram logits on two real grids and records upstream OpenDDE's own
compute_contact_prob (aurekaresearch/OpenDDE, opendde/model/sample_confidence.py) in float64.
Run with the stock OpenDDE source on PYTHONPATH:
    PYTHONPATH=<opendde>/stock/src python perf/fdx_confidence/make_contact_fixture.py
"""
from pathlib import Path

import numpy as np
import torch
from opendde.model.sample_confidence import compute_contact_prob

out = {}
g = torch.Generator().manual_seed(0)
for tag, (lo, hi, nb) in {"opendde": (2.25, 25.75, 96), "boltz2": (2.0, 22.0, 64)}.items():
    z = torch.randn(37, 37, nb, generator=g, dtype=torch.float64) * 3.0
    z = z + z.transpose(0, 1)                      # every head here symmetrises its logits
    out[f"{tag}_logits"] = z.numpy()
    for thr in (8.0, 12.0):
        out[f"{tag}_ref_{thr:g}"] = compute_contact_prob(z, lo, hi, nb, thres=thr).numpy()
dst = Path(__file__).resolve().parents[2] / "tests/fixtures/contact_probs_opendde_ref.npz"
np.savez_compressed(dst, **out)
print(dst, {k: v.shape for k, v in out.items()})
