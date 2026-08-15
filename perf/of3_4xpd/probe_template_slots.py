#!/usr/bin/env python3
"""K-T1: are OpenFold3's four template slots byte-identical for a template-free query?

The page fixture (perf/size512/fixtures/cdk2x2_512.yaml, one protein chain, templates off) still
runs the template embedder, and n_templates defaults to 4
(_vendor/openfold3/projects/of3_all_atom/config/dataset_config_components.py:124). For a query with
no templates, create_template_feature_precursor_of3 fills every slot with "GAP" / NaN and overwrites
nothing, so every downstream per-slot feature should come out identical. This asserts that on the
real vendored primitives rather than assuming it -- it is the one premise the template-dedup lever
rests on, and it costs no device time.

    PYTHONPATH=<worktree> python3 perf/of3_4xpd/probe_template_slots.py
"""
import numpy as np, torch
from tt_bio._vendor.openfold3.core.data.primitives.featurization.template import (
    create_template_restype, create_template_distogram, create_template_unit_vector)
from tt_bio.openfold3_host_prep import derive_template_feat

NT, N = 4, 512
res_names = np.full((NT, N), "GAP", dtype=np.dtype("U3"))
pb = np.full((NT, N, 3), np.nan, dtype=float)
fr = np.full((NT, N, 3, 3), np.nan, dtype=float)
asym = torch.ones(N, dtype=torch.int32)
mcm = (asym[..., None] == asym[..., None, :])[..., None, :, :, None]

feats = {
    "template_pseudo_beta_mask": torch.tensor(~np.isnan(pb).any(axis=-1), dtype=torch.float),
    "template_backbone_frame_mask": torch.tensor(~np.isnan(fr).any(axis=(-2, -1)), dtype=torch.float),
    "asym_id": asym,
}
feats["template_restype"] = create_template_restype(res_names, feats["template_pseudo_beta_mask"])
feats["template_distogram"] = create_template_distogram(
    pb, feats["template_pseudo_beta_mask"], mcm, 3.25, 50.75, 39)
feats["template_unit_vector"] = create_template_unit_vector(
    fr, feats["template_backbone_frame_mask"], mcm)

ok = True
for k, t in derive_template_feat(feats).items():
    same = all(torch.equal(t[0], t[i]) for i in range(1, t.shape[0]))
    ok &= same
    print(f"{k:32s} {tuple(t.shape)} slots_identical={same}")
print("ALL_SLOTS_IDENTICAL =", bool(ok))
assert ok, "template slots differ -- the dedup lever is dead on this input"
