"""place_ref_conformers reproduces upstream OpenDDE's reference-conformer placement (centre, then seeded random
translation and rotation per ref_space_uid) draw for draw. Needs an OpenDDE v1.2.0 checkout (OPENDDE_SRC)."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tt_bio.protenix_data import place_ref_conformers

GEOM = Path(os.environ.get("OPENDDE_SRC", "/home/moritz/tfg-ref/src/opendde")) / "opendde/utils/geometry.py"


def _feats(n_res=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    sizes = torch.randint(4, 15, (n_res,), generator=g)
    uid = torch.repeat_interleave(torch.arange(n_res), sizes)
    pos = torch.randn(int(sizes.sum()), 3, generator=g) * 2 + 5
    return {"ref_pos": pos, "ref_space_uid": uid}


def test_centred_without_seed():
    f = _feats()
    out = place_ref_conformers(f, None)["ref_pos"]
    for u in f["ref_space_uid"].unique():
        m = f["ref_space_uid"] == u
        assert out[m].mean(0).abs().max() < 1e-5
        assert torch.allclose(out[m] - out[m][0], f["ref_pos"][m] - f["ref_pos"][m][0], atol=1e-5)


@pytest.mark.skipif(not GEOM.exists(), reason="no OpenDDE checkout")
def test_matches_upstream_draws():
    spec = importlib.util.spec_from_file_location("up_geometry", GEOM)
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    f = _feats()
    pos, uid = f["ref_pos"].numpy(), f["ref_space_uid"].numpy()
    for seed in (101, 102):
        np.random.seed(seed)
        ref = np.empty_like(pos)
        for u in np.unique(uid):
            m = uid == u
            ref[m] = up.random_transform(pos[m], apply_augmentation=True, centralize=True)
        ours = place_ref_conformers(f, seed)["ref_pos"]
        assert torch.allclose(ours, torch.Tensor(ref), atol=1e-6), seed
    assert not torch.allclose(place_ref_conformers(f, 101)["ref_pos"], place_ref_conformers(f, 102)["ref_pos"])
