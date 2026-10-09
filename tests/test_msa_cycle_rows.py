"""msa_cycle_rows draws the rows upstream OpenDDE's MSAModule reads each recycling cycle
(model/msa_sampling.py subsample_msa_feature_dict_valid_first), row for row on the CPU generator.
Needs an OpenDDE v1.2.0 checkout (OPENDDE_SRC)."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from tt_bio.protenix import msa_cycle_rows

SAMPLER = Path(os.environ.get("OPENDDE_SRC", "/home/moritz/tfg-ref/src/opendde")) / "opendde/model/msa_sampling.py"


def _msa(rows=300, cols=40, gap_rows=60):
    g = torch.Generator().manual_seed(7)
    msa = torch.randint(0, 31, (rows, cols), generator=g)
    msa[torch.randperm(rows, generator=g)[:gap_rows]] = 31
    return msa


def test_whole_alignment_when_it_fits():
    assert msa_cycle_rows(_msa(rows=100), 10, 128, torch.Generator().manual_seed(1)) is None


def test_valid_rows_first_and_fresh_per_cycle():
    msa = _msa()
    rows = msa_cycle_rows(msa, 3, 256, torch.Generator().manual_seed(1))
    valid = (msa != 31).any(-1)
    for r in rows:
        assert len(set(r.tolist())) == 256 and valid[r[:240]].all() and not valid[r[240:]].any()
    assert not torch.equal(rows[0], rows[1])


@pytest.mark.skipif(not SAMPLER.exists(), reason="no OpenDDE checkout")
@pytest.mark.parametrize("depth", [100, 256])
def test_matches_upstream(depth):
    spec = importlib.util.spec_from_file_location("up_msa_sampling", SAMPLER)
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    msa = _msa()
    ours = msa_cycle_rows(msa, 4, depth, torch.Generator().manual_seed(101))
    torch.manual_seed(101)
    for r in ours:
        ref = up.subsample_msa_feature_dict_valid_first({"msa": msa}, {"msa": -2}, num_msa=depth,
                                                        msa_mask=torch.ones_like(msa), gap_token=31)
        assert torch.equal(ref["msa"], msa[r])
