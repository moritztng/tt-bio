"""The alignment rows each recycling cycle reads, row for row against the references on the CPU
generator: msa_cycle_rows is OpenDDE's MSAModule (model/msa_sampling.py
subsample_msa_feature_dict_valid_first, needs an OpenDDE v1.2.0 checkout, OPENDDE_SRC) and
msa_cycle_rows_random is Protenix v2.0.0's (protenix/model/utils.py sample_indices, PROTENIX_SRC).
Also the chunk sizes and OPM depth buckets that keep a per-cycle depth from recompiling."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from tt_bio.protenix import msa_cycle_rows, msa_cycle_rows_random
from tt_bio.tenstorrent import msa_row_bucket, msa_rows_host

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


# Protenix v2.0.0's rule (protenix/model/utils.py sample_indices, strategy "random", lower bound 1).
PROTENIX_UTILS = Path(os.environ.get("PROTENIX_SRC", "/home/moritz/.coworker/ref/Protenix")) / "protenix/model/utils.py"


def test_protenix_depth_uniform_and_rows_distinct():
    rows = msa_cycle_rows_random(50, 2000, torch.Generator().manual_seed(3))
    ks = torch.tensor([len(r) for r in rows])
    assert ks.min() == 1 and ks.max() == 50 and abs(ks.float().mean().item() - 25.5) < 1.0
    assert all(len(set(r.tolist())) == len(r) and r.max() < 50 for r in rows)


def test_protenix_cutoff():
    assert max(len(r) for r in msa_cycle_rows_random(100, 50, torch.Generator().manual_seed(1), cutoff=7)) == 7


@pytest.mark.skipif(not PROTENIX_UTILS.exists(), reason="no Protenix checkout")
@pytest.mark.parametrize("n", [1, 37, 9947])
def test_protenix_matches_upstream(n):
    """Upstream draws from torch's global generator; seeded alike, our CPU generator gives its rows."""
    src = PROTENIX_UTILS.read_text()
    ns = {"torch": torch}
    exec(src[src.index("def sample_indices"):src.index("def sample_msa_feature_dict_random_without_replacement")], ns)
    ours = msa_cycle_rows_random(n, 10, torch.Generator().manual_seed(101))
    torch.manual_seed(101)
    for r in ours:
        ref = ns["sample_indices"](n=n, lower_bound=1, strategy="random")[:16384]
        assert torch.equal(ref, r)


def test_row_buckets_few_and_tight():
    for n in (1, 37, 1000, 9947, 16384):
        seen = set()
        for d in range(1, n + 1):
            b = msa_row_bucket(d, n)
            assert d <= b < d + 1024 and b % 32 == 0
            seen.add(b)
        assert len(seen) <= -(-n // 1024)


def test_rows_host_pads_and_masks():
    feat = torch.randn(1, 50, 7, 3).to(torch.bfloat16)
    rows = torch.tensor([4, 0, 49])
    x, mask = msa_rows_host(feat, rows, 64)
    assert x.shape == (1, 64, 7, 3) and mask.shape == (64, 1, 1)
    assert torch.equal(x[:, :3], feat[:, rows]) and not x[:, 3:].any()
    assert mask[:3].eq(1).all() and not mask[3:].any()
