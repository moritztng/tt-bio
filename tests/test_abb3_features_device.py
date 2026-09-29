"""The on-card feature expansion against the untouched reference.

``abodybuilder3_reference.single_and_pair_features`` is the float64 reference every accuracy
number in this campaign is scored against, and it is deliberately not ported: the device path is
a second implementation that gets checked against it here, so the reference stays a reference and
the check cannot drift.

**The bar is exact equality, not a PCC**, and that is not this campaign's usual rule. A one-hot of
integers has no approximation to trade -- every value is exactly 0.0 or 1.0 and both are exact in
every float format -- so a device expansion that differs from the reference is wrong rather than
imprecise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tt_bio.abodybuilder3_reference import ABB3Config, single_and_pair_features
from tt_bio.train.abb3_features_device import AATYPE_DIM, CHAIN_PAIR_DIM

ROOT = Path(__file__).resolve().parents[1]
REAL_STRUCTURES = Path("/home/ttuser/abb3_data/data/structures/structures")
REAL_SPLIT = Path("/home/ttuser/abb3_src/ABodyBuilder3/data/split.csv")


def _device_available():
    return bool(os.environ.get("TT_VISIBLE_DEVICES")) and os.path.exists("/dev/tenstorrent")


needs_device = pytest.mark.skipif(not _device_available(),
                                  reason="needs a TT card and TT_VISIBLE_DEVICES pinned to it")


def _indices(b=2, n=64, seed=0):
    """Index maps at the shapes the assembler produces, including a padded tail."""
    g = torch.Generator().manual_seed(seed)
    aatype = torch.randint(0, 21, (b, n), generator=g)
    is_heavy = torch.zeros(b, n, dtype=torch.int64)
    residue_index = torch.zeros(b, n, dtype=torch.int64)
    for j in range(b):
        nl, nh = 20 + j, 25 - j
        is_heavy[j, nl:nl + nh] = 1
        residue_index[j, :nl] = torch.arange(nl)
        residue_index[j, nl:] = torch.arange(n - nl)
    return aatype, is_heavy, residue_index


def test_the_channel_offsets_match_the_reference_it_is_checked_against():
    """The two constants this module owns, read off the reference rather than assumed.

    They are the only numbers duplicated from `single_and_pair_features`, so they are the only
    way the device path can drift from it without a value ever disagreeing.
    """
    aatype, is_heavy, residue_index = _indices()
    single, pair = single_and_pair_features(aatype, is_heavy, residue_index)
    cfg = ABB3Config()
    assert single.shape[-1] == cfg.c_s and pair.shape[-1] == cfg.c_z
    # The amino-acid one-hot occupies [0, AATYPE_DIM) and the chain one-hot the rest.
    assert int(single[..., :AATYPE_DIM].sum()) == aatype.numel()
    assert int(single[..., AATYPE_DIM:].sum()) == aatype.numel()
    # The chain-PAIR one-hot occupies [0, CHAIN_PAIR_DIM) and relative position the rest.
    assert int(pair[..., :CHAIN_PAIR_DIM].sum()) == residue_index.shape[0] * residue_index.shape[1] ** 2
    assert int(pair[..., CHAIN_PAIR_DIM:].sum()) == residue_index.shape[0] * residue_index.shape[1] ** 2


@needs_device
@pytest.mark.parametrize("b,n", [(2, 64), (4, 256)])
def test_the_card_builds_exactly_what_the_reference_builds(b, n):
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.train.abb3_features_device import input_features_device

    aatype, is_heavy, residue_index = _indices(b, n, seed=b)
    ref_single, ref_pair = single_and_pair_features(aatype, is_heavy, residue_index)
    single_d, pair_d = input_features_device(aatype, is_heavy, residue_index,
                                             device=get_device())
    got_single = ttnn.to_torch(single_d).reshape(ref_single.shape)
    got_pair = ttnn.to_torch(pair_d).reshape(ref_pair.shape)
    assert torch.equal(got_single, ref_single), (got_single != ref_single).sum()
    assert torch.equal(got_pair, ref_pair), (got_pair != ref_pair).sum()


@needs_device
@pytest.mark.skipif(not (REAL_STRUCTURES.is_dir() and REAL_SPLIT.is_file()),
                    reason="the SAbDab structures are not staged on this host")
def test_the_assembler_uploads_exactly_the_reference_on_real_fvs():
    """The whole seam, on upstream's own structures: `host()` then `upload()` against the
    reference applied to the same index maps."""
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.train.abb3_dataset import SabdabFvs, resolve_split

    cfg = ABB3Config(use_plddt=False, no_blocks=8)
    ids = resolve_split(REAL_SPLIT)["train"]
    ds = SabdabFvs(ids, REAL_STRUCTURES, cfg, get_device(), tokens=256)
    hb = ds.host([0, 1, 2, 3])
    ref_single, ref_pair = single_and_pair_features(hb["aatype"], hb["is_heavy"],
                                                    hb["residue_index"])
    sample = ds.upload(hb)
    assert torch.equal(ttnn.to_torch(sample["single_d"]).reshape(ref_single.shape), ref_single)
    assert torch.equal(ttnn.to_torch(sample["pair_d"]).reshape(ref_pair.shape), ref_pair)


def test_the_reference_is_untouched():
    """`state/train/STATUS.md` marks `abodybuilder3_reference.py` nobody-territory, permanently.

    This module exists because of that line: the port goes beside the reference rather than into
    it, so the float64 reference every accuracy number is scored against keeps being able to
    falsify the device path.
    """
    src = (ROOT / "tt_bio/abodybuilder3_reference.py").read_text()
    assert "def single_and_pair_features" in src
    assert "ttnn" not in src, "the reference must stay a host-only float reference"
