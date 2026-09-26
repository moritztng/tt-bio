"""A crop is a SUBSET of the structure it is scored against, and `token_index` is the join.

Upstream crops the FEATURES to the token budget and hands back the whole deposited structure.
Measured on the campaign's own 8-target training subset at crop 384: 4ky2 arrives as 384 cropped
tokens against 480 ground-truth ones and 2wig as 384 against 2512. Lining the two up by POSITION
would score the crop's labels against the structure's first 384 tokens, which are different
tokens. Taking only the targets whose counts happen to match would train on the ones that fit,
which is the small pool and not the shipped one.

`token_index` is each token's index in the UNCROPPED structure and both sides carry it, so
matching the values is exact. On a target that fits, the permutation is the identity, and the
join runs unconditionally so that the easy case exercises it too.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ttnn")

from tt_bio.train.openfold3 import OpenFold3Dataset


def _fixture(tmp_path, crop_tokens, gt_tokens, n_real):
    """One `.pt` shaped the way the dataset reads it: a leading batch axis it strips."""
    n = len(crop_tokens)
    b = lambda t: t.unsqueeze(0)
    tok = torch.zeros(n)
    tok[:n_real] = 1.0
    n_atom = 3 * len(gt_tokens)
    xyz = torch.arange(n_atom, dtype=torch.float32).reshape(-1, 1).repeat(1, 3)
    f = {
        "token_mask": b(tok),
        "token_index": b(torch.tensor(crop_tokens, dtype=torch.int64)),
        "is_dna": b(torch.zeros(n)), "is_rna": b(torch.zeros(n)),
        "is_ligand": b(torch.zeros(n)),
        "token_bonds": b(torch.zeros(n, n)),
        "ground_truth": {
            "token_index": b(torch.tensor(gt_tokens, dtype=torch.int64)),
            "start_atom_index": b(torch.arange(0, n_atom, 3, dtype=torch.int64)),
            "atom_positions": b(xyz),
            "atom_resolved_mask": b(torch.ones(n_atom)),
        },
    }
    p = tmp_path / "00_test.pt"
    torch.save(f, p)
    return p


def test_a_strict_subset_lands_on_the_tokens_it_names(tmp_path):
    # crop keeps original tokens 10, 12, 15, 17 of a structure holding 10..20; two pad rows.
    _fixture(tmp_path, [10, 12, 15, 17, 0, 0], [10, 11, 12, 15, 17, 20], n_real=4)
    out = OpenFold3Dataset(tmp_path).batch([0])
    # gt rows 0, 2, 3, 4 -> representative atoms 0, 6, 9, 12, whose coordinates are the index.
    assert out["true_xyz"][:4, 0].tolist() == [0.0, 6.0, 9.0, 12.0]
    assert out["true_xyz"][4:].sum() == 0.0, "a padded row took a label"
    assert out["coord_mask"].tolist() == [1, 1, 1, 1, 0, 0]


def test_a_target_that_fits_is_the_identity(tmp_path):
    _fixture(tmp_path, [4, 5, 6, 7], [4, 5, 6, 7], n_real=4)
    out = OpenFold3Dataset(tmp_path).batch([0])
    assert out["true_xyz"][:, 0].tolist() == [0.0, 3.0, 6.0, 9.0]


def test_a_crop_naming_a_token_the_structure_does_not_have_is_refused(tmp_path):
    _fixture(tmp_path, [10, 99, 15, 17, 0, 0], [10, 11, 12, 15, 17, 20], n_real=4)
    with pytest.raises(ValueError, match="not in ground_truth by token_index"):
        OpenFold3Dataset(tmp_path).batch([0])
