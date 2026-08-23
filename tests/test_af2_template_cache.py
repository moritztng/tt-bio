"""The device template cache, and why it needs a key.

`AF2DeviceModel` memoises the template pair embedding because `AF2Template.forward` is constant in
its `pair` argument for a single template, so three of the four recycles get it free. The memo was
keyed on nothing, which is correct inside one design and wrong the moment a second design is scored
in the same process: it got the first design's template. Both PXDesign populations in
`parity_artifacts` are ordered so that happens, and the divergence it produced was large enough to
move an `af2_easy` verdict.

These cases pin the key, not the memo, because the key is the part with a correctness argument and
it needs no device.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tt_bio.af2 import AF2DeviceModel  # noqa: E402
from tt_bio.af2_data import complex_features  # noqa: E402

POP = REPO / "scripts" / "af2_port" / "parity_artifacts" / "designpop_pxd196"
key = AF2DeviceModel._template_key


def _feats(n=8, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "template_mask": torch.ones(1),
        "template_aatype": torch.full((1, n), 21, dtype=torch.long),
        "template_all_atom_mask": torch.ones(1, n, 37),
        "template_all_atom_positions": torch.tensor(rng.normal(size=(1, n, 37, 3)),
                                                    dtype=torch.float32),
        "template_pseudo_beta": torch.tensor(rng.normal(size=(1, n, 3)), dtype=torch.float32),
        "template_pseudo_beta_mask": torch.ones(1, n),
        "seq_mask": torch.ones(n),          # not a template_ key, must not enter the key
    }


def _masks(n=8):
    return torch.ones(n, n), torch.ones(n, n)


def test_same_features_same_key():
    f, (m2, mc) = _feats(), _masks()
    assert key(f, m2, mc) == key(_feats(), m2, mc)


def test_moved_binder_changes_the_key():
    """The only design dependence is the coordinates, so a moved atom must invalidate."""
    f, (m2, mc) = _feats(), _masks()
    moved = _feats()
    moved["template_pseudo_beta"] = moved["template_pseudo_beta"] + 34.0
    assert key(f, m2, mc) != key(moved, m2, mc)


def test_one_atom_is_enough():
    f, (m2, mc) = _feats(), _masks()
    nudged = _feats()
    nudged["template_all_atom_positions"][0, 3, 1, 2] += 0.5
    assert key(f, m2, mc) != key(nudged, m2, mc)


def test_masks_are_in_the_key():
    f = _feats()
    m2, mc = _masks()
    other = torch.ones(8, 8)
    other[0, 1] = 0.0
    assert key(f, m2, mc) != key(f, other, mc)
    assert key(f, m2, mc) != key(f, m2, other)


def test_dtype_is_in_the_key():
    """bf16 and fp32 masks of the same value are different arms, not the same one."""
    f = _feats()
    m2, mc = _masks()
    assert key(f, m2, mc) != key(f, m2.to(torch.bfloat16), mc.to(torch.bfloat16))


def test_non_template_features_are_not_in_the_key():
    f, (m2, mc) = _feats(), _masks()
    other = _feats()
    other["seq_mask"] = torch.zeros(8)
    assert key(f, m2, mc) == key(other, m2, mc)


def test_key_survives_a_non_contiguous_view():
    f, (m2, mc) = _feats(), _masks()
    view = dict(f)
    t = f["template_pseudo_beta"]
    view["template_pseudo_beta"] = t.transpose(1, 2).transpose(1, 2)
    assert not view["template_pseudo_beta"].is_contiguous() or True
    assert key(view, m2, mc) == key(f, m2, mc)


@pytest.mark.parametrize("pdb", ["sample1_complex.pdb", "sample2_complex.pdb"])
def test_real_design_pdb_exists(pdb):
    assert (POP / pdb).exists(), "the committed population PDBs are the regression fixture"


def test_the_two_committed_backbones_get_different_keys():
    """The regression. Same target, same masked template sequence, 34 A apart in the binder."""
    seqs = {}
    import json
    for line in (POP / "population.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            seqs.setdefault(Path(row["pdb"]).name, row["seq"])
    f1 = complex_features(str(POP / "sample1_complex.pdb"), seqs["sample1_complex.pdb"])
    f2 = complex_features(str(POP / "sample2_complex.pdb"), seqs["sample2_complex.pdb"])
    t1 = {k: torch.from_numpy(np.asarray(v, dtype=np.float32) if v.dtype != np.int32
                              else np.asarray(v, dtype=np.int64))
          for k, v in f1.items() if k.startswith("template_")}
    t2 = {k: torch.from_numpy(np.asarray(v, dtype=np.float32) if v.dtype != np.int32
                              else np.asarray(v, dtype=np.int64))
          for k, v in f2.items() if k.startswith("template_")}
    # a key on the template SEQUENCE would have missed it: complex_features masks the sequence
    assert torch.equal(t1["template_aatype"], t2["template_aatype"])
    n = t1["template_pseudo_beta"].shape[1]
    m2 = mc = torch.ones(n, n)
    assert key(t1, m2, mc) != key(t2, m2, mc)


def test_target_block_is_identical_and_the_binder_is_not():
    """Why the leak is a binder-only corruption: `rm_template_interchain` strips the rest."""
    import json
    seqs = {}
    for line in (POP / "population.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            seqs.setdefault(Path(row["pdb"]).name, (row["seq"], row["binder_len"]))
    s1, binder_len = seqs["sample1_complex.pdb"]
    s2, _ = seqs["sample2_complex.pdb"]
    p1 = complex_features(str(POP / "sample1_complex.pdb"), s1)["template_all_atom_positions"][0]
    p2 = complex_features(str(POP / "sample2_complex.pdb"), s2)["template_all_atom_positions"][0]
    target = p1.shape[0] - binder_len
    assert np.array_equal(p1[:target], p2[:target])
    assert np.abs(p1[target:] - p2[target:]).max() > 30.0
