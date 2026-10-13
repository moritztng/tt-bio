"""AF2-IG token bucketing (TT_BIO_AF2_BUCKET), on host, with a tiny random-weight AF2Model.

The flag pads every token axis of the feature and recycle dicts to the next multiple of 32 with
the masks zero on the padded rows, runs the whole model at that width and cuts each pass's output
back to the real residues. Two properties are what make that shippable, and both are checked here
against the torch reference model (no card, no checkpoint):

* at an already aligned length the flag changes nothing, bit for bit;
* at a ragged length the padded fold equals the unpadded one on the real residues, up to float
  reassociation, which is the statement that AF2's own masks keep the padding off the real rows.

Whether the DEVICE trunk keeps that property at a padded width is the A/B this does not replace.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tt_bio import af2ig  # noqa: E402
from tt_bio.af2_data import initial_recycle_state, pad_token_axis  # noqa: E402
from tt_bio.af2_reference import AF2Model  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "af2ig" / "designed_complex.pdb"
EXAMPLE = REPO / "examples" / "af2_designed_complex.yaml"


def _tiny_model(seed: int = 0) -> AF2Model:
    model = AF2Model(num_evoformer_blocks=1, num_extra_msa_blocks=1, trunk_dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return model.eval()


def _fold(monkeypatch, spec, flag: bool, recycles: int = 1):
    monkeypatch.setenv("TT_BIO_AF2_BUCKET", "1" if flag else "0")
    return af2ig.fold(_tiny_model(), spec, recycles=recycles)


def _aligned_spec(tmp_path) -> af2ig.AF2IGInput:
    """A 64-residue complex: the fixture's 32-residue target, and a copy of it as the binder."""
    lines = [ln for ln in FIXTURE.read_text().splitlines()
             if ln.startswith("ATOM") and ln[21] == "A"]
    chain_b = [ln[:21] + "B" + ln[22:] for ln in lines]
    pdb = "\n".join(lines + ["TER"] + chain_b + ["END"]) + "\n"
    from tt_bio._vendor.esm.utils import residue_constants as rc
    seen, seq = set(), []
    for ln in lines:
        key = ln[22:27]
        if key not in seen:
            seen.add(key)
            seq.append(rc.restype_3to1.get(ln[17:20], "X"))
    return af2ig.AF2IGInput(structure=pdb, binder_sequence="".join(seq),
                            target_chain="A", binder_chain="B")


def test_the_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("TT_BIO_AF2_BUCKET", raising=False)
    assert not af2ig.bucket_on()
    assert af2ig.bucket_width(195) == 195
    monkeypatch.setenv("TT_BIO_AF2_BUCKET", "1")
    assert af2ig.bucket_width(195) == 224 and af2ig.bucket_width(224) == 224
    monkeypatch.setenv("TT_BIO_TOKEN_BUCKET", "0")
    assert af2ig.bucket_width(195) == 195, "the fleet-wide off switch must win"


def test_pad_token_axis_is_identity_at_width_and_zero_masked_beyond():
    spec = af2ig.read_af2ig_input(EXAMPLE)
    feats = af2ig.features(spec)
    n = len(feats["residue_index"])
    feats.update(initial_recycle_state(feats, initial_guess=True))
    same = pad_token_axis(feats, n)
    assert all(same[k] is feats[k] for k in feats)
    padded = pad_token_axis(feats, 64)
    assert padded["seq_mask"].shape == (64,) and padded["seq_mask"][n:].sum() == 0
    assert padded["msa_mask"].shape == (1, 64) and padded["msa_mask"][:, n:].sum() == 0
    assert padded["template_all_atom_mask"][:, n:].sum() == 0
    assert padded["atom37_atom_exists"][n:].sum() == 0
    assert padded["prev_pair"].shape == (64, 64, 128)
    for k in feats:
        np.testing.assert_array_equal(
            padded[k][tuple(slice(0, s) for s in np.shape(feats[k]))], feats[k])
    with pytest.raises(KeyError):
        pad_token_axis({"something_new": np.zeros(n)}, 64)


def test_aligned_length_is_a_bit_identical_no_op(monkeypatch, tmp_path):
    spec = _aligned_spec(tmp_path)
    assert af2ig.token_count(spec) == 64
    off = _fold(monkeypatch, spec, False)
    on = _fold(monkeypatch, spec, True)
    assert on.tokens == off.tokens == 64
    np.testing.assert_array_equal(on.coords, off.coords)
    np.testing.assert_array_equal(on.b_factors, off.b_factors)
    np.testing.assert_array_equal(on.pae, off.pae)
    np.testing.assert_array_equal(on.distogram, off.distogram)
    assert on.metrics == off.metrics


def test_padded_fold_matches_the_ragged_one_on_real_residues(monkeypatch):
    spec = af2ig.read_af2ig_input(EXAMPLE)
    assert af2ig.token_count(spec) == 48          # pads to 64
    off = _fold(monkeypatch, spec, False, recycles=2)
    on = _fold(monkeypatch, spec, True, recycles=2)
    assert np.isfinite(off.coords).all() and np.isfinite(on.coords).all()
    assert on.tokens == off.tokens == 48
    assert on.pae.shape == off.pae.shape == (48, 48)
    assert on.distogram.shape == off.distogram.shape
    np.testing.assert_allclose(on.coords, off.coords, atol=1e-3, rtol=0)
    np.testing.assert_allclose(on.b_factors, off.b_factors, atol=2e-4, rtol=0)
    np.testing.assert_allclose(on.pae, off.pae, atol=2e-4, rtol=0)
    for k in ("plddt", "ptm", "iptm", "pae", "ipae", "interface_pae"):
        assert on.metrics[k] == pytest.approx(off.metrics[k], abs=1e-4), k


def test_the_comparison_catches_a_leaking_mask(monkeypatch):
    """Negative control: pad with the masks left ON and the same comparison must fail, or the
    test above would pass on a model whose outputs simply do not depend on the padding."""
    import tt_bio.af2_data as af2_data
    spec = af2ig.read_af2ig_input(EXAMPLE)
    off = _fold(monkeypatch, spec, False, recycles=2)
    real = af2_data.pad_token_axis

    def leaky(arrays, width):
        out = real(arrays, width)
        if "seq_mask" in out:
            out["seq_mask"][:] = 1
            out["msa_mask"][:] = 1
        return out

    monkeypatch.setattr(af2_data, "pad_token_axis", leaky)
    on = _fold(monkeypatch, spec, True, recycles=2)
    assert np.abs(on.coords - off.coords).max() > 1.0
    assert np.abs(on.pae - off.pae).max() > 2e-4
