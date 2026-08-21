"""AF2-IG featurizer parity, card-free and checkpoint-free.

Everything here runs off the committed capture
(`scripts/af2_port/parity_artifacts/laczc128_b80/ref_inputs.npz`) plus the committed target crop,
so the suite needs no device, no AF2 checkpoint and no JAX.

The keys most likely to be silently wrong get their own named test on top of the bulk comparison:
a partial capture or a half-built feature dict must not be able to read as a pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "af2_port"))

from tt_bio.af2_data import (  # noqa: E402
    RESTYPE_ATOM14_MASK,
    RESTYPE_ATOM14_TO_ATOM37,
    RESTYPE_ATOM37_MASK,
    RESTYPE_ATOM37_TO_ATOM14,
    add_virtual_cb,
    complex_features,
    initial_recycle_state,
    monomer_features,
    parse_pdb_chain,
)

REF_NPZ = REPO / "scripts" / "af2_port" / "parity_artifacts" / "laczc128_b80" / "ref_inputs.npz"
TARGET_CIF = REPO / "perf" / "pxdesign" / "targets" / "laczc_128.cif"
TARGET_LEN = 128
BINDER_LEN = 80


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    if not REF_NPZ.exists() or not TARGET_CIF.exists():
        pytest.skip("committed AF2-IG parity fixture is absent")
    from af2_fixture import build_fixture

    pdb = tmp_path_factory.mktemp("af2ig") / "complex.pdb"
    return build_fixture(str(TARGET_CIF), str(pdb), BINDER_LEN)


@pytest.fixture(scope="module")
def ref():
    if not REF_NPZ.exists():
        pytest.skip("committed AF2-IG reference capture is absent")
    return np.load(REF_NPZ)


@pytest.fixture(scope="module")
def features(fixture):
    return complex_features(fixture["pdb"], fixture["binder_seq"])


def test_every_featurizer_key_is_bit_exact():
    """The whole bar in one line, via the gate's own scorer: 33 complex keys + 27 monomer."""
    from parity_gate import FEATURE_KEYS, MONOMER_KEYS, featurizer_parity

    report = featurizer_parity()
    assert report["verdict"] == "PASS", report["mismatches"]
    assert (len(FEATURE_KEYS), len(MONOMER_KEYS)) == (33, 27)
    assert report["keys_bitexact"] == report["keys_total"] == 60


def test_capture_reproduces_the_measured_production_run(ref):
    """The capture is the production forward pass, not an approximation of one."""
    import json

    log = json.loads(bytes(ref["_meta/json"]).decode())["log"]
    assert log["plddt"] == pytest.approx(0.7392517, abs=1e-6)
    assert log["i_ptm"] == pytest.approx(0.4886602, abs=1e-6)
    assert log["recycles"] == 3.0


def test_chain_break_jumps_residue_index_by_fifty(features):
    index = features["residue_index"]
    assert index[TARGET_LEN - 1] == 128
    assert index[TARGET_LEN] == 128 + 50 + 1
    within = np.diff(index[:TARGET_LEN])
    assert (within == 1).all()


def test_template_sequence_is_masked_on_both_chains(features):
    """`rm_binder_seq` is in the production config, so the binder is masked too. A featurizer
    that kept the binder's template sequence would still pass a shape check."""
    assert (features["template_aatype"] == 21).all()


def test_template_is_backbone_plus_cb_on_both_chains(features):
    """`rm_target_sc=False` cannot survive `rm_target_seq=True`: `inputs.py:61` forces
    `rm_sc` wherever `rm_seq` is set. Assert the consequence, not the flag."""
    mask = features["template_all_atom_mask"][0]
    assert (mask[:, :5] == 1).all()
    assert (mask[:, 5:] == 0).all()
    # The raw flag really is False on the target -- the no-op is in the derivation, not the input.
    assert not features["rm_template_sc"][:TARGET_LEN].any()
    assert features["rm_template_sc"][TARGET_LEN:].all()
    # ... while the design's own coordinates keep their sidechains.
    assert features["batch/all_atom_mask"][:, 5:].sum() > 0


def test_extra_msa_track_is_masked_off(features):
    assert features["extra_msa_mask"].sum() == 0
    assert features["extra_msa_row_mask"].sum() == 0
    assert features["extra_msa"].shape == (1, TARGET_LEN + BINDER_LEN)


def test_msa_feat_writes_the_one_hot_into_both_blocks(features):
    msa = features["msa_feat"]
    assert np.array_equal(msa[..., 0:22], msa[..., 25:47])
    assert msa[..., 22:25].sum() == 0 and msa[..., 47:].sum() == 0
    one_hot = np.zeros((1, msa.shape[1], 22), np.float32)
    one_hot[0, np.arange(msa.shape[1]), features["aatype"]] = 1.0
    assert np.array_equal(msa[..., 0:22], one_hot)


def test_virtual_cb_is_built_for_glycine(fixture, features):
    """Glycine has no CB in the structure, and the template's pseudo-beta needs one anyway."""
    chain = parse_pdb_chain(fixture["pdb"], "A")
    glycine = chain.aatype == 7
    assert glycine.sum() > 0
    assert chain.mask[glycine, 3].sum() == 0
    filled = add_virtual_cb(chain)
    assert (filled.mask[glycine, 3] == 1).all()
    assert features["template_pseudo_beta_mask"].sum() == len(features["aatype"])


def test_virtual_cb_geometry_needs_float64(fixture, ref):
    """Doing the CB placement in float32 moves it by an ulp and breaks bit-exactness. This is
    the one numeric in the featurizer, so it gets its own test."""
    chain = add_virtual_cb(parse_pdb_chain(fixture["pdb"], "A"))
    glycine = chain.aatype == 7
    assert chain.positions.dtype == np.float64
    exact = chain.positions[glycine, 3].astype(np.float32)
    reference = ref["batch/all_atom_positions"][:TARGET_LEN][glycine, 3]
    assert np.array_equal(exact, reference)

    demoted = add_virtual_cb(
        type(chain)(chain.aatype, chain.positions.astype(np.float32).astype(np.float64),
                    chain.mask, chain.residue_index))
    # Sanity that the test can fail: a float32 round-trip of the *inputs* is harmless, but
    # float32 arithmetic is not. Recompute the same way in float32 and require it to differ.
    positions32 = chain.positions.astype(np.float32)
    unit = lambda x: x / np.sqrt(np.square(x).sum(-1, keepdims=True) + 1e-8)
    c, n, ca = positions32[:, 2], positions32[:, 0], positions32[:, 1]
    bc = unit(n - ca)
    normal = unit(np.cross(n - c, bc))
    naive = ca + np.float32(1.522 * np.cos(1.927)) * bc \
        + np.float32(1.522 * np.sin(1.927) * np.cos(-2.143)) * np.cross(normal, bc) \
        + np.float32(1.522 * np.sin(1.927) * np.sin(-2.143)) * -normal
    assert not np.array_equal(naive[glycine], reference)
    assert np.array_equal(demoted.positions[glycine, 3].astype(np.float32), reference)


def test_unknown_restype_has_no_atoms():
    """AF2 gives `UNK` an empty atom14 list; the vendored ESM table gives it N/CA/C. The row is
    unreachable from a 20-letter design sequence, so a silent vendor bump could move it without
    any other test noticing."""
    assert RESTYPE_ATOM14_MASK[20].sum() == 0
    assert RESTYPE_ATOM37_MASK[20].sum() == 0
    assert (RESTYPE_ATOM14_TO_ATOM37[20] == 0).all()
    assert (RESTYPE_ATOM37_TO_ATOM14[20] == 0).all()
    from tt_bio._vendor.esm.utils import residue_constants as rc

    assert rc.restype_atom14_mask[20].sum() == 3, "ESM's UNK row changed; re-check the pin"
    assert np.array_equal(RESTYPE_ATOM14_MASK[:20], rc.restype_atom14_mask[:20])


def test_initial_guess_seeds_prev_pos_with_the_design(features, ref):
    prev = initial_recycle_state(features)
    assert np.array_equal(prev["prev_pos"], ref["prev/prev_pos"])
    assert np.array_equal(prev["prev_pos"], features["batch/all_atom_positions"])
    assert prev["prev_pair"].sum() == 0 and prev["prev_msa_first_row"].sum() == 0
    assert not initial_recycle_state(features, initial_guess=False)["prev_pos"].any()


def test_monomer_capture_is_the_hallucination_protocol():
    """The monomer stage is a different model, not the complex with templates off. The capture's
    own provenance has to say so, or the 27 monomer keys are scored against the wrong thing."""
    import json

    path = (REPO / "scripts" / "af2_port" / "parity_artifacts" / "laczc128_b80"
            / "ref_inputs_monomer.npz")
    if not path.exists():
        pytest.skip("committed monomer capture is absent")
    meta = json.loads(bytes(np.load(path)["_meta/json"]).decode())
    assert meta["stage"] == "monomer"
    assert meta["production"]["protocol"] == "hallucination"
    assert meta["production"]["use_templates"] is False
    assert meta["production"]["use_initial_guess"] is False
    assert meta["log"]["recycles"] == 3.0
    assert meta["log"]["i_ptm"] == 0.0, "a single chain cannot have an interface pTM"


def test_monomer_stage_is_template_free_and_structure_free(fixture, features):
    """PXDesign's monomer stage is `protocol="hallucination"` with `use_templates=False`
    (`main_af2_monomer.py:120-128`): no PDB, no template, `residue_index` 0-based."""
    monomer = monomer_features(fixture["binder_seq"])
    assert monomer["template_mask"].sum() == 0
    assert monomer["template_aatype"].sum() == 0
    assert not any(k.startswith("batch/") for k in monomer)
    assert np.array_equal(monomer["residue_index"], np.arange(BINDER_LEN, dtype=np.int32))
    assert not monomer["mask_template_interchain"]
    # Same sequence as the complex's binder, or the RMSDs between the stages mean nothing.
    assert np.array_equal(monomer["aatype"], features["aatype"][TARGET_LEN:])
    assert np.array_equal(monomer["msa_feat"], features["msa_feat"][:, TARGET_LEN:])
    assert not initial_recycle_state(monomer)["prev_pos"].any()


def test_binder_sequence_length_must_match_the_chain(fixture):
    with pytest.raises(ValueError, match="binder sequence"):
        complex_features(fixture["pdb"], fixture["binder_seq"][:-1])


def test_insertion_codes_are_rejected(tmp_path):
    pdb = tmp_path / "icode.pdb"
    pdb.write_text(
        "ATOM      1  N   GLY A   1A     0.000   0.000   0.000  1.00  0.00           N\n")
    with pytest.raises(ValueError, match="insertion code"):
        parse_pdb_chain(str(pdb), "A")
