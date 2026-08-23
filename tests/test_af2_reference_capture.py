"""Assert the committed AF2-IG reference capture has the right *content*, not just the right keys.

`scripts/af2_port/parity_artifacts/laczc128_b80/ref_inputs.npz` is what the tt-bio featurizer is
scored bit-exact against, so a partial or mis-flagged capture would make the featurizer gate
agree with the wrong thing. These tests pin the features that encode a production flag, by name,
because a capture that silently lost one of them still has all 79 keys and still loads.

Two of them came out the opposite way from the port plan, so they are pinned here with the
mechanism rather than the expectation:

- `rm_target_sc=False` is a **no-op** in production. `colabdesign/af/inputs.py:61` computes
  `rm_sc = where(rm_seq, True, rm_template_sc)`, so removing the template sequence removes the
  sidechains too. Both chains set `rm_*_seq=True`, so the template is backbone+CB only
  everywhere, even though `rm_template_sc` is False on all 128 target residues.
- `template_aatype` is 21 on the binder as well as the target, because the production config sets
  `rm_binder_seq=True` alongside `rm_target_seq=True`.

Card-free, and the artifact is 103 KB.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ARTIFACT = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "af2_port" / "parity_artifacts" / "laczc128_b80" / "ref_inputs.npz"
)
TARGET, BINDER = 128, 80
CB_ATOMS = 5  # atom37 indices 0:5 are N, CA, C, CB, O; 5: is sidechain beyond CB.


@pytest.fixture(scope="module")
def ref():
    if not ARTIFACT.is_file():
        pytest.skip(f"{ARTIFACT} not present")
    with np.load(ARTIFACT) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def meta(ref):
    return json.loads(bytes(ref["_meta/json"]).decode())


def test_capture_reproduces_the_measured_production_scalars(meta):
    """The predecessor measured pLDDT 0.739 / i_pTM 0.489 on this fixture with upstream's own
    harness. The capture has to land on the same numbers or it is not the production run."""
    log = meta["log"]
    assert log["plddt"] == pytest.approx(0.739, abs=5e-4)
    assert log["i_ptm"] == pytest.approx(0.489, abs=5e-4)
    assert log["recycles"] == 3.0


def test_production_flags_are_the_ones_pxdesign_uses(meta):
    assert meta["production"] == {
        "num_recycles": 3, "protocol": "binder", "use_initial_atom_pos": False,
        "use_initial_guess": True, "use_multimer": False,
    }
    assert meta["prep"] == {
        "rm_target_sc": False, "rm_target_seq": True, "rm_template_ic": True,
        "use_binder_template": True,
    }
    assert meta["fixture"]["n_tokens"] == TARGET + BINDER


def test_chain_break_is_a_plus_fifty_residue_index_jump(ref):
    diffs = set(np.diff(ref["residue_index"]).tolist())
    assert diffs == {1, 51}, "the only index steps should be +1 within a chain and +51 at the break"
    assert ref["residue_index"][TARGET] - ref["residue_index"][TARGET - 1] == 51


def test_chains_are_split_by_asym_id(ref):
    asym = ref["asym_id"]
    assert int((asym == 0).sum()) == TARGET
    assert int((asym == 1).sum()) == BINDER
    assert bool(ref["mask_template_interchain"]) is True


def test_template_sequence_is_removed_on_both_chains(ref):
    """`rm_target_seq` and `rm_binder_seq` are both True, so every template residue is 21."""
    assert set(ref["template_aatype"][0].tolist()) == {21}


def test_template_has_no_sidechains_on_either_chain(ref):
    """rm_seq forces rm_sc (inputs.py:61), so rm_target_sc=False changes nothing."""
    mask = ref["template_all_atom_mask"][0]
    assert mask[:, CB_ATOMS:].sum() == 0, "template should be backbone+CB only"
    assert mask[:, :CB_ATOMS].sum() == (TARGET + BINDER) * CB_ATOMS
    # The flag really is False, so this is the forcing at inputs.py:61 and not a mis-set flag.
    assert ref["rm_template_sc"][:TARGET].sum() == 0
    assert ref["rm_template_sc"][TARGET:].sum() == BINDER
    # The underlying structure does have sidechains, so the zeros above are a mask, not missing
    # coordinates. Without this the test would pass on a backbone-only input file.
    assert ref["batch/all_atom_mask"][:, CB_ATOMS:].sum() > 0


def test_extra_msa_track_is_masked_off(ref):
    """The port plan's extra-MSA simplification rests on this being exactly zero."""
    assert ref["extra_msa_mask"].sum() == 0
    assert ref["extra_msa"].shape[0] == 1


def test_msa_is_single_sequence_and_fully_unmasked(ref):
    assert ref["msa_feat"].shape == (1, TARGET + BINDER, 49)
    assert ref["msa_mask"].sum() == TARGET + BINDER


def test_msa_feat_carries_the_sequence_in_two_blocks_only(ref):
    """`update_seq` writes the one-hot at 0:22 and the pssm at 25:47, and the sequence is 20-wide,
    so channels 20, 21, 45, 46 and everything from 47 stay zero."""
    nonzero = set(np.nonzero(ref["msa_feat"][0].sum(0))[0].tolist())
    assert nonzero == set(range(20)) | set(range(25, 45))
    assert np.allclose(ref["target_feat"].sum(1), 1.0)


def test_initial_guess_seeds_prev_pos_only(ref):
    """`use_initial_guess=True` puts the design coordinates in prev_pos; prev_pair and
    prev_msa_first_row start at zero (design.py:163-171)."""
    assert int((np.abs(ref["prev/prev_pos"]).sum((1, 2)) > 0).sum()) == TARGET + BINDER
    assert np.abs(ref["prev/prev_pair"]).max() == 0
    assert np.abs(ref["prev/prev_msa_first_row"]).max() == 0
