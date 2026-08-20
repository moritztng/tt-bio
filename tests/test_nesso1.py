"""Nesso-1 port: the de-hardcodes must not move Boltz-2, and bad input must fail loudly.

``tt_bio/nesso1.py`` reuses Boltz-2's ``AtomEncoder``, ``InputEmbedder`` and
``AffinityHeadsTransformer``, which needed three new keyword arguments: Nesso-1's
atom-feature concat order (390 dims, charge after the name chars), no MSA profile
track, and the pooled affinity representation. All three default to Boltz-2's
existing behaviour, and these tests pin that against the pre-change source pulled
straight out of git, so a future edit that changes the default fails here.

Device-free, checkpoint-free, CPU only.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent

# the commit that introduced the de-hardcode; its parent is the reference source
DEHARDCODE_COMMIT = "1cee2d27"

B, M, N = 1, 64, 8  # M must be a multiple of atoms_per_window_queries
ATOM_FEATURE_DIM = 3 + 1 + 128 + 4 * 64  # Boltz-2's own layout
CFG = dict(
    atom_s=32, atom_z=16, token_s=64, token_z=32,
    atoms_per_window_queries=32, atoms_per_window_keys=128,
    atom_feature_dim=ATOM_FEATURE_DIM, atom_encoder_depth=2, atom_encoder_heads=2,
)
ATOM_ENCODER_CFG = {k: v for k, v in CFG.items()
                    if k not in ("atom_encoder_depth", "atom_encoder_heads")}


@pytest.fixture(scope="module")
def old_boltz2():
    """boltz2.py as it was before the de-hardcode, imported as its own module."""
    rev = f"{DEHARDCODE_COMMIT}~1:tt_bio/boltz2.py"
    try:
        src = subprocess.check_output(["git", "show", rev], cwd=REPO, text=True,
                                      stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"{rev} not reachable from this checkout")
    path = Path(tempfile.mkdtemp()) / "boltz2_pre_dehardcode.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("boltz2_pre_dehardcode", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def feats():
    import tt_bio.boltz2 as new

    torch.manual_seed(7)
    f = {
        "ref_pos": torch.randn(B, M, 3),
        "ref_charge": torch.randn(B, M),
        "ref_element": torch.randn(B, M, 128),
        "ref_atom_name_chars": torch.randn(B, M, 4, 64),
        "ref_space_uid": torch.randint(0, 3, (B, M)),
        "atom_pad_mask": torch.ones(B, M),
        "atom_to_token": torch.zeros(B, M, N),
        "res_type": torch.randn(B, N, new.const.num_tokens),
        "profile": torch.randn(B, N, new.const.num_tokens),
        "deletion_mean": torch.randn(B, N),
    }
    for i in range(M):
        f["atom_to_token"][0, i, i % N] = 1.0
    return f


def _paired(old_cls, new_cls, seed=11, **kw):
    torch.manual_seed(seed)
    a = old_cls(**kw)
    torch.manual_seed(seed)
    b = new_cls(**kw)
    assert sorted(a.state_dict()) == sorted(b.state_dict())
    b.load_state_dict(a.state_dict())
    return a.eval(), b.eval()


def test_atom_encoder_default_is_bit_exact(old_boltz2, feats):
    import tt_bio.boltz2 as new

    a, b = _paired(old_boltz2.AtomEncoder, new.AtomEncoder,
                   structure_prediction=False, **ATOM_ENCODER_CFG)
    with torch.no_grad():
        q0, c0, p0, _ = a(feats)
        q1, c1, p1, _ = b(feats)
    assert torch.equal(q0, q1) and torch.equal(c0, c1) and torch.equal(p0, p1)


def test_input_embedder_default_is_bit_exact(old_boltz2, feats):
    import tt_bio.boltz2 as new

    a, b = _paired(old_boltz2.InputEmbedder, new.InputEmbedder, **CFG)
    with torch.no_grad():
        assert torch.equal(a(feats), b(feats))


def test_affinity_heads_default_returns_the_same_keys(old_boltz2):
    import tt_bio.boltz2 as new

    kw = dict(token_z=32, input_token_s=64, num_blocks=2, num_heads=4,
              activation_checkpointing=False)
    a, b = _paired(old_boltz2.AffinityHeadsTransformer,
                   new.AffinityHeadsTransformer, seed=3, **kw)
    f = {
        "token_pad_mask": torch.ones(B, N),
        "mol_type": torch.zeros(B, N, dtype=torch.long),
        "affinity_token_mask": torch.zeros(B, N),
    }
    f["affinity_token_mask"][0, -2:] = 1
    f["mol_type"][0, -2:] = 3
    z = torch.randn(B, N, N, 32)
    with torch.no_grad():
        ra, rb = a(z=z, feats=f), b(z=z, feats=f)
    assert sorted(ra) == sorted(rb)
    assert all(torch.equal(ra[k], rb[k]) for k in ra)


def test_return_repr_adds_the_pooled_vector():
    import tt_bio.boltz2 as new

    head = new.AffinityHeadsTransformer(
        token_z=32, input_token_s=64, num_blocks=2, num_heads=4,
        activation_checkpointing=False, return_repr=True,
    ).eval()
    f = {
        "token_pad_mask": torch.ones(B, N),
        "mol_type": torch.zeros(B, N, dtype=torch.long),
        "affinity_token_mask": torch.zeros(B, N),
    }
    f["affinity_token_mask"][0, -2:] = 1
    with torch.no_grad():
        out = head(z=torch.randn(B, N, N, 32), feats=f)
    assert out["affinity_repr"].shape == (B, 64)


def test_nesso_atom_features_reach_390_dims():
    """Nesso-1's concat order, with the dimension the checkpoint was trained on."""
    import tt_bio.boltz2 as new

    enc = new.AtomEncoder(
        **{**ATOM_ENCODER_CFG, "atom_feature_dim": 390},
        structure_prediction=False, add_additional_atom_features=True,
    ).eval()
    f = {
        "ref_pos": torch.randn(B, M, 3),
        "ref_charge": torch.randn(B, M),
        "ref_chirality": torch.randn(B, M),
        "ref_hybridization": torch.randn(B, M),
        "ref_element": torch.randn(B, M, 128),
        "ref_atom_name_chars": torch.randn(B, M, 4, 64),
        "ref_space_uid": torch.randint(0, 3, (B, M)),
        "atom_pad_mask": torch.ones(B, M),
    }
    with torch.no_grad():
        q, _, _, _ = enc(f)
    assert q.shape == (B, M, ATOM_ENCODER_CFG["atom_s"])


def test_input_embedder_without_msa_profile_drops_the_module():
    import tt_bio.boltz2 as new

    emb = new.InputEmbedder(**CFG, use_msa_profile=False)
    assert not hasattr(emb, "msa_profile_encoding")
    assert not any("msa_profile" in k for k in emb.state_dict())


def test_unknown_residue_code_raises_instead_of_degrading_to_unk():
    """Upstream maps any unmapped one-letter code to UNK with no warning.

    The map covers all 26 letters plus '-', and X already means "unknown residue",
    so a miss is always an input error: lowercase, a digit, whitespace. Silently
    substituting UNK turns a typo into a confident prediction on a different
    sequence.
    """
    from tt_bio._vendor.nesso.data.yaml_input import _protein_residues

    for seq, offender in (("ACDxG", "x"), ("ACD1G", "1"), ("ACD G", " ")):
        with pytest.raises(ValueError, match="unrecognized residue code"):
            _protein_residues(seq, None, ccd_dict={})
        # and the message names the offending character, not just its position
        try:
            _protein_residues(seq, None, ccd_dict={})
        except ValueError as exc:
            assert repr(offender) in str(exc)


def test_x_and_gap_are_still_accepted():
    """X is a legitimate unknown residue and '-' a legitimate gap."""
    from tt_bio._vendor.nesso.data import const
    from tt_bio._vendor.nesso.data.yaml_input import _protein_residues

    for c in ("X", "-"):
        assert c in const.prot_letter_to_token
    # validation happens before any RDKit lookup, so getting past it means accepted
    with pytest.raises(FileNotFoundError):
        _protein_residues("ACDX-G", None, ccd_dict={})
