"""Engine-side constraint handling: the Protenix data path and the BoltzGen design
parser.

Covers the two customer-reported bugs:
  * Protenix silently dropped every `constraints:` entry — including covalent `bond`
    constraints it can honour via the token-bond graph (now wired through), while
    pocket/contact (which need a constraint embedder it lacks) are rejected clearly.
  * BoltzGen crashed on a `constraints:` block — a cryptic "invalid keys" error for a
    pocket/contact constraint, and an IndexError on an empty list.

CPU-only (no device, no golden files); the few that need the bundled CCD mol library
skip when it is absent. The platform-gate counterpart lives in
test_platform_constraints.py (platform code is not on the engine branch).
"""
import os
from pathlib import Path

import pytest

_MOL_DIR = Path(os.path.expanduser("~/.boltz/mols"))
_needs_mols = pytest.mark.skipif(
    not (_MOL_DIR / "SO4.pkl").exists(), reason="needs bundled CCD mol library (~/.boltz/mols)")

_POCKET = """sequences:
  - protein: {id: A, sequence: MKVLAAAAAA}
  - ligand: {id: B, ccd: SO4}
constraints:
  - pocket: {binder: B, contacts: [[A, 3]]}
"""
_BOND = """sequences:
  - protein: {id: A, sequence: MKVLAAAAAA}
  - ligand: {id: B, ccd: SO4}
constraints:
  - bond: {atom1: [A, 3, CA], atom2: [B, 1, S]}
"""


def _write(tmp_path, name, body):
    p = tmp_path / f"{name}.yaml"
    p.write_text(body)
    return p


# --------------------------------------------------------------------------- #
# Protenix engine: parse `constraints`, reject pocket/contact, accept bonds.
# --------------------------------------------------------------------------- #
def test_read_bio_constraints_bond(tmp_path):
    from tt_bio.main import _read_bio_constraints
    bonds = _read_bio_constraints(_write(tmp_path, "b", _BOND))
    assert bonds == [(("A", 3, "CA"), ("B", 1, "S"))]


def test_read_bio_constraints_rejects_pocket(tmp_path):
    import click
    from tt_bio.main import _read_bio_constraints
    with pytest.raises(click.ClickException, match="pocket"):
        _read_bio_constraints(_write(tmp_path, "p", _POCKET))


def test_read_bio_constraints_none(tmp_path):
    from tt_bio.main import _read_bio_constraints
    spec = "sequences:\n  - protein: {id: A, sequence: MKVLA}\n"
    assert _read_bio_constraints(_write(tmp_path, "n", spec)) == []


@_needs_mols
def test_protenix_bond_wires_exactly_one_token_pair():
    """A bond constraint marks exactly its two endpoint tokens in token_bonds and
    leaves everything else (incl. the automatic intra-ligand bonds) untouched."""
    from tt_bio.protenix_data import build_complex_features
    chains = [("MKVLA", None, "protein"), ("CCD_SO4", None, "ligand")]
    ids = ["A", "B"]
    base = build_complex_features(chains, mol_dir=str(_MOL_DIR), chain_ids=ids)["token_bonds"]
    bonded = build_complex_features(
        chains, mol_dir=str(_MOL_DIR), chain_ids=ids,
        bonds=[(("A", 3, "CA"), ("B", 1, "S"))])["token_bonds"]
    # protein residue 3 -> token 2; ligand sulfur is the first ligand token (5).
    t_prot, t_lig = 2, 5
    assert bonded[t_prot, t_lig] == 1.0 and bonded[t_lig, t_prot] == 1.0
    changed = {tuple(ix) for ix in (bonded - base).nonzero(as_tuple=False).tolist()}
    assert changed == {(t_prot, t_lig), (t_lig, t_prot)}


@_needs_mols
def test_protenix_bond_bad_reference_raises():
    from tt_bio.protenix_data import build_complex_features
    chains = [("MKVLA", None, "protein")]
    with pytest.raises(ValueError, match="chain"):
        build_complex_features(chains, mol_dir=str(_MOL_DIR), chain_ids=["A"],
                               bonds=[(("A", 1, "CA"), ("Z", 1, "CA"))])
    with pytest.raises(ValueError, match="residue"):
        build_complex_features(chains, mol_dir=str(_MOL_DIR), chain_ids=["A"],
                               bonds=[(("A", 1, "CA"), ("A", 99, "CA"))])


# --------------------------------------------------------------------------- #
# BoltzGen design parser: no crash, clear guidance.
# --------------------------------------------------------------------------- #
def test_boltzgen_constraint_helpers():
    from tt_bio.boltzgen.data.parse.schema import (
        _design_constraints, _total_len_spec, _reject_unsupported_constraints)
    assert _design_constraints({}) == []
    assert _design_constraints({"constraints": []}) == []          # was an IndexError source
    assert _total_len_spec([]) is None
    assert _total_len_spec([{"bond": {}}, {"total_len": {"min": 5}}]) == {"min": 5}
    _reject_unsupported_constraints([{"total_len": {"min": 5}}])    # no raise
    with pytest.raises(ValueError, match="pocket"):
        _reject_unsupported_constraints([{"pocket": {}}])
    with pytest.raises(ValueError, match="contact"):
        _reject_unsupported_constraints([{"contact": {}}])


@_needs_mols
def test_boltzgen_parse_constraint_cases(tmp_path):
    from tt_bio.boltzgen.data.parse.schema import YamlDesignParser
    parser = YamlDesignParser(_MOL_DIR)

    empty = _write(tmp_path, "empty",
                   "entities:\n  - protein: {id: A, sequence: MKKAVINGE}\nconstraints: []\n")
    parser.parse_yaml(empty, {}, _MOL_DIR)        # no IndexError

    total_len = _write(tmp_path, "tl",
                       "entities:\n  - protein: {id: B, sequence: 50..80}\n"
                       "constraints:\n  - total_len: {min: 50, max: 80}\n")
    parser.parse_yaml(total_len, {}, _MOL_DIR)    # parses fine

    pocket = _write(tmp_path, "pk",
                    "entities:\n  - protein: {id: A, sequence: MKKAVINGE}\n"
                    "constraints:\n  - pocket: {binder: A, contacts: [[A, 3]]}\n")
    with pytest.raises(ValueError, match="pocket"):
        parser.parse_yaml(pocket, {}, _MOL_DIR)
