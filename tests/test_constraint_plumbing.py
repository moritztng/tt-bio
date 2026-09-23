"""`cyclic: true`, `bond` and `modifications:` reach each model's features, not just its
capability row. Host-only, no card.

The capability table only says a model honours a key; these check the key changes the
features that model's trunk reads. A row that said `yes` over a door that dropped the key is
the defect class tests/test_input_capabilities.py cannot see.
"""
from __future__ import annotations

import json

import click
import pytest
import torch

from tt_bio.main import _read_bio_bonds, _read_bio_chains, _read_cyclic

SFTI = "GRCTKSIPPICFPD"   # SFTI-1, head-to-tail cyclic with a Cys3-Cys11 disulfide (PDB 1SFI)


def _yaml(tmp_path, text):
    p = tmp_path / "q.yaml"
    p.write_text(text)
    return p


def test_cyclic_becomes_the_head_to_tail_amide(tmp_path):
    p = _yaml(tmp_path, f"version: 1\nsequences:\n  - protein:\n      id: [A, B]\n"
                        f"      sequence: {SFTI}\n      cyclic: true\n  - protein:\n"
                        f"      id: C\n      sequence: {SFTI}\n")
    assert _read_cyclic(p) == ["A", "B"]
    assert _read_bio_bonds(p, _read_bio_chains(p)) == [
        (("A", 14, "C"), ("A", 1, "N")), (("B", 14, "C"), ("B", 1, "N"))]


def test_a_cyclic_nucleic_acid_is_refused_on_a_bond_model(tmp_path):
    p = _yaml(tmp_path, "version: 1\nsequences:\n  - rna:\n      id: R\n"
                        "      sequence: GAUCGAUC\n      cyclic: true\n")
    with pytest.raises(click.ClickException, match="only a protein chain can be cyclic"):
        _read_bio_bonds(p, _read_bio_chains(p))


def test_the_ring_closure_reaches_protenix_token_bonds(tmp_path):
    from tt_bio.protenix_data import build_complex_features

    p = _yaml(tmp_path, f"version: 1\nsequences:\n  - protein:\n      id: A\n"
                        f"      sequence: {SFTI}\n      cyclic: true\n")
    chains = _read_bio_chains(p)
    tb = build_complex_features([(SFTI, None, "protein")], chain_ids=["A"],
                                bonds=_read_bio_bonds(p, chains))["token_bonds"]
    assert tb[0, 13] == tb[13, 0] == 1 and tb.sum() == 2


def test_rf3_spec_carries_modifications_and_bonds():
    from tt_bio.worker import _rf3_bonds, _rf3_sequence

    assert _rf3_sequence("NKLCG", [{"position": 2, "ccd": "mly"}]) == "N(MLY)LCG"
    comps = [{"seq": "GCGSQ", "chain_id": "A"}, {"smiles": "C(=O)CCN", "chain_id": "L"}]
    # the portable SMILES name C1 is atomworks' C0
    assert _rf3_bonds(comps, [(("A", 2, "SG"), ("L", 1, "C1"))]) == [
        ["A/CYS/2/SG", "L/L:0/0/C0"]]
    with pytest.raises(ValueError, match="Its atoms: C1, O1, C2, C3, N1"):
        _rf3_bonds(comps, [(("A", 2, "SG"), ("L", 1, "C9"))])


def test_rf3_features_see_the_ligand_bond_and_the_ring(tmp_path):
    from tt_bio.rf3.featurize import featurize
    from tt_bio.worker import _rf3_bonds

    def feats(comps, bonds=None, cyclic=None):
        spec = tmp_path / "t.json"
        spec.write_text(json.dumps([{"name": "t", "components": comps,
                                     "bonds": _rf3_bonds(comps, bonds)}]))
        return featurize(spec, n_recycles=1, diffusion_batch_size=1, seed=0,
                         cyclic_chains=cyclic)[0]["feats"]

    lig = [{"seq": "GCGSQWDRSGR", "chain_id": "A"}, {"smiles": "C(=O)CCN", "chain_id": "L"}]
    assert feats(lig, [(("A", 2, "SG"), ("L", 1, "C1"))])["token_bonds"].sum() > \
        feats(lig)["token_bonds"].sum()
    assert list(feats([{"seq": SFTI, "chain_id": "A"}], cyclic=["A"])["cyclic_asym_ids"]) == [0]


def test_of3_features_see_the_bond_and_the_ring():
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        Query)
    from tt_bio.openfold3_data import build_openfold3_features
    from tt_bio.openfold3_host_prep import derive_relpos

    def q(chains):
        return Query.model_validate({"query_name": "t", "use_msas": False,
                                     "use_main_msas": False, "use_paired_msas": False,
                                     "chains": chains})

    lig = q([{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "GCGSQWDRSGR"},
             {"molecule_type": "ligand", "chain_ids": ["L"], "smiles": "C(=O)CCN"}])
    assert build_openfold3_features(lig, bonds=[(("A", 2, "SG"), ("L", 1, "C1"))])[
        "token_bonds"].sum() > build_openfold3_features(lig)["token_bonds"].sum()
    with pytest.raises(ValueError, match="Its atoms: C1, O1, C2, C3, N1"):
        build_openfold3_features(lig, bonds=[(("A", 2, "SG"), ("L", 1, "C9"))])

    pep = q([{"molecule_type": "protein", "chain_ids": ["A"], "sequence": SFTI}])
    r0, r1 = derive_relpos(build_openfold3_features(pep)), derive_relpos(
        build_openfold3_features(pep, cyclic=["A"]))
    # on the ring the last residue sits one step before the first, as residue 1 does on a
    # line (the first 66 channels are the residue-offset one-hot, 2 * 32 + 2 bins)
    assert r1[0, 13, :66].argmax() == r0[1, 0, :66].argmax() != r0[0, 13, :66].argmax()
