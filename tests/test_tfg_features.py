"""TFG input features (tt_bio.tfg.features) against OpenDDE v1.2.0's constraint schema.

Schema cases mirror upstream tests/test_input_validation.py and the checks in
json_to_feature.py. Resolution runs on upstream's examples/tfg/1a14 inputs translated to tt-bio
YAML. The upstream comparison tests need an OpenDDE v1.2.0 checkout (OPENDDE_SRC, default
/tmp/tfgsrc/up) and its CCD cache; they skip otherwise.
"""
import copy
import json
import logging
import os
import sys
import typing
from pathlib import Path

import numpy as np
import pytest
import torch

from tt_bio.protenix_data import build_complex_features
from tt_bio.tfg.features import (
    GEOMETRY_FEATURES, RDKIT_GEOMETRY_FEATURES, AtomTable, constraint_features, entity_groups,
    geometry_features, validate_constraint)

H = ("QVQLQQSGAELVKPGASVRMSCKASGYTFTNYNMYWVKQSPGQGLEWIGIFYPGNGDTSYNQKFKDKATLTADKSSNTAYMQLSSL"
     "TSEDSAVYYCARSGGSYRYDGGFDYWGQGTTVTV")
L = ("DIELTQTTSSLSASLGDRVTISCRASQDISNYLNWYQQNPDGTVKLLIYYTSNLHSEVPSRFSGSGSGTDYSLTISNLEQEDIATYF"
     "CQQDFTLPFTFGGGTAA")
N = ("RDFNNLTKGLCTINSWHIYGKDNAVRIGEDSDVLVTREPYVSCDPDECRFYALSQGTTIRGKHSNGTIHDRSQYRALISWPLSSPPTV"
     "YNSRVECIGWSSTSCHDGKTRMSICISGPNNNASAVIWYNRRPVTEINTWARNILRTQESECVCHNGVCPVVFTDGSATGPAETRIYY"
     "FKEGKILKWEPLAGTAKHIEECSCYGERAEITCTCRDNWQGSNRPVIRIDPVAMTHTSQYICSPVLTDNPRPNDPTVGKCNDPYPGN"
     "NNNGVKGFSYLDGVNTWLGRTISIASRSGYEMLKVPNALTDDKSKPTQGQTIVLNTDWSGYSGSFMDYWAEGECYRACFYVELIRGRPK"
     "EDKVWWTSNSIVSMCSSTEFLGQWDWPDGAKIEYFL")

CONTACT_1A14 = {
    "contact": [
        {"entity1": "3", "copy1": 1, "position1": "248", "atom1": "CA",
         "entity2": "2", "copy2": 1, "position2": "93", "atom2": "CA",
         "min_distance": 3.5, "max_distance": 8.0},
        {"entity1": "3", "copy1": 1, "position1": "249", "atom1": "CA",
         "entity2": "2", "copy2": 1, "position2": "92", "atom2": "CA",
         "min_distance": 3.5, "max_distance": 8.0},
        {"entity1": "3", "copy1": 1, "position1": "319", "atom1": "CA",
         "entity2": "1", "copy2": 1, "position2": "55", "atom2": "CA",
         "min_distance": 3.5, "max_distance": 8.0},
        {"entity1": "3", "copy1": 1, "position1": "251", "atom1": "CA",
         "entity2": "2", "copy2": 1, "position2": "30", "atom2": "CA",
         "min_distance": 3.5, "max_distance": 8.0},
    ],
    "movable_chains": ["H", "L"],
}
POCKET_1A14 = {
    "movable_chains": ["H", "L"],
    "epitope": {
        "residues": [
            {"entity": "3", "copy": 1, "position": "248", "residue": "P"},
            {"entity": "3", "copy": 1, "position": "249", "residue": "N"},
            {"entity": "3", "copy": 1, "position": "318", "residue": "N"},
            {"entity": "3", "copy": 1, "position": "321", "residue": "W"},
        ],
        "paratope": "all",
        "min_fraction": 0.5,
    },
}


def _yaml_complex(tmp_path, entries):
    """Write a tt-bio YAML; returns (chains via main._read_bio_chains, entity groups, feats)."""
    import yaml
    from tt_bio.main import _read_bio_chains
    path = tmp_path / "in.yaml"
    path.write_text(yaml.safe_dump({"sequences": entries}))
    chains = _read_bio_chains(path)
    feats = build_complex_features([(s, None, mt) for _c, s, _m, mt, _x in chains],
                                   modifications=[c[4] for c in chains])
    return chains, entity_groups(path), feats


@pytest.fixture(scope="module")
def ab_1a14(tmp_path_factory):
    entries = [{"protein": {"id": c, "sequence": s, "msa": "empty"}}
               for c, s in (("H", H), ("L", L), ("N", N))]
    return _yaml_complex(tmp_path_factory.mktemp("1a14"), entries)


# ------------------------------------------------------------------------------------- schema

def _contact(**overrides):
    pair = {"entity1": "1", "position1": "1", "atom1": "CA", "entity2": "2", "position2": "2",
            "atom2": "CA", "min_distance": 3.5, "max_distance": 8.0}
    pair.update(overrides)
    return pair


@pytest.mark.parametrize("overrides, match", [
    ({"min_distance": float("nan")}, "min_distance must be a finite number"),
    ({"max_distance": float("inf")}, "max_distance must be a finite number"),
    ({"min_distance": True}, "min_distance must be a finite number"),
    ({"max_distance": "8.0"}, "max_distance must be a finite number"),
    ({"min_distance": None}, "min_distance must be a finite number"),
    ({"min_distance": -5.0, "max_distance": -1.0}, "must be >= 0"),
    ({"min_distance": 9.0, "max_distance": 8.0}, r"max_distance \(8.0\) < min_distance \(9.0\)"),
    ({"entity1": True}, "entity1 must be an integer, got the boolean"),
    ({"position1": 9.9}, "position1 must be an integer"),
    ({"copy1": "x"}, "copy1 must be an integer"),
    ({"entity1": None}, "missing entity1/left_entity"),
    ({"position2": None}, "missing position2/right_position"),
])
def test_contact_values_are_validated(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_constraint({"contact": [_contact(**overrides)]}, True)


@pytest.mark.parametrize("constraint, match", [
    ([1, 2], "'constraint' field must be an object"),
    ("contact", "'constraint' field must be an object"),
    ({"contact": {"a": 1}}, "constraint.contact must be a list; got dict"),
    ({"contact": "x"}, "constraint.contact must be a list; got str"),
    ({"contact": [1]}, r"constraint.contact\[0\] must be an object"),
    ({"pocket": []}, "Unsupported constraint field.*pocket"),
    ({"epitope": None}, "constraint.epitope is null"),
    ({"epitope": {}, "contact": []}, "cannot be combined"),
    ({"movable_chains": []}, "movable_chains must be a non-empty list"),
    ({"movable_chains": "H"}, "movable_chains must be a non-empty list"),
    ({"movable_chains": ["H", "H"]}, "repeated chain id"),
    ({"epitope": []}, "constraint.epitope must be an object"),
    ({"epitope": {"residues": [], "foo": 1}}, r"Unknown constraint.epitope field\(s\): \['foo'\]"),
    ({"epitope": {"residues": [{"entity": 1, "copy": 1, "position": 1}]}},
     "requires constraint.movable_chains"),
    ({"movable_chains": ["H"], "epitope": {"residues": []}}, "residues must be a non-empty list"),
    ({"movable_chains": ["H"], "epitope": {"residues": [1]}}, r"residues\[0\] must be an object"),
    ({"movable_chains": ["H"], "epitope": {"residues": [{"entity": 1, "copy": 1}]}},
     r"residues\[0\] missing position"),
    ({"movable_chains": ["H"],
      "epitope": {"residues": [{"entity": 1, "copy": 1, "position": 1, "chain": "A"}]}},
     r"unknown field\(s\) \['chain'\]"),
    ({"movable_chains": ["H"],
      "epitope": {"residues": [{"entity": 1, "copy": 1.5, "position": 1}]}},
     r"residues\[0\].copy must be an integer"),
    ({"movable_chains": ["H"],
      "epitope": {"residues": [{"entity": 1, "copy": 1, "position": 1}], "min_fraction": 0}},
     r"min_fraction must be in \(0, 1\]"),
    ({"movable_chains": ["H"],
      "epitope": {"residues": [{"entity": 1, "copy": 1, "position": 1}], "min_fraction": 1.5}},
     r"min_fraction must be in \(0, 1\]"),
    ({"movable_chains": ["H"],
      "epitope": {"residues": [{"entity": 1, "copy": 1, "position": 1}], "paratope": "cdrs"}},
     'paratope must be "cdr", "all"'),
    ({"movable_chains": ["H"],
      "epitope": {"residues": [{"entity": 1, "copy": 1, "position": 1}],
                  "paratope": {"windows": {"L": [[1, 2]]}}}},
     "names 'L', which is not a movable chain"),
    ({"movable_chains": ["H"],
      "epitope": {"residues": [{"entity": 1, "copy": 1, "position": 1}],
                  "paratope": {"windows": {"H": [[5, 2]]}}}},
     "has a window with low > high"),
    ({"movable_chains": ["H"],
      "epitope": {"residues": [{"entity": 1, "copy": 1, "position": 1}],
                  "paratope": {"windows": {"H": [5]}}}},
     r"must be a list of \[low, high\] pairs"),
])
def test_malformed_constraint_is_an_error(constraint, match):
    with pytest.raises(ValueError, match=match):
        validate_constraint(constraint, True)


@pytest.mark.parametrize("value", [True, "0.5", None, float("nan")])
def test_epitope_min_fraction_must_be_a_finite_number(value):
    c = {"movable_chains": ["H"],
         "epitope": {"residues": [{"entity": "1", "copy": 1, "position": "1"}],
                     "min_fraction": value}}
    with pytest.raises(ValueError, match="min_fraction must be a finite number"):
        validate_constraint(c, True)


def test_epitope_requires_tfg_guidance():
    with pytest.raises(ValueError, match="use_tfg_guidance"):
        validate_constraint({"movable_chains": ["H"], "epitope": {"residues": []}}, False)


def test_contact_without_tfg_warns_and_null_is_absent(caplog):
    assert validate_constraint(None, True) is None
    with caplog.at_level(logging.WARNING):
        out = validate_constraint({"contact": [_contact()]}, False)
    assert "TFG guidance is disabled" in caplog.text
    assert len(out["contact"]) == 1


def test_contact_defaults_and_aliases():
    c = validate_constraint({"contact": [{"left_entity": 1, "left_position": "4",
                                          "right_entity": "2", "right_position": 7,
                                          "right_copy": "2", "right_atom": "N"}]}, True)
    r = c["contact"][0]
    assert (r["min_distance"], r["max_distance"]) == (3.5, 8.0)
    assert r["left"] == {"entity_id": 1, "copy_id": None, "position": 4, "atom_name": None}
    assert r["right"] == {"entity_id": 2, "copy_id": 2, "position": 7, "atom_name": "N"}
    assert c["movable_chains"] is None and c["epitope"] is None


# --------------------------------------------------------------------------------- resolution

def test_1a14_contact_resolves_to_the_named_atoms(ab_1a14):
    chains, ents, feats = ab_1a14
    assert ents == [["H"], ["L"], ["N"]]
    out = constraint_features(validate_constraint(CONTACT_1A14, True), feats, chains, ents)
    t = AtomTable(feats, chains, ents)
    idx = out["user_distance_restraint_index"]
    assert idx.dtype == torch.int64 and idx.shape == (2, 4)
    got = [[(t.chain_id[i], int(t.res_id[i]), t.atom_name[i]) for i in col] for col in idx.T.tolist()]
    assert got == [[("N", 248, "CA"), ("L", 93, "CA")], [("N", 249, "CA"), ("L", 92, "CA")],
                   [("N", 319, "CA"), ("H", 55, "CA")], [("N", 251, "CA"), ("L", 30, "CA")]]
    assert out["user_distance_restraint_lower_bound"].dtype == torch.float32
    assert out["user_distance_restraint_lower_bound"].tolist() == [3.5] * 4
    assert out["user_distance_restraint_upper_bound"].tolist() == [8.0] * 4
    mov = out["user_rigid_movable_atom"]
    assert mov.dtype == torch.bool and mov.shape == (feats["ref_pos"].shape[0],)
    assert np.array_equal(mov.numpy(), np.isin(t.chain_id, ["H", "L"]))


def test_1a14_pocket_resolves_epitope_and_paratope(ab_1a14, caplog):
    chains, ents, feats = ab_1a14
    with caplog.at_level(logging.INFO):
        out = constraint_features(validate_constraint(POCKET_1A14, True), feats, chains, ents)
    t = AtomTable(feats, chains, ents)
    idx = out["user_epitope_atom_index"]
    assert idx.dtype == torch.int64 and idx.shape[0] == 4
    for row, (pos, aa, n_heavy) in zip(idx.tolist(), [(248, "P", 7), (249, "N", 8),
                                                      (318, "N", 8), (321, "W", 14)]):
        atoms = [a for a in row if a >= 0]
        assert len(atoms) == n_heavy and row[n_heavy:] == [-1] * (len(row) - n_heavy)
        assert {(t.chain_id[a], int(t.res_id[a])) for a in atoms} == {("N", pos)}
        assert N[pos - 1] == aa
        assert t.heavy[atoms].all()
    assert idx.shape[1] == 14
    assert out["user_epitope_k"].tolist() == [2] and out["user_epitope_k"].dtype == torch.int64
    para = out["user_epitope_paratope_atom"]
    assert para.dtype == torch.bool and torch.equal(para, out["user_rigid_movable_atom"])
    assert "K=2" in caplog.text and "paratope_residues_per_chain={'H': 120, 'L': 104}" in caplog.text


def test_cdr_paratope_and_explicit_windows(ab_1a14):
    chains, ents, feats = ab_1a14
    t = AtomTable(feats, chains, ents)
    c = copy.deepcopy(POCKET_1A14)
    c["epitope"]["paratope"] = "cdr"
    para = constraint_features(validate_constraint(c, True), feats, chains, ents)[
        "user_epitope_paratope_atom"].numpy()
    in_cdr = np.zeros(len(t), dtype=bool)
    for lo, hi in [(24, 40), (50, 66), (88, 115)]:
        in_cdr |= (t.res_id >= lo) & (t.res_id <= hi)
    assert np.array_equal(para, in_cdr & np.isin(t.chain_id, ["H", "L"]))
    c["epitope"]["paratope"] = {"windows": {"H": [[26, 35]]}}
    para = constraint_features(validate_constraint(c, True), feats, chains, ents)[
        "user_epitope_paratope_atom"].numpy()
    assert set(zip(t.chain_id[para], t.res_id[para].tolist())) == {("H", r) for r in range(26, 36)}


@pytest.mark.parametrize("mutate, match", [
    (lambda c: c["epitope"]["residues"][0].update(residue="A"),
     r"residue 'A' given, the sequence has 'P' at entity=3 copy=1 position=248"),
    (lambda c: c["epitope"]["residues"][0].update(position=9999), "no atom found for entity=3"),
    (lambda c: c["epitope"]["residues"][0].update(entity=1, position=50, residue=None),
     "lies in a movable chain"),
    (lambda c: c.update(movable_chains=["H", "X"]), r"absent: \['X'\]"),
    (lambda c: c.update(movable_chains=["H", "L", "N"]), "names every chain"),
    (lambda c: c["epitope"].update(paratope={"windows": {"H": [[500, 600]]}}),
     "the paratope selects no atom"),
])
def test_structural_epitope_errors(ab_1a14, mutate, match):
    chains, ents, feats = ab_1a14
    c = copy.deepcopy(POCKET_1A14)
    mutate(c)
    with pytest.raises(ValueError, match=match):
        constraint_features(validate_constraint(c, True), feats, chains, ents)


def test_repeated_epitope_residue_counts_once_and_k_rounds_up(ab_1a14):
    chains, ents, feats = ab_1a14
    c = copy.deepcopy(POCKET_1A14)
    c["epitope"]["residues"].append(dict(c["epitope"]["residues"][0]))
    c["epitope"]["min_fraction"] = 0.6
    out = constraint_features(validate_constraint(c, True), feats, chains, ents)
    assert out["user_epitope_atom_index"].shape[0] == 4
    assert out["user_epitope_k"].tolist() == [3]          # ceil(0.6 * 4)


def test_entity_copy_mapping_follows_yaml_entries(tmp_path):
    """`id: [A, B]` is one entity (copies 1, 2); a separate entry with the same sequence is
    another entity, although tt-bio's entity_id feature merges all three."""
    seq = "MKTAYIAKQRQISFVKSHFSRQ"
    chains, ents, feats = _yaml_complex(tmp_path, [
        {"protein": {"id": ["A", "B"], "sequence": seq, "msa": "empty"}},
        {"protein": {"id": "C", "sequence": seq, "msa": "empty"}},
        {"ligand": {"id": "D", "ccd": "ATP"}},
    ])
    assert ents == [["A", "B"], ["C"], ["D"]]
    assert feats["entity_id"].unique().tolist() == [0, 1]
    t = AtomTable(feats, chains, ents)

    def resolve(contacts):
        out = constraint_features(validate_constraint({"contact": contacts}, True),
                                  feats, chains, ents)
        return [[(t.chain_id[i], int(t.res_id[i]), t.atom_name[i]) for i in col]
                for col in out["user_distance_restraint_index"].T.tolist()]

    assert resolve([{"entity1": 1, "copy1": 2, "position1": 5, "entity2": 2, "position2": 7}]) \
        == [[("B", 5, "CA"), ("C", 7, "CA")]]
    # no copy on either side: every copy pairs up in order
    assert resolve([{"entity1": 1, "position1": 3, "atom1": "N", "entity2": 1, "position2": 9}]) \
        == [[("A", 3, "N"), ("A", 9, "CA")], [("B", 3, "N"), ("B", 9, "CA")]]
    assert resolve([{"entity1": 3, "position1": 1, "atom1": "PG", "entity2": 2, "copy2": 1,
                     "position2": 2, "atom2": "CB"}]) == [[("D", 1, "PG"), ("C", 2, "CB")]]
    with pytest.raises(ValueError, match="asymmetric copy counts"):
        resolve([{"entity1": 1, "position1": 3, "entity2": 2, "position2": 9}])
    with pytest.raises(ValueError, match=r"atom/atom is required for non-protein entity 3 "
                                         r"\(type='non-polymer'\)"):
        resolve([{"entity1": 3, "position1": 1, "entity2": 2, "position2": 2}])
    with pytest.raises(ValueError, match="No atom found for constraint.contact\\[0\\] left"):
        resolve([{"entity1": 1, "copy1": 3, "position1": 1, "entity2": 2, "position2": 2}])


def test_fasta_entity_groups(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">A,B|protein|empty\nMKV\n>C|protein|empty\nMKV\n>L|ccd\nATP\n")
    assert entity_groups(p) == [["A", "B"], ["C"], ["L"]]


def test_absent_constraint_gives_empty_tensors(ab_1a14):
    chains, ents, feats = ab_1a14
    out = constraint_features(None, feats, chains, ents)
    assert out["user_distance_restraint_index"].shape == (2, 0)
    assert out["user_distance_restraint_lower_bound"].shape == (0,)
    assert out["user_rigid_movable_atom"].shape == (0,)
    assert "user_epitope_atom_index" not in out


def test_geometry_features_protein_only(ab_1a14):
    chains, ents, feats = ab_1a14
    g = geometry_features(feats, chains, ents)
    assert set(g) == set(GEOMETRY_FEATURES)
    assert g["pairwise_distance_index"].shape == (2, 0)
    assert g["planar_improper_index"].shape == (4, 0)
    assert g["symmetric_chain_index"].shape == (2, 0)


# ------------------------------------------------------------------------- against upstream

UPSTREAM = Path(os.environ.get("OPENDDE_SRC", "/tmp/tfgsrc/up"))


@pytest.fixture(scope="module")
def upstream():
    if not (UPSTREAM / "opendde/data/inference/json_to_feature.py").exists():
        pytest.skip(f"no OpenDDE checkout at {UPSTREAM}")
    if not hasattr(typing, "Self"):                       # upstream needs Python 3.11 typing
        import typing_extensions
        typing.Self = typing_extensions.Self
    sys.path.append(str(UPSTREAM))
    try:
        from opendde.data.core.ccd import get_ccd_cache_paths
        from opendde.data.inference.json_to_feature import SampleDictToFeatures
    except Exception as e:                                # noqa: BLE001
        pytest.skip(f"upstream does not import here: {e}")
    if not all(Path(p).exists() for p in get_ccd_cache_paths()):
        pytest.skip("upstream CCD cache absent")
    return SampleDictToFeatures


def _upstream_feats(upstream, job):
    feats, atom_array, _ = upstream(copy.deepcopy(job), extract_features_for_tfg=True).get_feature_dict()
    return feats, atom_array


@pytest.mark.parametrize("name", ["contact", "pocket"])
def test_1a14_matches_upstream(upstream, ab_1a14, name):
    job = json.loads((UPSTREAM / f"examples/tfg/1a14/1a14_{name}.json").read_text())[0]
    for e in job["sequences"]:
        e["proteinChain"].pop("pairedMsaPath"), e["proteinChain"].pop("unpairedMsaPath")
    uf, aa = _upstream_feats(upstream, job)
    chains, ents, feats = ab_1a14
    t = AtomTable(feats, chains, ents)
    # same atom axis: (chain, residue, atom name) at every index
    assert [(c, int(r), a) for c, r, a in zip(aa.chain_id, aa.res_id, aa.atom_name)] == \
        [(t.chain_id[i], int(t.res_id[i]), t.atom_name[i]) for i in range(len(t))]
    out = constraint_features(validate_constraint(job["constraint"], True), feats, chains, ents)
    for k, v in out.items():
        assert v.dtype == uf[k].dtype and torch.equal(v, uf[k]), k


def test_geometry_matches_upstream(upstream, tmp_path):
    seq = "MKSAEGLLVK"
    job = {"name": "g", "sequences": [
        {"proteinChain": {"sequence": seq, "count": 2, "id": ["A", "B"],
                          "modifications": [{"ptmType": "CCD_SEP", "ptmPosition": 3}]}},
        {"ligand": {"ligand": "CCD_ATP", "count": 1, "id": ["C"]}},
        {"ligand": {"ligand": "CCD_RET", "count": 1, "id": ["D"]}},
        {"ligand": {"ligand": "CCD_0LI", "count": 1, "id": ["E"]}},
        {"ligand": {"ligand": "CCD_ZN", "count": 1, "id": ["F"]}},
    ]}
    uf, aa = _upstream_feats(upstream, job)
    chains, ents, feats = _yaml_complex(tmp_path, [
        {"protein": {"id": ["A", "B"], "sequence": seq, "msa": "empty",
                     "modifications": [{"position": 3, "ccd": "SEP"}]}},
        {"ligand": {"id": "C", "ccd": "ATP"}}, {"ligand": {"id": "D", "ccd": "RET"}},
        {"ligand": {"id": "E", "ccd": "0LI"}}, {"ligand": {"id": "F", "ccd": "ZN"}},
    ])
    t = AtomTable(feats, chains, ents)
    assert [(c, int(r), a) for c, r, a in zip(aa.chain_id, aa.res_id, aa.atom_name)] == \
        [(t.chain_id[i], int(t.res_id[i]), t.atom_name[i]) for i in range(len(t))]
    g = geometry_features(feats, chains, ents)
    assert g["stereo_bond_index"].shape[1] > 0 and g["linear_triple_bond_index"].shape[1] > 0
    assert g["symmetric_chain_index"].tolist() == [[0], [1]]
    for k in GEOMETRY_FEATURES:
        assert g[k].dtype == uf[k].dtype and torch.equal(g[k], uf[k]), k
    assert set(RDKIT_GEOMETRY_FEATURES) < set(g)
