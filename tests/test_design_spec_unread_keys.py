"""A design spec key the model will not read is an error, not a default.

RFD3 and PXDesign both used to take conditioning fields they never pass to the model:
the run cost the same and came back unconditioned, which is indistinguishable from a
design that ignored its own epitope. Both readers now refuse by name through the one
helper in `tt_bio.data.yaml_input`, the way BoltzGen's schema check already did.
"""
import pytest
import yaml

from tt_bio.data.yaml_input import refuse_unread_keys, refuse_unresolved
from tt_bio.rfd3.input import InputSpecification

RFD3_BASE = {"input": "target.cif", "contig": "A1-50,70"}

# Every upstream field the featurizer never reads, with a value that asks for something.
RFD3_UNREAD = [
    ("select_hotspots", "A10-20"),
    ("select_hotspots", True),
    ("select_hotspots", {"A10-20": "ALL"}),
    ("select_partially_buried", "A10-20"),
    ("select_hbond_donor", "A10-20"),
    ("select_hbond_acceptor", "A10-20"),
    ("select_unfixed_sequence", False),
    ("redesign_motif_sidechains", True),
    ("ori_token", [1.0, 2.0, 3.0]),
    ("infer_ori_strategy", "com"),
    ("plddt_enhanced", False),
    ("dialect", 1),
    ("cif_parser_args", {"cache_dir": "/tmp/x"}),
    ("extra", {"anything": 1}),
    # ligand-only: on a spec with no `ligand` these reach no feature
    ("select_buried", "A10-20"),
    ("select_exposed", "A10-20"),
]


@pytest.mark.parametrize("key,value", RFD3_UNREAD)
def test_rfd3_refuses_a_field_it_never_reads(key, value):
    spec = InputSpecification.from_dict({**RFD3_BASE, key: value})
    with pytest.raises(ValueError, match=key):
        spec.validate()


def test_rfd3_refuses_an_unknown_key_by_name():
    spec = InputSpecification.from_dict({**RFD3_BASE, "select_hotspot": "A10"})
    with pytest.raises(ValueError, match=r"unknown key\(s\).*select_hotspot"):
        spec.validate()


@pytest.mark.parametrize("key,value", [
    ("select_unfixed_sequence", True),
    ("redesign_motif_sidechains", False),
    ("plddt_enhanced", True),
    ("dialect", 2),
])
def test_rfd3_allows_a_field_spelled_out_at_its_default(key, value):
    """Spelling out a default asks for nothing, so it is not an error."""
    InputSpecification.from_dict({**RFD3_BASE, key: value}).validate()


def test_rfd3_still_accepts_the_fields_it_honours():
    InputSpecification.from_dict({
        "input": "target.cif", "contig": "A1-50,70", "length": "70",
        "select_fixed_atoms": {"A5": "N,CA,C,O"}, "is_non_loopy": True, "partial_t": 0.5,
    }).validate()


def test_rfd3_ligand_spec_may_use_the_rasa_bins():
    """select_buried/select_exposed do reach the model for ligand atoms."""
    InputSpecification.from_dict({
        "input": "target.cif", "contig": "A1-50,70", "ligand": "IAI",
        "select_buried": {"IAI": "C1,C2"},
    }).validate()


def _px(tmp_path, chain_props, extra=None):
    cfg = {"target": {"file": "t.cif", "chains": {"A": chain_props}}, "binder_length": 80}
    cfg.update(extra or {})
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.mark.parametrize("props,needle", [
    ({"crop": ["1-10"], "hotspot": [4]}, "hotspot"),
    ({"crops": ["1-10"]}, "crops"),
    ({"crop": ["1-10"], "symmetry": "C3"}, "symmetry"),
    ({"crop": ["1-10"], "msa": "./msa/0"}, "msa"),
])
def test_pxdesign_refuses_a_chain_key_it_does_not_read(tmp_path, props, needle):
    from tt_bio.pxdesign.inputs import read_design_yaml

    with pytest.raises(ValueError, match=needle):
        read_design_yaml(_px(tmp_path, props))


def test_pxdesign_refuses_an_unknown_top_level_key(tmp_path):
    from tt_bio.pxdesign.inputs import read_design_yaml

    with pytest.raises(ValueError, match="num_designs"):
        read_design_yaml(_px(tmp_path, {"crop": ["1-10"]}, extra={"num_designs": 7}))


def test_refuse_unresolved_names_the_numbering_it_matched_against():
    with pytest.raises(ValueError) as e:
        refuse_unresolved("hotspots", [455, 486], list(range(1, 117)), what="chain A")
    msg = str(e.value)
    assert "455" in msg and "486" in msg and "label_seq 1..116" in msg


def test_refuse_unresolved_passes_when_every_residue_is_there():
    refuse_unresolved("hotspots", [40, 99], list(range(1, 117)), what="chain A")


def test_refuse_unread_keys_reports_both_kinds_in_one_message():
    with pytest.raises(ValueError) as e:
        refuse_unread_keys({"a": 1, "b": 2, "typo": 3},
                           honoured=("a",), unimplemented={"b": (None, "not built")},
                           what="spec")
    msg = str(e.value)
    assert "b=2" in msg and "not built" in msg and "typo" in msg
