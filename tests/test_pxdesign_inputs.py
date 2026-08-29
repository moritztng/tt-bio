"""The PXDesign coordinate input path: a user's structure file to the model input dict.

The headline arm is the hydrogen one. PXDesign reads a target through protenix's
`DistillationMMCIFParser`, a parallel entry point that skips the `remove_water` /
`remove_hydrogens` the production `MMCIFParser.get_bioassembly` runs, so hydrogens reach the
CCD atom-name match and a residue with at least as many hydrogens as heavy atoms is thrown
away. `5o45.cif` carries 1248 of them and loses 61 of its 116 target residues that way, each
one then conditioned on at the origin. This path filters first, so all 116 survive.
"""
import os

import pytest
import torch

from tt_bio.protenix_data import parse_crop_spec, structure_token_coords
from tt_bio.pxdesign.featurize import RESTYPE_VOCAB
from tt_bio.pxdesign.inputs import (MODEL_INPUT_KEYS, design_inputs,
                                    design_inputs_from_yaml, read_design_yaml)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "pxdesign")
ARTS = os.path.join(os.path.dirname(__file__), "..", "scripts", "pxdesign_port",
                    "parity_artifacts")
PDL1, RBD = os.path.join(FIX, "PDL1.yaml"), os.path.join(FIX, "RBD.yaml")


def test_fixture_5o45_really_carries_hydrogens():
    """The hydrogen arm below is only an arm if its fixture has hydrogens in it."""
    import gemmi
    st = gemmi.read_structure(os.path.join(FIX, "5o45.cif.gz"))
    h = sum(1 for ch in st[0] for r in ch for a in r if a.element == gemmi.Element("H"))
    assert h == 1248, f"expected 1248 hydrogens in 5o45.cif, found {h}"


def test_hydrogen_bearing_target_keeps_every_residue():
    """116 of 116, where PXDesign's own parse of the same file keeps 55."""
    ch = structure_token_coords(os.path.join(FIX, "5o45.cif.gz"), ["A"], "1-116")["A"]
    assert len(ch["label_seq"]) == 116
    assert bool(ch["is_resolved"].all())
    assert ch["label_seq"][0] == 1 and ch["label_seq"][-1] == 116
    assert (ch["coord"].abs().sum(-1) > 0).all(), "a conditioned token landed on the origin"


def test_hydrogen_free_control_loses_nothing():
    """6m0j has no hydrogens, so the filter has nothing to do and must change nothing."""
    ch = structure_token_coords(os.path.join(FIX, "6m0j.cif.gz"), ["B"], "15-208")["B"]
    assert len(ch["label_seq"]) == 194
    assert bool(ch["is_resolved"].all())
    assert (ch["coord"].abs().sum(-1) > 0).all()


def test_chain_keys_are_label_asym_not_auth():
    """6m0j's RBD is auth chain E and label_asym B, and a CIF's YAML means the label one.
    Asking for label_asym B must give the 194-residue RBD, not ACE2's 597 residues, and
    label_asym A must give ACE2. The auth key only comes into play as a fallback for a PDB,
    which has no label_asym_id at all (test_pdb_and_cif_agree)."""
    got = structure_token_coords(os.path.join(FIX, "6m0j.cif.gz"), ["A", "B"])
    assert len(got["B"]["label_seq"]) == 194 and len(got["A"]["label_seq"]) == 597
    with pytest.raises(ValueError, match="no polymer chain"):
        structure_token_coords(os.path.join(FIX, "6m0j.cif.gz"), ["Z"])


def test_distogram_rep_atom_is_cb_not_ca():
    """CB for protein, CA for glycine. CA everywhere shifts every conditioning distance by
    a C-alpha-to-C-beta bond and still returns a plausible complex, so it is a silent edge."""
    import gemmi
    from tt_bio.protenix_data import distogram_rep_atom
    assert distogram_rep_atom("GLY", "protein") == "CA"
    assert distogram_rep_atom("ALA", "protein") == "CB"
    assert distogram_rep_atom("DA", "dna") == "C4" and distogram_rep_atom("U", "rna") == "C2"

    ch = structure_token_coords(os.path.join(FIX, "5o45.cif.gz"), ["A"], "1-116")["A"]
    st = gemmi.read_structure(os.path.join(FIX, "5o45.cif.gz"))
    st.setup_entities(); st.remove_hydrogens(); st.assign_label_seq_id()
    by_seq = {r.label_seq: r for c in st[0] for sc in c.subchains()
              if sc.subchain_id() == "A" for r in sc}
    for seq, name, xyz in zip(ch["label_seq"], ch["res_name"], ch["coord"]):
        want = by_seq[seq]["CA" if name == "GLY" else "CB"][0].pos
        assert torch.allclose(xyz, torch.tensor([want.x, want.y, want.z]))


@pytest.mark.parametrize("spec,want", [
    ("1-5", {1, 2, 3, 4, 5}), (["1-3", "7"], {1, 2, 3, 7}), ("1-3,7", {1, 2, 3, 7}),
    (40, {40}), ("all", None), ("full", None), (None, None), ([], None),
])
def test_crop_spec(spec, want):
    assert parse_crop_spec(spec) == want


def test_crop_selects_what_it_says():
    ch = structure_token_coords(os.path.join(FIX, "5o45.cif.gz"), ["A"], ["1-10", "20-30"])["A"]
    assert ch["label_seq"] == list(range(1, 11)) + list(range(20, 31))
    full = structure_token_coords(os.path.join(FIX, "5o45.cif.gz"), ["A"])["A"]
    assert len(full["label_seq"]) > 116, "no crop must mean the whole chain"


def test_pdb_and_cif_agree(tmp_path):
    """A user's PDB has to work as well as their CIF. A PDB has no label_asym_id, so the
    RBD is reached by its auth name E where the CIF reaches it as label_asym B, and both
    have to yield the same 194 residues at the same coordinates."""
    import gemmi
    st = gemmi.read_structure(os.path.join(FIX, "6m0j.cif.gz"))
    st.setup_entities()
    pdb = str(tmp_path / "6m0j.pdb")
    st.write_pdb(pdb)
    a = structure_token_coords(os.path.join(FIX, "6m0j.cif.gz"), ["B"], "15-208")["B"]
    b = structure_token_coords(pdb, ["E"], "15-208")["E"]
    assert a["sequence"] == b["sequence"] and a["label_seq"] == b["label_seq"]
    assert torch.allclose(a["coord"], b["coord"], atol=1e-3)


def test_coordinates_only_pdb_keeps_the_author_numbering(tmp_path):
    """The test above writes its PDB through gemmi, which emits SEQRES, so label_seq
    could still be assigned by aligning the observed residues against it. A user pastes
    ATOM records and nothing else: there is then no sequence to align to, gemmi leaves
    every label_seq None, and this used to die on `int(None)` deep in the featurizer.

    The author numbering is the fallback, because on such a file it is the only residue
    number the user can see -- and therefore the only one their hotspots and crop can
    mean. 6m0j says that plainly: its RBD is label_seq 15-208 and auth 333-526, so the
    same 194 residues at the same coordinates come back under the numbering that file
    actually shows."""
    import gemmi
    st = gemmi.read_structure(os.path.join(FIX, "6m0j.cif.gz"))
    st.setup_entities()
    full = str(tmp_path / "with_seqres.pdb")
    st.write_pdb(full)
    bare = tmp_path / "atoms_only.pdb"
    bare.write_text("".join(ln + "\n" for ln in open(full).read().splitlines()
                            if ln.startswith(("ATOM", "HETATM", "TER", "END"))))
    assert "SEQRES" not in bare.read_text()
    assert all(r.label_seq is None
               for ch in gemmi.read_structure(str(bare))[0] for r in ch)

    a = structure_token_coords(os.path.join(FIX, "6m0j.cif.gz"), ["B"], "15-208")["B"]
    b = structure_token_coords(str(bare), ["E"], "333-526")["E"]
    assert b["label_seq"][0] == 333 and b["label_seq"][-1] == 526
    assert a["sequence"] == b["sequence"]
    assert torch.allclose(a["coord"], b["coord"], atol=1e-3)


def test_yaml_reader():
    spec = read_design_yaml(PDL1)
    assert spec["chains"] == ["A"] and spec["binder_length"] == 80
    assert spec["crop"] == {"A": ["1-116"]} and spec["hotspots"] == {"A": [40, 99, 107]}
    assert spec["structure"].name == "5o45.cif.gz" and spec["structure"].exists()


@pytest.mark.parametrize("yaml_path,n_target,n_atom", [(PDL1, 116, 1250), (RBD, 194, 1857)])
def test_model_input_shape_and_binder(yaml_path, n_target, n_atom):
    d = design_inputs_from_yaml(yaml_path)
    n_token = n_target + 80
    assert d["restype"].shape == (n_token, 36) and d["ref_pos"].shape == (n_atom, 3)
    # 32 is `xpb`, the binder placeholder, and it is the last 80 tokens
    rt = d["restype"].argmax(-1)
    assert RESTYPE_VOCAB[32] == "xpb"
    assert bool((rt[n_target:] == 32).all()) and not bool((rt[:n_target] == 32).any())
    assert int(d["hotspot"].sum()) == 3 and bool((d["hotspot"][n_target:] == 0).all())
    # the binder is never a condition, and no condition sits at the origin
    c = d["condition"]
    conditioned = torch.tensor([r != "xpb" for r in c["res_name"]]) & c["is_resolved"]
    assert int(conditioned.sum()) == n_target
    assert int(((c["coord"].abs().sum(-1) == 0) & conditioned).sum()) == 0
    assert bool(torch.equal(d["conditional_templ_mask"].bool().any(1), conditioned))


@pytest.mark.parametrize("yaml_path,art", [(PDL1, "pdl1_protenix05_noH"), (RBD, "rbd_6m0j")])
def test_bit_exact_against_the_capture(yaml_path, art):
    """The regression proof: the new parser did not change the working case. 17 of 18 keys
    are bit-exact; `ref_pos` is upstream's unseeded per-residue random rigid transform
    (`ref_pos_augment` defaults True) and does not reproduce upstream either."""
    want = torch.load(os.path.join(ARTS, art, "ref_design_inputs.pt"), weights_only=False)
    got = design_inputs_from_yaml(yaml_path)
    for k in MODEL_INPUT_KEYS:
        if k == "ref_pos":
            continue
        assert torch.equal(got[k].to(want[k].dtype), want[k]), f"{k} is not bit-exact"
    assert got["ref_pos"].shape == want["ref_pos"].shape


def test_his_nd1_carries_the_ccd_formal_charge():
    """The CCD's ideal histidine is the protonated imidazolium, and protenix reads the CCD
    straight through: ND1 is +1 in both the 0.5.5 and the 2.0 captures. `protenix_ref_out.pkl`,
    which `test_protein_atom_metadata_exact` scores against, contains no histidine, so this
    pins the case that golden cannot see."""
    d = design_inputs_from_yaml(PDL1)
    names = [RESTYPE_VOCAB[i] for i in d["restype"].argmax(-1).tolist()]
    a2t = d["atom_to_token_idx"].long()
    chars = d["ref_atom_name_chars"].argmax(-1)
    charged = {(names[int(a2t[i])], "".join(chr(c + 32) for c in chars[i].tolist()).strip())
               for i in (d["ref_charge"] != 0).nonzero().flatten().tolist()}
    assert charged == {("HIS", "ND1"), ("LYS", "NZ"), ("ARG", "NH2")}


def test_binder_length_must_be_positive():
    with pytest.raises(ValueError, match="binder_length"):
        design_inputs(os.path.join(FIX, "5o45.cif.gz"), ["A"], 0, crop="1-116")
