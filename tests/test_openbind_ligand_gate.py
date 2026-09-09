"""The ligand gate: OpenBind folds protein-ligand complexes, OF3-preview2 still refuses.

Host-only, no card. Guards an asymmetry that was introduced deliberately, because it is the
kind of thing a later "simplification" removes on the reasoning that the two models share
every module. They do share every module; they do not share training. OpenBind is the
checkpoint upstream trained and evaluated for protein-ligand co-folding. preview2 was
released as a polymer model, and its featurizer would happily build a ligand and its sampler
would happily emit a status=ok structure for it -- garbage from a checkpoint never trained
for the task, which is the silent-garbage class the gate exists to stop.

The refusal itself lives in the capability table (tt_bio/capabilities.py) and is exercised
across every model in tests/test_input_capabilities.py. What is specific here is the
asymmetry and the Query translation: a LIGAND chain is shaped differently from a polymer one
upstream (inference_query_format.Chain), and getting that wrong does not raise, it folds
something else.
"""
import pytest

from tt_bio.capabilities import CAPABILITY, HONOURED, REFUSED, check_input

_PROT = ("A", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
         None, "protein", None)
_SMILES = ("B", "c1ccccc1", None, "ligand", None)
_CCD = ("B", "CCD_ATP", None, "ligand", None)


def test_the_asymmetry_is_declared():
    assert CAPABILITY["openbind"]["ligand"] == HONOURED
    assert CAPABILITY["openfold3"]["ligand"] == REFUSED


@pytest.mark.parametrize("lig", [_SMILES, _CCD], ids=["smiles", "ccd"])
def test_openbind_accepts_ligands_and_preview2_refuses_them(tmp_path, lig):
    p = tmp_path / "q.yaml"
    p.write_text("version: 1\nsequences: []\n")
    check_input(p, [_PROT, lig], "openbind", echo=None)
    with pytest.raises(RuntimeError) as e:
        check_input(p, [_PROT, lig], "openfold3", echo=None)
    # the refusal has to say why and point somewhere useful, not just say no
    assert "polymer-only" in str(e.value) and "openbind" in str(e.value)


def test_a_blank_ligand_spec_never_reaches_a_model(tmp_path):
    """An empty ligand spec builds no molecule; it must not fold to a status=ok structure.
    The spec rides the sequence slot, so this is the same blank check a polymer gets, and it
    is now in the one reader instead of per model."""
    from tt_bio.main import _read_bio_chains

    p = tmp_path / "q.yaml"
    p.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MQIF\n"
                 "  - ligand:\n      id: B\n      smiles: '   '\n")
    with pytest.raises(Exception, match="empty/whitespace"):
        _read_bio_chains(p)


def test_ligand_query_chain_shape():
    """A LIGAND Chain upstream takes smiles OR ccd_codes and NO sequence; a polymer takes a
    sequence and neither. Validated through the real upstream pydantic model, so a field
    rename or a tightened validator upstream fails here rather than at fold time."""
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        Chain,
    )

    smiles = Chain(molecule_type="LIGAND", chain_ids=["B"], smiles="c1ccccc1")
    assert smiles.molecule_type.name == "LIGAND"
    assert smiles.sequence is None and smiles.ccd_codes is None

    ccd = Chain(molecule_type="LIGAND", chain_ids=["B"], ccd_codes=["ATP"])
    assert ccd.ccd_codes == ["ATP"] and ccd.smiles is None and ccd.sequence is None

    prot = Chain(molecule_type="PROTEIN", chain_ids=["A"], sequence="MQIF")
    assert prot.smiles is None and prot.ccd_codes is None
