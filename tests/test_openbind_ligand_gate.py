"""The ligand gate: OpenBind folds protein-ligand complexes, OF3-preview2 still refuses.

Host-only, no card. Guards the asymmetry deliberately introduced when ligands were enabled,
because it is the kind of thing a later "simplification" removes on the reasoning that the
two models share every module. They do share every module; they do not share training.
OpenBind is the checkpoint upstream trained and evaluated for protein-ligand co-folding.
preview2 was released as a polymer model, and its featurizer would happily build a ligand
and its sampler would happily emit a status=ok structure for it -- garbage from a
checkpoint that was never trained for the task, which is the same silent-garbage class the
gate exists to stop.

Also pins the Query translation, since a LIGAND chain is shaped differently from a polymer
one upstream (inference_query_format.Chain): smiles OR ccd_codes, and no sequence. Getting
that wrong does not raise; it folds something else.
"""
import pytest

from tt_bio.worker import _validate_openfold3_chains

# (chain_id, sequence-or-ligand-spec, msa_spec, molecule_type) -- what _read_bio_chains emits.
_PROT = ("A", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
         None, "protein")
_SMILES = ("B", "c1ccccc1", None, "ligand")
_CCD = ("B", "CCD_ATP", None, "ligand")


@pytest.mark.parametrize("lig", [_SMILES, _CCD], ids=["smiles", "ccd"])
def test_openbind_accepts_ligands(lig):
    _validate_openfold3_chains([_PROT, lig], "openbind")


@pytest.mark.parametrize("lig", [_SMILES, _CCD], ids=["smiles", "ccd"])
def test_openfold3_still_refuses_ligands(lig):
    with pytest.raises(RuntimeError, match="polymer-only"):
        _validate_openfold3_chains([_PROT, lig], "openfold3")
    # and the refusal has to point somewhere useful, not just say no
    with pytest.raises(RuntimeError, match="openbind"):
        _validate_openfold3_chains([_PROT, lig], "openfold3")


def test_blank_ligand_spec_is_refused_on_both():
    """An empty ligand spec builds no molecule; it must not fold to a status=ok structure.
    The ligand spec rides the sequence slot, so it is the same blank check as a polymer."""
    for model in ("openbind", "openfold3"):
        with pytest.raises(RuntimeError):
            _validate_openfold3_chains([_PROT, ("B", "   ", None, "ligand")], model)


def test_polymer_only_models_unaffected():
    """preview2's polymer behaviour is byte-identical to before the ligand work."""
    for model in ("openbind", "openfold3"):
        _validate_openfold3_chains([_PROT], model)
        _validate_openfold3_chains([_PROT, ("B", "GAUC", None, "rna")], model)
        with pytest.raises(RuntimeError, match="empty/whitespace"):
            _validate_openfold3_chains([("A", "", None, "protein")], model)


def test_ligand_query_chain_shape():
    """A LIGAND Chain upstream takes smiles OR ccd_codes and NO sequence; a polymer takes
    a sequence and neither. Validated through the real upstream pydantic model, so a field
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
