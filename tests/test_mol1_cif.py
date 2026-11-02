"""Mol-1's mmCIF vs upstream's for the same input. CPU only.

The goldens (tests/data/mol1_io/cif/<case>_sample_0.cif.gz) are Mol-1 e6e64866's own prediction CIFs for the jobs
beside them, from the mol1-ref capture (qb2 ~/mol1-goldens/n4, state/mol1/goldens/INDEX.md). The jobs are folded
here through the worker with the network stubbed, so everything but the coordinates and the B-factors (pLDDT) must
read as upstream wrote it: every category other than atom_site whole, and atom_site column by column."""
import gzip
import io
import json
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data" / "mol1_io" / "cif"
COORD = {"Cartn_x", "Cartn_y", "Cartn_z", "B_iso_or_equiv"}


def _fold(tmp_path, case, src=DATA):
    from test_protenix_output import _LogitStub

    from tt_bio import protenix_json
    from tt_bio.worker import _WorkerState

    yaml = sorted(protenix_json.stage(src / f"{case}.json", tmp_path / "staged").glob(f"{case}*.yaml"))[0]
    state = object.__new__(_WorkerState)
    state.model, state.accelerator = _LogitStub(), "cpu"
    work = tmp_path / "out"
    work.mkdir()
    state.predict_one(yaml, {"model": "mol1", "msa_dir": str(tmp_path), "struct_dir": str(work), "output_format": "cif",
                             "recycling_steps": 4, "sampling_steps": 1, "diffusion_samples": 1,
                             "single_sequence": True, "template_structures": str(tmp_path)})
    (cif,) = work.rglob(f"{case}_sample_0.cif")
    return cif


def _block(text):
    import biotite.structure.io.pdbx as pdbx
    return pdbx.CIFFile.read(io.StringIO(text)).block


@pytest.mark.parametrize("case", ["copies", "extras"])
def test_mol1_cif_matches_upstream(tmp_path, case):
    ours_path = _fold(tmp_path, case)
    ours, ref = _block(ours_path.read_text()), _block(gzip.open(DATA / f"{case}_sample_0.cif.gz", "rt").read())
    assert sorted(ours) == sorted(ref)
    for cat in ref:
        if cat == "atom_site":
            continue
        assert set(ours[cat]) == set(ref[cat]), cat
        for col in ref[cat]:
            assert ours[cat][col].as_array().tolist() == ref[cat][col].as_array().tolist(), (cat, col)
    a, b = ours["atom_site"], ref["atom_site"]
    assert set(a) == set(b)
    for col in set(b) - COORD:
        assert a[col].as_array().tolist() == b[col].as_array().tolist(), ("atom_site", col)
    assert ours_path.read_text().startswith(f"data_{case}_sample_0_predicted_by_mol\n")


def test_gemmi_reads_the_entities(tmp_path):
    gemmi = pytest.importorskip("gemmi")
    st = gemmi.read_structure(str(_fold(tmp_path, "copies")))
    st.setup_entities()
    got = {e.name: (e.entity_type.name, e.polymer_type.name, sorted(e.subchains)) for e in st.entities}
    assert got["1"] == ("Polymer", "PeptideL", ["A", "B"])
    assert got["4"] == ("Polymer", "Dna", ["G", "H"])
    assert got["5"] == ("Polymer", "Rna", ["I", "J"])
    assert {k: v[0] for k, v in got.items() if k in "236"} == {"2": "NonPolymer", "3": "NonPolymer", "6": "NonPolymer"}
    seq = {e.name: gemmi.one_letter_code(e.full_sequence) for e in st.entities if e.full_sequence}
    assert seq == {"1": "MKQLEDKVEELLSKNYHLENEVARLKKLVGER", "4": "GATTACA", "5": "GGAUCC"}


def test_smiles_and_file_ligands_are_named_as_upstream(tmp_path):
    """Upstream names each SMILES/file ligand entry l01, l02, ... in input order; copies share the name and a CCD
    ligand keeps its code. The old "LIG" is a CCD compound, so OpenStructure bonded the ligand as that and matched
    none of FoldBench's SMILES/file references."""
    gemmi = pytest.importorskip("gemmi")
    sdf = tmp_path / "eth.sdf"
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles("CCO")); AllChem.EmbedMolecule(m, randomSeed=0)
    Chem.MolToMolFile(Chem.RemoveHs(m), str(sdf))
    job = [{"name": "ligs", "sequences": [
        {"proteinChain": {"sequence": "MKQLEDKVEELLSKNYHLENEVARLKKLVGER", "count": 1}},
        {"ligand": {"ligand": "CC(=O)Oc1ccccc1C(=O)O", "count": 2}},
        {"ligand": {"ligand": "CCD_ATP", "count": 1}},
        {"ligand": {"ligand": f"FILE_{sdf}", "count": 1}}]}]
    (tmp_path / "ligs.json").write_text(json.dumps(job))
    cif = _fold(tmp_path, "ligs", src=tmp_path)
    st = gemmi.read_structure(str(cif))
    names = {ch.name: sorted({r.name for r in ch}) for ch in st[0]}
    del names["A"]
    assert names == {"B": ["l01"], "C": ["l01"], "D": ["ATP"], "E": ["l02"]}
