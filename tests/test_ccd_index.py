"""tt_bio.ccd.CcdMols: a per-code index over ccd.pkl must give the mols pickle.load gives."""

import pickle

import pytest

Chem = pytest.importorskip("rdkit.Chem")


@pytest.fixture(autouse=True)
def _private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_BIO_CACHE", str(tmp_path / "cache"))


def _mol(smiles, names):
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=0)
    m.SetProp("PDB_NAME", smiles)
    m.GetConformer().SetProp("name", "Computed")
    for a, n in zip(m.GetAtoms(), names):
        a.SetProp("name", n)
        a.SetBoolProp("leaving_atom", n.startswith("O"))
    return m


def _key(m):
    return repr(
        (
            Chem.MolToMolBlock(m),
            m.GetPropsAsDict(True, False),
            [a.GetPropsAsDict(True, False) for a in m.GetAtoms()],
            [(c.GetPropsAsDict(True, False), c.GetPositions().tobytes()) for c in m.GetConformers()],
        )
    )


def _write_ccd(path):
    prev = Chem.GetDefaultPickleProperties()
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    try:
        ccd = {"ALA": _mol("CC(N)C(=O)O", ["CB", "CA", "N", "C", "O", "OXT"]), "HOH": _mol("O", ["O"]), "NUL": None}
        with open(path, "wb") as f:
            pickle.dump(ccd, f)
    finally:
        Chem.SetDefaultPickleProperties(prev)


def test_index_matches_the_full_load_written_and_read(tmp_path):
    from tt_bio import ccd as C

    pkl = tmp_path / "ccd.pkl"
    _write_ccd(pkl)
    with open(pkl, "rb") as f:
        full = pickle.load(f)
    for _ in range(2):  # first builds the index, second maps it
        mols = C.CcdMols(pkl)
        assert sorted(mols) == sorted(full) and len(mols) == len(full)
        assert "NOPE" not in mols and "ALA" in mols
        assert mols["NUL"] is None
        for k in ("ALA", "HOH"):
            assert _key(mols[k]) == _key(full[k])
        assert mols["ALA"] is mols["ALA"], "a code is built once"
    assert C.index_path(pkl.resolve()).exists()


def test_index_falls_back_to_memory_when_the_cache_cannot_be_written(tmp_path, monkeypatch):
    from tt_bio import ccd as C

    pkl = tmp_path / "ccd.pkl"
    _write_ccd(pkl)
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the cache dir should be")
    monkeypatch.setenv("TT_BIO_CACHE", str(blocked))
    with open(pkl, "rb") as f:
        full = pickle.load(f)
    mols = C.CcdMols(pkl)
    assert _key(mols["ALA"]) == _key(full["ALA"])


def test_esmfold2_load_ccd_uses_the_index(tmp_path, monkeypatch):
    from tt_bio._vendor.esm.models.esmfold2 import conformers
    from tt_bio.ccd import CcdMols

    pkl = tmp_path / "ccd.pkl"
    _write_ccd(pkl)
    monkeypatch.setattr(conformers, "_CCD_MOLECULES", None)
    monkeypatch.setattr(conformers, "CCD_PICKLE_PATH", pkl)
    monkeypatch.setattr(conformers, "_CCD_CONFORMERS", {})
    assert isinstance(conformers.load_ccd(), CcdMols)
    assert set(conformers.get_ccd_conformer("ALA")) == {"CB", "CA", "N", "C", "O", "OXT"}
