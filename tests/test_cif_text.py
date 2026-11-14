"""tt_bio.cif_text writes biotite's CIF text byte for byte, host side."""
import numpy as np
import pytest

pdbx = pytest.importorskip("biotite.structure.io.pdbx")
import biotite.structure as struc  # noqa: E402

from tt_bio import cif_text  # noqa: E402


def _file(categories):
    return pdbx.CIFFile({"t": pdbx.CIFBlock({k: pdbx.CIFCategory(v) for k, v in categories.items()})})


def test_escapes_and_padding_match_biotite():
    odd = ["A", "", "_x", "it's", 'say "hi"', "a b", "a\tb", "CA", "longer_value", "1.250"]
    f = _file({"entry": {"id": "x y"},
               "odd": {"v": odd, "w": [str(i) for i in range(len(odd))], "f": np.linspace(0, 9, len(odd))},
               "num": {"i": np.arange(5), "s": ["H", "HE", "C", "N", "O"]}})
    assert cif_text.serialize(f) == f.serialize()


def test_masked_and_multiline_fall_back_to_the_same_text():
    col = pdbx.CIFColumn(pdbx.CIFData(np.array(["1.5", "2.25", "3.0"])),
                         pdbx.CIFData(np.array([pdbx.MaskValue.PRESENT, pdbx.MaskValue.MISSING,
                                                pdbx.MaskValue.INAPPLICABLE])))
    f = _file({"m": {"a": col, "b": ["p", "q", "r"]},
               "multi": {"a": ["one\ntwo", "x"], "b": ["both ' and \"", "y"]}})
    assert cif_text.serialize(f) == f.serialize()


def test_a_structure_as_the_writers_build_it(tmp_path):
    n = 300
    arr = struc.AtomArray(n)
    arr.coord = np.random.default_rng(0).normal(0, 15, (n, 3)).astype(np.float32)
    arr.chain_id = np.where(np.arange(n) < 150, "A", "B")
    arr.res_id = np.arange(n) // 8 + 1
    arr.res_name = np.where(arr.res_id % 3 == 0, "GLY", "ALA")
    arr.atom_name = np.array(["N", "CA", "C", "O", "CB", "O5'", "C1'", "OXT"])[np.arange(n) % 8]
    arr.element = np.array([a[0] for a in arr.atom_name])
    arr.hetero = np.zeros(n, bool)
    arr.add_annotation("b_factor", float)
    arr.b_factor = np.linspace(0, 100, n)
    f = pdbx.CIFFile()
    pdbx.set_structure(f, arr)
    cif_text.write(f, tmp_path / "a.cif")
    assert (tmp_path / "a.cif").read_text() == f.serialize()
