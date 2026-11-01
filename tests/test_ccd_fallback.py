"""A CCD code the `mols` library lacks is built from the model's own CCD table (protenix_data.ccd_mols). CPU only.

Feature equality with upstream Mol-1 for such codes is the `ccd_newer` golden job in test_mol1_features.py. Here: each
of the three quality-set codes that used to be refused builds with upstream's heavy atoms, a code in neither source
is still refused by name, and a model without a table (Protenix v1/v2, OpenDDE, PXDesign) is byte-identical to before.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from tt_bio.protenix_data import (CcdTable, _default_mol_dir, build_complex_features, ccd_leaving_groups,
                                  load_conformer_table)

DATA = Path(__file__).parent / "data" / "mol1_io"
GOLDEN_TABLE = DATA / "ccd_conformers_golden.npz"
NEWER = ("A1A5U", "YKX", "A1EKL")      # quality cases pfm__9DBP, 8g3w, pfm__9LLG

pytestmark = pytest.mark.skipif(not (Path(_default_mol_dir()) / "ATP.pkl").exists(),
                                reason="the CCD `mols` library is not cached here")


@pytest.fixture(scope="module")
def table():
    return load_conformer_table(GOLDEN_TABLE)


@pytest.mark.parametrize("code", NEWER)
def test_newer_code_folds_with_its_ccd_atoms(table, code):
    assert not (Path(_default_mol_dir()) / f"{code}.pkl").exists()
    with pytest.raises(ValueError, match=f"'{code}' is not a CCD component"):
        build_complex_features([("MKVL", None, "protein"), (f"CCD_{code}", None, "ligand")])
    f = build_complex_features([("MKVL", None, "protein"), (f"CCD_{code}", None, "ligand")], conformer_table=table)
    heavy = [(n, xyz) for n, xyz in table[code].items() if not n.startswith("H")]
    lig = (f["atom_to_token_idx"] >= 4).numpy()
    names = ["".join(chr(int(c) + 32) for c in row.argmax(-1)).strip() for row in f["ref_atom_name_chars"][lig]]
    assert names == [n for n, _ in heavy]
    assert np.array_equal(f["ref_pos"][lig].numpy(), np.stack([xyz for _, xyz in heavy]))
    assert int(f["token_bonds"][4:, 4:].sum()) > 0


def test_table_mol_carries_leaving_atoms(table):
    """A1APB's O1 leaves when C1 bonds (the ccd_newer golden checks the drop against upstream)."""
    assert ccd_leaving_groups(table.mol("A1APB")) == {"C1": [["O1"]]}


def test_unknown_code_is_still_refused(table):
    with pytest.raises(ValueError, match="'ZZZZZ' is not a CCD component this build knows"):
        build_complex_features([("MKVL", None, "protein"), ("CCD_ZZZZZ", None, "ligand")], conformer_table=table)


def test_conformer_only_table_builds_nothing(tmp_path):
    """A table without topology (the layout before ccd_table.npz) adds no codes."""
    with np.load(GOLDEN_TABLE) as z:
        np.savez(tmp_path / "old.npz", **{k: z[k] for k in ("codes", "counts", "names", "xyz")})
    old = load_conformer_table(tmp_path / "old.npz")
    assert isinstance(old, CcdTable) and old.mol("A1A5U") is None and "A1A5U" in old


def _digest(f):
    h = hashlib.sha256()
    for k in sorted(f):
        v = f[k]
        if hasattr(v, "detach"):
            h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


#: features of known-code inputs on a model without a CCD table, digested on main e8f85baae before the fallback
TABLELESS = {
    "e8f85baae-atp_nag_glycan": (dict(chains=[("MKVLNGTAAS", None, "protein"), ("CCD_NAG,NAG", None, "ligand"),
                                              ("CCD_ATP", None, "ligand")],
                                      chain_ids=["A", "B", "C"], bonds=[(("A", 5, "ND2"), ("B", 1, "C1"))]),
                                 "da4402264fa00f8e"),
    "e8f85baae-mods_na": (dict(chains=[("PREACHINGS", None, "protein"), ("GATTACA", None, "dna"),
                                       ("GUAC", None, "rna")],
                               modifications=[[{"position": 1, "ccd": "HY3"}, {"position": 5, "ccd": "SEP"}],
                                              [{"position": 1, "ccd": "6OG"}], [{"position": 1, "ccd": "2MG"}]]),
                          "69130ac91b3ef073"),
}


@pytest.mark.parametrize("name", TABLELESS)
def test_tableless_models_are_byte_identical(name):
    kw, want = TABLELESS[name]
    assert _digest(build_complex_features(**kw)) == want
