"""Chain ids the engine used to mangle, crash on, or accept when it should not.

Six bugs agx-chain-names found once JapanFold stopped renaming chains: a long id crashed
OpenFold3's MSA mapper (biotite's <U4), a repeated id folded as one merged chain, ESMFold2's
mmCIF writer truncated ids at 16 characters, RF3/OpenFold3 PDB output raised after the fold on a
multi-character id, a space in an id crashed RF3's spec parser, and AF2-IG accepted the same
chain as target and binder. Everything here runs the shipped host path on CPU; the network is
stubbed or never reached.
"""
from __future__ import annotations

import os

import click
import pytest
import torch

from tests.test_chain_ids_survive_the_fold import (_cfg, _chains_in, _ESMFold2Stub,
                                                   _Featurized, _state)
from tt_bio.af2ig_input import read_af2ig_input
from tt_bio.capabilities import check_capabilities, check_chain_ids
from tt_bio.main import _read_bio_chains

needs_ccd = pytest.mark.skipif(not os.path.exists(os.path.expanduser("~/.boltz/mols")),
                               reason="needs the bundled CCD mol library (~/.boltz/mols)")

LONG = "heavy_chain"


def _yaml(tmp_path, *chains, name="in.yaml"):
    """A protein-only input, one entry per (id, sequence)."""
    p = tmp_path / name
    p.write_text("version: 1\nsequences:\n" + "".join(
        f"  - protein:\n      id: {cid!r}\n      sequence: {seq}\n" for cid, seq in chains))
    return p


def _of3_atom_array(tmp_path, monkeypatch, model, path):
    """OpenFold3's real featurizer (MSA token mapper included) on ``path``, its atom array."""
    import tt_bio.openfold3_data as mod

    real, seen = mod.build_openfold3_features, []

    def capture(*a, **kw):
        seen.append(real(*a, **kw))
        raise _Featurized

    monkeypatch.setattr(mod, "build_openfold3_features", capture)
    with pytest.raises(_Featurized):
        _state(None).predict_one(path, _cfg(tmp_path, model))
    return seen[0]["atom_array"]


# 1. OpenFold3 / OpenBind: an id over four characters matched no atom in the MSA mapper.
@needs_ccd
@pytest.mark.parametrize("model", ["openfold3", "openbind"])
def test_of3_featurizes_an_id_longer_than_four(tmp_path, monkeypatch, model):
    arr = _of3_atom_array(tmp_path, monkeypatch, model,
                          _yaml(tmp_path, (LONG, "MKTAYIAKQRQ"), ("B", "GSHMLE")))
    assert list(dict.fromkeys(arr.chain_id)) == [LONG, "B"]


@needs_ccd
def test_of3_short_ids_keep_the_default_width(tmp_path, monkeypatch):
    """An id of four or fewer keeps biotite's <U4, so existing folds see the same array."""
    arr = _of3_atom_array(tmp_path, monkeypatch, "openfold3",
                          _yaml(tmp_path, ("A", "MKTAYIAKQRQ"), ("ABCD", "GSHMLE")))
    assert arr.chain_id.dtype.str == "<U4"


# 2. A repeated chain id is refused by the one reader every model shares.
@pytest.mark.parametrize("text, suffix", [
    ("version: 1\nsequences:\n  - protein:\n      sequence: MKT\n"
     "  - protein:\n      sequence: GSH\n", ".yaml"),
    ("version: 1\nsequences:\n  - protein:\n      id: [A, A]\n      sequence: MKT\n", ".yaml"),
    (">A|protein\nMKT\n>A|protein\nGSH\n", ".fasta"),
])
def test_a_repeated_chain_id_is_refused(tmp_path, text, suffix):
    p = tmp_path / f"dup{suffix}"
    p.write_text(text)
    with pytest.raises(click.ClickException, match="'A'.*more than once"):
        _read_bio_chains(p)


def test_distinct_ids_still_read(tmp_path):
    chains = _read_bio_chains(_yaml(tmp_path, ("A", "MKT"), ("B", "GSH")))
    assert [c[0] for c in chains] == ["A", "B"]


# 3. ESMFold2's mmCIF writer kept 16 characters of an id and merged ids sharing that prefix.
def test_esmfold2_writes_ids_longer_than_sixteen(tmp_path):
    a, b = "antibody_heavy_chain_1", "antibody_heavy_chain_2"
    p = _yaml(tmp_path, (a, "MKTAYIAKQRQ"), (b, "GSHMLE"), name="long.yaml")
    _state(_ESMFold2Stub()).predict_one(p, _cfg(tmp_path, "esmfold2"))
    assert _chains_in(tmp_path / "long.cif") == {a: 11, b: 6}


# 4. RF3 / OpenFold3 / OpenBind PDB output goes through tt_bio.data.pdb like the others.
def _small_array(chain_ids):
    import biotite.structure as struc
    import numpy as np

    arr = struc.AtomArray(len(chain_ids))
    arr.chain_id = np.array(chain_ids)
    arr.res_id[:] = np.arange(1, len(chain_ids) + 1)
    arr.res_name[:], arr.atom_name[:], arr.element[:] = "ALA", "CA", "C"
    return arr


def test_atom_array_pdb_relabels_a_long_id(tmp_path):
    from tt_bio.worker import _write_atom_array_structure

    out = tmp_path / "x.pdb"
    _write_atom_array_structure(_small_array([LONG, "B"]), torch.zeros(2, 3), out, "pdb")
    text = out.read_text()
    assert f"REMARK 999 CHAIN A IS {LONG}" in text
    assert sorted({ln[21] for ln in text.splitlines() if ln.startswith("ATOM")}) == ["A", "B"]


def test_atom_array_pdb_with_one_letter_ids_is_byte_identical(tmp_path):
    """The biotite write the worker used before, for an array that was always PDB-legal."""
    import biotite.structure.io.pdb as _pdb

    from tt_bio.worker import _write_atom_array_structure

    arr = _small_array(["A", "B"])
    new = tmp_path / "new.pdb"
    _write_atom_array_structure(arr, torch.zeros(2, 3), new, "pdb")
    old_arr = arr.copy()
    old_arr.coord[:] = 0.0
    old_arr.add_annotation("b_factor", float)
    old_arr.add_annotation("occupancy", float)
    old_arr.occupancy[:] = 1.0
    pf = _pdb.PDBFile()
    pf.set_structure(old_arr)
    pf.write(str(tmp_path / "old.pdb"))
    assert new.read_bytes() == (tmp_path / "old.pdb").read_bytes()


@needs_ccd
def test_of3_pdb_output_with_a_long_id(tmp_path, monkeypatch):
    from tt_bio.worker import _write_atom_array_structure

    arr = _of3_atom_array(tmp_path, monkeypatch, "openfold3",
                          _yaml(tmp_path, (LONG, "MKTAYIAKQRQ")))
    out = tmp_path / "of3.pdb"
    _write_atom_array_structure(arr, torch.zeros(arr.array_length(), 3), out, "pdb")
    assert f"REMARK 999 CHAIN A IS {LONG}" in out.read_text()


def test_pdb_output_past_62_chains_is_refused_before_the_fold():
    chains = [(f"c{i}", "MKT", None, "protein", None) for i in range(63)]
    with pytest.raises(ValueError, match="at most 62 chains"):
        check_chain_ids(chains, "openfold3", "pdb")
    check_chain_ids(chains, "openfold3", "cif")


# 5. RF3 writes ids unquoted into an mmCIF line, so whitespace in one is refused up front.
def test_rf3_refuses_whitespace_in_a_chain_id(tmp_path):
    p = _yaml(tmp_path, ("heavy chain", "MKTAYIAKQRQ"))
    chains = _read_bio_chains(p)
    with pytest.raises(ValueError, match="whitespace in a chain id"):
        check_capabilities(p, chains, "rf3", echo=None)
    check_capabilities(p, chains, "protenix-v2", echo=None)


def test_rf3_front_door_refuses_whitespace_before_featurizing(tmp_path, monkeypatch):
    import tt_bio.rf3.featurize as feat

    monkeypatch.setattr(feat, "featurize", lambda *a, **k: pytest.fail("featurized"))
    with pytest.raises(ValueError, match="whitespace"):
        _state(None).predict_one(_yaml(tmp_path, ("heavy chain", "MKTAYIAKQRQ")),
                                 _cfg(tmp_path, "rf3"))


# 6. AF2-IG: one chain cannot be both the target and the binder.
def test_af2ig_refuses_the_same_chain_twice(tmp_path):
    p = tmp_path / "af2ig.yaml"
    p.write_text("target:\n  structure: ATOM\n  chain: A\nbinder:\n  sequence: MKT\n  chain: A\n")
    with pytest.raises(ValueError, match="both 'A'"):
        read_af2ig_input(p)
