"""A chain comes back under the id it was submitted with, on every predict model.

Protenix-v1/-v2 and OpenDDE wrote chains as A, B, C in input order whatever the input called
them, and RF3 let its featurizer pick the next free letter for a ligand. So a ligand submitted
as L came back as B, and every downstream script that selects a chain by id read the wrong one
(mgx-matrix, 2026-09-23: 23 folded cells on whglx with ``chain_ids_kept: false``).

Each model runs its own front door on the host with the network stubbed out: the reader, the
capability check, the featurizer and the structure writer are the shipped code, and only the
coordinates are fake. RF3 and the OF3 family need their weights for the host-side prep after
featurization, so for those two the featurizer's own atom array goes to the shared writer
instead of a fold.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tt_bio.capabilities import CAPABILITY, HONOURED
from tt_bio.main import PREDICT_MODELS

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.expanduser("~/.boltz/mols")),
                                reason="needs the bundled CCD mol library (~/.boltz/mols)")

#: chain id -> (yaml entry, residues it must come back with). None of the ids is the letter the
#: chain's position would give it, so a writer that relabels by order fails on every chain.
CHAINS = {
    "P": ("  - protein:\n      id: P\n      sequence: MKTAYIAKQRQ\n", 11),
    "R": ("  - rna:\n      id: R\n      sequence: GAUC\n", 4),
    "L": ("  - ligand:\n      id: L\n      ccd: ATP\n", 1),
}
_KIND = {"R": "rna", "L": "ligand"}


def _expected(model):
    """The chains this model folds: a molecule type it refuses is left out of its input."""
    return {c: n for c, (_y, n) in CHAINS.items()
            if c not in _KIND or CAPABILITY[model][_KIND[c]] == HONOURED}


def _input(tmp_path, model):
    p = tmp_path / "ids.yaml"
    p.write_text("version: 1\nsequences:\n"
                 + "".join(CHAINS[c][0] for c in _expected(model)))
    return p


def _chains_in(path):
    """chain id -> residue count, as a reader of the written file sees it."""
    import biotite.structure.io.pdbx as pdbx

    arr = pdbx.get_structure(pdbx.CIFFile.read(str(path)), model=1)
    return {c: len(set(arr.res_id[arr.chain_id == c])) for c in dict.fromkeys(arr.chain_id)}


def _cfg(tmp_path, model):
    return {"model": model, "msa_dir": str(tmp_path), "struct_dir": str(tmp_path),
            "output_format": "cif", "recycling_steps": 1, "sampling_steps": 1,
            "diffusion_samples": 1, "template_structures": str(tmp_path)}


def _state(model):
    from tt_bio.worker import _WorkerState

    state = object.__new__(_WorkerState)
    state.model, state.accelerator = model, "cpu"
    return state


class _ProtenixStub:
    """Protenix.fold / OpenDDE.fold with the network replaced by random coordinates."""

    def fold(self, feats, **_kw):
        n = feats["atom_to_token_idx"].shape[0]
        return [torch.randn(n, 3)], [{"plddt": 0.5, "plddt_atom": torch.zeros(n)}]


class _ESMFold2Stub:
    """The ESMFold2 forward: the vendored builder featurizes before it and decodes after it."""

    device, msa_encoder = torch.device("cpu"), None

    def __call__(self, **f):
        return {"sample_atom_coords": torch.zeros(1, f["ref_pos"].shape[1], 3),
                "plddt": torch.zeros(1, f["token_index"].shape[1])}


class _Featurized(Exception):
    """Raised by the stubbed model once the featurizer's output has been captured."""


def _fold(tmp_path, model, stub):
    p = _input(tmp_path, model)
    _state(stub).predict_one(p, _cfg(tmp_path, model))
    return _chains_in(tmp_path / f"{p.stem}.cif")


def _write_featurized(tmp_path, model, monkeypatch, module, name, atom_array_of):
    """Run the front door up to the featurizer, then write its atom array."""
    import importlib

    from tt_bio.worker import _write_atom_array_structure

    mod = importlib.import_module(module)
    real, seen = getattr(mod, name), []

    def capture(*a, **kw):
        seen.append(real(*a, **kw))
        raise _Featurized

    monkeypatch.setattr(mod, name, capture)
    with pytest.raises(_Featurized):
        _state(None).predict_one(_input(tmp_path, model), _cfg(tmp_path, model))
    arr = atom_array_of(seen[0])
    out = tmp_path / "written.cif"
    _write_atom_array_structure(arr, torch.zeros(arr.array_length(), 3), out, "cif")
    return _chains_in(out)


def _boltz2(tmp_path):
    from tt_bio.data.parse import parse_yaml
    from tt_bio.data.write import to_mmcif

    target = parse_yaml(_input(tmp_path, "boltz2"), {},
                        Path.home() / ".boltz" / "mols", boltz2=True)
    out = tmp_path / "written.cif"
    out.write_text(to_mmcif(target.structure, boltz2=True))
    return _chains_in(out)


def _probe(model, tmp_path, monkeypatch):
    if model == "boltz2":
        return _boltz2(tmp_path)
    if model in ("esmfold2", "esmfold2-fast"):
        return _fold(tmp_path, model, _ESMFold2Stub())
    if model in ("protenix-v1", "protenix-v2", "opendde", "opendde-abag"):
        return _fold(tmp_path, model, _ProtenixStub())
    if model == "rf3":
        return _write_featurized(tmp_path, model, monkeypatch, "tt_bio.rf3.featurize",
                                 "featurize", lambda out: out[0]["atom_array"])
    if model in ("openfold3", "openbind"):
        return _write_featurized(tmp_path, model, monkeypatch, "tt_bio.openfold3_data",
                                 "build_openfold3_features", lambda f: f["atom_array"])
    raise AssertionError(f"no chain-id probe for --model {model}; add one here")


@pytest.mark.parametrize("model", PREDICT_MODELS)
def test_every_chain_comes_back_under_its_submitted_id(tmp_path, monkeypatch, model):
    assert _probe(model, tmp_path, monkeypatch) == _expected(model)


def test_the_protenix_writer_keeps_an_id_longer_than_four_characters(tmp_path):
    """biotite's default chain_id annotation is <U4; assigning into it truncates silently."""
    p = tmp_path / "long.yaml"
    p.write_text("version: 1\nsequences:\n  - protein:\n      id: heavy_chain\n"
                 "      sequence: MKTAYIAKQRQ\n")
    _state(_ProtenixStub()).predict_one(p, _cfg(tmp_path, "protenix-v2"))
    assert _chains_in(tmp_path / "long.cif") == {"heavy_chain": 11}
