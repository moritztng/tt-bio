"""`tt-bio json` (upstream `mol json`) against Mol-1 e6e64866's own output, no network.

Goldens: upstream's tojson run on pc (biotite 1.7.1, MOL_ROOT_DIR = the official common/ data) for seven structures
(ubiquitin; hemoglobin with heme, also as assembly 1; tRNA-Phe with 14 modified bases and Mg/Mn; B-DNA; SARS-CoV-2
Mpro with the N3 peptide inhibitor; ACE2-RBD with NAG glycans, Zn, Cl; a .pdb input) under the four option sets.
components.cif.gz holds only the CCD entries those structures use, cut from Mol-1's common/components.cif.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tt_bio import weights
from tt_bio.main import cli

D = Path(__file__).parent / "data" / "mol1_json"
VARIANTS = {"default": [], "assembly1": ["--assembly_id", "1"], "discont": ["--include_discont_poly_poly_bonds"],
            "altlocA": ["--altloc", "A"]}


@pytest.fixture(scope="module")
def components(tmp_path_factory):
    p = tmp_path_factory.mktemp("ccd") / "components.cif"
    p.write_bytes(gzip.decompress((D / "components.cif.gz").read_bytes()))
    return p


@pytest.mark.parametrize("variant", VARIANTS)
def test_json_equals_upstreams(components, monkeypatch, tmp_path, variant):
    monkeypatch.setattr(weights, "fetch", lambda key, root=None, **_: components)
    out = tmp_path / "out"
    r = CliRunner().invoke(cli, ["json", str(D / "in"), "--out_dir", str(out), *VARIANTS[variant]])
    assert r.exit_code == 0, r.output
    golden = sorted((D / "golden" / variant).iterdir())
    assert sorted(p.name for p in out.iterdir()) == [p.name for p in golden]
    for g in golden:
        assert json.loads((out / g.name).read_text()) == json.loads(g.read_text()), g.name


def test_a_directory_without_structures_is_an_error(components, monkeypatch, tmp_path):
    monkeypatch.setattr(weights, "fetch", lambda key, root=None, **_: components)
    r = CliRunner().invoke(cli, ["json", str(tmp_path), "--out_dir", str(tmp_path / "o")])
    assert r.exit_code == 1 and "no .pdb or .cif" in r.output


def test_the_components_file_is_mol1s_own():
    art = weights.ARTIFACTS["mol1-components"]
    assert art.sha256 == "64f53e4d72c8f919d9d2afd34183df7c0b6211929dfe0eeb865fad82cd1374d5"
    assert art.models == ()   # predict never fetches it
