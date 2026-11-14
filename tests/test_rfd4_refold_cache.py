"""The refold cache: what the Tenstorrent oracle writes, and what evaluate reads back.

Upstream folds a design only when its confidence JSON is missing and otherwise scores the cached
refold, so these two files are the entire seam between a TT refold and every oracle-bound column.
What can be wrong without a card is the naming (upstream's `of3` oracle_name, not `af3`), the
order the pair is written in (a run killed mid-fold must not leave a cache entry), and whether
evaluate's reader finds an OpenFold3 refold at all.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perf" / "rfd4"))

from tt_bio.rfd4 import eval_refold as R
from tt_bio.rfd4 import oracle_of3 as adapter
from tt_bio.rfd4.oracle import OracleRequest


def _oracle(**kw):
    o = adapter.OpenFold3Oracle(msa_dir="/tmp/x", template_structures="/tmp/y", **kw)
    o.model = object()
    return o


def _written(monkeypatch, tmp_path, fail=False):
    import tt_bio.worker as worker

    def write(atom_array, coords, outpath, fmt, b_factors=None):
        if fail:
            raise RuntimeError("structure writer failed")
        outpath.write_text("data_refold\n")

    monkeypatch.setattr(worker, "_write_atom_array_structure", write)
    job = OracleRequest(structure="", example_id="d0_1234abcd", chain_of_interest="A",
                        out_dir=tmp_path / "evaluation_samples" / "d0_1234abcd")
    return job


def test_the_cache_pair_carries_upstreams_of3_names(monkeypatch, tmp_path):
    job = _written(monkeypatch, tmp_path)
    confidences, refolded = _oracle().write_refold(
        job, {"overall_plddt": 81.5, "plddt_coi": 80.0}, None, None, np.zeros(3))

    assert refolded.name == "d0_1234abcd_of3_refolded.cif"
    assert confidences.name == "d0_1234abcd_of3_confidences.json"
    values = json.loads(confidences.read_text())
    # Flat, unprefixed reward keys plus the COI, which is the shape upstream's own file has and
    # `eval_refold.oracle_metrics` reads; the config key prefixes the column, not the file.
    assert values == {"overall_plddt": 81.5, "plddt_coi": 80.0, "chain_of_interest": "A"}


def test_a_ppi_oracle_writes_the_ppi_cache_name(monkeypatch, tmp_path):
    job = _written(monkeypatch, tmp_path)
    confidences, _ = _oracle(name="oracle_ppi", oracle_name="of3_ppi").write_refold(
        job, {"min_ipae_coi": 3.1}, None, None, np.zeros(3))
    assert confidences.name == "d0_1234abcd_of3_ppi_confidences.json"


def test_a_failed_structure_write_leaves_no_cache_entry(monkeypatch, tmp_path):
    """The JSON is the resume key, so it must never exist without its refold beside it."""
    job = _written(monkeypatch, tmp_path, fail=True)
    with pytest.raises(RuntimeError):
        _oracle().write_refold(job, {"overall_plddt": 81.5}, None, None, np.zeros(3))
    assert not (job.out_dir / "d0_1234abcd_of3_confidences.json").exists()


def test_evaluate_reads_an_openfold3_refold(tmp_path):
    samples = tmp_path / "evaluation_samples" / "d0_1234abcd"
    samples.mkdir(parents=True)
    (samples / "d0_1234abcd_of3_confidences.json").write_text(
        json.dumps({"overall_plddt": 77.0, "chain_of_interest": "A"}))
    (samples / "d0_1234abcd_of3_refolded.cif").write_text("data_refold\n")

    columns, refolds, coi = R.oracle_metrics(samples.parent, "d0_1234abcd",
                                             {"unconditional_refolding"})
    assert columns["oracle.overall_plddt"] == 77.0
    assert coi == "A"
    assert refolds["oracle"].name == "d0_1234abcd_of3_refolded.cif"
    assert "oracle.chain_of_interest" not in columns


def test_an_af3_refold_still_wins_when_both_are_cached(tmp_path):
    """AF3 is upstream's default; a directory holding both grades on it, not on the substitute."""
    samples = tmp_path / "evaluation_samples" / "e0_dead1234"
    samples.mkdir(parents=True)
    for name, plddt in (("af3", 90.0), ("of3", 70.0)):
        (samples / f"e0_dead1234_{name}_confidences.json").write_text(json.dumps({"overall_plddt": plddt}))
        (samples / f"e0_dead1234_{name}_refolded.cif").write_text("data_refold\n")

    columns, _refolds, _coi = R.oracle_metrics(samples.parent, "e0_dead1234",
                                               {"unconditional_refolding"})
    assert columns["oracle.overall_plddt"] == 90.0


def test_the_refold_driver_and_evaluate_agree_on_the_example_id(tmp_path):
    """The cache lands under evaluate's own id, so the two cannot drift apart."""
    from tt_bio.rfd4.evaluate import example_id

    design = tmp_path / "sub" / "d0_generated.cif"
    design.parent.mkdir()
    design.write_text("data_x\n")
    assert example_id(design, tmp_path).startswith("d0_generated_")


def test_every_family_names_an_oracle_key_the_metric_set_uses():
    import metrics as mspec
    import refold as driver

    assert set(driver.FAMILY_ORACLE) <= set(mspec.FEATURE_TAGS)
    for key, name in driver.FAMILY_ORACLE.values():
        assert (key, name) in [(k, n) for k, names, _ in R.ORACLES for n in names]


def test_structure_chains_reads_an_mmcif_design(monkeypatch):
    """`tt-bio design --model rfd4` writes mmCIF; reading it as PDB finds no chains at all."""
    import io

    import biotite.structure.io.pdb as pdb
    import biotite.structure.io.pdbx as pdbx

    from tests.test_rfd4_oracle_af2ig import _pdb

    arr = pdb.PDBFile.read(io.StringIO(_pdb())).get_structure(model=1)
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, arr)
    buf = io.StringIO()
    cif.write(buf)
    assert adapter.structure_chains(buf.getvalue()) == ["A", "B"]
