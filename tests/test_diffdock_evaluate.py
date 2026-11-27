"""The DiffDock evaluation metrics: do they say what upstream's evaluate.py says?

Card-free. Every case is built from molecules written here rather than from the benchmark set,
so the suite runs anywhere and a failure names the arithmetic, not the data.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from tt_bio.diffdock import evaluate

Chem = pytest.importorskip("rdkit.Chem")
AllChem = pytest.importorskip("rdkit.Chem.AllChem")
# spyrmsd ships in the `eval` extra, not the base install.
needs_spyrmsd = pytest.mark.skipif(importlib.util.find_spec("spyrmsd") is None,
                                   reason="spyrmsd not installed (tt-bio[eval])")


def _mol(smiles="c1ccccc1C(=O)O", seed=7):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    return Chem.RemoveHs(m)


def _shift(mol, d):
    out = Chem.Mol(mol)
    c = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        p = c.GetAtomPosition(i)
        c.SetAtomPosition(i, (p.x + d[0], p.y + d[1], p.z + d[2]))
    return out


@needs_spyrmsd
def test_symmetry_rmsd_matches_upstream():
    """A rigid translation moves the RMSD by exactly its length, and a symmetric ligand is
    scored on its shape: benzoic acid relabelled around the ring is the same pose, so the
    symmetry-corrected RMSD is 0 while a naive atom-order RMSD is not."""
    m = _mol()
    assert evaluate.symmetry_rmsd(m, m) == (0.0, False)

    d = (0.3, -0.4, 0.0)
    r, naive = evaluate.symmetry_rmsd(m, _shift(m, d))
    assert not naive
    assert r == pytest.approx(math.sqrt(sum(x * x for x in d)), abs=1e-6)

    # Same molecule, atoms renumbered: upstream's get_symmetry_rmsd is invariant to it.
    order = list(range(m.GetNumAtoms()))
    order = order[3:] + order[:3]
    renum = Chem.RenumberAtoms(m, order)
    sym, naive = evaluate.symmetry_rmsd(m, renum)
    assert not naive and sym == pytest.approx(0.0, abs=1e-6)
    x, y = m.GetConformer().GetPositions(), renum.GetConformer().GetPositions()
    assert float(np.sqrt(((x - y) ** 2).sum(axis=1).mean())) > 0.5


@needs_spyrmsd
def test_best_symmetry_rmsd_takes_the_minimum_over_crystal_copies():
    """PoseBusters ships every crystal copy and upstream's evaluator minimises over them."""
    m = _mol()
    near, far = _shift(m, (0.2, 0, 0)), _shift(m, (5.0, 0, 0))
    best, naive = evaluate.best_symmetry_rmsd([far, near], m)
    assert not naive and best == pytest.approx(0.2, abs=1e-6)


@needs_spyrmsd
def test_metric_block_matches_upstream_csv():
    """success_rates reproduces upstream's performance_metrics on a hand-checked matrix.

    Four complexes x 5 samples. Rank 1 is column 0, so `rmsds_below_2` is the pooled rate over
    all 20 samples, `min_rmsds_below_2` the any-sample rate, `top5_rmsds_below_2` the best of
    the first five. The expected values are computed by hand below, not by re-running the code.
    """
    r = np.array([
        [1.0, 1.5, 3.0, 4.0, 9.0],   # rank1 hit
        [3.0, 1.2, 6.0, 7.0, 8.0],   # rank1 miss, a hit deeper in
        [6.0, 6.5, 7.0, 8.0, 9.0],   # no hit at all
        [0.5, 0.6, 0.7, 0.8, 0.9],   # all hits
    ])
    m = evaluate.success_rates(r)
    assert m["rmsds_below_2"] == pytest.approx(100 * 8 / 20)     # 2 + 1 + 0 + 5 cells under 2 A
    assert m["rmsds_below_5"] == pytest.approx(100 * 11 / 20)    # 4 + 2 + 0 + 5
    assert m["min_rmsds_below_2"] == pytest.approx(75.0)         # 3 of 4 complexes
    assert m["min_rmsds_below_5"] == pytest.approx(75.0)
    assert m["top5_rmsds_below_2"] == pytest.approx(75.0)
    assert m["mean_rmsd"] == pytest.approx(r.mean(), abs=5e-4)
    assert m["rmsds_percentile_50"] == pytest.approx(np.percentile(r, 50).round(2))
    assert m["n_complexes"] == 4 and m["n_samples"] == 5
    assert "top10_rmsds_below_2" not in m                        # only 5 samples

    # Upstream's top-5 centroid is the centroid of the BEST-RMSD pose among the first five,
    # not the smallest centroid in the row.
    cen = np.array([[9.0, 0.1, 0.1, 0.1, 0.1]] * 4)
    m2 = evaluate.success_rates(r, centroid_distances=cen)
    assert m2["top5_centroid_below_2"] == pytest.approx(25.0)    # only row 1 picks a column
                                                             # other than 0, where the centroid is 9 A


def test_steric_metrics():
    """Cross distance is the closest ligand-receptor approach; the self-intersection fraction
    counts poses whose two nearest atoms sit under upstream's 0.4 A bar."""
    lig = np.array([[0.0, 0, 0], [1.5, 0, 0]])
    rec = np.array([[0.0, 3.0, 0], [10.0, 0, 0]])
    assert evaluate.min_cross_distance(lig, rec) == pytest.approx(3.0)
    assert evaluate.min_self_distance(lig) == pytest.approx(1.5)

    clash = np.array([[0.0, 0, 0], [0.3, 0, 0]])
    assert evaluate.min_self_distance(clash) == pytest.approx(0.3)
    assert evaluate.self_intersect_fraction([1.5, 0.3, 0.39, 0.41]) == pytest.approx(50.0)
    assert evaluate.CLASH_CUTOFF_A == 0.4


@needs_spyrmsd
def test_benchmark_driver_small_set(tmp_path: Path):
    """grade_set reads a two-complex set off disk, and a complex that wrote no pose stays in
    the list as unscored rather than leaving the denominator."""
    truth, poses = tmp_path / "truth", tmp_path / "poses"
    m = _mol()
    for cid, offset in (("AAAA_LIG", (0.0, 0, 0)), ("BBBB_LIG", (0.0, 0, 0))):
        (truth / cid).mkdir(parents=True)
        w = Chem.SDWriter(str(truth / cid / f"{cid}_ligand.sdf"))
        w.write(_shift(m, offset))
        w.close()
    (poses / "AAAA_LIG").mkdir(parents=True)
    for rank, d in ((1, (0.5, 0, 0)), (2, (4.0, 0, 0))):
        w = Chem.SDWriter(str(poses / "AAAA_LIG" / f"rank{rank}.sdf"))
        w.write(_shift(m, d))
        w.close()
    (poses / "BBBB_LIG").mkdir(parents=True)
    (poses / "BBBB_LIG" / "rank1.sdf").write_text("")   # the 0-byte file a crashed writer leaves

    out = evaluate.grade_set(["AAAA_LIG", "BBBB_LIG"], poses, truth)
    assert out["n_attempted"] == 2 and out["n_scored"] == 1
    assert [r["id"] for r in out["per_complex"] if not r["scored"]] == ["BBBB_LIG"]
    scored = next(r for r in out["per_complex"] if r["scored"])
    assert scored["rmsd"][0] == pytest.approx(0.5, abs=1e-6)
    assert scored["rmsd"][1] == pytest.approx(4.0, abs=1e-6)
    assert out["samples_graded"] == 2
    assert out["metrics"]["min_rmsds_below_2"] == pytest.approx(100.0)
    assert out["metrics"]["rmsds_below_2"] == pytest.approx(50.0)
    assert out["nan_or_collapse"] == 0 and out["n_naive_rmsd"] == 0


def test_posebusters_validity_rate(tmp_path: Path):
    """The validity rate is the posebusters suite's, with the physics-only rate beside it so a
    pose is not counted twice against an RMSD success rate."""
    pytest.importorskip("posebusters")
    m = _mol()
    pred, true = tmp_path / "pred.sdf", tmp_path / "true.sdf"
    for path, mol in ((pred, _shift(m, (0.4, 0, 0))), (true, m)):
        w = Chem.SDWriter(str(path))
        w.write(mol)
        w.close()
    out = evaluate.posebusters_validity([{"mol_pred": str(pred), "mol_true": str(true)}],
                                        config="mol", table_path=tmp_path / "pb.csv")
    assert out["n"] == 1
    assert 0.0 <= out["validity_rate"] <= 1.0
    assert out["physics_validity_rate"] >= out["validity_rate"]   # one check fewer to fail
    assert (tmp_path / "pb.csv").exists()


def test_gnina_passthrough_skips_without_binary(tmp_path: Path, monkeypatch):
    """gnina minimisation is optional and off by default. Without the binary it must raise:
    reporting an unminimised pose as minimised would overstate the result."""
    monkeypatch.setattr(evaluate.shutil, "which", lambda _b: None)
    assert not evaluate.gnina_available()
    with pytest.raises(FileNotFoundError):
        evaluate.gnina_minimize(tmp_path / "p.sdf", tmp_path / "r.pdb", tmp_path / "o.sdf")
