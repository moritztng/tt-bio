"""The RFD4 quality harness, graded on inputs whose right answer is known in advance."""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perf" / "rfd4"))
import geometry  # noqa: E402
import metrics as mspec  # noqa: E402
import quality  # noqa: E402
import stats  # noqa: E402

FEATURE = "unconditional"
FAST = 2000  # permutations; the bar is 0.05, so 2000 draws resolve it with room to spare


def _write_arm(tmp: pathlib.Path, name: str, rows: list[dict]) -> pathlib.Path:
    d = tmp / name / "evaluation_metrics"
    d.mkdir(parents=True)
    path = d / "per_example_metrics.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return tmp / name


def _rows(n: int, seed: int, rmsd_mu: float = 1.0, pass_rate: float = 0.7) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        rmsd = float(abs(rng.normal(rmsd_mu, 0.25)))
        out.append({
            "example_id": f"{seed}_{i}",
            "designed_structure_path": f"/designs/{seed}_{i}.cif",
            "refolded_backbone_rmsd.rmsd": rmsd,
            "oracle.overall_plddt": float(rng.normal(85, 4)),
            "clash_metrics.frac_atoms_clashing_1p5A": float(abs(rng.normal(0.001, 0.0005))),
            "unconditional.pass": float(rng.random() < pass_rate),
            "amino_acid_composition.n_generated_unknown_residues": 0.0,
        })
    return out


# ----------------------------------------------------------------------------- statistics
def test_kolmogorov_q_matches_the_published_value():
    assert abs(stats.kolmogorov_q(1.0) - 0.26999967) < 1e-7
    assert stats.kolmogorov_q(0.0) == 1.0
    assert stats.kolmogorov_q(2.5) < 1e-5


def test_ks_separates_a_shifted_distribution_and_passes_an_unshifted_one():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 60), rng.normal(0, 1, 60)
    same = stats.ks_2samp(a, b, permutations=FAST)
    assert same.p > 0.05
    shifted = stats.ks_2samp(a + 3.0, b, permutations=FAST)
    assert shifted.p < 0.01 and shifted.d > 0.8


def test_permutation_p_never_reaches_zero():
    a, b = np.arange(40.0), np.arange(40.0) + 1000
    assert stats.ks_2samp(a, b, permutations=200).p == pytest.approx(1 / 201)


def test_fisher_is_one_sided_on_a_drop():
    # 30/30 vs 5/30 is a collapse and must be caught; the same gap the other way is not a failure.
    assert stats.fisher_exact_greater(5, 30, 30, 30) < 1e-6
    assert stats.fisher_exact_greater(30, 30, 5, 30) == pytest.approx(1.0)
    assert stats.fisher_exact_greater(21, 30, 22, 30) > 0.05
    # Hand-checkable hypergeometric tails on a 2x2 with 4 designs an arm.
    # [3 pass, 1 fail] vs [4 pass, 0 fail]: P(X<=3) = C(4,3)C(4,4)/C(8,7) = 4/8.
    assert stats.fisher_exact_greater(3, 4, 4, 4) == pytest.approx(0.5, abs=1e-9)
    # [1, 3] vs [4, 0]: P(X<=1) = C(4,1)C(4,4)/C(8,5) = 4/56.
    assert stats.fisher_exact_greater(1, 4, 4, 4) == pytest.approx(1 / 14, abs=1e-9)


# ----------------------------------------------------------------------------- geometry
def _helix_cif(path: pathlib.Path, n: int = 30, chain: str = "A", nonfinite: bool = False) -> pathlib.Path:
    """An ideal alpha helix: 1.5 A rise, 100 deg per residue, CA-CA = 3.80 A."""
    lines = ["data_test", "loop_", "_atom_site.group_PDB", "_atom_site.label_atom_id",
             "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id",
             "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"]
    r, rise, dphi = 2.3, 1.5, math.radians(100.0)
    for i in range(n):
        x, y, z = r * math.cos(i * dphi), r * math.sin(i * dphi), i * rise
        if nonfinite and i == 3:
            x = float("nan")
        lines.append(f"ATOM CA ALA {chain} {i + 1} {x:.3f} {y:.3f} {z:.3f}")
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_geometry_reads_an_ideal_helix(tmp_path):
    g = geometry.design_geometry(_helix_cif(tmp_path / "h.cif"))
    assert g["geom.n_ca"] == 30
    assert g["geom.frac_ca_bond_outlier"] == 0.0  # ideal helix CA-CA is 3.80 A
    assert g["geom.ca_clash_frac"] == 0.0
    assert 9.0 < g["geom.radius_of_gyration"] < 16.0
    assert g["geom.n_nonfinite_coords"] == 0.0


def test_geometry_flags_nonfinite_coordinates(tmp_path):
    g = geometry.design_geometry(_helix_cif(tmp_path / "nan.cif", nonfinite=True))
    assert g["geom.n_nonfinite_coords"] == 1.0


def test_collapsed_backbone_is_a_clash(tmp_path):
    path = tmp_path / "collapse.cif"
    lines = ["data_x", "loop_", "_atom_site.group_PDB", "_atom_site.label_atom_id",
             "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id",
             "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"]
    for i in range(20):  # a straight line with a 1 A step: every pair within 3.6 A clashes
        lines.append(f"ATOM CA GLY A {i + 1} {i * 1.0:.3f} 0.000 0.000")
    path.write_text("\n".join(lines) + "\n#\n")
    g = geometry.design_geometry(path)
    assert g["geom.ca_clash_frac"] > 0.1
    assert g["geom.frac_ca_bond_outlier"] == 1.0


def test_sequence_of_reads_residue_identities(tmp_path):
    assert geometry.sequence_of(_helix_cif(tmp_path / "s.cif", n=5)) == "AAAAA"


# ----------------------------------------------------------------------------- the grade
def _calibrate(tmp_path, out, feature=None):
    """The real workflow: the floor is measured before any grade is taken."""
    a = _write_arm(tmp_path, "floor_a", _rows(40, 11))
    b = _write_arm(tmp_path, "floor_b", _rows(40, 12))
    assert quality.main(["floor", "--feature", feature or FEATURE, "--a", str(a), "--b", str(b),
                         "--permutations", str(FAST), "--out", str(out)]) == 0


def test_a_grade_without_a_measured_seed_floor_is_not_a_pass(tmp_path):
    """Every bar is quoted beside the floor, so a family with no floor has no justified bar."""
    tt = _write_arm(tmp_path, "tt", _rows(40, 1))
    ref = _write_arm(tmp_path, "ref", _rows(40, 2))
    out = tmp_path / "QUALITY.json"
    rc = quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                       "--permutations", str(FAST), "--out", str(out)])
    assert rc == 1 and json.loads(out.read_text())["features"][FEATURE]["verdict"] == "NO_FLOOR"


def test_two_seeds_of_the_same_distribution_pass(tmp_path):
    tt = _write_arm(tmp_path, "tt", _rows(40, 1))
    ref = _write_arm(tmp_path, "ref", _rows(40, 2))
    out = tmp_path / "QUALITY.json"
    _calibrate(tmp_path, out)
    rc = quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                       "--permutations", str(FAST), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert rc == 0 and doc["verdict"] == "PASS"
    assert doc["features"][FEATURE]["metrics"]["refolded_backbone_rmsd.rmsd"]["pass"] is True
    assert doc["features"][FEATURE]["n_graded_metrics"] >= 4


def test_a_shifted_refold_rmsd_fails(tmp_path):
    tt = _write_arm(tmp_path, "tt", _rows(40, 1, rmsd_mu=2.0))
    ref = _write_arm(tmp_path, "ref", _rows(40, 2, rmsd_mu=1.0))
    out = tmp_path / "QUALITY.json"
    rc = quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                       "--permutations", str(FAST), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert rc == 1 and doc["verdict"] == "FAIL"
    assert "refolded_backbone_rmsd.rmsd" in doc["features"][FEATURE]["failed"]


def test_a_collapsed_pass_rate_fails(tmp_path):
    tt = _write_arm(tmp_path, "tt", _rows(40, 1, pass_rate=0.1))
    ref = _write_arm(tmp_path, "ref", _rows(40, 2, pass_rate=0.8))
    out = tmp_path / "QUALITY.json"
    quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                  "--permutations", str(FAST), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert "unconditional.pass" in doc["features"][FEATURE]["failed"]


def test_unknown_residues_fail_the_guard_even_when_the_reference_has_them_too(tmp_path):
    rows_tt = _rows(40, 1)
    rows_ref = _rows(40, 2)
    for r in rows_tt + rows_ref:
        r["amino_acid_composition.n_generated_unknown_residues"] = 2.0
    tt, ref = _write_arm(tmp_path, "tt", rows_tt), _write_arm(tmp_path, "ref", rows_ref)
    out = tmp_path / "QUALITY.json"
    quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                  "--permutations", str(FAST), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert "amino_acid_composition.n_generated_unknown_residues" in doc["features"][FEATURE]["failed"]


def test_a_move_inside_the_seed_floor_is_not_a_regression(tmp_path):
    """A shift the reference's own seeds already produce passes; a 4x bigger one does not."""
    out = tmp_path / "QUALITY.json"
    a = _write_arm(tmp_path, "ref_a", _rows(40, 2, rmsd_mu=1.0))
    b = _write_arm(tmp_path, "ref_b", _rows(40, 3, rmsd_mu=1.12))  # seed-to-seed spread
    quality.main(["floor", "--feature", FEATURE, "--a", str(a), "--b", str(b),
                  "--permutations", str(FAST), "--out", str(out)])
    floor = json.loads(out.read_text())["seed_floor"][FEATURE]["refolded_backbone_rmsd.rmsd"]
    assert floor["delta_median_abs"] > 0

    inside = _write_arm(tmp_path, "tt_in", _rows(40, 2, rmsd_mu=1.0))
    quality.main(["grade", "--feature", FEATURE, "--tt", str(inside), "--ref", str(a),
                  "--permutations", str(FAST), "--out", str(out)])
    entry = json.loads(out.read_text())["features"][FEATURE]["metrics"]["refolded_backbone_rmsd.rmsd"]
    assert entry["pass"] is True and entry["seed_floor"] == floor

    far = _write_arm(tmp_path, "tt_far", _rows(40, 2, rmsd_mu=1.0 + 4 * floor["delta_median_abs"] + 0.4))
    quality.main(["grade", "--feature", FEATURE, "--tt", str(far), "--ref", str(a),
                  "--permutations", str(FAST), "--out", str(out)])
    entry = json.loads(out.read_text())["features"][FEATURE]["metrics"]["refolded_backbone_rmsd.rmsd"]
    assert entry["pass"] is False and entry["inside_seed_floor"] is False


def test_too_few_designs_is_not_a_pass(tmp_path):
    tt = _write_arm(tmp_path, "tt", _rows(10, 1))
    ref = _write_arm(tmp_path, "ref", _rows(10, 2))
    out = tmp_path / "QUALITY.json"
    rc = quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                       "--permutations", str(FAST), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert rc == 1 and doc["features"][FEATURE]["verdict"] == "INSUFFICIENT_N"


def test_grading_one_feature_leaves_the_others_alone(tmp_path):
    out = tmp_path / "QUALITY.json"
    tt, ref = _write_arm(tmp_path, "tt", _rows(40, 1)), _write_arm(tmp_path, "ref", _rows(40, 2))
    _calibrate(tmp_path, out)
    quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                  "--permutations", str(FAST), "--out", str(out)])
    doc = json.loads(out.read_text())
    doc["features"]["binder"] = {"verdict": "PASS", "metrics": {}}
    out.write_text(json.dumps(doc))
    quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                  "--permutations", str(FAST), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert set(doc["features"]) == {FEATURE, "binder"} and doc["verdict"] == "PASS"


def test_a_feature_with_no_data_is_not_a_pass(tmp_path):
    empty = tmp_path / "empty_a"
    (empty / "evaluation_metrics").mkdir(parents=True)
    other = tmp_path / "empty_b"
    (other / "evaluation_metrics").mkdir(parents=True)
    out = tmp_path / "QUALITY.json"
    rc = quality.main(["grade", "--feature", FEATURE, "--tt", str(empty), "--ref", str(other),
                       "--permutations", str(FAST), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert rc == 1 and doc["features"][FEATURE]["verdict"] == "NO_DATA" and doc["verdict"] == "FAIL"


def test_every_feature_family_has_a_metric_set_and_tags():
    for feature in mspec.SPECIFIC:
        assert mspec.columns_for(feature) and mspec.FEATURE_TAGS[feature]
        assert any(k == "rate" or needs for _, k, needs in mspec.columns_for(feature)) or feature in {
            "inverse_folding"
        }


def test_only_the_generated_cif_counts_as_a_design(tmp_path):
    """Upstream writes the input copy, refolds and chain extracts next to the design."""
    d = tmp_path / "run" / "inference_samples" / "ex"
    d.mkdir(parents=True)
    for name in ("ex_generated.cif", "ex_input.cif", "ex_af2_refolded.cif", "ex_chain.cif"):
        _helix_cif(d / name, n=12)
    (d / "trajectories").mkdir()
    _helix_cif(d / "trajectories" / "frame0.cif", n=12)
    arm = quality.load_arm(tmp_path / "run")
    assert arm["n_designs_parsed"] == 1

    plain = tmp_path / "plain"
    plain.mkdir()
    _helix_cif(plain / "a.cif", n=12)
    _helix_cif(plain / "b_refolded.cif", n=12)
    assert quality.load_arm(plain)["n_designs_parsed"] == 1


def test_a_family_with_no_oracle_columns_is_not_a_pass(tmp_path):
    """The pass criteria and every refold RMSD come from the oracle: without it, nothing is graded."""
    rows_tt, rows_ref = _rows(40, 1), _rows(40, 2)
    for r in rows_tt + rows_ref:  # as if no oracle had run
        for col in ("refolded_backbone_rmsd.rmsd", "oracle.overall_plddt", "unconditional.pass"):
            r[col] = ""
    tt, ref = _write_arm(tmp_path, "tt", rows_tt), _write_arm(tmp_path, "ref", rows_ref)
    out = tmp_path / "QUALITY.json"
    rc = quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                       "--permutations", str(FAST), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert rc == 1 and doc["features"][FEATURE]["verdict"] == "ORACLE_MISSING"
    assert "refolded_backbone_rmsd.rmsd" in doc["features"][FEATURE]["missing_oracle_metrics"]

    rc = quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                       "--allow-no-oracle", "--allow-no-floor",
                       "--permutations", str(FAST), "--out", str(out)])
    assert rc == 0 and json.loads(out.read_text())["features"][FEATURE]["verdict"] == "PASS"


def test_an_evaluate_tree_merges_into_the_arm_without_double_counting(tmp_path):
    """Generation and `rfd4-evaluate` write two trees for one arm, and evaluate re-exports every
    input design under `evaluation_samples/`. The arm is 2 designs, not 4."""
    gen = tmp_path / "gen" / "inference_samples"
    ev = tmp_path / "eval" / "evaluation_samples"
    paths = []
    for i in (0, 1):
        (gen / f"ex{i}").mkdir(parents=True)
        _helix_cif(gen / f"ex{i}" / f"ex{i}_generated.cif", n=12)
        paths.append(str(gen / f"ex{i}" / f"ex{i}_generated.cif"))
        (ev / f"ex{i}_generated_abc{i}").mkdir(parents=True)
        _helix_cif(ev / f"ex{i}_generated_abc{i}" / f"ex{i}_generated.cif", n=12)
    (gen.parent / "inference_metrics").mkdir()
    (ev.parent / "evaluation_metrics").mkdir()
    for d, cols in ((gen.parent / "inference_metrics", {"seed": 0}),
                    (ev.parent / "evaluation_metrics", {"clash_metrics.frac_atoms_clashing_1p5A": 0.01})):
        with (d / "per_example_metrics.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["designed_structure_path", *cols])
            w.writeheader()
            for p in paths:
                w.writerow({"designed_structure_path": p, **cols})
    arm = quality.load_arm([tmp_path / "gen", tmp_path / "eval"])
    assert arm["n"] == 2 and arm["n_designs_parsed"] == 2
    # both CSVs landed on the same rows, and geometry landed once
    assert all("seed" in r and "clash_metrics.frac_atoms_clashing_1p5A" in r and "geom.n_ca" in r
               for r in arm["rows"])


def test_diversity_is_tested_against_the_reference_not_a_ratio_bar():
    """6/8 vs 4/8 unique is one duplicate at n=8. Upstream against itself must not fail it."""
    tt = {"n": 8, "n_unique_sequences": 4}
    ref = {"n": 8, "n_unique_sequences": 6}
    assert quality._grade_diversity(tt, ref)["pass"] is True
    # a real collapse at a real n does fail
    worse = quality._grade_diversity({"n": 40, "n_unique_sequences": 8}, {"n": 40, "n_unique_sequences": 38})
    assert worse["pass"] is False and worse["p"] < mspec.P_BAR
    # every design identical is a broken port whatever the reference did
    dead = quality._grade_diversity({"n": 30, "n_unique_sequences": 1}, {"n": 30, "n_unique_sequences": 1})
    assert dead["pass"] is False and dead["collapsed"] is True


def test_the_floor_watcher_calibrates_from_the_reference_index(tmp_path, monkeypatch):
    """rfd4-fair-gpu's two-seed reference arms become the floor without this row being relaunched."""
    import floor_from_reference as ffr

    seeds = {}
    for seed in ("1234", "5678"):
        root = _write_arm(tmp_path, f"ref_{seed}", _rows(40, int(seed)))
        seeds[seed] = str(root)
    index = tmp_path / "reference-designs.json"
    index.write_text(json.dumps({FEATURE: {"recipe": "thinking_medium bf16",
                                           "session": "vast:57063695", "seeds": seeds}}))
    out = tmp_path / "QUALITY.json"
    monkeypatch.setattr(ffr, "INDEX", index)
    monkeypatch.setattr(quality, "QUALITY_JSON", out)

    assert ffr.main() == 0
    floor = json.loads(out.read_text())["seed_floor"][FEATURE]
    assert floor["_source"] == "thinking_medium bf16 | vast:57063695 | seeds 1234/5678"
    assert floor["_n"] == [40, 40]
    assert "refolded_backbone_rmsd.rmsd" in floor

    # idempotent: a second tick on the same index entry does not recalibrate
    before = out.read_text()
    assert ffr.main() == 0 and out.read_text() == before

    # and a family with one seed arm is reported, not silently skipped
    index.write_text(json.dumps({"binder": {"seeds": {"1234": seeds["1234"]}}}))
    ffr.main()


def test_the_floor_watcher_refuses_arms_below_the_n_bar(tmp_path, monkeypatch, capsys):
    import floor_from_reference as ffr

    seeds = {s: str(_write_arm(tmp_path, f"small_{s}", _rows(10, int(s)))) for s in ("1", "2")}
    index = tmp_path / "reference-designs.json"
    index.write_text(json.dumps({FEATURE: {"seeds": seeds}}))
    out = tmp_path / "QUALITY.json"
    monkeypatch.setattr(ffr, "INDEX", index)
    monkeypatch.setattr(quality, "QUALITY_JSON", out)
    assert ffr.main() == 0
    assert "the floor needs >= 30" in capsys.readouterr().out
    assert not out.exists()
