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
            # upstream writes the presence column beside every pass criterion
            "unconditional.all_metrics_present": 1.0,
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
    assert rc == 1 and doc["features"][FEATURE]["verdict"] == "NO_DATA" and doc["verdict"] == "NO_DATA"


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
    (plain / "inputs").mkdir()
    _helix_cif(plain / "inputs" / "template.cif", n=12)
    assert quality.load_arm(plain)["n_designs_parsed"] == 1


def test_a_family_with_no_oracle_columns_is_not_a_pass(tmp_path):
    """The pass criteria and every refold RMSD come from the oracle: without it, nothing is graded."""
    rows_tt, rows_ref = _rows(40, 1), _rows(40, 2)
    for r in rows_tt + rows_ref:  # as if no oracle had run
        for col in ("refolded_backbone_rmsd.rmsd", "oracle.overall_plddt", "unconditional.pass",
                    "unconditional.all_metrics_present"):
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
    assert floor["_source"] == "thinking_medium bf16 | vast:57063695 | two-seed arms 1234/5678"
    assert floor["_kind"] == "two-seed"
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


def test_an_arm_rejected_by_a_pre_oracle_filter_is_no_data(tmp_path):
    """A filter the designs cannot satisfy fails all of them and upstream skips every metric.
    Observed for real: adding `unindexed` and `3d_coordinate_conditioning` to the tags of a plain
    monomer run took the CSV from 41 columns to 9, all metrics gone, pre_oracle_pass 0 throughout."""
    rows_tt, rows_ref = _rows(40, 1), _rows(40, 2)
    for r in rows_ref:
        r["pre_oracle_pass"] = 1
    for r in rows_tt:  # as upstream writes it when nothing cleared the filters
        r["pre_oracle_pass"] = 0
        for col in list(r):
            if col not in ("designed_structure_path", "example_id", "pre_oracle_pass"):
                r[col] = ""
    tt, ref = _write_arm(tmp_path, "tt", rows_tt), _write_arm(tmp_path, "ref", rows_ref)
    out = tmp_path / "QUALITY.json"
    rc = quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                       "--permutations", str(FAST), "--out", str(out)])
    res = json.loads(out.read_text())["features"][FEATURE]
    assert rc == 1 and res["verdict"] == "NO_DATA"
    assert any("pre-oracle filter" in n for n in res["notes"])


def test_the_reference_index_names_one_entry_per_seed_and_refuses_a_single_seed(tmp_path, capsys):
    import reference_index as ri

    root = tmp_path / "reference"
    for seed in ("1234", "5678"):
        d = root / f"binder_seed{seed}" / "inference_samples" / "ex"
        d.mkdir(parents=True)
        _helix_cif(d / "ex_generated.cif", n=12)
    out = tmp_path / "reference-designs.json"
    rc = ri.main(["--root", str(root), "--family", "binder", "--recipe", "thinking_medium bf16",
                  "--session", "vast:57071268", "--out", str(out)])
    entry = json.loads(out.read_text())["binder"]
    assert rc == 0 and sorted(entry["seeds"]) == ["1234", "5678"]
    assert entry["recipe"] == "thinking_medium bf16" and entry["session"] == "vast:57071268"

    only = tmp_path / "one"
    d = only / "binder_seed1234" / "inference_samples" / "ex"
    d.mkdir(parents=True)
    _helix_cif(d / "ex_generated.cif", n=12)
    assert ri.main(["--root", str(only), "--family", "binder", "--recipe", "r", "--session", "s",
                    "--out", str(tmp_path / "one.json")]) == 1
    assert "needs TWO seeds" in capsys.readouterr().out


def test_the_overall_verdict_says_why_it_is_not_a_pass():
    """FAIL is reserved for a metric that moved. An ungraded campaign must not look like one."""
    assert quality.overall_verdict({"features": {}}) == "NOT_GRADED"
    assert quality.overall_verdict({"features": {"a": {"verdict": "PASS"}}}) == "PASS"
    assert quality.overall_verdict(
        {"features": {"a": {"verdict": "PASS"}, "b": {"verdict": "INSUFFICIENT_N"}}}) == "INSUFFICIENT_N"
    # a real failure outranks every other reason
    assert quality.overall_verdict(
        {"features": {"a": {"verdict": "FAIL"}, "b": {"verdict": "ORACLE_MISSING"}}}) == "FAIL"


def _two_chain_cif(path: pathlib.Path, binder_n: int, target_n: int) -> pathlib.Path:
    """A complex: chain A is the designed binder, chain B a big fixed target."""
    lines = ["data_x", "loop_", "_atom_site.group_PDB", "_atom_site.label_atom_id",
             "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id",
             "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"]
    i = 0
    for ch, n, res, offset in (("A", binder_n, "ALA", 0.0), ("B", target_n, "GLY", 200.0)):
        for k in range(n):
            i += 1
            lines.append(f"ATOM CA {res} {ch} {k + 1} {offset + k * 3.8:.3f} 0.000 0.000")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_complex_is_graded_on_its_chain_of_interest(tmp_path):
    """A radius of gyration dominated by a fixed target barely moves whatever the port does."""
    cif = _two_chain_cif(tmp_path / "ex_generated.cif", binder_n=10, target_n=200)
    whole = geometry.design_geometry(cif)
    binder = geometry.design_geometry(cif, chain="A")
    assert binder["geom.n_ca"] == 10 and whole["geom.n_ca"] == 210
    assert binder["geom.radius_of_gyration"] < whole["geom.radius_of_gyration"] / 5
    assert geometry.sequence_of(cif, chain="A") == "A" * 10

    # load_arm takes the chain from upstream's own per-design column
    d = tmp_path / "run" / "inference_samples" / "ex"
    d.mkdir(parents=True)
    _two_chain_cif(d / "ex_generated.cif", binder_n=10, target_n=200)
    m = tmp_path / "run" / "evaluation_metrics"
    m.mkdir(parents=True)
    with (m / "per_example_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["designed_structure_path", "chain_of_interest"])
        w.writeheader()
        w.writerow({"designed_structure_path": str(d / "ex_generated.cif"), "chain_of_interest": "A"})
    arm = quality.load_arm(tmp_path / "run")
    assert arm["rows"][0]["geom.n_ca"] == 10 and arm["rows"][0]["geom.chain_not_found"] == 0.0


def test_an_unmatched_chain_id_keeps_the_design_and_flags_it(tmp_path):
    """Dropping the design instead would quietly shrink n and bias the grade."""
    cif = _two_chain_cif(tmp_path / "ex_generated.cif", binder_n=10, target_n=20)
    geom = geometry.design_geometry(cif, chain="Z")
    assert geom["geom.n_ca"] == 30 and geom["geom.chain_not_found"] == 1.0


def test_the_chain_of_interest_falls_back_to_upstreams_chain_extract(tmp_path):
    """`chain_of_interest` came back EMPTY on both real evaluate runs (a monomer and a PD-L1
    binder, 2026-10-21). Reading only that column grades a complex on the whole assembly and
    nothing says so. `designed_chain_path` is the column upstream actually fills."""
    d = tmp_path / "run" / "inference_samples" / "ex"
    d.mkdir(parents=True)
    design = _two_chain_cif(d / "ex_generated.cif", binder_n=10, target_n=200)
    # upstream's extract: the chain of interest alone, keeping its label_asym_id
    extract = tmp_path / "chains" / "ex_generated_chain.cif"
    extract.parent.mkdir(parents=True)
    extract.write_text("\n".join(
        l for l in design.read_text().splitlines() if not l.startswith("ATOM") or " B " in l[:40]
    ) + "\n")
    m = tmp_path / "run" / "evaluation_metrics"
    m.mkdir(parents=True)
    with (m / "per_example_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "designed_structure_path", "chain_of_interest", "designed_chain_path"])
        w.writeheader()
        w.writerow({"designed_structure_path": str(design), "chain_of_interest": "",
                    "designed_chain_path": str(extract)})
    arm = quality.load_arm(tmp_path / "run")
    assert arm["rows"][0]["geom.n_ca"] == 200  # chain B, not the 210-atom complex
    assert arm["rows"][0]["geom.chain_not_found"] == 0.0


def test_a_chain_extract_with_several_chains_is_not_a_chain_of_interest(tmp_path):
    """Then there is no single designed chain and the whole structure is the right scope."""
    extract = tmp_path / "ex_generated_chain.cif"
    _two_chain_cif(extract, binder_n=4, target_n=4)
    assert quality._chain_of_interest({"designed_chain_path": str(extract)}) is None
    assert quality._chain_of_interest({"designed_chain_path": str(tmp_path / "gone.cif")}) is None
    assert quality._chain_of_interest({}) is None


def test_a_pass_criterion_with_no_inputs_is_not_a_clean_rate(tmp_path):
    """Upstream writes pass=0 with all_metrics_present=0 when a criterion's inputs are absent
    (observed on two VHH complexes, 2026-10-21). Reading only `.pass` makes a family that was
    never evaluated look like one that failed honestly, and both arms agree, so the rate test
    is happy. The presence column is the guard that says so."""
    rows_tt, rows_ref = _rows(40, 1), _rows(40, 2)
    for r in rows_tt + rows_ref:
        r["unconditional.pass"] = 0.0
        r["unconditional.all_metrics_present"] = 0.0
    tt, ref = _write_arm(tmp_path, "tt", rows_tt), _write_arm(tmp_path, "ref", rows_ref)
    out = tmp_path / "QUALITY.json"
    rc = quality.main(["grade", "--feature", FEATURE, "--tt", str(tt), "--ref", str(ref),
                       "--permutations", str(FAST), "--out", str(out)])
    feat = json.loads(out.read_text())["features"][FEATURE]
    assert rc == 1 and feat["verdict"] == "FAIL"
    assert "unconditional.all_metrics_present" in feat["failed"]
    assert feat["metrics"]["unconditional.pass"]["pass"] is True  # the rate alone sees nothing wrong


def test_a_resample_floor_is_labelled_and_reaches_the_grade_that_reads_it(tmp_path):
    """Two halves of one arm bound sampling noise, not seed noise. A grade must be able to say
    which it cleared, so the kind is structural in QUALITY.json and copied into the grade."""
    a = quality.load_arm(str(_write_arm(tmp_path, "a", _rows(40, 1))))
    b = quality.load_arm(str(_write_arm(tmp_path, "b", _rows(40, 2))))

    floor = quality.calibrate_floor(FEATURE, a, b, permutations=FAST, source="halves", kind="resample")
    assert floor["_kind"] == "resample"
    assert quality.calibrate_floor(FEATURE, a, b, permutations=FAST)["_kind"] == "two-seed"
    with pytest.raises(ValueError):
        quality.calibrate_floor(FEATURE, a, b, permutations=FAST, kind="two_seeds")

    res = quality.grade_feature(FEATURE, a, b, {FEATURE: floor}, permutations=FAST, allow_no_oracle=True)
    assert res["seed_floor_kind"] == "resample" and res["seed_floor_source"] == "halves"


def test_the_floor_watcher_refuses_an_index_kind_it_does_not_know(tmp_path, capsys):
    import floor_from_reference as ffr

    index = {"binder": {"recipe": "r", "session": "s", "kind": "vibes",
                        "seeds": {"A": str(tmp_path), "B": str(tmp_path)}}}
    idx = tmp_path / "reference-designs.json"
    idx.write_text(json.dumps(index))
    doc = tmp_path / "QUALITY.json"
    monkey = {"INDEX": ffr.INDEX, "QUALITY_JSON": quality.QUALITY_JSON}
    ffr.INDEX, quality.QUALITY_JSON = idx, doc
    try:
        assert ffr.main() == 0
    finally:
        ffr.INDEX, quality.QUALITY_JSON = monkey["INDEX"], monkey["QUALITY_JSON"]
    assert "is not one of" in capsys.readouterr().out
    assert not doc.exists()


# ----------------------------------------------------------------------------- the paired leg
def test_the_sign_test_needs_twelve_of_sixteen_and_recomputes_after_ties():
    worse = [0.4] * 12 + [-0.1] * 4
    assert stats.sign_test(worse)["verdict"] == "FAIL"
    assert stats.sign_test([0.4] * 11 + [-0.1] * 5)["verdict"] == "INCONCLUSIVE"
    assert stats.sign_test([-0.4] * 12 + [0.1] * 4)["verdict"] == "PASS"
    # two exact ties removed: the threshold moves from 12 of 16 to 11 of 14, so 11 now decides
    tied = stats.sign_test([0.4] * 11 + [-0.1] * 3 + [0.0, 0.0])
    assert tied["ties"] == 2 and tied["n"] == 14 and tied["needed_one_side"] == 11
    assert tied["verdict"] == "FAIL"
    assert stats.sign_test([0.1, -0.1, 0.1, -0.1])["needed_one_side"] is None  # n=4 cannot reject


def test_an_equivalent_arm_reads_inconclusive_not_pass():
    rng = np.random.default_rng(7)
    d = rng.normal(0.0, 0.3, 16)
    assert stats.sign_test(d)["verdict"] == "INCONCLUSIVE"


def test_one_huge_difference_cannot_carry_the_verdict_but_moves_hodges_lehmann():
    d = [-50.0] + [0.2] * 15
    assert stats.sign_test(d)["verdict"] == "FAIL"  # 15 of 16 worse, whatever the one magnitude says
    assert stats.hodges_lehmann(d)["estimate"] > 0


def _cases(n_worse, n_better, stratum="pdl1-128", seed0=1, incomplete=0):
    cases = [{"stratum": stratum, "seed": seed0 + i, "tt_value": 1.5, "ref_value": 1.0} for i in range(n_worse)]
    cases += [{"stratum": stratum, "seed": seed0 + 100 + i, "tt_value": 1.0, "ref_value": 1.5}
              for i in range(n_better)]
    cases += [{"stratum": stratum, "seed": seed0 + 200 + i, "tt_value": float("nan"), "ref_value": 1.0}
              for i in range(incomplete)]
    return cases


def test_the_completeness_floor_overrides_the_signs():
    res = quality.paired_leg(_cases(11, 0, incomplete=5), "refolded_backbone_rmsd.rmsd")
    st = res["strata"]["pdl1-128"]
    assert st["sign_test"]["verdict"] == "FAIL" and st["n_complete"] == 11
    assert st["verdict"] == "INCONCLUSIVE" and res["verdict"] == "INCONCLUSIVE"
    assert [c["seed"] for c in st["incomplete"]] == [201, 202, 203, 204, 205]  # named, never dropped


def test_a_stratum_that_fails_decides_the_leg_and_the_leg_never_gates():
    res = quality.paired_leg(_cases(12, 4) + _cases(0, 16, stratum="pdl1-256"),
                             "refolded_backbone_rmsd.rmsd")
    assert res["strata"]["pdl1-128"]["verdict"] == "FAIL"
    assert res["strata"]["pdl1-256"]["verdict"] == "PASS"
    assert res["verdict"] == "FAIL" and res["gates"] is False


def test_the_holdout_seeds_are_refused_without_the_flag(tmp_path, capsys):
    hold = quality.holdout()
    ref = _write_arm(tmp_path, "ref", _rows(1, 0))
    tt = _write_arm(tmp_path, "tt", _rows(1, 1))
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"cases": [{"stratum": "pdl1-128", "seed": hold["seeds"][0],
                                               "tt": str(tt), "ref": str(ref)}]}))
    out = tmp_path / "QUALITY.json"
    argv = ["paired", "--feature", FEATURE, "--manifest", str(manifest), "--out", str(out)]
    with pytest.raises(SystemExit) as exc:
        quality.main(argv)
    assert exc.value.code == 2 and "allow-holdout" in capsys.readouterr().err
    assert not out.exists()
    assert quality.main(argv + ["--allow-holdout"]) == 0
    doc = json.loads(out.read_text())
    assert doc["paired"][FEATURE]["on_holdout_seeds"] == [hold["seeds"][0]]
    assert doc["paired"][FEATURE]["verdict"] == "INCONCLUSIVE"  # one case, far under the floor
    assert doc["verdict"] == "NOT_GRADED"  # the paired leg does not write a grade of its own
