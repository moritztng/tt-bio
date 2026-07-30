"""The orphan reconciler must count artifacts exactly as the harness does, apply the harness's
failed-results cross-check, and never claim provenance it cannot establish.

Before these, it globbed `*_pae.npz` (which also matches the aggregate `<target>_pae.npz`, so a
50-sample fold reported 51), skipped the results.json cross-check entirely (so a fold whose
results.json said failed was recorded `ok` from its leftover files), and always wrote
`tt_bio_commit: null` -- which, once "done" meant "defensible", made every record it produced
worthless.
"""
import importlib.util
import json
import os
import pathlib
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gen():
    argv, sys.argv = sys.argv, ["abag_xm_generate"]
    try:
        spec = importlib.util.spec_from_file_location(
            "abag_xm_generate", ROOT / "scripts" / "abag_xm_generate.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        sys.argv = argv


def _fold_dir(tmp_path, target, n, *, aggregate_pae=True, status="ok", n_runs=None):
    """A result dir shaped like a real one: a winner CIF with no _model_ suffix, n-1 ranked
    CIFs, n per-sample PAEs, and the backwards-compat aggregate PAE that must NOT be counted."""
    rd = tmp_path / f"boltz2_results_{target}"
    sd = rd / "structures"
    sd.mkdir(parents=True)
    (sd / f"{target}.cif").write_text("winner")
    for i in range(1, n):
        (sd / f"{target}_model_{i}.cif").write_text("x")
    for i in range(n):
        (sd / f"{target}_model_{i}_pae.npz").write_bytes(b"x")
    if aggregate_pae:
        (sd / f"{target}_pae.npz").write_bytes(b"x")   # the one that caused n_paes=51
    (rd / "results.json").write_text(json.dumps(
        {"status": status, "all_runs": [{}] * (n_runs if n_runs is not None else n)}))
    return rd


def test_aggregate_pae_is_not_counted(gen, tmp_path):
    """The bug that produced 8 records claiming 51 PAEs on a 50-sample campaign."""
    n = 7
    rd = _fold_dir(tmp_path, "T1", n, aggregate_pae=True)
    cifs, paes = gen.count_artifacts(rd, "T1")
    assert len(cifs) == n, [p.name for p in cifs]
    assert len(paes) == n, [p.name for p in paes]
    # the loose glob the reconciler used to carry would have found one more
    assert len(sorted((rd / "structures").glob("*_pae.npz"))) == n + 1


def test_winner_cif_is_counted_once_and_only_if_present(gen, tmp_path):
    rd = _fold_dir(tmp_path, "T2", 5)
    assert len(gen.count_artifacts(rd, "T2")[0]) == 5
    (rd / "structures" / "T2.cif").unlink()
    assert len(gen.count_artifacts(rd, "T2")[0]) == 4


def test_another_targets_files_are_not_counted(gen, tmp_path):
    """Result dirs sit under one output directory per model, so the glob must be target-scoped."""
    rd = _fold_dir(tmp_path, "T3", 4)
    (rd / "structures" / "OTHER_model_9.cif").write_text("x")
    (rd / "structures" / "OTHER_model_9_pae.npz").write_bytes(b"x")
    cifs, paes = gen.count_artifacts(rd, "T3")
    assert len(cifs) == 4 and len(paes) == 4


def test_results_entry_reads_list_and_dict_and_survives_junk(gen, tmp_path):
    rd = tmp_path / "boltz2_results_T4"
    rd.mkdir()
    (rd / "results.json").write_text('[{"status": "ok"}]')
    assert gen.results_entry(rd)["status"] == "ok"
    (rd / "results.json").write_text('{"status": "failed"}')
    assert gen.results_entry(rd)["status"] == "failed"
    (rd / "results.json").write_text("not json")
    assert gen.results_entry(rd) is None
    (rd / "results.json").unlink()
    assert gen.results_entry(rd) is None


def test_harness_and_reconciler_agree_on_completeness(gen, tmp_path):
    """The reconciler's completeness rule must be the harness's, including the cross-check it
    used to skip: a fold whose results.json says failed is not a recoverable ok fold."""
    n = gen.N_SAMPLES
    good = _fold_dir(tmp_path / "a", "G", n)
    entry = gen.results_entry(good)
    cifs, paes = gen.count_artifacts(good, "G")
    assert (entry["status"] == "ok" and len(entry["all_runs"]) == n
            and len(cifs) == n and len(paes) == n)

    failed = _fold_dir(tmp_path / "b", "F", n, status="failed")
    assert gen.results_entry(failed)["status"] != "ok"

    dropped = _fold_dir(tmp_path / "c", "D", n, n_runs=n - 2)
    assert len(gen.results_entry(dropped)["all_runs"]) != n


def test_provenance_comes_from_the_sidecar_not_from_inference(gen, tmp_path):
    """The driver knows the engine tree exactly when it launches a fold, so it writes it down then.
    Two earlier attempts inferred it after the fact and both were subtly wrong: a fold's tree lives
    in whichever worktree its DRIVER ran in, more than one worktree can fold into the same campaign
    directory (p4's drivers folded tree 3dc9db33 while p6 on disk was 871c8992), and neither a
    directory-level stamp nor file mtimes can tell two concurrent owners apart."""
    rd = _fold_dir(tmp_path, "T5", 4)
    assert gen.read_fold_provenance(rd) is None          # nothing written -> not stateable

    gen.write_fold_provenance(rd, "T5", "a" * 40, "abc1234", "deadbeef00000000", 5, 50)
    prov = gen.read_fold_provenance(rd)
    assert prov["tt_bio_tree"] == "a" * 40
    assert prov["tt_bio_commit"] == "abc1234"
    assert prov["msa_sha"] == "deadbeef00000000"
    assert prov["worktree"] == str(gen.ROOT)
    assert prov["mps"] == 5 and prov["n_samples"] == 50


def test_a_sidecar_without_a_tree_is_not_provenance(gen, tmp_path):
    """A dirty worktree makes _head_tree() None, and the driver still writes the file. A record
    built from that must not be treated as defensible."""
    rd = _fold_dir(tmp_path, "T6", 4)
    gen.write_fold_provenance(rd, "T6", None, "abc1234-dirty", "x" * 16, 5, 50)
    assert gen.read_fold_provenance(rd) is None


def test_a_corrupt_sidecar_is_not_provenance(gen, tmp_path):
    rd = _fold_dir(tmp_path, "T7", 4)
    (rd / gen.FOLD_PROVENANCE).write_text("{truncated")
    assert gen.read_fold_provenance(rd) is None


def test_the_sidecar_does_not_pollute_the_artifact_counts(gen, tmp_path):
    """It lives in the result dir, not in structures/, and must not be mistaken for output."""
    rd = _fold_dir(tmp_path, "T8", 6)
    gen.write_fold_provenance(rd, "T8", "b" * 40, "abc", "y" * 16, 5, 50)
    cifs, paes = gen.count_artifacts(rd, "T8")
    assert len(cifs) == 6 and len(paes) == 6


def test_a_sidecar_older_than_the_artifacts_it_would_describe_is_refused(gen, tmp_path):
    """Found by accident, live. The driver writes the sidecar and THEN launches the fold, so a fold
    interrupted before producing anything leaves the sidecar next to whatever a previous run left in
    the same directory -- a 21tw sidecar written 07-28 13:02 sat beside a structures/ from 07-27
    22:28. Read naively, the new tree gets attributed to the old output."""
    rd = _fold_dir(tmp_path, "T9", 5)
    cifs, paes = gen.count_artifacts(rd, "T9")
    for a in cifs + paes:                      # artifacts from a previous run
        os.utime(a, (1000, 1000))
    gen.write_fold_provenance(rd, "T9", "c" * 40, "abc", "z" * 16, 5, 50)   # sidecar written after
    assert gen.read_fold_provenance(rd, cifs + paes) is None
    # ignoring the ordering is what made it look fine
    assert gen.read_fold_provenance(rd)["tt_bio_tree"] == "c" * 40


def test_a_sidecar_written_before_its_artifacts_is_trusted(gen, tmp_path):
    """The normal case: driver writes the sidecar, the fold then writes its output."""
    rd = tmp_path / "boltz2_results_TA"
    (rd / "structures").mkdir(parents=True)
    gen.write_fold_provenance(rd, "TA", "d" * 40, "abc", "w" * 16, 5, 50)
    time.sleep(0.01)
    rd2 = _fold_dir(tmp_path, "TA2", 4)
    for a in gen.count_artifacts(rd2, "TA2")[0]:
        (rd / "structures" / a.name.replace("TA2", "TA")).write_text("x")
    cifs, paes = gen.count_artifacts(rd, "TA")
    assert gen.read_fold_provenance(rd, cifs + paes)["tt_bio_tree"] == "d" * 40
