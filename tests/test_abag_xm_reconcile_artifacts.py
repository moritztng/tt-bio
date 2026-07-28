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


def test_tree_unchanged_since_is_conservative(gen, tmp_path):
    """A fold loads tt_bio/ before it writes anything, so an artifact NEWER than every tt_bio
    file means this tree is the tree that ran. Older means we must not claim it."""
    newer = tmp_path / "newer"
    newer.write_text("x")
    os.utime(newer, (time.time() + 86400, time.time() + 86400))
    assert gen.tree_unchanged_since(newer) is True

    older = tmp_path / "older"
    older.write_text("x")
    os.utime(older, (0, 0))
    assert gen.tree_unchanged_since(older) is False

    assert gen.tree_unchanged_since(tmp_path / "absent") is False


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
