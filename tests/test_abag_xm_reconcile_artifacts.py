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


def test_orphan_provenance_needs_the_launch_stamp(gen, tmp_path, monkeypatch):
    """A fold loads tt_bio/ from whichever worktree its DRIVER ran in, which is not necessarily
    this one. While this campaign was moved from the p4 worktree to p6, p4's drivers kept folding
    under engine tree 3dc9db33 while p6 on disk was 871c8992 -- and because p6's files were checked
    out BEFORE those folds wrote their artifacts, a pure mtime test says "unchanged" and would
    stamp p6's tree onto p4's folds. The launcher's stamp is what rules that out."""
    artifact = tmp_path / "art"
    artifact.write_text("x")
    os.utime(artifact, (time.time() + 86400, time.time() + 86400))   # newer than all sources
    stamp = tmp_path / "launched_from"
    monkeypatch.setattr(gen, "LAUNCH_OWNER", stamp)

    # no stamp at all -> cannot state
    assert gen.provenance_for_orphan(artifact) is None

    # a stamp from a DIFFERENT worktree -> cannot state, even though mtimes look fine
    stamp.write_text(json.dumps({"worktree": "/some/other/worktree",
                                 "tt_bio_tree": gen._head_tree()}))
    assert gen.provenance_for_orphan(artifact) is None

    # this worktree but a different tree than is checked out now -> cannot state
    stamp.write_text(json.dumps({"worktree": str(gen.ROOT), "tt_bio_tree": "f" * 40}))
    assert gen.provenance_for_orphan(artifact) is None

    # this worktree, this tree, artifact newer than every source -> stateable
    stamp.write_text(json.dumps({"worktree": str(gen.ROOT), "tt_bio_tree": gen._head_tree()}))
    assert gen.provenance_for_orphan(artifact) == gen._head_tree()


def test_orphan_provenance_refuses_when_sources_moved_after_the_fold(gen, tmp_path, monkeypatch):
    """Condition 2: an artifact OLDER than a tt_bio source means the code changed after the fold."""
    old = tmp_path / "old"
    old.write_text("x")
    os.utime(old, (0, 0))
    stamp = tmp_path / "launched_from"
    stamp.write_text(json.dumps({"worktree": str(gen.ROOT), "tt_bio_tree": gen._head_tree()}))
    monkeypatch.setattr(gen, "LAUNCH_OWNER", stamp)
    assert gen.provenance_for_orphan(old) is None


def test_pycache_is_not_treated_as_a_source_change(gen):
    """A .pyc is rewritten on import, so it records when a fold RAN rather than when the code
    changed. Counting it would make the mtime test refuse at random."""
    srcs = list(gen._tt_bio_sources())
    assert srcs, "no tt_bio sources found"
    assert not any("__pycache__" in p.parts or p.suffix == ".pyc" for p in srcs)


def test_record_launch_owner_round_trips(gen, tmp_path, monkeypatch):
    stamp = tmp_path / "launched_from"
    monkeypatch.setattr(gen, "LAUNCH_OWNER", stamp)
    tree = gen.record_launch_owner()
    d = json.loads(stamp.read_text())
    assert d["worktree"] == str(gen.ROOT) and d["tt_bio_tree"] == tree == gen._head_tree()
