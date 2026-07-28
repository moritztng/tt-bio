"""A card that was busy must not be recorded as a failed fold.

33 of the Tier-A slab's pairs failed because the fold never reached the model -- the card was
held by another process, or the chip would not initialise -- and the harness wrote each one
down as a result. These tests pin the three guards that stop that: the classifier that tells
an infrastructure event from a model failure, the provenance check that refuses to fold at all
when the record could not be defended later, and the agreement between the resume filter and
the acceptance gate that a pair is "done".
"""
import importlib.util
import pathlib
import sys

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


# The two stderrs below are copied from real progress.jsonl records, not invented: the whole
# point of the classifier is that it recognises what the slab actually produced.
A1 = ("all local workers exited before the run finished. Re-run with --debug to see the "
      "worker's own error (the Rich display suppresses it); a common cause is another "
      "process already holding the card (DeviceInUseError).")
A2 = "Failed to open device 2: firmware did not respond within the init timeout"
RAGGED = ("TT_FATAL @ binary_ng_device_operation.cpp:346: a_dim == b_dim || a_dim == 1 || "
          "b_dim == 1  Broadcasting rule violation")
OOM = "Out of Memory: Not enough space to allocate 1585971200 B DRAM buffer across 8 banks"


@pytest.mark.parametrize("status,stderr,is_infra", [
    ("fold_failed", A1, True),      # lease theft: constant 133 s = 120 s timeout + startup
    ("killed", A1, True),           # same thing, killed before its display flushed
    ("fold_failed", A2, True),      # the chip never initialised, ~23 s
    ("incomplete", RAGGED, False),  # a real device bug -- must be recorded, not retried
    ("incomplete", OOM, False),     # a real allocation failure -- likewise
    ("fold_failed", "ModuleNotFoundError: No module named 'click'", False),
    ("timed_out", A1, False),       # a timeout is its own class; the retry path is not it
    ("ok", "", False),
])
def test_only_card_availability_is_retried(gen, status, stderr, is_infra):
    assert gen._is_infra_failure({"status": status, "stderr": stderr}) is is_infra


def test_a_model_failure_is_never_silently_retried(gen):
    """Retrying a model failure would burn a card re-deriving the same number and hide the
    defect. Every non-empty stderr that is not one of the device-open signatures stays a
    result."""
    for sig in gen.INFRA_SIGNATURES:
        assert gen._is_infra_failure({"status": "fold_failed", "stderr": f"... {sig} ..."})
    assert not gen._is_infra_failure(
        {"status": "fold_failed", "stderr": "RuntimeError: shape mismatch in trunk"})


def test_engine_tree_is_the_subtree_not_the_commit(gen):
    """`tt_bio_commit` counts commits to scripts the folding interpreter never loads, which is
    how 34 commit strings looked like 34 procedures when they were 5 engine trees."""
    tree = gen._head_tree()
    assert tree and len(tree) == 40, tree
    assert tree != gen._head_commit()


def test_unstateable_provenance_aborts_before_folding(gen, monkeypatch):
    """Fold-then-record-`unknown` is worse than not folding: the record is indistinguishable
    from a good one downstream, and 21 pairs reached exactly that state."""
    monkeypatch.setattr(gen, "_head_tree", lambda: None)
    with pytest.raises(SystemExit) as e:
        gen.provenance_or_die()
    assert "cannot be stated" in str(e.value)


def test_provenance_ok_returns_the_tree(gen):
    assert gen.provenance_or_die() == gen._head_tree()


def test_msa_hash_covers_every_chain_and_is_stable(gen):
    """The MSA fall-through (build one with whichever mmseqs is on PATH) is the only path that
    can change a fold's input, so it is settled with bytes rather than argument."""
    target = next(iter(sorted(p.stem for p in gen.YAML_DIR.glob("*.yaml"))))
    a, miss_a = gen._msa_sha(target)
    b, miss_b = gen._msa_sha(target)
    assert a == b and len(a) == 16 and miss_a == miss_b


def test_msa_hash_is_target_specific(gen):
    ts = sorted(p.stem for p in gen.YAML_DIR.glob("*.yaml"))[:8]
    digests = {gen._msa_sha(t)[0] for t in ts}
    assert len(digests) == len(ts), "two different targets hashed to the same MSA digest"


def test_done_means_what_acceptance_accepts(gen, tmp_path, monkeypatch):
    """The resume filter and the release gate had drifted apart, so a pair could be skipped as
    done and rejected as unpublishable at the same time. One definition, or the slab
    deadlocks."""
    import json
    n = gen.N_SAMPLES
    good = {"target": "T1", "model": "boltz2", "status": "ok", "mps": gen.MPS,
            "tt_bio_commit": "abc1234", "n_cifs": n, "n_paes": n}
    rows = [
        good,
        # dirty worktree: provenance cannot be stated
        {**good, "target": "T2", "tt_bio_commit": "abc1234-dirty"},
        # reconstructed post-hoc by a looser-globbing tool: no commit, and n+1 PAEs because it
        # counted the aggregate <target>_pae.npz the harness deliberately excludes
        {**good, "target": "T3", "tt_bio_commit": None, "n_paes": n + 1},
        # right provenance, dropped samples
        {**good, "target": "T4", "n_cifs": n - 3},
        # a different sampling configuration is a different procedure
        {**good, "target": "T5", "mps": 3},
        # not ok at all
        {**good, "target": "T6", "status": "incomplete"},
        # a p6-era record states its engine tree, so a dirty commit string is not disqualifying
        {**good, "target": "T7", "tt_bio_commit": "abc1234-dirty", "tt_bio_tree": "d" * 40},
    ]
    p = tmp_path / "progress.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(gen, "PROGRESS", p)
    done = gen.done_pairs()
    assert done == {("T1", "boltz2"), ("T7", "boltz2")}, done


def test_lease_timeout_outlasts_a_fold(gen):
    """tt_bio's 120 s default is shorter than any fold here, so a sibling that took the card
    legitimately always won the race and the fold recorded a failure at a constant 133 s."""
    assert gen.LEASE_TIMEOUT_S >= 900
    assert gen.INFRA_RETRIES >= 1 and gen.INFRA_BACKOFF_S >= 60


def test_peer_mirror_is_scheduling_only_not_evidence(gen, tmp_path, monkeypatch):
    """A pair the peer folded must be skipped by the resume, but must NOT become locally done:
    the artifacts, labels and provenance still live on the peer, and moving them is
    abag_xm_merge_hosts.py's job at release time."""
    import json
    n = gen.N_SAMPLES
    good = {"model": "boltz2", "status": "ok", "mps": gen.MPS,
            "tt_bio_commit": "abc1234", "n_cifs": n, "n_paes": n}
    local = tmp_path / "progress.jsonl"
    local.write_text(json.dumps({**good, "target": "MINE"}) + "\n")
    peer = tmp_path / "peer_progress.jsonl"
    peer.write_text("".join(json.dumps(r) + "\n" for r in [
        {**good, "target": "THEIRS"},
        # the peer's junk is junk here too -- one predicate, both files
        {**good, "target": "THEIRS_DIRTY", "tt_bio_commit": "abc1234-dirty"},
        {**good, "target": "THEIRS_SHORT", "n_paes": n + 1},
    ]))
    monkeypatch.setattr(gen, "PROGRESS", local)
    monkeypatch.setattr(gen, "PEER_PROGRESS", peer)
    assert gen.done_pairs() == {("MINE", "boltz2")}
    assert gen.peer_done_pairs() == {("THEIRS", "boltz2")}


def test_peer_mirror_absent_is_not_an_error(gen, tmp_path, monkeypatch):
    """The common case is a single-host campaign with no mirror at all."""
    monkeypatch.setattr(gen, "PEER_PROGRESS", tmp_path / "nope.jsonl")
    assert gen.peer_done_pairs() == set()


def test_one_predicate_for_both_files(gen, tmp_path, monkeypatch):
    """done_pairs and peer_done_pairs must not drift apart the way done_pairs and the acceptance
    gate did -- same record, same verdict, whichever file it came from."""
    import json
    n = gen.N_SAMPLES
    rows = [
        {"target": "A", "model": "boltz2", "status": "ok", "mps": gen.MPS,
         "tt_bio_commit": "abc1234", "n_cifs": n, "n_paes": n},
        {"target": "B", "model": "boltz2", "status": "ok", "mps": 3,
         "tt_bio_commit": "abc1234", "n_cifs": n, "n_paes": n},
        {"target": "C", "model": "protenix-v2", "status": "incomplete", "mps": gen.MPS,
         "tt_bio_commit": "abc1234", "n_cifs": n, "n_paes": n},
    ]
    blob = "".join(json.dumps(r) + "\n" for r in rows)
    a = tmp_path / "a.jsonl"; a.write_text(blob)
    b = tmp_path / "b.jsonl"; b.write_text(blob)
    monkeypatch.setattr(gen, "PROGRESS", a)
    monkeypatch.setattr(gen, "PEER_PROGRESS", b)
    assert gen.done_pairs() == gen.peer_done_pairs() == {("A", "boltz2")}
