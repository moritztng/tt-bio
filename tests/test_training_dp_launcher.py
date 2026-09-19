"""The launcher, checked where it can be checked without a card, plus the two loud failures.

Host-only and importless of ttnn, which is most of the point: the collective sums fp32 host
arrays, so the whole exchange -- publish, barrier, add, and that it SUMS rather than averages --
runs in a temp directory with two threads and no device. What needs a card is the part that
opens one, and that is measured in ``perf/train_d_dp/`` rather than asserted here.

The two arms that matter are the negative controls at the bottom. A data-parallel run that
silently diverged, and one whose ranks shared a chip, both produce a plausible throughput number
and a falling loss curve; the launcher raises on each, and these prove it raises rather than
that it has a docstring saying it would.
"""

import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tt_bio.train import launcher as L          # noqa: E402
from tt_bio.train.mesh import Axis, Mesh, UnreducedGradients   # noqa: E402


@pytest.fixture
def as_rank(monkeypatch, tmp_path):
    """Make this process look like rank ``r`` of ``w``, the way the driver's env does."""
    def setup(r, w, run=None):
        monkeypatch.setenv(L.RANK_ENV, str(r))
        monkeypatch.setenv(L.WORLD_ENV, str(w))
        monkeypatch.setenv(L.RUN_ENV, str(run or tmp_path))
        return Mesh({"dp": list(range(w))}).axis("dp")
    return setup


# --------------------------------------------------------------- where am I

def test_a_process_nobody_launched_is_the_driver_and_is_rank_zero(monkeypatch):
    monkeypatch.delenv(L.RANK_ENV, raising=False)
    assert L.driving() and not L.inside()
    assert L.rank() == 0 and L.world() == 1


def test_a_rank_knows_it_is_one(as_rank):
    as_rank(1, 2)
    assert L.inside() and not L.driving()
    assert L.rank() == 1 and L.world() == 2


def test_rank_zero_owns_out_dir_and_the_others_get_a_subdirectory(as_rank, monkeypatch):
    monkeypatch.delenv(L.RANK_ENV, raising=False)
    assert L.out_dir("runs/a") == Path("runs/a")
    as_rank(0, 2)
    assert L.out_dir("runs/a") == Path("runs/a")
    as_rank(1, 2)
    assert L.out_dir("runs/a") == Path("runs/a/rank1")


def test_replicas_is_empty_on_one_chip_and_one_deep_on_a_rank(as_rank, monkeypatch):
    class P:
        def __init__(self, g):
            self.grad = g

    params = {"a.A": P("g0"), "a.B": P(None)}
    monkeypatch.delenv(L.RANK_ENV, raising=False)
    # Empty, not absent: the optimizer refuses a non-empty replicas without a wide axis, and
    # `opt.step(replicas=replicas(params))` has to be the same call at width 1.
    assert L.replicas(params) == {}
    as_rank(0, 2)
    assert L.replicas(params) == {"a.A": ["g0"]}


# --------------------------------------------------------------- the collective

def _axis(r, w, run):
    return L.ProcessAxis(name="dp", device_ids=tuple(range(w)), dp_rank=r, run=str(run))


def test_the_collective_sums_and_does_not_average(tmp_path):
    """Sum, not mean. The divisor is the global batch the caller pinned.

    Averaging here is what makes one recipe mean something different on a 2-chip box than on a
    4-chip one, so it is asserted on the number and not left to a docstring.
    """
    grads = {0: np.array([1.0, 2.0, 3.0], np.float32),
             1: np.array([10.0, 20.0, 30.0], np.float32)}
    got = {}

    def run(r):
        ax = _axis(r, 2, tmp_path)
        got[r] = ax.reduce_all({"w": [grads[r]]})["w"]

    ts = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    [t.start() for t in ts]
    [t.join(timeout=60) for t in ts]
    for r in (0, 1):
        assert got[r] == pytest.approx([11.0, 22.0, 33.0]), (
            f"rank {r} got {got[r]}; the mean would be [5.5, 11, 16.5]")


def test_every_rank_gets_the_same_bits_so_the_masters_stay_identical(tmp_path):
    """Bit-identical sums across ranks, which is what the sync invariant rests on.

    Every rank adds the same files in rank order, so the fp32 sum is the same rounding on every
    one of them. Summed in a different order per rank they would differ in the last bits, the
    masters would drift apart, and the only thing that would notice is the hash check.
    """
    rng = np.random.default_rng(0)
    grads = {r: rng.normal(0, 1, 4096).astype(np.float32) for r in range(3)}
    got = {}

    def run(r):
        got[r] = _axis(r, 3, tmp_path).reduce_all({"w": [grads[r]]})["w"]

    ts = [threading.Thread(target=run, args=(r,)) for r in range(3)]
    [t.start() for t in ts]
    [t.join(timeout=60) for t in ts]
    b = [got[r].tobytes() for r in range(3)]
    assert b[0] == b[1] == b[2], "the ranks' sums differ in their bits, so their masters will"


def test_a_rank_may_only_offer_its_own_one_gradient(tmp_path):
    ax = _axis(0, 2, tmp_path)
    with pytest.raises(ValueError, match="holds one replica"):
        ax.reduce_all({"w": [np.zeros(4, np.float32), np.zeros(4, np.float32)]})


def test_a_plain_axis_reduces_per_parameter_through_reduce(monkeypatch):
    """The default ``reduce_all`` is the per-name loop, so overriding it is opt-in."""
    seen = []

    class Fake(Axis):
        def reduce(self, tensors):
            seen.append(list(tensors))
            return ["summed"]

    ax = Fake(name="dp", device_ids=(0, 1))
    assert ax.reduce_all({"a": ["x", "y"], "b": ["p", "q"]}) == {"a": "summed", "b": "summed"}
    assert seen == [["x", "y"], ["p", "q"]]


def test_a_width_one_axis_is_handed_back_unchanged(monkeypatch):
    monkeypatch.delenv(L.RANK_ENV, raising=False)
    ax = Mesh({"dp": [0]}).axis("dp")
    assert L.reducer(ax) is ax


def test_a_wide_axis_outside_the_launcher_says_there_is_nobody_to_reduce_with(monkeypatch):
    monkeypatch.delenv(L.RUN_ENV, raising=False)
    monkeypatch.delenv(L.RANK_ENV, raising=False)
    with pytest.raises(RuntimeError, match="nobody to reduce with"):
        L.reducer(Mesh({"dp": [0, 1]}).axis("dp"))


# --------------------------------------------------------------- what a driver refuses

def test_a_program_that_cannot_be_re_run_is_refused_with_what_to_do(monkeypatch):
    """``python -c`` and a REPL have no program to re-run, and the message says so.

    A rank is a fresh interpreter because ``TT_VISIBLE_DEVICES`` is read at ``import ttnn``, so
    there is no way to hand it the caller's live objects; it rebuilds them by re-running the
    caller's own program. Without a program that is a refusal, not a fallback.
    """
    monkeypatch.setattr(sys, "argv", ["-c"])
    monkeypatch.setitem(sys.modules, "__main__", type("M", (), {"__spec__": None})())
    with pytest.raises(RuntimeError, match="cannot be re-run"):
        L._relaunch_argv()


def test_a_module_entry_point_is_re_run_as_a_module(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["whatever", "--steps", "3"])
    spec = type("S", (), {"name": "pkg.mod"})()
    monkeypatch.setitem(sys.modules, "__main__", type("M", (), {"__spec__": spec})())
    assert L._relaunch_argv() == [sys.executable, "-u", "-m", "pkg.mod", "--steps", "3"]


def test_a_driver_holding_a_card_refuses_to_spawn_and_names_the_fix(monkeypatch):
    from tt_bio.train import provenance
    monkeypatch.setattr(provenance, "open_nodes", lambda pid=None: [3])
    with pytest.raises(RuntimeError, match="already has /dev/tenstorrent"):
        L._refuse_if_holding_a_card()
    monkeypatch.setattr(provenance, "open_nodes", lambda pid=None: [])
    L._refuse_if_holding_a_card()


# --------------------------------------------------------------- the two loud failures

def _rank_payload(rank, sha, node, loss=1.0):
    return {"rank": rank, "world": 2, "visible": str(rank), "node": node,
            "master_sha": sha, "params": ["a.A", "a.B"],
            "history": [{"step": 0, "loss": loss, "breakdown": {}, "grad_norm": 1.0, "lr": 1e-4},
                        {"step": 1, "loss": loss / 2, "breakdown": {}, "grad_norm": 1.0,
                         "lr": 1e-4}],
            "step_s": [0.5], "publish_s": [0.01], "wait_s": [0.02], "add_s": [0.01],
            "comm_bytes": 1024, "displacement": {"ratio": 1.0},
            "provenance": {"aiclk": {"median": 1350.0, "samples": 9}, "device_nodes": [node]},
            "plan": "plan: fits", "checkpoints": []}


def test_two_ranks_that_diverged_are_refused_however_good_the_throughput_looks(tmp_path):
    """The check that earned its place: it caught the shipped adapter init being unseeded.

    `lora_factors` seeded itself from entropy, so every rank started from a different A and
    trained a different model. Nothing else noticed: both loss curves fell and the step times
    were normal.
    """
    res = [_rank_payload(0, "a" * 64, 3), _rank_payload(1, "b" * 64, 1)]
    with pytest.raises(AssertionError, match="SYNC FAILED"):
        L._aggregate(res, wall_s=10.0, steps=2, out_dir=tmp_path, run=str(tmp_path))


def test_two_ranks_on_one_chip_are_refused_even_though_both_ran(tmp_path):
    """TT_VISIBLE_DEVICES=N does not open /dev/tenstorrent/N, and this is where that bites.

    Two ranks on one chip with a third sitting idle still finish, still agree on their masters,
    and still show a speedup. The node is read from each rank's own fds, so what is compared is
    what they held rather than what they were told to hold.
    """
    res = [_rank_payload(0, "a" * 64, 3), _rank_payload(1, "a" * 64, 3)]
    with pytest.raises(AssertionError, match="not 2 distinct chips"):
        L._aggregate(res, wall_s=10.0, steps=2, out_dir=tmp_path, run=str(tmp_path))


def test_a_healthy_pair_aggregates_the_loss_as_the_mean_over_the_shards(tmp_path):
    res = [_rank_payload(0, "a" * 64, 3, loss=1.0), _rank_payload(1, "a" * 64, 1, loss=3.0)]
    out = L._aggregate(res, wall_s=10.0, steps=2, out_dir=tmp_path, run=str(tmp_path))
    assert out["dp"]["sync_ok"] and out["dp"]["nodes"] == [3, 1]
    assert out["dp"]["distinct_master_sha"] == 1
    # each rank's loss is over its own equal-sized shard, so the global batch's loss is the mean
    assert out["history"][0]["loss"] == pytest.approx(2.0)
    assert out["history"][0]["per_rank_loss"] == [1.0, 3.0]
    assert out["params"] == {} and out["optimizer"] is None, (
        "the driver never opened a card and the ranks have exited, so there is no live tensor "
        "to hand back and pretending otherwise would be worse than saying so")


def test_a_run_directory_whose_driver_is_gone_is_swept(tmp_path, monkeypatch):
    """/dev/shm is RAM on a shared box, so a crashed run must not keep holding it.

    20 leaked directories from this row's own measurement pass held 268 MB before this existed.
    A live driver's directory is left alone, which is what makes the sweep safe to run at the
    start of every launch.
    """
    monkeypatch.setattr(L, "SHM_ROOT", str(tmp_path))
    dead = tmp_path / "999999999-1700000000"
    mine = tmp_path / f"{os.getpid()}-1700000000"
    junk = tmp_path / "not-a-pid"
    for d in (dead, mine, junk):
        d.mkdir()
        (d / "s1_r0.npy").write_bytes(b"x" * 16)
    L._sweep_shm()
    assert not dead.exists(), "a directory whose pid is gone was kept"
    assert mine.exists() and junk.exists(), "the sweep took a directory it should not have"


# --------------------------------------------------------------- the claim in the README

def test_the_readme_data_parallelism_claim_is_backed_by_the_recipe_reaching_the_launcher():
    """The positive form of the docs gate.

    ``tests/test_training_docs_match_reality.py`` goes quiet by design once the recipe stops
    refusing a wide axis, which leaves nothing asserting that it now DOES something. This is
    that assertion: the Tier-1 recipe hands a wide axis to the launcher, and the launcher has a
    driver to hand it to.
    """
    src = (REPO_ROOT / "tt_bio" / "train" / "recipes.py").read_text()
    assert "launcher.drive(" in src, (
        "the Tier-1 recipe no longer hands a wide dp axis to tt_bio.train.launcher, so "
        "`tt-bio finetune --chips 2` and `train.finetune(mesh=...)` reach no launcher and the "
        "README's works-today claim is false again")
    assert "NotImplementedError" not in src.split("def train_loop", 1)[1][:2000]
    readme = (REPO_ROOT / "README.md").read_text()
    import re
    claim = re.search(r"\*\*What works today:\*\*(.{0,900}?)\*\*What does not", readme, re.S)
    assert claim and re.search(r"data[- ]parallel", claim.group(1), re.I), (
        "the README's works-today sentence no longer claims data parallelism, but the launcher "
        "is built and reachable from both entry points. Either the claim or this test is stale.")


def test_the_optimizer_still_refuses_an_unreduced_wide_step(as_rank):
    """The guard the launcher must not have quietly removed.

    Handing `step()` a process-reducing axis is not the same as handing it a way around the
    check: a rank that forgets its own gradient is still refused, which is the failure that has
    no signature in any metric a user watches.
    """
    from tt_bio.train.optim import AdamW

    ax = L.reducer(as_rank(0, 2))

    class T:
        def __init__(self):
            self.value = np.zeros((2, 2), np.float32)
            self.grad = None

    opt = AdamW.__new__(AdamW)
    opt.data_parallel = ax
    opt.params = {"a.A": T()}
    with pytest.raises(UnreducedGradients, match="no per-chip gradients"):
        opt._reduce(None)
