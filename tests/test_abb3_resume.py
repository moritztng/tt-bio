"""The reproduction's resume and its data-parallel invariant, on the host.

Four things are checked here and each one is a failure that has no signature in a loss curve:

1. a resume restores the masters, BOTH RAdam moments, RAdam's own step counter and the dropout
   generator -- and the trajectory after it is bit-identical to the uninterrupted one;
2. the NEGATIVE CONTROL: a resume that restores only the weights diverges immediately. Without
   it, test 1 would pass on a checkpoint that saved nothing the optimizer needs, because both
   runs would simply be wrong in the same way;
3. the cross-rank sum is taken in the same order on every rank, so every rank computes the same
   bits. Floating point addition is not associative, and two ranks that disagree in the last
   mantissa bit diverge from the next step onward;
4. the master-weight equality check FAILS on a divergence. A check that cannot fail is the
   check removed.

No device: everything above is host state, and a gate that needs a card is a gate CI cannot run.
``scripts/abb3_port/resume_gate.py`` is the same claim on the card at the real configuration,
with a real SIGKILL and an unattended restart.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tt_bio.train.abb3_checkpoint import (FORMAT, latest_checkpoint, load_run_state,
                                          save_run_state)
from tt_bio.train.hostreduce import HostReduce, master_hash


class FakeTensor:
    """Stands in for a taped device tensor: the resume only writes ``.value`` on it."""

    def __init__(self):
        self.value = None


class FakeStep:
    """The host half of a ``TrainStep``: the mirror, the optimizer, the dropout, the recipe.

    Deliberately not a mock of the device forward. What a resume has to get right is entirely
    in these four, and building the real step would make the gate need a card for no added
    coverage of the thing being checked.
    """

    def __init__(self, n=3, size=5, seed=0, accumulate=16):
        from tt_bio.abodybuilder3 import Dropout
        g = torch.Generator().manual_seed(seed)
        self.mirror = [torch.randn(size, size, generator=g).requires_grad_(True)
                       for _ in range(n)]
        self.params = [FakeTensor() for _ in range(n)]
        self.optimizer = torch.optim.RAdam(self.mirror, lr=1e-3, weight_decay=1e-4)
        self.dropout = Dropout.__new__(Dropout)
        self.dropout.rate, self.dropout.calls = 0.1, 0
        self.dropout.generator = torch.Generator().manual_seed(seed)
        self.recipe = {"lr": 1e-3}
        self.accumulate = accumulate

    def step(self, i: int):
        """One synthetic optimizer step. The gradient is derived from the STEP NUMBER.

        Index-derived and not drawn from a carried stream, which is the run's own arrangement:
        ``run()`` places the data order with ``islice(batches(...), start, None)`` off the step
        number, so a resume does not have to replay a generator to reach the right batch.

        The dropout generator IS advanced, and is the piece that has to be carried rather than
        derived -- it is part of the recipe, so a resume that re-seeds it replays masks the run
        already used. Without this line test 1 would pass with the RNG forgotten.
        """
        g = torch.Generator().manual_seed(1000 + i)
        scale = torch.rand(1, generator=self.dropout.generator).item() + 0.5
        self.dropout.calls += 1
        for m in self.mirror:
            m.grad = torch.randn(m.shape, generator=g) * scale
        self.optimizer.step()
        return master_hash([m.detach().numpy() for m in self.mirror])


def _noop_upload(t):
    return t


def _trajectory(step, steps, first=1):
    return [step.step(i) for i in range(first, first + steps)]


def test_a_resume_reproduces_the_uninterrupted_trajectory_bit_for_bit(tmp_path):
    control = _trajectory(FakeStep(), 6)

    killed = FakeStep()
    first = _trajectory(killed, 3)
    ck = save_run_state(tmp_path / "step-000000003.safetensors", killed, global_step=3)

    # A genuinely fresh process's worth of state: a new step object that has never trained.
    revived = FakeStep()
    state = load_run_state(ck, revived, upload=_noop_upload)
    assert state.step == 3
    # Steps 4, 5, 6 -- placed from the step number the checkpoint carries, which is exactly how
    # the run places them after a resume.
    after = _trajectory(revived, 3, first=state.step + 1)

    assert first == control[:3]
    assert after == control[3:], (
        "the resumed trajectory diverged from the uninterrupted one. The saved set is the "
        "masters, both RAdam moments, RAdam's step counter and the dropout generator; a "
        "divergence here means one of them is missing")
    assert all(p.value is not None for p in revived.params), (
        "the resume did not write the device copies back from the masters")


def test_restoring_only_the_weights_diverges_so_the_check_above_is_not_vacuous(tmp_path):
    """The negative control. Omitting the optimizer state is SILENT, which is the whole point.

    ``load_adapter``'s ``opt`` argument is optional and this is what omitting it costs: the
    weights are right, both moments are a fresh optimizer's zeros, the step count driving bias
    correction is back at one, and the run carries on with a plausible loss curve for a
    different optimisation problem. Over a nine-day run with 9-47 watchdog resets, that is
    9-47 silent restarts of the optimiser.
    """
    control = _trajectory(FakeStep(), 6)
    killed = FakeStep()
    _trajectory(killed, 3)
    ck = save_run_state(tmp_path / "step-000000003.safetensors", killed, global_step=3)

    revived = FakeStep()
    state = load_run_state(ck, revived, upload=_noop_upload)
    # The whole of the omission: the masters are right and the optimizer is a fresh one.
    revived.optimizer = torch.optim.RAdam(revived.mirror, lr=1e-3, weight_decay=1e-4)
    after = _trajectory(revived, 3, first=state.step + 1)
    assert state.step == 3
    assert after != control[3:], (
        "a weights-only resume produced the same trajectory as a complete one, so the "
        "positive test proves nothing. Either RAdam's first step does not depend on its "
        "moments or the harness is not exercising them")


def test_a_checkpoint_from_a_different_global_batch_is_refused(tmp_path):
    """Resuming across a change in the global batch would continue a different recipe.

    The chip count is deliberately NOT part of this: the checkpoint holds the global batch's
    divisor, so a 2-chip run interrupted by a reset can resume on 1 chip when only one card is
    free. Refusing that would turn a card-availability problem into a lost run.
    """
    step = FakeStep(accumulate=16)
    ck = save_run_state(tmp_path / "step-000000001.safetensors", step, global_step=1)
    with pytest.raises(ValueError, match="accumulate"):
        load_run_state(ck, FakeStep(accumulate=8), upload=_noop_upload)
    with pytest.raises(ValueError, match="parameters"):
        load_run_state(ck, FakeStep(n=4, accumulate=16), upload=_noop_upload)


def test_latest_checkpoint_reads_the_step_number_not_the_mtime(tmp_path):
    for n in (5, 120, 9):
        save_run_state(tmp_path / f"step-{n:09d}.safetensors", FakeStep(), global_step=n)
    assert latest_checkpoint(tmp_path).name == "step-000000120.safetensors"
    assert latest_checkpoint(tmp_path / "empty") is None
    assert FORMAT >= 1


def test_the_cross_rank_sum_is_order_fixed_so_every_rank_computes_the_same_bits(tmp_path):
    """Two ranks in one process, exercised through the real rendezvous.

    Values chosen so the sum is order-dependent in float32: 1.0 plus two half-eps terms sums to
    1.0 one way and to 1.0000001 the other. A reduce that did not fix the order would pass on
    friendlier numbers and diverge on a real gradient.
    """
    eps = np.float32(np.finfo(np.float32).eps)
    a = np.array([1.0, eps / 2, eps / 2], dtype=np.float32)
    b = np.array([eps / 2, 1.0, eps / 2], dtype=np.float32)
    r0 = HostReduce(tmp_path / "rv", 0, 2, timeout=30)
    r1 = HostReduce(tmp_path / "rv", 1, 2, timeout=30)
    import threading
    out = {}

    def go(comm, vec, key):
        out[key] = comm.allreduce(vec, step=1)

    ts = [threading.Thread(target=go, args=(r0, a, 0)), threading.Thread(target=go, args=(r1, b, 1))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert out[0].tobytes() == out[1].tobytes(), (
        "the two ranks reduced the same inputs to different bits, so they will train "
        "different models from the next step onward")
    assert out[0].tobytes() == (a + b).tobytes()


def test_the_master_equality_check_fails_on_a_divergence(tmp_path):
    """A check that cannot fail is the check removed rather than met."""
    import threading
    r0 = HostReduce(tmp_path / "rv", 0, 2, timeout=30)
    r1 = HostReduce(tmp_path / "rv", 1, 2, timeout=30)
    errs = []

    def go(comm, digest):
        try:
            comm.check_equal(digest, step=1)
        except RuntimeError as e:
            errs.append(e)

    ts = [threading.Thread(target=go, args=(r0, b"\x00" * 16)),
          threading.Thread(target=go, args=(r1, b"\x01" * 16))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert len(errs) == 2, "a divergence must fail every rank, not just the one that noticed"
    assert "diverged" in str(errs[0])
    # And it passes when they agree, so it is not simply always failing.
    HostReduce(tmp_path / "rv1", 0, 1).check_equal(b"\x00" * 16, step=1)


def test_a_dead_rank_times_out_rather_than_stalling_the_survivors(tmp_path):
    """A rendezvous that waits forever turns one watchdog reset into a silently hung run."""
    comm = HostReduce(tmp_path / "rv", 0, 2, timeout=0.3, poll=0.05)
    with pytest.raises(TimeoutError, match="watchdog"):
        comm.allreduce(np.ones(4, dtype=np.float32), step=1)


def test_master_hash_moves_on_the_low_mantissa_bits(tmp_path):
    """The invariant is only worth having if its digest sees the bits that diverge first."""
    a = np.ones(8, dtype=np.float32)
    b = a.copy()
    b[3] = np.nextafter(b[3], np.float32(2.0))
    assert master_hash([a]) != master_hash([b])
    assert master_hash([a]) == master_hash([a.copy()])
    # Shape is part of the digest, so two ranks holding the same values in different shapes
    # do not read as equal.
    assert master_hash([a.reshape(2, 4)]) != master_hash([a.reshape(4, 2)])
