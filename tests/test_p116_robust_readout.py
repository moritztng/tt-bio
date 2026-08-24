"""The p116 cap ladder's load-robust readout, pinned on the case that broke it.

qb2 is shared and cannot be quieted, so the cap ladder reports a second cost per arm: the q10 of
the fold's diffusion-module calls times the calls that arm makes per design. Contention is
one-sided -- a co-tenant can make a call slower, never faster -- so a low quantile recovers the
uncontended cost from the 200-400 samples already inside a fold.

That is only true if the quantile is taken within one batch size. The R2 rung folded 6 designs at
b=4, which runs batches [4, 2]; half its calls were the tail batch's cheap b=2 calls, the pooled
q10 landed among them, and the arm read 31.0 s/design against its own 49.0 s wall -- a phantom
1.54x on the arm that decides a shipped default. These tests pin the fix and the equivalence that
makes it safe.
"""
from __future__ import annotations

import pytest

from _port_module import port_module

p116 = port_module("rfd3_port", "p116_cap_ladder")


def _fold(batch_designs, per_call):
    """Synthesise one fold: `batch_designs` batches, `per_call[i]` seconds per call in batch i,
    199 calls each, in the (designs, first_call, last_call) form the harness records."""
    calls, batches, k = [], [], 0
    for d, c in zip(batch_designs, per_call):
        calls.extend([c] * 199)
        batches.append((d, k, k + 199))
        k += 199
    return calls, batches


def test_ragged_tail_does_not_set_the_whole_arm_cost():
    """The measured R2 b=4 fold: batches [4, 2] at 0.93 and 0.468 s per call, 6 designs."""
    calls, batches = _fold([4, 2], [0.93, 0.468])
    seg = p116.split_batches(calls, batches)
    got = p116.robust_s_per_design(seg, 6)

    assert [b["designs"] for b in seg] == [4, 2]
    assert [b["q10"] for b in seg] == [0.93, 0.468]
    assert got == pytest.approx((0.93 * 199 + 0.468 * 199) / 6, rel=1e-9)
    assert got == pytest.approx(46.36, abs=0.05)

    # what the pooled form reported, and why it was wrong: the 10th percentile of the pooled
    # calls is a tail-batch call, so it prices all 398 calls at the tail's rate.
    pooled = p116.q10(calls) * len(calls) / 6
    assert p116.q10(calls) == 0.468
    assert pooled == pytest.approx(31.05, abs=0.05)
    assert got / pooled == pytest.approx(1.49, abs=0.02)


@pytest.mark.parametrize("designs,batch", [(6, 1), (6, 2), (6, 6), (2, 1), (2, 2), (4, 4)])
def test_matches_the_pooled_form_when_every_batch_is_the_same_size(designs, batch):
    """No behaviour change on the arms that were already right, which is every arm whose design
    count divides evenly by its batch size."""
    n_batches = designs // batch
    calls, batches = _fold([batch] * n_batches, [0.25 * batch] * n_batches)
    seg = p116.split_batches(calls, batches)
    assert p116.robust_s_per_design(seg, designs) == pytest.approx(
        p116.q10(calls) * len(calls) / designs, rel=1e-9)


def test_ragged_flag_is_what_the_artifact_records():
    """`ragged` is what tells a later reader which folds needed the per-batch form."""
    assert len({b["designs"] for b in p116.split_batches(*_fold([4, 2], [0.9, 0.5]))}) == 2
    assert len({b["designs"] for b in p116.split_batches(*_fold([2, 2], [0.5, 0.5]))}) == 1


def test_contention_in_one_batch_does_not_leak_into_another():
    """The whole point of the statistic. A co-tenant stalls the b=1 batches; the q10 of each
    batch still recovers its uncontended cost, because contention is one-sided."""
    calls, batches = [], []
    for i in range(6):
        seg = [0.24] * 180 + [3.0] * 19          # 90 % clean, 10 % stalled
        batches.append((1, len(calls), len(calls) + len(seg)))
        calls.extend(seg)
    got = p116.robust_s_per_design(p116.split_batches(calls, batches), 6)
    assert got == pytest.approx(0.24 * 199, rel=1e-9)
    assert sum(calls) / 6 > got * 1.2            # the sum-of-walls readout is 20 %+ inflated


def test_empty_fold_reports_nothing_rather_than_zero():
    assert p116.robust_s_per_design(p116.split_batches([], []), 4) is None
