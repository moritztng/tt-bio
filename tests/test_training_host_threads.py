"""The host's cores are divided across the ranks, and nothing may quietly stop dividing.

The defect this pins was measured on qb1 on 2026-09-19 (`perf/train_w_fourchip`): torch sizes
its intra-op pool from the machine and cannot see its sibling ranks, so a four-rank run took
4 x 16 threads on 16 physical cores and spent 913.98 s of a 931.05 s ABodyBuilder3 step in
`losses` and `host_backward`. The same per-rank work with the cores divided spends 2.50 s.
Nothing about that is visible at world 1, and at world 2 it costs 8 % -- which is why it
survived every two-chip measurement the campaign took.

So the property under test is not "a number is set" but "it goes DOWN as the world goes UP".
A test that only asserted `OMP_NUM_THREADS in env` would pass against the defect.
"""

import os

import pytest

from tt_bio.train.launcher import host_threads


def test_cores_are_divided_by_the_world():
    one = host_threads(1)
    assert one >= 1
    # Strictly decreasing while there are cores left to give away. On a host with fewer cores
    # than ranks every rung floors at 1, which is correct and is not a division failure, so the
    # comparison is skipped rather than asserted into a false red on a small machine.
    for w in (2, 4):
        if one >= w:
            assert host_threads(w) == one // w, (
                f"world {w} must get the host's {one} cores divided {w} ways, not "
                f"{host_threads(w)}. A rank that keeps the whole box is the 931 s step")
        assert host_threads(w) <= one
        assert host_threads(w) >= 1


def test_the_widest_world_never_asks_for_more_than_the_host_has():
    # world x per-rank threads <= cores, for every width. This is the invariant the defect
    # violated by a factor of the rank count.
    one = host_threads(1)
    for w in (1, 2, 3, 4, 8, 64):
        assert w * host_threads(w) <= max(one, w), (
            f"world {w} would run {w * host_threads(w)} threads on {one} cores")


def test_negative_control_torch_default_does_not_divide():
    """The control: torch's own default is the whole machine whatever the world is.

    Seen to fail against the shipped behaviour before the fix -- that is the point of it. If
    `host_threads` is ever reverted to returning the machine width, the first test above goes
    red and this one explains why.
    """
    import torch
    default = torch.get_num_threads()
    one = host_threads(1)
    if one < 2:
        pytest.skip("single-core host: there is nothing to divide and no defect to pin")
    assert host_threads(4) < default or default < 4, (
        f"torch's default is {default} threads a rank regardless of the world, which is the "
        f"oversubscription; host_threads(4) must be smaller")


def test_an_explicit_setting_is_an_override_and_survives(monkeypatch):
    """A caller who set OMP_NUM_THREADS meant it. `drive` uses setdefault, never assignment."""
    import inspect

    from tt_bio.train import launcher
    src = inspect.getsource(launcher.drive)
    assert 'setdefault("OMP_NUM_THREADS"' in src, (
        "drive must setdefault OMP_NUM_THREADS so an explicit caller value survives; a plain "
        "assignment would silently overrule someone who pinned it on purpose")
    assert 'env["OMP_NUM_THREADS"] =' not in src


def test_smt_is_not_counted_as_a_core():
    """Physical cores, not logical. Two threads on one core share the vector units."""
    smt = "/sys/devices/system/cpu/smt/active"
    if not os.path.exists(smt):
        pytest.skip("no SMT topology exposed on this host")
    active = open(smt).read().strip() == "1"
    logical = os.cpu_count() or 1
    expected = max(1, logical // 2) if active else logical
    assert host_threads(1) == expected
