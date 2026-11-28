"""Unit tests for `tt_bio.autograd.suspended()` (pure CPU, no device)."""
from __future__ import annotations

import pytest

from tt_bio import autograd as ag
from tt_bio import ops


def test_suspended_clears_and_restores_hooks():
    was_installed = ag.installed()
    if not was_installed:
        ag.install()
    try:
        assert ops.taping() is True
        assert ops.grad_hook() is not None
        assert ops.host_softmax_hook() is not None
        assert ops.declines_under_tape("pair_relu") is True
        with ops.recycle_region(0, 1):
            assert ag.is_grad_enabled() is False
        with ag.suspended():
            assert ops.taping() is False
            assert ops.grad_hook() is None
            assert ops.host_softmax_hook() is None
            assert ops.declines_under_tape("pair_relu") is False
            with ops.recycle_region(0, 1):
                assert ag.is_grad_enabled() is True
        assert ops.taping() is True
        assert ag.installed() is True
        assert ops.host_softmax_hook() is not None
        assert ops.declines_under_tape("pair_relu") is True
        with ops.recycle_region(0, 1):
            assert ag.is_grad_enabled() is False
    finally:
        if not was_installed:
            ag.uninstall()


def test_suspended_restores_hooks_on_exception():
    was_installed = ag.installed()
    if not was_installed:
        ag.install()
    try:
        with pytest.raises(ValueError, match="boom"):
            with ag.suspended():
                assert ops.taping() is False
                raise ValueError("boom")
        assert ops.taping() is True
        assert ag.installed() is True
    finally:
        if not was_installed:
            ag.uninstall()


def test_suspended_noop_when_not_installed():
    was_installed = ag.installed()
    if was_installed:
        ag.uninstall()
    try:
        assert ops.taping() is False
        with ag.suspended():
            assert ops.taping() is False
        assert ops.taping() is False
    finally:
        if was_installed:
            ag.install()


def test_suspended_refuses_inside_tape():
    with ag.tape():
        with pytest.raises(RuntimeError, match="outside autograd.tape"):
            with ag.suspended():
                pass


def test_suspend_tape_unshims_and_restores_inside_tape():
    from tt_bio import taped_ttnn as TT

    with ag.tape():
        assert bool(TT._SHIMMED) is True
        assert ops.taping() is True
        with ops.suspend_tape():
            assert not TT._SHIMMED
            assert ops.taping() is False
        assert bool(TT._SHIMMED) is True
        assert ops.taping() is True
        wrapped = ops.wrap_untaped("raw_handle")
        assert isinstance(wrapped, ag.Tensor) and wrapped.requires_grad is False
