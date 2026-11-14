"""A Transition height the device refuses is recorded and the call re-run at half of it.

Host-only: `_transition` is stubbed to throw tt-metal's static-CB clash above a height, the way
fc2 did for OpenDDE's c=64 MSA Transition at 256 tokens on a p150a.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

CLASH = ("TT_THROW @ program.cpp:1052: Statically allocated circular buffers in program 269 clash "
         "with L1 buffers on core range [(x=0,y=0) - (x=10,y=8)]. L1 buffer allocated at 890880 "
         "and static circular buffer region ends at 893440")
KEY = (256, 64, 256)


class _X:
    def is_allocated(self):
        return True


def _module(derive, fits):
    """A Transition whose height derivation is `derive()` and which refuses any height > fits."""
    m = T.Transition.__new__(T.Transition)
    tried = []

    def run(x, mc, add):
        h = derive()
        if KEY in T.TRANSITION_H_CLASH:
            h = min(h, max(1, T.TRANSITION_H_CLASH[KEY] // 2))
        m._h_last = (KEY, h)
        tried.append(h)
        if h > fits:
            raise RuntimeError(CLASH)
        return h

    m._transition = run
    return m, tried


@pytest.fixture(autouse=True)
def _clean():
    T.TRANSITION_H_CLASH.clear()
    yield
    T.TRANSITION_H_CLASH.clear()


def test_a_refused_height_halves_until_it_fits_and_is_remembered():
    m, tried = _module(lambda: 192, fits=60)
    assert m._transition_fit(_X(), None, False) == 48
    assert tried == [192, 96, 48]
    assert T.TRANSITION_H_CLASH[KEY] == 96
    tried.clear()
    assert m._transition_fit(_X(), None, False) == 48
    assert tried == [48]            # later calls start below the refused height


def test_a_forced_height_that_was_refused_raises_instead_of_looping():
    T.TRANSITION_H_CLASH[KEY] = 96
    m = T.Transition.__new__(T.Transition)

    def run(x, mc, add):
        m._h_last = (KEY, 96)       # TT_BIO_TRANSITION_H_CHUNK pins it past the record
        raise RuntimeError(CLASH)

    m._transition = run
    with pytest.raises(RuntimeError):
        m._transition_fit(_X(), None, False)


def test_other_errors_and_written_inputs_are_not_retried():
    m, _ = _module(lambda: 192, fits=60)
    real = m._transition

    def other(x, mc, add):
        m._h_last = (KEY, 192)
        raise RuntimeError("shape mismatch")

    m._transition = other
    with pytest.raises(RuntimeError, match="shape mismatch"):
        m._transition_fit(_X(), None, False)

    def wrote(x, mc, add):
        T.PAIR_INPLACE_STATS[1] += 1
        return real(x, mc, add)

    m._transition = wrote
    with pytest.raises(RuntimeError, match="circular buffers"):
        m._transition_fit(_X(), None, False)
