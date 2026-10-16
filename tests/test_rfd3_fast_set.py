"""`--fast` turns on every member of the fast set, and turning it off restores the environment.

The set is a dict of module globals, so a member whose name drifts away from its global would be
silently dropped: `set_fast(True)` would write a name nothing reads and the lever would ship off.
That is exactly how a measured win goes missing, so the membership is pinned here rather than
inferred from a fold's timing.
"""
import pytest

from tt_bio.rfd3 import model as M


@pytest.fixture(autouse=True)
def _restore():
    before = {name: getattr(M, name) for name in M._FAST_SET}
    yield
    for name, value in before.items():
        setattr(M, name, value)


def test_every_member_names_a_real_global():
    for name in M._FAST_SET:
        assert hasattr(M, name), name


def test_fast_on_turns_every_member_on():
    for name in M._FAST_SET:
        setattr(M, name, False)
    M.set_fast(True)
    assert all(getattr(M, name) for name in M._FAST_SET)


def test_fast_off_restores_the_environment(monkeypatch):
    for name, env in M._FAST_SET.items():
        monkeypatch.setenv(env, "0")
    M.set_fast(True)
    M.set_fast(False)
    for name in M._FAST_SET:
        assert getattr(M, name) is False, name


def test_pair_ws_ships_in_both_modes(monkeypatch):
    """Graded on the normal bar too, so it is a default rather than a fast-set member."""
    monkeypatch.delenv("RFD3_PAIR_WS", raising=False)
    assert "_PAIR_WS" not in M._FAST_SET
    assert M.env_flag("RFD3_PAIR_WS", True) is True
    M.set_fast(False)
    assert M._PAIR_WS is True


def test_dit_pair_bias_ships_in_both_modes_on_wormhole(monkeypatch):
    """The DiT mask program is graded on the normal bar on Wormhole; Blackhole waits for its own grade."""
    class Dev:
        def __init__(self, arch):
            self._arch = arch

        def arch(self):
            return self._arch

    monkeypatch.delenv("RFD3_DPB", raising=False)
    M.set_fast(False)
    assert M._dpb_on(Dev(M.ttnn.Arch.WORMHOLE_B0)) is True
    assert M._dpb_on(Dev(M.ttnn.Arch.BLACKHOLE)) is False
    monkeypatch.setenv("RFD3_DPB", "1")
    assert M._dpb_on(Dev(M.ttnn.Arch.BLACKHOLE)) is True
