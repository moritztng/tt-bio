"""`--fast` turns on every member of the fast set, and turning it off restores the environment.

The set is a dict of module globals, so a member whose name drifts away from its global would be
silently dropped: `set_fast(True)` would write a name nothing reads and the lever would ship off.
That is exactly how a measured win goes missing, so the membership is pinned here rather than
inferred from a fold's timing.
"""
import os

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


def test_dit_pair_bias_ships_in_both_modes(monkeypatch):
    """The DiT mask program is graded on the normal bar on Wormhole and p150a, so it is a default on both."""
    monkeypatch.delenv("RFD3_DPB", raising=False)
    assert "_DIT_DPB" not in M._FAST_SET
    assert M.env_flag("RFD3_DPB", True) is True
    M.set_fast(False)
    assert M._DIT_DPB is True
    assert callable(M._DPB.eligible)   # the flag once shadowed the dit_pair_bias module and every default design crashed


def test_css_ships_in_both_modes(monkeypatch):
    """The chunk-skipping atom SDPA is graded on the normal bar, so it is a default rather than a fast-set member."""
    monkeypatch.delenv("RFD3_CSS", raising=False)
    assert "_CSS" not in M._FAST_SET
    assert M.env_flag("RFD3_CSS", True) is True
    M.set_fast(False)
    assert M._CSS is True


def test_pair_ws_resid_ships_in_both_modes(monkeypatch):
    """The in-place z + transition(z) is graded on the normal bar, so it is a default rather than a fast-set member."""
    monkeypatch.delenv("RFD3_PAIR_WS_RESID", raising=False)
    assert "_PAIR_WS_RESID" not in M._FAST_SET
    assert M.env_flag("RFD3_PAIR_WS_RESID", True) is True
    M.set_fast(False)
    assert M._PAIR_WS_RESID is True
    assert callable(M.Transition.add_into)


def test_head_split_kernels_ship_in_both_modes(monkeypatch):
    """qkh / gch move data only (digests equal the reshape + permute path on WH and p150a), so they are defaults."""
    for name, env in (("_QKV_HEADS", "RFD3_QKV_HEADS"), ("_GCA_HEADS", "RFD3_GCA_HEADS")):
        monkeypatch.delenv(env, raising=False)
        assert name not in M._FAST_SET
        assert M.env_flag(env, True) is True
    M.set_fast(False)
    assert M._QKV_HEADS is True and M._GCA_HEADS is True


def test_process_z_collapse_ships_in_both_modes(monkeypatch):
    """The collapsed process_z is graded on the normal bar (8 paired WH seeds), so it left the fast set."""
    monkeypatch.delenv("RFD3_PROCESS_Z_COLLAPSE", raising=False)
    assert "_PROCESS_Z_COLLAPSE" not in M._FAST_SET
    assert M.env_flag("RFD3_PROCESS_Z_COLLAPSE", True) is True


def test_dit_shoist_ships_in_both_modes(monkeypatch):
    """The hoisted DiT conditioning is graded on the normal bar (8 paired p150a seeds), so it is a default."""
    monkeypatch.delenv("RFD3_DIT_SHOIST", raising=False)
    assert "_DIT_SHOIST" not in M._FAST_SET
    assert M.env_flag("RFD3_DIT_SHOIST", True) is True
    M.set_fast(False)
    assert M._DIT_SHOIST is True


def test_pair_ws_lofi_is_fast_only(monkeypatch):
    """pws in bfp8 at LoFi passed the fast bar (8 paired seeds on WH and p150a), not graded on the normal one."""
    monkeypatch.delenv("RFD3_PAIR_WS_LOFI", raising=False)
    assert M._FAST_SET["_PAIR_WS_LOFI"] == "RFD3_PAIR_WS_LOFI"
    M.set_fast(True)
    assert M._PAIR_WS_LOFI is True
    M.set_fast(False)
    assert M._PAIR_WS_LOFI is False


def test_pz_spread_is_a_default(monkeypatch):
    """The bank-spread gather table reads the same bits (digests equal on WH and p150a), so it is on in both modes."""
    monkeypatch.delenv("RFD3_PZ_SPREAD", raising=False)
    assert "_PZ_SPREAD" not in M._FAST_SET
    assert M.env_flag("RFD3_PZ_SPREAD", True) is True


def test_exact_data_movement_defaults(monkeypatch):
    """pzg, the grouped mask writer and the WH DiT q chunk change only the work (digests equal on WH and p150a)."""
    from tt_bio import rfd3_bias
    monkeypatch.delenv("RFD3_PZ_GATHER", raising=False)
    assert "_PZ_GATHER" not in M._FAST_SET
    assert M.env_flag("RFD3_PZ_GATHER", True) is True
    if "RFD3_MASKW_HG" not in os.environ:
        assert rfd3_bias.HG == 4
    monkeypatch.setattr(M, "_DIT_SDPA_CHUNKS_ENV", None)
    monkeypatch.setattr(M, "is_wormhole", lambda: True)
    assert M._dit_sdpa_chunks() == (352, 352)
    monkeypatch.setattr(M, "is_wormhole", lambda: False)
    assert M._dit_sdpa_chunks() == (64, 352)


def test_dknn_default_off_on_wormhole_only(monkeypatch):
    """The card neighbour index (same rows as the k-d tree) is on by default where it has run on a card."""
    monkeypatch.setattr(M, "_DKNN", None)
    assert "_DKNN" not in M._FAST_SET
    monkeypatch.setattr(M, "is_wormhole", lambda: False)
    assert M._dknn_on() is True
    monkeypatch.setattr(M, "is_wormhole", lambda: True)
    assert M._dknn_on() is False
    monkeypatch.setattr(M, "_DKNN", True)
    assert M._dknn_on() is True


def test_pf_dpb_wormhole_only(monkeypatch):
    """The pairformer pair bias through dit_pair_bias graded and paid on Wormhole."""
    monkeypatch.setattr(M, "_PF_DPB", None)
    assert "_PF_DPB" not in M._FAST_SET
    monkeypatch.setattr(M, "is_wormhole", lambda: True)
    assert M._pf_dpb_on() is True
    monkeypatch.setattr(M, "is_wormhole", lambda: False)
    assert M._pf_dpb_on() is False
    monkeypatch.setattr(M, "_PF_DPB", True)
    assert M._pf_dpb_on() is True


@pytest.mark.skipif(bool(os.environ.get("RFD3_PZ_ADDCMUL") or os.environ.get("RFD3_PZ_ROWS")), reason="forced by env")
def test_pz_addcmul_rows_default_on(monkeypatch):
    """The process_z addcmul and its row form are on on both arches; the row form follows the addcmul flag."""
    assert M._PZ_ADDCMUL is True and "_PZ_ADDCMUL" not in M._FAST_SET
    assert M._pz_rows_on() is True
    monkeypatch.setattr(M, "_PZ_ADDCMUL", False)
    assert M._pz_rows_on() is False


@pytest.mark.skipif(bool(os.environ.get("RFD3_PF_SDPA") or os.environ.get("RFD3_PAIR_WS_PSILU")), reason="forced by env")
def test_pf_sdpa_and_pair_ws_psilu_default_on():
    """The fused pairformer SDPA and the pair Transition's packer silu graded and paid on both arches."""
    assert M._PF_SDPA is True and M._PAIR_WS_PSILU is True
