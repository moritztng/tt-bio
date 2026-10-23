"""T2CM (`trimul_tail.fused_tail_ln_cm`, the `trimul_t2cm` lever) host side, no device.

T2 reading the trimul einsum's channel-major [1, K, S, S] output itself, so `reblock_permute_back` never
runs. Checked here: the lever is named and in no mode, eligibility declines cleanly (a reason string,
nothing thrown or touched), and the program is T2m's with c_off 0 (whole-tile staging at c_z 128 on both
L1 budgets). T2m's gather is checked in tests/test_trimul_t2m.py. The device A/B is perf/spd_trimul/t2cm.py.

Run: TT_VISIBLE_DEVICES= python3 -m pytest tests/test_trimul_t2cm.py
"""
import re
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")
from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TTL

KDIR = Path(TTL.__file__).resolve().parent / "kernels" / "trimul_tail_res"
L1_BH_BANK = 1_461_760           # `_l1_bank_bytes` on Blackhole
L1_WH_BANK = 1_499_136 * 0.98    # the Wormhole figure tests/test_transition_t3.py budgets against


def test_lever_named_and_scoped():
    assert "trimul_t2cm" in T.LEVERS
    assert "trimul_t2cm" not in T.UNGRADED_LEVERS | T.NORMAL_LEVERS
    from tt_bio import openfold3_fold as of3
    assert "trimul_t2cm" in of3.FAST_LEVERS and "trimul_t2cm" in of3.NORMAL_LEVERS   # Blackhole normal, nbg
    assert "trimul_t2cm" not in of3.WH_NORMAL_LEVERS | of3.WH_FAST_LEVERS


# ---------------------------------------------------------------- eligibility

class _Fake:
    def __init__(self, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=None, padded=None):
        self.shape, self.padded_shape = tuple(shape), tuple(padded or shape)
        self.dtype, self.layout = dtype, layout
        self._mem = mem or ttnn.DRAM_MEMORY_CONFIG

    def memory_config(self):
        return self._mem


def _args(S=512, K=128, N=128):
    return dict(xcm=_Fake([1, K, S, S]), ln_w=_Fake([K], padded=[32, K]), ln_b=_Fake([K], padded=[32, K]),
                wp=_Fake([K, N]), g=_Fake([1, S, S, N]), resid=_Fake([1, S, S, N]))


def test_eligibility_declines_cleanly(monkeypatch):
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: L1_BH_BANK)
    monkeypatch.setattr(T, "arch_name", lambda: "blackhole")
    monkeypatch.setattr(TTL, "T2", False)
    monkeypatch.setattr(TTL, "T2CM", False)
    assert TTL.t2cm_eligible(**_args()) == "off"
    with T.levers("trimul_t2cm"):                 # without T2 there is nothing to feed
        assert TTL.t2cm_eligible(**_args()) == "off"
    with T.levers("trimul_t2"):                   # T2 alone keeps its permuted input
        assert TTL.t2cm_eligible(**_args()) == "off"
    with T.levers("trimul_t2,trimul_t2cm"):
        assert TTL.t2cm_eligible(**_args()) is None
        assert TTL.t2cm_eligible(**_args(S=256)) is None
        assert TTL.t2cm_eligible(**_args(K=256, N=256)) is None          # protenix-v2's c_z: two stage parts, which Blackhole takes
        a = _args()
        a["xcm"] = _Fake([1, 128, 500, 500], padded=[1, 128, 512, 512])
        assert TTL.t2cm_eligible(**a) == "cm_shape"                      # ragged: the permute keeps it
        a = _args()
        a["xcm"] = _Fake([1, 512, 512, 128])                             # the permuted layout is not T2CM's
        assert TTL.t2cm_eligible(**a) in ("cm_shape", "act_k")
        a = _args()
        a["xcm"] = _Fake([1, 128, 512, 512], mem=ttnn.L1_MEMORY_CONFIG)
        assert TTL.t2cm_eligible(**a) == "memcfg"
        a = _args()
        a["resid"] = _Fake([1, 256, 256, 128])
        assert TTL.t2cm_eligible(**a) == "out_shape"
        a = _args()
        a["ln_w"] = _Fake([128], dtype=ttnn.float32)
        assert TTL.t2cm_eligible(**a) == "ln_params"
        a = _args()
        a["g"] = _Fake([1, 512, 512, 128], dtype=ttnn.bfloat8_b)
        assert TTL.t2cm_eligible(**a) == "dtype"
        monkeypatch.setattr(T, "_l1_bank_bytes", lambda: 100_000)        # a smaller part: refuse, not throw
        assert TTL.t2cm_eligible(**_args()) == "l1"


def test_off_declines_without_counting_or_touching(monkeypatch):
    monkeypatch.setattr(TTL, "T2", False)
    monkeypatch.setattr(TTL, "T2CM", False)
    before, rej = list(TTL.T2CM_STATS), dict(TTL.REJECTS)
    assert TTL.fused_tail_ln_cm(None, None, None, None, None, None, (0, 0, 0, 0), (1, 1)) is None
    assert TTL.T2CM_STATS == before and TTL.REJECTS == rej


# ---------------------------------------------------------------- the program is T2m's

@pytest.mark.parametrize("bank", [L1_BH_BANK, L1_WH_BANK])
def test_c128_takes_t2m_whole_tile_staging(monkeypatch, bank):
    """T2CM is T2m with c_off 0; at c_z 128 its stage fits beside T2's buffers in one part (P = 1)."""
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: bank)
    wp = torch.zeros(128, 128)
    assert TTL._t2cm_parts(wp) == 1


@pytest.mark.parametrize("arch,bank", [("blackhole", L1_BH_BANK), ("wormhole_b0", L1_WH_BANK)])
def test_c384_declines_on_its_part_count(monkeypatch, arch, bank):
    """OpenDDE's c_z 384 stage needs 4 parts, above T2CM_MAX_P: the trimul keeps the move + T2 instead."""
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: bank)
    monkeypatch.setattr(T, "arch_name", lambda: arch)
    assert TTL._t2cm_parts(torch.zeros(384, 384)) > TTL.T2CM_MAX_P[arch]
    assert TTL.t2cm_fit(torch.zeros(384, 384)) == "parts"
    assert TTL.t2cm_fit(torch.zeros(128, 128)) is None


@pytest.mark.parametrize("arch,bank,fit", [("blackhole", L1_BH_BANK, None), ("wormhole_b0", L1_WH_BANK, "parts")])
def test_c256_two_parts_only_on_blackhole(monkeypatch, arch, bank, fit):
    """Protenix-v2's c_z 256 stage needs 2 parts: Blackhole takes T2CM, Wormhole keeps the move + T2."""
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: bank)
    monkeypatch.setattr(T, "arch_name", lambda: arch)
    assert TTL._t2cm_parts(torch.zeros(256, 256)) == 2
    assert TTL.t2cm_fit(torch.zeros(256, 256)) == fit


def test_routes_through_the_moved_program():
    src = Path(TTL.__file__).read_text()
    assert "moved=(xcm, c_off, P)" in src
    assert not (KDIR / "reader_ln_cm.cpp").exists() and not (KDIR / "writer_ln_cm.cpp").exists()
    assert "CM" not in re.sub(r"//.*", "", (KDIR / "compute_ln.cpp").read_text())


def test_rfd4_bf16_stream_form(monkeypatch):
    """RFD4's pair_tail on the bf16 pair stream (`pair_tail_cm`): hidden 256 -> c_z 384, a key plain T2CM does not
    take, admitted only with `stream16`; its stage fits Blackhole's L1 whole (P = 1), double-buffered."""
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: L1_BH_BANK)
    monkeypatch.setattr(T, "arch_name", lambda: "blackhole")
    monkeypatch.setattr(TTL, "T2", False)
    monkeypatch.setattr(TTL, "T2CM", False)
    from tt_bio.rfd4 import trunk
    assert "pair_tail_cm" in T.LEVERS and "pair_tail_cm" not in T.NORMAL_LEVERS
    assert "pair_tail_cm" in trunk.NORMAL_LEVERS
    a = _args(K=256, N=384)
    assert TTL.t2cm_eligible(**a, stream16=True) == "off"
    with T.levers("pair_tail_cm"):
        assert TTL.t2cm_eligible(**a) == "off"                         # trimul_t2cm's own gate is unchanged
        assert TTL.t2cm_eligible(**a, stream16=True) is None
        assert TTL.t2cm_eligible(**_args(S=384, K=256, N=384), stream16=True) is None
        assert TTL.t2cm_eligible(**a, stream16=True, wg=_Fake([256, 384])) == "stream16_form"
    with T.levers("trimul_t2,trimul_t2cm"):
        assert TTL.t2cm_eligible(**a) == "key=8x12"                    # no other model's T2CM eligibility moves
    assert TTL._t2cm_parts(torch.zeros(256, 384)) == 1
    # measured on 110 cores: wins at 384 / 512, loses or is noise at 256 / 288 / 320 / 352 / 448, so those keep the move
    assert [n for n in (256, 288, 320, 352, 384, 448, 512) if TTL.t2cm_stream16_fits(n, 110)] == [384, 512]
