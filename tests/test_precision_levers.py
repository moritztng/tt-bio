"""Protenix's precision levers: one named switch each, a mode is a set of names, and a model's set
is active only inside its own build and fold. Host only: no device is opened."""
from __future__ import annotations

import pytest

import tt_bio.tenstorrent as T


def test_fast_set_never_holds_the_grid_dependent_lever_or_an_ungraded_one():
    assert T.FAST_LEVERS <= frozenset(T.LEVERS) and "triatt_b8" not in T.FAST_LEVERS
    assert not T.UNGRADED_LEVERS & (T.FAST_LEVERS | T.NORMAL_LEVERS)
    assert T.NORMAL_LEVERS <= T.FAST_LEVERS


def test_a_model_names_its_own_fast_set_and_the_harness_switch_still_wins(monkeypatch):
    from tt_bio import openfold3_fold
    monkeypatch.delenv("TT_BIO_LEVERS", raising=False)
    monkeypatch.setattr(T, "_FAST_MODE", True)
    assert T.mode_levers() == T.FAST_LEVERS
    assert T.mode_levers(fast=openfold3_fold.FAST_LEVERS) == T.NORMAL_LEVERS
    monkeypatch.setenv("TT_BIO_LEVERS", "fast")
    assert T.mode_levers(fast=openfold3_fold.FAST_LEVERS) == T.FAST_LEVERS
    monkeypatch.setattr(T, "_FAST_MODE", False)
    monkeypatch.delenv("TT_BIO_LEVERS")
    assert T.mode_levers(fast=openfold3_fold.FAST_LEVERS) == T.NORMAL_LEVERS


def test_parse_expands_modes_and_rejects_unknown_names():
    assert T.parse_levers("fast") == T.FAST_LEVERS
    assert T.parse_levers("normal,opm_b8") == T.NORMAL_LEVERS | {"opm_b8"}
    assert T.parse_levers("") == T.parse_levers("none") == frozenset()
    with pytest.raises(ValueError, match="unknown precision lever"):
        T.parse_levers("lofi,bfp2")


def test_parse_adds_and_drops_after_the_first_term():
    assert T.parse_levers("fast-lofi-acc_off") == T.FAST_LEVERS - {"lofi", "acc_off"}
    assert T.parse_levers("normal+opm_b8+lofi") == T.NORMAL_LEVERS | {"opm_b8", "lofi"}
    assert T.parse_levers("fast+triatt_b8") == T.FAST_LEVERS | {"triatt_b8"}


def test_levers_restore_the_previous_set_even_on_error():
    assert not T.lever("lofi")
    with pytest.raises(RuntimeError):
        with T.levers("lofi,opm_b8"):
            assert T.lever("lofi") and T.lever("opm_b8") and not T.lever("acc_off")
            with T.levers(()):
                assert not T.lever("lofi")
            assert T.lever("lofi")
            raise RuntimeError
    assert T._LEVERS == frozenset()


def test_triatt_formats_follow_the_levers():
    with T.levers("triatt_bias_b8"):
        assert T._triatt_bias_b8() and not T._triatt_b8()
    assert not T._triatt_bias_b8()


def test_dit_lowp_formats_follow_the_dit_dtype_and_the_levers():
    ttnn, ckc = T.ttnn, object()
    with T.levers("dit_mm16+dit_b8"):
        mm16, b8 = T.dit_lowp(ttnn.float32, ckc), T.dit_lowp(ttnn.bfloat16, ckc)
    assert (mm16.w, mm16.act, mm16.out, mm16.k1, mm16.ckc) == (ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, True, ckc)
    assert (b8.w, b8.act, b8.mid, b8.out, b8.k1) == (ttnn.bfloat8_b,) * 3 + (ttnn.bfloat16, False)
    assert not b8.ckc.fp32_dest_acc_en
    with T.levers("dit_mm16"):
        assert T.dit_lowp(ttnn.bfloat16, ckc) is None
    assert T.dit_lowp(ttnn.float32, ckc) is None


@pytest.mark.parametrize("fast,want", [(False, T.NORMAL_LEVERS), (True, T.FAST_LEVERS)])
def test_checkpoint_entry_takes_the_modes_set(monkeypatch, tmp_path, fast, want):
    import torch
    import tt_bio.protenix as P

    seen = {}

    def fake_init(self, sd, ckc, dev, **kw):
        seen.update(kw)

    monkeypatch.setattr(P.Protenix, "__init__", fake_init)
    monkeypatch.setattr(T, "_FAST_MODE", fast)
    ckpt = tmp_path / "w.pt"
    torch.save({"model": {}}, ckpt)
    P.Protenix.load_from_checkpoint(str(ckpt), compute_kernel_config=object(), device=object())
    assert seen["levers"] == want
    monkeypatch.setenv("TT_BIO_LEVERS", "fast-lofi")
    P.Protenix.load_from_checkpoint(str(ckpt), compute_kernel_config=object(), device=object())
    assert T.parse_levers(seen["levers"]) == T.FAST_LEVERS - {"lofi"}


def test_fold_runs_under_the_models_levers_and_restores():
    import tt_bio.protenix as P

    class M:
        _levers = frozenset({"acc_off"})

        @P._under_levers
        def fold(self):
            return T.lever("acc_off")

    assert M().fold() is True and not T.lever("acc_off")


def test_trimul_levers_reach_their_kernels_only_inside_the_set():
    import tt_bio.reblock_permute as RB
    import tt_bio.trimul_tail as TTL

    assert (T._trimul_ibw_full(), TTL._epi(), RB._gate_lean()) == (False, 0, 2)
    with T.levers("trimul_ibw,trimul_tail"):
        assert (T._trimul_ibw_full(), TTL._epi(), RB._gate_lean()) == (True, 2, 2)
        assert T._trimul_in0_block_w(23, T._trimul_ibw_full()) == 23
    assert T._trimul_in0_block_w(23, T._trimul_ibw_full()) == 1


def test_bfp8_fidelity_drops_to_hifi2_only_when_both_operands_are_bfp8():
    import types
    import ttnn
    b8, bf = types.SimpleNamespace(dtype=ttnn.bfloat8_b), types.SimpleNamespace(dtype=ttnn.bfloat16)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                           fp32_dest_acc_en=False, packer_l1_acc=True)
    out = T.bfp8_fidelity(ckc, b8, b8)
    assert out.math_fidelity == ttnn.MathFidelity.HiFi2
    assert (out.math_approx_mode, out.fp32_dest_acc_en, out.packer_l1_acc) == (True, False, True)
    assert T.bfp8_fidelity(ckc, b8, bf) is ckc and T.bfp8_fidelity(ckc, bf, b8) is ckc
    lofi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi)
    assert T.bfp8_fidelity(lofi, b8, b8) is lofi
