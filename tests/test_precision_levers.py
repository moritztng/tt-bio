"""Protenix's precision levers: one named switch each, a mode is a set of names, and a model's set
is active only inside its own build and fold. Host only: no device is opened."""
from __future__ import annotations

import pytest

import tt_bio.tenstorrent as T


def test_fast_set_is_every_lever_but_the_grid_dependent_one():
    assert T.FAST_LEVERS == frozenset(T.LEVERS) - {"triatt_b8"}
    assert T.NORMAL_LEVERS <= T.FAST_LEVERS


def test_parse_expands_modes_and_rejects_unknown_names():
    assert T.parse_levers("fast") == T.FAST_LEVERS
    assert T.parse_levers("normal,opm_b8") == T.NORMAL_LEVERS | {"opm_b8"}
    assert T.parse_levers("") == T.parse_levers("none") == frozenset()
    with pytest.raises(ValueError, match="unknown precision lever"):
        T.parse_levers("lofi,bfp2")


def test_parse_adds_and_drops_after_the_first_term():
    assert T.parse_levers("fast-lofi-acc_off") == T.FAST_LEVERS - {"lofi", "acc_off"}
    assert T.parse_levers("normal+opm_b8+lofi") == T.NORMAL_LEVERS | {"opm_b8", "lofi"}
    assert T.parse_levers("fast+triatt_b8") == frozenset(T.LEVERS)


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
