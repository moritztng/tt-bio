"""ESMFold2 is fast-only on Wormhole for every loader caller, not just the CLI.

The bf16 ESMC-6B (~12.8 GB) does not fit a Wormhole chip's 12 GiB. main.py turned --fast on for
a CLI user, but the worker path (perf_regression, a server) loaded normal precision and died in
the allocator with TT_FATAL on the 0.15.0 gate's Wormhole perf leg.
"""
import sys
import types

import pytest

pytest.importorskip("ttnn")


@pytest.mark.parametrize("wormhole,fast,want", [(True, False, True), (False, False, False),
                                                 (False, True, True)])
def test_load_runs_fast_on_wormhole(monkeypatch, wormhole, fast, want):
    from tt_bio import esmfold2_runtime as rt
    from tt_bio import tenstorrent

    class Model:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def eval(self):
            return self

    mod = types.ModuleType("modeling_esmfold2")
    mod.ESMFold2Model = Model
    monkeypatch.setitem(sys.modules, "tt_bio._vendor.esmfold2_hf.modeling_esmfold2", mod)
    monkeypatch.setattr(rt, "is_wormhole", lambda: wormhole)
    built = {}
    monkeypatch.setattr(rt, "patch_esmfold2", lambda model, **k: built.update(k) or model)
    monkeypatch.setattr(tenstorrent, "_FAST_MODE", False)
    rt.load_ttnn_esmfold2(fast=fast)
    # The model is built fast, on both archs under LEVERS_FAST with the older `_FAST_MODE` off
    # (esmfold2_runtime.fold_scope).
    assert built["fast"] is want
    assert tenstorrent._FAST_MODE is False


@pytest.mark.parametrize("wormhole", [True, False])
def test_normal_set_takes_hifi2_only_where_graded(monkeypatch, wormhole):
    # trimul_gin_hifi2 passed ESMFold2's normal bar on Blackhole; Wormhole was never graded on it.
    from tt_bio import esmfold2_runtime as rt
    monkeypatch.delenv("TT_BIO_LEVERS", raising=False)
    monkeypatch.setattr(rt, "is_wormhole", lambda: wormhole)
    assert ("trimul_gin_hifi2" in rt.fold_levers(False).split("+")) is not wormhole
    # ws_resid passed ESMFold2's fast bar on Blackhole only, so Wormhole's fast set drops it.
    assert ("ws_resid" in rt.fold_levers(True).split("+")) is not wormhole
    # The ef2_* levers ride both sets on both archs, except device pair init on Wormhole (out of DRAM at 1024 tokens).
    for fast in (False, True):
        names = rt.fold_levers(fast).split("+")
        assert "ef2_relpos_lookup" in names and "ef2_bit_draw" in names
        assert ("ef2_pair_init" in names) is not wormhole
    assert "ef2_opm_kq" not in rt.fold_levers(False).split("+")
