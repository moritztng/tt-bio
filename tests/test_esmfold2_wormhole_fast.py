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
    monkeypatch.setattr(rt, "patch_esmfold2", lambda model, **k: model)
    monkeypatch.setattr(tenstorrent, "_FAST_MODE", False)
    rt.load_ttnn_esmfold2(fast=fast)
    assert tenstorrent._FAST_MODE is want
