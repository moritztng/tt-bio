"""The row-blocked pair FFN must use the fc1 layout its module actually built.

`_row_blocked` calls `_ffn(part, split=True, ...)` for every module. Only a module constructed
with `fuse_swiglu=True` builds the split halves `fc1_a_weight`/`fc1_b_weight`; ESMFold2's msa and
pair transitions (`esmfold2.py:1252`, `:1255`) and ESMC's LM block FFN build the single
`fc1_weight`. Nothing reached the split branch on those modules until the DRAM-refusal fallback
started routing them into `_row_blocked`, and then esmfold2 died on

    AttributeError: 'SwiGLUFFN' object has no attribute 'fc1_a_weight'
        [tt_bio origin: esmc.py:642 in _ffn]

at 1300, 1408 and 1536 tokens, verbatim from `perf/bh1536/runs/esmfold2_1300/fold.log`. The
1536 run reached it AFTER the fallback had already caught the 9.00 GiB refusal, so the failure
recorded there as a third OOM was this crash, not a wall.

Host-only. ttnn is stubbed, so what runs is the branch choice in `_ffn` and nothing else.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio import esmc as E  # noqa: E402


class _T:
    """Enough of a tensor for the stubs to hand around."""

    def __init__(self, name):
        self.name = name
        self.shape = (1, 32, 1536, 256)
        self.padded_shape = self.shape


def _stub_ttnn(monkeypatch):
    """Every ttnn call `_ffn` can make, recorded rather than executed."""
    ns = types.SimpleNamespace(
        layer_norm=lambda x, **kw: _T("norm"),
        chunk=lambda h, n, dim: [_T("x1"), _T("x2")],
        silu=lambda x: _T("silu"),
        multiply=lambda a, b, **kw: _T("gated"),
        deallocate=lambda t: None,
        L1_MEMORY_CONFIG=object(),
        UnaryOpType=types.SimpleNamespace(SILU=object()),
        bfloat16=object(),
        experimental=types.SimpleNamespace(minimal_matmul=None),
    )
    monkeypatch.setattr(E, "ttnn", ns)
    return ns


class _Unsplit:
    """A SwiGLUFFN built the way esmfold2's pair transition is: one fc1 weight, no halves."""

    _ffn = E.SwiGLUFFN._ffn
    _fc1_full = E.SwiGLUFFN._fc1_full

    def __init__(self):
        self.split_swiglu = False
        self.fuse_swiglu = False
        self.fc1_weight = _T("fc1")
        self.fc2_weight = _T("fc2")
        self.norm_weight = _T("ln_w")
        self.norm_bias = _T("ln_b")
        self.compute_kernel_config = None
        self.lins: list[str] = []

    def _lin(self, x, w, **kw):
        self.lins.append(w.name)
        return _T("lin")


def test_a_row_blocked_unsplit_module_does_not_reach_for_the_split_halves(monkeypatch):
    _stub_ttnn(monkeypatch)
    ffn = _Unsplit()
    out = ffn._ffn(_T("block"), split=True, l1_gated=True)
    assert out.name == "lin"
    assert ffn.lins == ["fc1", "fc2"], (
        f"the unsplit fc1 weight is what this module holds; it used {ffn.lins}")


def test_the_negative_control_is_the_recorded_crash(monkeypatch):
    """Without the guard, the same call raises the AttributeError from the 1300 log. Asserting
    the attribute is genuinely absent is what makes the test above non-vacuous."""
    _stub_ttnn(monkeypatch)
    ffn = _Unsplit()
    assert not hasattr(ffn, "fc1_a_weight")
    with pytest.raises(AttributeError, match="fc1_a_weight"):
        _ = ffn.fc1_a_weight


def test_a_split_module_still_takes_the_split_path(monkeypatch):
    """The guard must not turn the trunk transition's measured path off."""
    _stub_ttnn(monkeypatch)
    ffn = _Unsplit()
    ffn.split_swiglu = True
    ffn.fc1_weight = None
    ffn.fc1_a_weight = _T("fc1_a")
    ffn.fc1_b_weight = _T("fc1_b")
    monkeypatch.setattr(E, "_PAIR_FFN_L1_FC1", False)   # the plain split arm, no L1 levers
    ffn._ffn(_T("block"), split=True, l1_gated=True)
    assert ffn.lins == ["fc1_a", "fc1_b", "fc2"], ffn.lins
