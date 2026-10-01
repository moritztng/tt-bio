"""`autograd.FANIN_CAST_FUSED`: a bf16 value's last widened contribution is held back and added
with a bf16 output when the closure reads it; every other read flushes it through `widen_add`."""
import types

import pytest

ttnn = pytest.importorskip("ttnn")
from tt_bio import autograd as ag  # noqa: E402


def _t(name, dtype=None):
    return types.SimpleNamespace(name=name, shape=(1, 32, 32, 32), layout=ttnn.TILE_LAYOUT,
                                 dtype=dtype or ttnn.bfloat16)


@pytest.fixture
def calls(monkeypatch):
    log = []
    monkeypatch.setattr(ag._rne_add, "widen_eligible", lambda a, b: True)
    monkeypatch.setattr(ag._rne_add, "widen_add",
                        lambda a, b: log.append(("widen", a.name, b.name)) or _t(f"({a.name}+{b.name})", ttnn.float32))
    monkeypatch.setattr(ag._rne_add, "round_add",
                        lambda a, b: log.append(("round", a.name, b.name)) or _t(f"[{a.name}+{b.name}]"))
    monkeypatch.setattr(ag.ttnn, "typecast", lambda g, dt: log.append(("cast", g.name)) or _t(g.name, dt))
    return log


def _three(on, monkeypatch):
    monkeypatch.setattr(ag, "FANIN_CAST_FUSED", on)
    x = ag.Tensor(_t("x"), requires_grad=True)
    for n in "abc":
        x.add_grad(_t(n))
    return x


def test_on_the_closure_read_rounds_the_last_add_and_casts_nothing(calls, monkeypatch):
    g = _three(True, monkeypatch).closure_grad()
    assert calls == [("widen", "a", "b"), ("round", "(a+b)", "c")]
    assert g.name == "[(a+b)+c]" and g.dtype == ttnn.bfloat16


def test_off_is_widen_widen_cast(calls, monkeypatch):
    g = _three(False, monkeypatch).closure_grad()
    assert calls == [("widen", "a", "b"), ("widen", "(a+b)", "c"), ("cast", "((a+b)+c)")]
    assert g.dtype == ttnn.bfloat16


def test_any_other_read_flushes_in_float32(calls, monkeypatch):
    g = _three(True, monkeypatch).grad
    assert calls == [("widen", "a", "b"), ("widen", "(a+b)", "c")]
    assert g.dtype == ttnn.float32


def test_a_single_contribution_is_handed_over_as_is(calls, monkeypatch):
    monkeypatch.setattr(ag, "FANIN_CAST_FUSED", True)
    x = ag.Tensor(_t("x"), requires_grad=True)
    x.add_grad(_t("a"))
    assert x.closure_grad().name == "a" and calls == []


def test_a_float32_value_is_not_held(calls, monkeypatch):
    monkeypatch.setattr(ag, "FANIN_CAST_FUSED", True)
    x = ag.Tensor(_t("x", ttnn.float32), requires_grad=True)
    x.add_grad(_t("a"))
    x.add_grad(_t("b"))
    assert x._pend is None and calls == [("widen", "a", "b")]
