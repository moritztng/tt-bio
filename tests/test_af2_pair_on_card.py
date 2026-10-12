"""The precondition behind `TT_BIO_AF2_PAIR_ON_CARD`, checked without a device.

The optimisation is a deletion, not a change of arithmetic: `AF2DeviceModel.extra_msa_stack`
downloads the pair and `evoformer_stack` uploads the same bytes back, and between the two calls
`AF2Model.forward` does not read or rewrite the pair at all. That is the whole claim, so it is
what this file pins -- in the source, where a later edit would break it silently. The numbers it
buys are a device question and live in `perf/spd/af2ig_census.py`'s A/B.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from tt_bio import af2, af2_reference


def _forward_body() -> list[ast.stmt]:
    src = inspect.getsource(af2_reference.AF2Model.forward)
    return ast.parse(textwrap.dedent(src)).body[0].body


def _call_index(body: list[ast.stmt], attr: str) -> int:
    for i, node in enumerate(body):
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == attr):
                return i
    raise AssertionError(f"forward never calls self.{attr}")


def test_forward_leaves_the_pair_alone_across_the_seam():
    """Nothing between the two stack calls reads or rebinds `pair`."""
    body = _forward_body()
    start = _call_index(body, "extra_msa_stack")
    end = _call_index(body, "evoformer_stack")
    assert start < end
    for node in body[start + 1:end]:
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        assert "pair" not in names, (
            f"{ast.unparse(node)!r} sits between the extra-MSA stack and the Evoformer stack "
            "and touches the pair, so it cannot stay on card across the seam")


def test_the_pair_is_the_argument_each_stack_hands_the_next():
    """The handoff is one variable: the extra-MSA stack's result is the Evoformer's `pair`."""
    body = _forward_body()
    assign = body[_call_index(body, "extra_msa_stack")]
    assert isinstance(assign, ast.Assign) and len(assign.targets) == 1
    assert isinstance(assign.targets[0], ast.Name) and assign.targets[0].id == "pair"
    call = next(n for n in ast.walk(body[_call_index(body, "evoformer_stack")])
                if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "evoformer_stack")
    assert [a.id for a in call.args if isinstance(a, ast.Name)][1] == "pair"


def test_default_is_off_and_every_instrument_declines_the_handoff():
    model = af2.AF2DeviceModel.__new__(af2.AF2DeviceModel)
    assert af2.AF2DeviceModel.pair_on_card is False, "the gate ships off"
    for name, value in (("pair_on_card", True),):
        setattr(model, name, value)
    for name, value in (("extra_msa_host", False), ("block_tap", None),
                        ("substitute", frozenset()), ("skip", frozenset()),
                        ("trace_stacks", False)):
        setattr(model, name, value)
    assert model._pair_handoff() is True
    for name, value in (("extra_msa_host", True), ("block_tap", lambda *a: None),
                        ("substitute", frozenset({"opm"})), ("skip", frozenset({"opm"})),
                        ("trace_stacks", True)):
        before = getattr(model, name)
        setattr(model, name, value)
        assert model._pair_handoff() is False, f"{name} must keep the host download"
        setattr(model, name, before)
    model.pair_on_card = False
    assert model._pair_handoff() is False
