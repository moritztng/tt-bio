"""The ABodyBuilder3 gradient glue against the tape it now sits on, without a card.

`tt_bio/train/abodybuilder3_grad.py` was written against a 1026-line `tt_bio/autograd.py`. The
Protenix-v2 tape took that file to 1517 lines and git merged the two without a conflict marker,
which proves nothing: the glue builds `ag.Tensor` and `ag._Node` by hand and replays the tape with
its own multi-root `backward`, so it depends on the tape's internals and not only on its names.

Everything here runs on plain objects. The op rules themselves call `ttnn` in their backwards and
need a card (`scripts/abb3_port/op_gradcheck.py` is that layer); the graph machinery underneath
them does not, and it is the half the merge could have broken.
"""

import pytest

pytest.importorskip("ttnn")

from tt_bio import autograd as ag                          # noqa: E402
from tt_bio.train import abodybuilder3_grad as grad        # noqa: E402


class _Val:
    """Stands in for a ttnn tensor. Answers nothing, on purpose: a rule that reached for a
    device method would fail here rather than pass by accident."""


def _leaf(requires_grad=True):
    return ag.Tensor(_Val(), requires_grad=requires_grad)


def test_the_glue_reads_the_tape_it_is_merged_onto():
    """Every name the glue takes off `autograd` still exists and still means what it used to."""
    assert grad.Tensor is ag.Tensor
    assert grad._Node is ag._Node, "two node classes for one tape is the thing this is checked for"
    assert ag.is_grad_enabled() is True
    with ag.no_grad():
        assert grad._grad_enabled() is False
    assert grad._grad_enabled() is True


def test_a_node_built_by_the_glue_pins_its_values():
    """`ag.Tensor.free` releases a buffer unless it is pinned, and the glue builds its nodes by
    hand. Unpinned, a free would drop a value the backward still reads."""
    x, w = _leaf(), _leaf()
    out = grad._tape(_Val(), [x, w, None, "not a tensor"], lambda: (lambda g: None))
    assert out.pinned and x.pinned and w.pinned
    assert out.node.parents == [x, w], "raw and None operands must not become tape parents"


def test_the_glues_backward_fires_each_closure_once_after_all_its_gradients_land():
    """The fan-in the multi-root replay exists for: a trunk read by two heads must be visited
    once, and only after both heads have contributed."""
    fired = []
    trunk = _leaf()
    trunk.node = ag._Node(lambda g: fired.append(("trunk", g)), [])
    trunk.pinned = True

    heads = []
    for name in ("a", "b"):
        h = ag.Tensor(_Val(), requires_grad=True)
        h.node = ag._Node(
            (lambda n: lambda g: (fired.append((n, g)),
                                  setattr(trunk, "grad", g if trunk.grad is None
                                          else trunk.grad + g)))(name),
            [trunk])
        heads.append(h)

    grad.backward(heads, [1.0, 10.0])

    assert sorted(n for n, _ in fired[:2]) == ["a", "b"], "a head did not fire, or fired twice"
    assert [n for n, _ in fired[2:]] == ["trunk"], (
        "the shared trunk fired before both heads had contributed, or fired once per head, so "
        "its gradient is partial or double-counted")
    assert fired[-1][1] == 11.0, "the fan-in was not summed"


def test_no_grad_keeps_the_hook_off_the_tape():
    """Under `no_grad` the hook must hand back the shipped value rather than tape it, which is
    what makes a frozen block free."""
    calls = []

    def shipped(v, factor):
        calls.append((v, factor))
        return _Val()

    x = _leaf()
    with ag.no_grad():
        out = grad._hook("scale", shipped, (x, 2.0), {})
    assert isinstance(out, ag.Tensor) and out.node is None
    assert calls == [(x.value, 2.0)], "the hook did not unwrap the taped operand for the shipped call"


def test_an_op_with_no_backward_refuses_instead_of_dropping_the_gradient():
    x = _leaf()
    with pytest.raises(NotImplementedError, match="has no backward"):
        grad._hook("an_op_nobody_taped", lambda v: _Val(), (x,), {})


def test_an_operand_nobody_is_differentiating_falls_through_to_production():
    x = ag.Tensor(_Val(), requires_grad=False)
    assert grad._hook("scale", lambda v, f: _Val(), (x, 2.0), {}) is not None
    assert grad._hook("scale", lambda v, f: _Val(), (_Val(), 2.0), {}) is None
