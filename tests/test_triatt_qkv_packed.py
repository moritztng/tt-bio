"""`autograd._packed_qkv_source`: triatt_bw writes the packed qkv gradient only when q, k and v are
slots 0/1/2 of one `nlp_create_qkv_heads` parent whose layout matches, and declines otherwise."""
import types

import pytest

ttnn = pytest.importorskip("ttnn")
from tt_bio import autograd as ag  # noqa: E402


def _v(shape, dtype=None, layout=None):
    return types.SimpleNamespace(shape=shape, dtype=dtype or ttnn.bfloat16,
                                 layout=layout or ttnn.TILE_LAYOUT)


def _slot(x, s, shape=(288, 4, 288, 32)):
    def fn(g):
        pass
    fn.qkv_slot = s
    return types.SimpleNamespace(node=types.SimpleNamespace(fn=fn, parents=[x]), value=_v(shape))


def _x(shape=(288, 1, 288, 384), **kw):
    return types.SimpleNamespace(requires_grad=True, value=_v(shape, **kw))


def test_three_slots_of_one_split_route_to_the_parent():
    x = _x()
    assert ag._packed_qkv_source(*(_slot(x, s) for s in range(3))) is x


@pytest.mark.parametrize("case", ["order", "two_parents", "no_grad", "untaped", "head_dim",
                                  "ragged", "dtype", "row_major"])
def test_anything_else_declines(case):
    x, y = _x(), _x()
    q, k, v = (_slot(x, s) for s in range(3))
    if case == "order":
        q, k = k, q
    elif case == "two_parents":
        v = _slot(y, 2)
    elif case == "no_grad":
        x.requires_grad = False
    elif case == "untaped":
        k.node = None
    elif case == "head_dim":
        x2 = _x((288, 1, 288, 768))
        q, k, v = (_slot(x2, s, (288, 4, 288, 64)) for s in range(3))
    elif case == "ragged":
        x2 = _x((290, 1, 290, 384))
        q, k, v = (_slot(x2, s, (290, 4, 290, 32)) for s in range(3))
    elif case == "dtype":
        x.value = _v((288, 1, 288, 384), dtype=ttnn.float32)
    elif case == "row_major":
        x.value = _v((288, 1, 288, 384), layout=ttnn.ROW_MAJOR_LAYOUT)
    assert ag._packed_qkv_source(q, k, v) is None
