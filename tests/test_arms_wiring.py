"""Every arm name in `perf/bcx_bytes/bytes.py` flips the engine flag it claims to.

Card-free by construction: `Arms.set()` only patches module attributes, so the wiring can be
checked before a card is spent discovering that an arm name is a no-op. That is worth a test
rather than a run: this fleet's most repeated failure is a lever that is wired and inert, and an
arm whose name matches nothing fails SILENTLY -- `set()` parses `name.split("+")` and an
unrecognised token simply selects no lever, so the arm runs as `base` and reports a 1.00x that
looks like a measurement.

`smbf16+moreh` is the combination the row's headline rests on: `moreh_softmax_backward` refuses
fp32 and admits bfloat16, so only that pair can execute the fused route at all.
"""
import sys

import pytest

sys.path.insert(0, ".")

BIG = 1 << 30
CASES = {
    "base":         dict(perm=False, tree=BIG, dt="keep", route="chain", widen=True),
    "perm":         dict(perm=True,  tree=BIG, dt="keep", route="chain", widen=True),
    "tree":         dict(perm=False, tree=256, dt="keep", route="chain", widen=True),
    "smbf16":       dict(perm=False, tree=BIG, dt="bf16", route="chain", widen=True),
    "fanin":        dict(perm=False, tree=BIG, dt="keep", route="chain", widen=False),
    "moreh":        dict(perm=False, tree=BIG, dt="keep", route="moreh", widen=True),
    "smbf16+moreh": dict(perm=False, tree=BIG, dt="bf16", route="moreh", widen=True),
    "perm+tree":    dict(perm=True,  tree=256, dt="keep", route="chain", widen=True),
}


@pytest.fixture(scope="module")
def arms():
    from perf.bcx_bytes.bytes import Arms
    return Arms()


def _state():
    from tt_bio import autograd as ag, taped_ttnn as T
    return dict(perm=T.PERMUTE_BW_REBLOCK, tree=ag.LEADING_SUM_TREE_ROWS,
                dt=ag.SOFTMAX_BW_DTYPE, route=ag.SOFTMAX_BW_ROUTE,
                widen=ag.FANIN_WIDEN_INCOMING)


@pytest.mark.parametrize("name", list(CASES))
def test_arm_sets_exactly_its_levers(arms, name):
    arms.set(name)
    assert _state() == CASES[name]


def test_an_unknown_arm_name_is_indistinguishable_from_base(arms):
    """The control, and it is the reason this file exists rather than a docstring.

    `set()` selects levers by membership, so a typo selects none and the arm runs as `base`
    while reporting under its own name. Nothing raises. Pinning it here means a future arm
    added to the CLI and not to `set()` shows up as this test failing on the name, not as a
    1.00x in an A/B six hours later.
    """
    arms.set("base")
    base = _state()
    arms.set("smbf17")                      # one character off
    assert _state() == base
