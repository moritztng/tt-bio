"""Weight discovery by walking the built model, and the three ways it used to miss.

A census hooked at `Module.torch_to_tt` finds what the loader produced and nothing a module
derived for itself, so `TriangleMultiplication`'s fused in-projection -- 232 of OpenFold3's
tensors, at every triangle multiplication in the trunk, the MSA stack and the template stack --
was a constant to the tape. The backward completed, the input got a gradient, and no parameter
did. These tests are that defect in miniature, host-only: a `ttnn.Tensor` with no device on it
is still a `ttnn.Tensor`, which is all the walk tests against.

The third miss is an ORDERING one and is the reason this file exists rather than a one-line
change: the fused weights do not exist after `__init__` either. They are pushed on the first
call at a given chunk width, so a walk of a freshly built model misses exactly the tensors that
motivated the walk, and it misses them silently -- the paths are simply absent from the total.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

ttnn = pytest.importorskip("ttnn")

from tt_bio import autograd as ag                                      # noqa: E402
from tt_bio import ops                                                 # noqa: E402
from tt_bio.tenstorrent import device_weights, walk_device_weights     # noqa: E402
from tt_bio.train.checks import weight_coverage                        # noqa: E402
from tt_bio.train.lora import Parameters, weights_for                  # noqa: E402


def _host(n=2):
    """A `ttnn.Tensor` that never touched a card. Identity is all the walk reads."""
    return ttnn.from_torch(torch.zeros(n, n), dtype=ttnn.bfloat16)


class _Fused:
    """A module in the shape that broke it: one loaded weight, one fused, one lazy.

    `loaded` stands for `torch_to_tt`'s output. `fused` stands for the host `torch.cat` a
    triangle multiplication pushes in `__init__`. `cache` stands for `_gp_cache`, which is not
    filled until the first call.
    """

    def __init__(self):
        self.loaded = _host()
        self.fused = _host()
        self.cache = {}
        self.torch_weight = torch.zeros(2, 2)      # host torch: not a device weight

    def __call__(self):
        self.cache.setdefault((32, 1), [_host(), _host()])
        return self.cache[(32, 1)]


class _Stack:
    def __init__(self, n=3):
        self.blocks = [_Fused() for _ in range(n)]
        self.numpy = pytest.importorskip("numpy")     # the `self.np = numpy` hazard

    def __call__(self):
        for b in self.blocks:
            b()


def test_walk_reaches_a_weight_the_loader_never_produced():
    """The K29 instance. A loader hook sees `loaded`; the walk sees `fused` as well."""
    m = _Fused()
    found = device_weights(m)
    assert set(found) == {"loaded", "fused"}, found
    assert found["fused"] is m.fused


def test_a_walk_before_the_forward_misses_the_lazily_fused_weights():
    """The ordering finding, and it is the one a denominator alone cannot show.

    Before the call the cache is empty, so the missing weights are missing from the TOTAL as
    well as from the leaves -- coverage reads 2 of 2 and looks perfect.
    """
    m = _Fused()
    before = device_weights(m)
    m()
    after = device_weights(m)
    assert len(before) == 2, before
    assert len(after) == 4, after
    assert set(after) - set(before) == {"cache.(32, 1).0", "cache.(32, 1).1"}, sorted(after)


def test_the_walk_does_not_descend_into_a_module_object():
    """`vars()` of a module is its whole namespace, so one `self.np = numpy` walks the world."""
    found = device_weights(_Stack())
    assert len(found) == 6, sorted(found)
    assert all(n.startswith("blocks.") for n in found), sorted(found)


def test_every_walked_weight_becomes_a_leaf_and_coverage_says_so():
    m = _Stack()
    m()
    params = Parameters()
    for path, owner, key, t in walk_device_weights(m):
        params[path] = ag.parameter(t)
        params.slots[path] = (owner, key)
    try:
        cov = weight_coverage(m)
        assert cov.total == 12, str(cov)
        assert cov.registered == 12, str(cov)
        assert not cov.unregistered, str(cov)
        # Nothing ran a backward, so none of them carries a gradient yet. Reported, not hidden:
        # registered and with_grad are different questions and R22 is the two being conflated.
        assert cov.with_grad == 0 and len(cov.without_grad) == 12, str(cov)
    finally:
        ag.forget_parameters()


def test_negative_control_one_unregistered_weight_fails_on_exactly_that_weight():
    """PROTOCOL SS3e. A leaf-count check nobody has watched fail is not a check.

    One weight is deliberately left out of the registration. The check must name it, and name
    nothing else -- a check that fails everywhere on one fault is as useless as one that never
    fails.
    """
    m = _Stack()
    m()
    walked = list(walk_device_weights(m))
    skipped = "blocks.1.fused"
    assert any(p == skipped for p, _, _, _ in walked), sorted(p for p, *_ in walked)
    try:
        for path, _, _, t in walked:
            if path != skipped:
                ag.parameter(t)
        cov = weight_coverage(m)
        assert cov.total == 12, str(cov)
        assert cov.registered == 11, str(cov)
        assert cov.unregistered == [skipped], str(cov)
        assert not cov.ok
    finally:
        ag.forget_parameters()


def test_rebind_puts_the_optimizer_s_new_tensor_back_where_the_walk_found_it():
    """Without this the forward reads the checkpoint's weights for the whole run.

    `AdamW.step` replaces the leaf's value rather than writing into it, so the model's own
    attribute still holds the handle discovery saw. Gradients stay real, the loss curve falls
    and the model stands still.
    """
    m = _Stack()
    m()
    params = Parameters()
    for path, owner, key, t in walk_device_weights(m):
        params[path] = ag.parameter(t)
        params.slots[path] = (owner, key)
    try:
        stepped = _host()
        params["blocks.0.fused"].value = stepped
        assert m.blocks[0].fused is not stepped     # the model is still on the old handle
        assert params.rebind() == 1
        assert m.blocks[0].fused is stepped
        assert params.rebind() == 0                 # idempotent; nothing left to move
        # And a weight held in a cache list is written back through its container.
        newer = _host()
        params["blocks.2.cache.(32, 1).1"].value = newer
        assert params.rebind() == 1
        assert m.blocks[2].cache[(32, 1)][1] is newer
    finally:
        ag.forget_parameters()


def test_rebind_refuses_a_weight_held_in_a_tuple_instead_of_skipping_it():
    """An unwritable slot is a forward that will read a stale weight. Loud, not silent."""
    class _Tupled:
        def __init__(self):
            self.pair = (_host(), _host())

    m = _Tupled()
    params = Parameters()
    try:
        for path, owner, key, t in walk_device_weights(m):
            params[path] = ag.parameter(t)
            params.slots[path] = (owner, key)
        params["pair.0"].value = _host()
        with pytest.raises(TypeError, match="tuple"):
            params.rebind()
    finally:
        ag.forget_parameters()


def test_a_census_over_an_autograd_tensor_input_completes_instead_of_raising():
    """The second defect: the census hook did not compose with the shipped layer norm.

    With no hook installed under it the census declined, the dispatch fell through to
    production, and `ttnn.layer_norm` got an `autograd.Tensor` where a raw one is required:
    `TypeError: ttnn.layer_norm(): incompatible function arguments`. `weights_for` reports "no
    adaptable site" for a census that raised as readily as for one that found nothing, which is
    how 0 of 8 weights read as a clean result.
    """
    seen = []

    def shipped_linear(x, w, bias=None, **kw):
        seen.append(type(x).__name__)
        assert not isinstance(x, ag.Tensor), "production got a taped operand"
        assert not isinstance(w, ag.Tensor), "production got a taped weight"
        return x

    def forward(x):
        # Two routed calls, the second reading the first's output, so a wrapped return value
        # would surface on the next call rather than only on this one.
        y = ops.linear(x, _host(), compute_kernel_config=None)
        return ops.linear(y, _host(), compute_kernel_config=None)

    prev_lin = ops.linear
    ops.linear = lambda *a, **k: (ops.grad_hook()(
        "linear", shipped_linear, a, k) if ops.grad_hook() else shipped_linear(*a, **k))
    try:
        params = weights_for(forward, None, ag.Tensor(_host(), requires_grad=True))
    finally:
        ops.linear = prev_lin
        ag.forget_parameters()
    assert len(params) == 2, sorted(params)
    assert seen == ["Tensor", "Tensor"], seen
