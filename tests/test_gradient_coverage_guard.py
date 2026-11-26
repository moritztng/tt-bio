"""The every-parameter-trains gate (R4T charter G2), card-free.

The gate must pass a healthy model whose output layer starts at zero (dead for step 1 only, the
transient a one-step check misreads), and must FAIL the same model with one parameter frozen or
detached. A gate never seen failing is not evidence.
"""
import numpy as np
import pytest
import torch

from tt_bio.train.checks import (COVERAGE_MIN_STEPS, GradientCoverage, dead_parameters)


class _Net(torch.nn.Module):
    """Two residual-free linears with the output layer zero-initialised, upstream's `final` init."""

    def __init__(self, detach_first=False):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.inp = torch.nn.Linear(8, 16)
        self.out = torch.nn.Linear(16, 4)
        with torch.no_grad():
            self.inp.weight.copy_(torch.randn(16, 8, generator=g) * 0.3)
            self.out.weight.zero_()
            self.out.bias.zero_()
        self.detach_first = detach_first

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        return self.out(h.detach() if self.detach_first else h)


def _run(net, steps, cov, frozen=()):
    for name, p in net.named_parameters():
        p.requires_grad_(name not in frozen)
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1)
    g = torch.Generator().manual_seed(1)
    names = [n for n, _ in net.named_parameters()]
    first = None
    for _ in range(steps):
        x, y = torch.randn(32, 8, generator=g), torch.randn(32, 4, generator=g)
        opt.zero_grad()
        ((net(x) - y) ** 2).mean().backward()
        grads = {n: (None if p.grad is None else p.grad.numpy().copy()) for n, p in net.named_parameters()}
        dead = dead_parameters(grads)
        first = dead if first is None else first
        cov.require(dead, len(names))
        opt.step()
    return first


def test_zero_init_output_is_a_transient_not_a_failure():
    first = _run(_Net(), COVERAGE_MIN_STEPS, GradientCoverage())
    # Step 1 really does read the input layer dead: the gate passes only because it intersects.
    assert first == {"inp.weight", "inp.bias"}


@pytest.mark.parametrize("arm", ["frozen", "detached"])
def test_gate_fails_on_one_dead_parameter(arm):
    net = _Net(detach_first=arm == "detached")
    frozen = ("inp.bias",) if arm == "frozen" else ()
    with pytest.raises(RuntimeError, match=r"received no gradient on any of the first 5 steps"):
        _run(net, COVERAGE_MIN_STEPS, GradientCoverage(), frozen=frozen)


def test_exemption_needs_a_reason_and_then_holds():
    with pytest.raises(ValueError, match="need a reason"):
        GradientCoverage(exempt={"inp.bias": ""})
    _run(_Net(), COVERAGE_MIN_STEPS, GradientCoverage(exempt={"inp.bias": "test: frozen on purpose"}),
         frozen=("inp.bias",))


def test_window_below_the_floor_is_refused():
    with pytest.raises(ValueError, match="floor"):
        GradientCoverage(steps=COVERAGE_MIN_STEPS - 1)


def test_decides_once_on_the_closing_step():
    cov = GradientCoverage()
    outs = [cov.observe({"a"} if k == 0 else set()) for k in range(COVERAGE_MIN_STEPS + 2)]
    assert outs[:COVERAGE_MIN_STEPS - 1] == [None] * (COVERAGE_MIN_STEPS - 1)
    assert outs[COVERAGE_MIN_STEPS - 1] == frozenset()
    assert outs[COVERAGE_MIN_STEPS:] == [None, None] and not cov.open
    assert dead_parameters({"z": np.zeros(3), "m": None, "ok": np.ones(1)}) == {"z", "m"}
