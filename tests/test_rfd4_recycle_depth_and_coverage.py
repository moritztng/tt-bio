"""A step that draws one recycle trains neither RFD4 cycle projection, and the guard must know.

Upstream's `_run_trunk` (`nn/local_latents_transformer.py:396-414`, pinned 2db07bb4) starts the
recycle state at `torch.zeros_like`, feeds `z_init + z_cycle_proj(z_cycle_norm(z))` into every
cycle, and keeps the gradient for the last cycle only (`recycle_grad: last`). Both cycle
projections are zero-initialised (`:164,167`).

So at `n_recycle = 1` the single taped cycle reads the zero state: the projections' gradient is
`x^T g` with `x = norm(0) = 0`, and the norms behind those zero projections get `W^T g = 0`.
Measured below: all four are dead at depth 1, and the two norms are still dead at depth 2 and 3
from a fresh init, because they only start training once an update has moved the projection
in front of them.

Training draws the depth per step, Poisson(3) clamped to [1, 9] (`hero_pretrain.yaml:449-454`),
so p(depth = 1) = 0.20. Five consecutive ones is 3.2e-4, and fewer than two depth >= 2 steps in
five is 6.7e-3 -- about one launch in 150 would be refused for a model that is training
normally. **An RFD4 run must therefore count a step toward the guard only when its drawn depth
is >= 2**, which costs nothing: the loop owns `observe()` and simply does not call it on a
depth-1 step.

The wiring is transcribed rather than imported because it lives inside
`LocalLatentsTransformer.__init__`, which cannot be built without the whole feature stack;
`tests/rfd4_reference.py` is the same arrangement. Shapes are shrunk; the result is structural.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

TOKEN, PAIR, N = 32, 16, 6
SEED_BASE, LAM, MAX_DEPTH = 0x5EED0000, 3.0, 9
CYCLE_PARAMS = {"s_cycle_norm.weight", "s_cycle_proj.weight",
                "z_cycle_norm.weight", "z_cycle_proj.weight"}


class _Recycled(torch.nn.Module):
    """Upstream's recycle wiring: zero state, zero-initialised projections, last-cycle grad."""

    def __init__(self):
        super().__init__()
        self.s_cycle_norm = torch.nn.RMSNorm(TOKEN, eps=1e-5)
        self.s_cycle_proj = torch.nn.Linear(TOKEN, TOKEN, bias=False)
        torch.nn.init.zeros_(self.s_cycle_proj.weight)
        self.z_cycle_norm = torch.nn.RMSNorm(PAIR, eps=1e-5)
        self.z_cycle_proj = torch.nn.Linear(PAIR, PAIR, bias=False)
        torch.nn.init.zeros_(self.z_cycle_proj.weight)
        self.trunk_s = torch.nn.Linear(TOKEN, TOKEN)
        self.trunk_z = torch.nn.Linear(PAIR, PAIR)

    def forward(self, s_init, z_init, n_recycle, grad_k=1):
        s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)
        for cycle in range(n_recycle):
            z_in = z_init + self.z_cycle_proj(self.z_cycle_norm(z))
            s_in = s_init + self.s_cycle_proj(self.s_cycle_norm(s))
            grad_on = torch.is_grad_enabled() and cycle >= n_recycle - grad_k
            with torch.set_grad_enabled(grad_on):
                s, z = self.trunk_s(s_in), self.trunk_z(z_in)
            if not grad_on:
                s, z = s.detach(), z.detach()
        return s, z


def _dead_at(depth: int) -> set:
    torch.manual_seed(0)
    m = _Recycled()
    s, z = m(torch.randn(1, N, TOKEN), torch.randn(1, N, N, PAIR), depth)
    (s.square().mean() + z.square().mean()).backward()
    return {n for n, p in m.named_parameters() if p.grad is None or not torch.any(p.grad)}


def _depths(n: int) -> list:
    """Upstream's own per-step draw, seeded `0x5EED0000 + global_step`."""
    return [int(np.clip(np.random.default_rng(SEED_BASE + s).poisson(LAM), 1, MAX_DEPTH))
            for s in range(n)]


def test_depth_one_trains_neither_cycle_projection_nor_its_norm():
    assert _dead_at(1) == CYCLE_PARAMS


def test_deeper_steps_train_the_projections_but_not_yet_the_norms():
    """From a fresh init: the norm in front of a zero projection waits for one update."""
    for depth in (2, 3):
        dead = _dead_at(depth)
        assert "s_cycle_proj.weight" not in dead and "z_cycle_proj.weight" not in dead
        assert dead == {"s_cycle_norm.weight", "z_cycle_norm.weight"}


def test_the_trunk_itself_always_trains():
    for depth in (1, 2, 3):
        assert not {n for n in _dead_at(depth) if n.startswith("trunk_")}


def test_depth_one_is_common_enough_to_matter():
    """0.20 per step, so a five-step window that is all depth 1 is not a curiosity."""
    d = _depths(100_000)
    p1 = sum(x == 1 for x in d) / len(d)
    assert 0.19 < p1 < 0.21
    five = sum(all(x == 1 for x in d[i:i + 5]) for i in range(len(d) - 4)) / (len(d) - 4)
    assert five > 1e-4, "if this draw stopped producing depth-1 runs, re-derive the guard rule"


def test_counting_only_deeper_steps_clears_the_window():
    """The rule the RFD4 loop must follow: observe a step only when its depth is >= 2.

    Modelled as the loop runs it -- the dead set of a depth-1 step is never observed, so the
    intersection over five OBSERVED steps is empty and the guard passes a healthy run.
    """
    from tt_bio.train.parity import GradientCoverage

    depths = [1, 1, 3, 1, 4, 2, 1, 5, 3]
    names = sorted(CYCLE_PARAMS | {"trunk_s.weight", "trunk_z.weight"})
    guard = GradientCoverage(names, steps=5)
    moved = False
    for step, depth in enumerate(depths):
        if guard.done:
            break
        if depth == 1:
            continue
        # After the first deeper step the projections have been updated, so the norms behind
        # them train from the next deeper step on.
        dead = set() if moved else {"s_cycle_norm.weight", "z_cycle_norm.weight"}
        moved = True
        guard.observe(dead)
    assert guard.done and not guard.never


def test_counting_every_step_would_refuse_the_same_healthy_run():
    """The control: the same run, observed on every step, is refused. The rule is load-bearing."""
    from tt_bio.train.parity import GradientCoverage

    names = sorted(CYCLE_PARAMS | {"trunk_s.weight", "trunk_z.weight"})
    guard = GradientCoverage(names, steps=5)
    with pytest.raises(RuntimeError, match="parameters received no gradient"):
        for _ in range(5):  # five consecutive depth-1 steps: p = 3.2e-4 per launch
            guard.observe(CYCLE_PARAMS)
