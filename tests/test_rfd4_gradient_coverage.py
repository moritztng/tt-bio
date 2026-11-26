"""The every-parameter-trains guard, driven on upstream RFD4's own modules.

`test_training_parity_bars.py` drives `GradientCoverage` with hand-made dead sets. This file
drives it with REAL ones: upstream's `MultiheadAttnAndTransition` and `PairReprUpdate`, built
with upstream's own initialisers, four layers at 24 tokens so it runs on a CPU in under a
second. Three things are asserted, and the middle one is why the guard is written the way it is:

* a one-step check fires on parameters that are fine -- upstream zero-initialises the triangle
  gate and `to_out`, the adaLN gammas and the pair transition, so the weights behind them have
  an exactly zero gradient on step 1 only. Measured here: **36 of 174, every one of them alive
  by step 2**;
* the >= 5-step intersection is empty on that same honest model, so the guard passes it;
* one parameter detached makes the guard RAISE. A gate nobody has seen fail is not evidence.

Skipped where the pinned upstream checkout is absent or its deps are not importable;
`/home/moritz/rfd4-cpu-venv` is the venv that runs it on pc.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

from tt_bio.train.parity import GradientCoverage, zero_or_missing

UPSTREAM = pathlib.Path("/home/moritz/rfd4-upstream/RFD4-Proteina")
pytestmark = pytest.mark.skipif(not UPSTREAM.exists(),
                                reason="upstream reference clone not on this host")

N_LAYERS, EVERY_N = 4, 2
TOKEN_DIM, PAIR_DIM, COND_DIM, N_HEADS, TRI_C = 128, 64, 64, 4, 64
B, N, STEPS = 1, 24, 6


def _trunk():
    """`N_LAYERS` upstream sequence blocks with a pair update every `EVERY_N`.

    The released flow config's shapes scaled down. What this test reads is upstream's
    INITIALISATION, which does not depend on the widths.
    """
    if str(UPSTREAM / "src") not in sys.path:
        sys.path.insert(0, str(UPSTREAM / "src"))
    try:
        from rfproteina.nn.modules.attn_n_transition import MultiheadAttnAndTransition
        from rfproteina.nn.modules.pair_update import PairReprUpdate
    except ImportError as exc:  # upstream's own runtime deps, not tt-bio's
        pytest.skip(f"upstream rfproteina is not importable in this venv: {exc}")

    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                MultiheadAttnAndTransition(
                    dim_token=TOKEN_DIM, dim_pair=PAIR_DIM, nheads=N_HEADS, dim_cond=COND_DIM,
                    residual_mha=True, residual_transition=True,
                    parallel_mha_transition=False, use_attn_pair_bias=True, use_qkln=True,
                    attention_type="torch")
                for _ in range(N_LAYERS))
            self.pair = torch.nn.ModuleList(
                PairReprUpdate(token_dim=TOKEN_DIM, pair_dim=PAIR_DIM, use_tri_mult=True,
                               tri_mult_c=TRI_C, tri_mult_bias=False, pair_write="hadamard",
                               checkpoint_tri_mult=False, tri_mult_backend="torch")
                for _ in range(N_LAYERS // EVERY_N))

        def forward(self, s, z, cond, mask):
            p = 0
            for i, blk in enumerate(self.blocks):
                s = blk(s, z, cond, mask)
                if (i + 1) % EVERY_N == 0:
                    z = self.pair[p](s, z, mask)
                    p += 1
            return s, z

    return Trunk()


def _dead_sets(detach: str | None = None):
    """Run `STEPS` Adam steps; return (parameter names, per-step no/zero-gradient sets)."""
    torch.manual_seed(0)
    model = _trunk()
    if detach is not None:
        dict(model.named_parameters())[detach].requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    per_step = []
    for step in range(STEPS):
        g = torch.Generator().manual_seed(step)
        s = torch.randn(B, N, TOKEN_DIM, generator=g)
        z = torch.randn(B, N, N, PAIR_DIM, generator=g)
        cond = torch.randn(B, N, COND_DIM, generator=g)
        mask = torch.ones(B, N, dtype=torch.bool)
        out_s, out_z = model(s, z, cond, mask)
        opt.zero_grad(set_to_none=True)
        (out_s.square().mean() + out_z.square().mean()).backward()
        per_step.append(set(zero_or_missing({n: p.grad
                                             for n, p in model.named_parameters()})))
        opt.step()
    return [n for n, _ in model.named_parameters()], per_step


@pytest.fixture(scope="module")
def honest():
    return _dead_sets()


def test_a_one_step_check_fires_on_upstreams_zero_initialised_branches(honest):
    """The false positive, quantified: 36 of 174 at step 1, all of them alive at step 2."""
    names, per_step = honest
    assert len(per_step[0]) > 0, "no zero-init transient at all; the model is not upstream's"
    assert not (per_step[0] & per_step[1]), (
        "a parameter dead on both of the first two steps: this test can no longer tell the "
        "transient from a dead parameter")


def test_the_guard_passes_the_honest_model_over_five_steps(honest):
    names, per_step = honest
    guard = GradientCoverage(names, steps=5, where="[rfd4 honest] ")
    for dead in per_step[:5]:
        guard.observe(dead)
    assert guard.done and not guard.never


def test_the_guard_fails_on_one_detached_parameter(honest):
    """The arm that makes a pass meaningful."""
    names, _ = honest
    victim = next(n for n in names if n.endswith("weight"))
    names_b, per_step_b = _dead_sets(victim)
    guard = GradientCoverage(names_b, steps=5, where="[rfd4 detached] ")
    with pytest.raises(RuntimeError, match=r"1 of \d+ parameters received no gradient"):
        for dead in per_step_b[:5]:
            guard.observe(dead)
