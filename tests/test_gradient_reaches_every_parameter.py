"""Real training steps, and every trainable parameter has to come out of them with a gradient.

This is the check the ABodyBuilder3 reproduction was missing. On 2026-09-19 its `base-loss` leg
had run 5.4 h and 1,248 optimizer steps while **356 of its 436 parameters had an identically zero
RAdam first moment**, which means every gradient they ever received was exactly zero. The scorer
agreed: step 1,248 came back at 15.262 A mean CDR-H3 against an untrained 15.262 A.

Nothing in the stack could see it. 4,124 tests passed, so did the cross-chip resume gate, the
heartbeat, the dry run and the lr-schedule check, and three resumes replayed 56 of 57 steps
bit-identically, because a frozen parameter is perfectly reproducible. `grad_norm` read a plausible
0.18 and even declined, since it is the norm over all 436 mirrors and the 80 live ones carried it.
`tt_bio/train/abodybuilder3_step.py:241` is what makes the failure silent: a missing gradient
becomes `torch.zeros_like`, so RAdam steps every parameter, `opt_steps` advances on every
parameter, and from there an untrained parameter is indistinguishable from a trained one.

**The gate reads `p.grad`, which is what the tape produced, before any substitution.** The
substitution is written into the host mirror and never back into `p.grad`, so a real step followed
by a look at `p.grad` is the honest question. This file leaves that line alone and needs no change
to it.

**The weights are the run's own, and that is why this gate sees what the older one does not.**
`scripts/abb3_port/step_gate.py` has carried an every-parameter-takes-a-gradient assertion since
the port's first complete step, and it passed throughout: its fixture redraws every parameter from
`N(0, 0.05)` first. The run does not. `scripts/abb3_port/repro.py:37` starts from
`ABB3StructureModule(cfg).state_dict()`, and `tt_bio/af2_reference.py:98` allocates every `Linear`
weight with `torch.zeros` because that class exists to hold weights loaded from a checkpoint and
never to initialise them, so 300 of the 316 tensors the run starts from are identically zero. Zero
is a fixed point: no gradient passes upstream through `W = 0` (`dx = g W^T`), so the layer's input
stays zero and `dW = x^T g` is zero as well. A fixture that redraws the weights deletes the defect
before the check runs, which :func:`test_a_dense_initialisation_makes_the_same_gate_green` states
as a control.

**Three steps, not one, and the third test is why.** Upstream initialises each residual branch's
output projection to zero, so at step 1 those branches contribute nothing and the weights BEHIND
them have an exactly zero gradient -- upstream's behaviour, not a defect, and it clears at step 2
because the projection itself trains from `x^T g`. Measured here: with dense weights and the IPA
output projections zeroed, 186 of 436 parameters look dead at step 1 and none do at step 2. A
one-step check would fail a correctly initialised model. What separates the transient from a dead
parameter is time, so the zero sets are intersected across `STEPS` steps and a parameter has to be
zero on all of them to count.

Sizes: the shipped 8 blocks, because each block carries its own copy of the weights and only the
full depth covers the whole parameter set; one 32-token micro-batch, one tile, which is
`abb3_dataset.TOKEN_MULTIPLE` and the smallest token axis the kernels accept. Three steps at that
crop cost ~1 s on a Blackhole p150a after the compile, and the whole file runs in ~15 s.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

pytest.importorskip("ttnn")

import ttnn                                                                 # noqa: E402

from tt_bio.abodybuilder3_reference import ABB3Config, ABB3StructureModule  # noqa: E402
from tt_bio.tenstorrent import get_device                                   # noqa: E402
from tt_bio.train.abodybuilder3_step import TrainStep                       # noqa: E402

# The run's own starting weights and the run's own micro-batch shapes, imported rather than
# restated: a gate built on a second copy of either would pass while the run failed, which is
# exactly the history above.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "abb3_port"))
from repro import initial_state_dict                                        # noqa: E402
from step_gate import synthetic_micro_batch                                 # noqa: E402

BLOCKS = 8
TOKENS = 32
MICRO = 1
STEPS = 3


def _train(state_dict: dict, cfg: ABB3Config, steps: int = STEPS):
    """Run `steps` real steps and report, per step, which parameters took no usable gradient.

    `p.grad is None` means the tape never reached the parameter at all. An identically zero
    gradient means it reached it and carried nothing, which RAdam treats the same way: no moment,
    no update, ever. Both are collected; only the first is a failure on its own.
    """
    device = get_device()
    step = TrainStep(state_dict, cfg, accumulate=1)
    absent, dead = set(), []
    for i in range(steps):
        step.step([synthetic_micro_batch(cfg, MICRO, TOKENS, 100 + i, device)])
        no_grad = {i for i, p in enumerate(step.params) if p.grad is None}
        zero = {i for i, p in enumerate(step.params)
                if p.grad is not None and float(ttnn.to_torch(p.grad).abs().max()) == 0.0}
        absent |= no_grad
        dead.append(no_grad | zero)
    never_moved = set.intersection(*dead)
    return step, absent, never_moved, dead


def _dense_state_dict(seed: int = 0) -> dict:
    """`step_gate.py`'s fixture: the model with every parameter redrawn from `N(0, 0.05)`."""
    cfg = ABB3Config(use_plddt=False, no_blocks=BLOCKS)
    torch.manual_seed(seed)
    ref = ABB3StructureModule(cfg)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.05)
    return ref.state_dict()


def _shapes(step, idx) -> dict:
    return dict(Counter(tuple(ttnn.to_torch(step.params[i].value).shape) for i in sorted(idx)))


@pytest.mark.device
def test_a_real_training_step_reaches_every_trainable_parameter():
    """The gate. Steps from the weights the run starts from, and no parameter left behind."""
    cfg = ABB3Config(use_plddt=False, no_blocks=BLOCKS)
    step, absent, never_moved, dead = _train(initial_state_dict(cfg, seed=0), cfg)
    total = len(step.params)
    live = set(range(total)) - never_moved
    assert not absent and not never_moved, (
        f"{len(never_moved)} of {total} parameters took no usable gradient on any of {STEPS} real "
        f"training steps ({[len(d) for d in dead]} dead per step, {len(absent)} of them with "
        f"`p.grad is None`). {len(live)} trained, and their shapes are {_shapes(step, live)} "
        f"against {_shapes(step, never_moved)} dead -- not one weight matrix is training.\n"
        f"abodybuilder3_step.py:241 substitutes torch.zeros_like for a missing gradient, so RAdam "
        f"steps all {total} of them, opt_steps advances on all of them, and grad_norm averages "
        f"the zeros in with the live ones. A run in this state is reproducible, resumes "
        f"bit-identically, passes every other gate, and does not train.")


@pytest.mark.device
def test_a_dense_initialisation_makes_the_same_gate_green():
    """The control, in the other direction: make the condition true and the gate has to pass.

    Same model, same crop, same assertion, every parameter redrawn from `N(0, 0.05)`. If this is
    red the gate is broken rather than discriminating, and the failure above says nothing about
    gradients.
    """
    cfg = ABB3Config(use_plddt=False, no_blocks=BLOCKS)
    step, absent, never_moved, dead = _train(_dense_state_dict(), cfg)
    assert not absent and not never_moved, (
        f"the control arm is red: {len(never_moved)} of {len(step.params)} parameters took no "
        f"gradient across {STEPS} steps from densely initialised weights, {len(absent)} of them "
        f"absent ({[len(d) for d in dead]} per step). The gate above cannot be read as evidence "
        f"about the initialisation until this passes")


@pytest.mark.device
def test_a_zero_initialised_residual_branch_is_a_transient_and_not_a_failure():
    """The second control: the gate must tolerate upstream's own zero-initialised projections.

    Upstream zeroes each residual branch's output projection at init. The weights behind such a
    branch have an exactly zero gradient on the first step and a non-zero one on the second, so a
    one-step check fails a correct model. Modelled with the IPA output projections zeroed over
    dense weights -- not upstream's full `final` set, which is the initialiser's business, just
    enough of it to produce the transient.

    The first assertion is what keeps the control honest: if step 1 came back clean there would be
    no transient here and the second assertion would pass for the wrong reason.
    """
    cfg = ABB3Config(use_plddt=False, no_blocks=BLOCKS)
    sd = _dense_state_dict()
    zeroed = [k for k in sd if k.endswith(("linear_out.weight", "linear_out.bias"))]
    for key in zeroed:
        sd[key] = torch.zeros_like(sd[key])
    step, absent, never_moved, dead = _train(sd, cfg)
    assert len(zeroed) == 32 and dead[0], (
        f"zeroing {len(zeroed)} residual output tensors produced no first-step transient "
        f"({[len(d) for d in dead]} dead per step), so this control proves nothing about "
        f"intersecting over {STEPS} steps")
    assert not absent and not never_moved, (
        f"{len(never_moved)} of {len(step.params)} parameters counted as dead across {STEPS} "
        f"steps with only the residual output projections zeroed ({[len(d) for d in dead]} per "
        f"step). That is upstream's own initialisation, so the gate is failing a correct model "
        f"and needs more steps, not fewer")


def test_the_runs_starting_weights_are_not_already_at_zero():
    """The same defect one layer earlier, and this one needs no card.

    A parameter that starts at exactly zero is not merely badly initialised, it is unreachable:
    `dx = g W^T` is zero through a zero weight, so nothing upstream of it receives a gradient
    either, and `dW = x^T g` is zero for every layer whose input the zeros have already killed.
    The all-zero model is a fixed point of its own optimizer, which is how the leg ran 1,248 steps
    without moving. Its checkpoint says the same thing from the other side: all 356 parameters
    with a dead moment also have an identically zero master weight, and none of the 80 live ones
    do (`scripts/abb3_port/moment_audit.py`).

    `tt_bio/af2_reference.py:98` is the line -- `Linear.__init__` allocates `torch.zeros`, correct
    for a module that exists to hold a loaded checkpoint and wrong for the one the reproduction
    initialises from.
    """
    cfg = ABB3Config(use_plddt=False, no_blocks=BLOCKS)
    sd = initial_state_dict(cfg, seed=0)
    zeros = [k for k, v in sd.items() if v.is_floating_point() and float(v.abs().max()) == 0.0]
    assert not zeros, (
        f"{len(zeros)} of {len(sd)} tensors in the run's starting weights are identically zero, "
        f"e.g. {zeros[:6]}. A zero weight cannot receive a gradient and cannot pass one upstream, "
        f"so the model the reproduction starts from is a fixed point of its own optimizer")
