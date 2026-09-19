"""One real training step, and every trainable parameter has to come out of it with a gradient.

This is the check the ABodyBuilder3 reproduction was missing. On 2026-09-19 its `base-loss` leg
had run 5.4 h and 1,248 optimizer steps while **356 of its 436 parameters had an identically zero
first moment**, which for RAdam means every gradient they ever received was exactly zero. The
scorer agreed: step 1,248 came back at 15.262 A mean CDR-H3 against an untrained 15.262 A.

Nothing in the stack could see it. 4,124 tests passed, so did the cross-chip resume gate, the
heartbeat, the dry run and the lr-schedule check, and three resumes replayed 56 of 57 steps
bit-identically -- a frozen parameter is perfectly reproducible. `grad_norm` averaged the zeros in
with the live parameters and declined convincingly as the handful of live ones converged.
`tt_bio/train/abodybuilder3_step.py:241` is why: a missing gradient becomes `torch.zeros_like`, so
RAdam steps every parameter, `opt_steps` advances on every parameter, and an untrained parameter
is indistinguishable from a trained one from there on.

**The gate reads `p.grad`, which is the gradient the tape produced, before any substitution.** The
substitution is written into the host mirror, never back into `p.grad`, so one real step followed
by a look at `p.grad` is the honest question. This file does not touch that line and does not need
it changed to work.

**The weights are the run's own, and that is the whole reason this gate sees what the older one
does not.** `scripts/abb3_port/step_gate.py` has carried an every-parameter-takes-a-gradient
assertion since the port's first complete step, and it passed throughout. Its fixture redraws
every parameter from `N(0, 0.05)` first. The run does not: `scripts/abb3_port/repro.py:37` starts
from `ABB3StructureModule(cfg).state_dict()`, and `tt_bio/af2_reference.py:98` allocates every
`Linear` weight with `torch.zeros`, because that class was written to hold weights loaded from a
checkpoint and never to initialise them. 300 of the 316 tensors the run starts from are identically
zero, and zero is a fixed point: with `W = 0` no gradient passes upstream (`dx = g W^T`), so the
layer's input stays zero and `dW = x^T g` is zero too. The only leaves that move are the ones that
are not zero to begin with -- the LayerNorm scales -- plus the biases directly under a live
gradient. A fixture that redraws the weights deletes the defect before the check runs, which is
what :func:`test_a_dense_initialisation_makes_the_same_gate_green` is here to state as a control.

Sizes: the shipped 8 blocks, because each block carries its own copy of the weights and only the
full depth covers the whole parameter set; one 32-token micro-batch, one tile, which is
`abb3_dataset.TOKEN_MULTIPLE` and the smallest token axis the kernels accept. One step at that
crop is ~3 s on a Blackhole p150a after the compile.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

pytest.importorskip("ttnn")

import ttnn                                                              # noqa: E402

from tt_bio.abodybuilder3_reference import ABB3Config, ABB3StructureModule  # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402
from tt_bio.train.abodybuilder3_step import TrainStep                    # noqa: E402

# The run's own starting weights and the run's own micro-batch shapes, imported rather than
# restated: a gate built on a second copy of either would pass while the run failed, which is
# exactly the history above.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "abb3_port"))
from repro import initial_state_dict                                     # noqa: E402
from step_gate import synthetic_micro_batch                              # noqa: E402

BLOCKS = 8
TOKENS = 32
MICRO = 1


def _one_real_step(state_dict: dict, cfg: ABB3Config):
    """Forward, host losses, device backward and one RAdam step, on the leased card."""
    step = TrainStep(state_dict, cfg, accumulate=1)
    step.step([synthetic_micro_batch(cfg, MICRO, TOKENS, 100, get_device())])
    return step


def _without_a_gradient(step) -> tuple[list, list]:
    """The parameters one step left untrainable, split by how.

    `p.grad is None` means the tape never reached the parameter. An identically zero gradient
    means it reached it and carried nothing, which RAdam treats the same way: no moment, no
    update, ever. Both are reported because they have different causes and the same consequence.
    """
    absent = [i for i, p in enumerate(step.params) if p.grad is None]
    zero = [i for i, p in enumerate(step.params)
            if p.grad is not None and float(ttnn.to_torch(p.grad).abs().max()) == 0.0]
    return absent, zero


def _shapes(step, idx) -> dict:
    return dict(Counter(tuple(ttnn.to_torch(step.params[i].value).shape) for i in idx))


@pytest.mark.device
def test_a_real_training_step_reaches_every_trainable_parameter():
    """The gate. One step from the weights the run starts from, and no parameter left behind."""
    cfg = ABB3Config(use_plddt=False, no_blocks=BLOCKS)
    step = _one_real_step(initial_state_dict(cfg, seed=0), cfg)
    absent, zero = _without_a_gradient(step)
    dead = sorted(set(absent) | set(zero))
    live = [i for i in range(len(step.params)) if i not in set(dead)]
    assert not dead, (
        f"{len(dead)} of {len(step.params)} parameters came out of one real training step with no "
        f"usable gradient: {len(absent)} with `p.grad is None`, {len(zero)} with an identically "
        f"zero gradient. {len(live)} trained, and their shapes are {_shapes(step, live)} against "
        f"{_shapes(step, dead)} dead -- not one weight matrix is training.\n"
        f"abodybuilder3_step.py:241 substitutes torch.zeros_like for a missing gradient, so RAdam "
        f"steps all {len(step.params)} of them, opt_steps advances on all of them, and grad_norm "
        f"averages the zeros in with the live ones. A run in this state is reproducible, resumes "
        f"bit-identically, passes every other gate, and does not train.")


@pytest.mark.device
def test_a_dense_initialisation_makes_the_same_gate_green():
    """The control, in the other direction: make the condition true and the gate has to pass.

    Same model, same crop, same assertion, with every parameter redrawn from `N(0, 0.05)` --
    `step_gate.py`'s fixture. If this is red the gate is broken rather than discriminating, and
    if both arms are red the failure above says nothing about gradients.
    """
    cfg = ABB3Config(use_plddt=False, no_blocks=BLOCKS)
    torch.manual_seed(0)
    ref = ABB3StructureModule(cfg)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.05)
    step = _one_real_step(ref.state_dict(), cfg)
    absent, zero = _without_a_gradient(step)
    assert not absent and not zero, (
        f"the control arm is red: {len(absent)} parameters took no gradient and {len(zero)} took "
        f"an identically zero one from densely initialised weights. The gate above cannot be read "
        f"as evidence about the initialisation until this passes")


def test_the_runs_starting_weights_are_not_already_at_zero():
    """The same defect one layer earlier, and this one needs no card.

    A parameter that starts at exactly zero is not merely badly initialised, it is unreachable:
    `dx = g W^T` is zero through a zero weight, so nothing upstream of it receives a gradient
    either, and `dW = x^T g` is zero for every layer whose input the zeros have already killed.
    The all-zero model is a fixed point of the update, which is why the leg could run 1,248 steps
    without moving.

    `tt_bio/af2_reference.py:98` is the line: `Linear.__init__` allocates `torch.zeros`, correct
    for a module that exists to hold a loaded checkpoint and wrong for the one the reproduction
    initialises from. Card-free on purpose -- this is the half of the failure any host can check.
    """
    cfg = ABB3Config(use_plddt=False, no_blocks=BLOCKS)
    sd = initial_state_dict(cfg, seed=0)
    zeros = [k for k, v in sd.items() if v.is_floating_point() and float(v.abs().max()) == 0.0]
    assert not zeros, (
        f"{len(zeros)} of {len(sd)} tensors in the run's starting weights are identically zero, "
        f"e.g. {zeros[:6]}. A zero weight cannot receive a gradient and cannot pass one upstream, "
        f"so the model the reproduction starts from is a fixed point of its own optimizer")
