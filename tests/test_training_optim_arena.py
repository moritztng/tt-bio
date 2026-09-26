"""The arena rewrite of `AdamW.step` is bit-identical to the expression it replaces.

`step()` used to allocate about eleven full-size temporaries per parameter and read every
gradient back from the card twice -- once for the global clip norm, once in the update loop.
At OF3T's crop-384 census (3,152 parameters, 381,302,188 elements) that was 1.847 s of PCIe
and 2.046 s of numpy out of AdamW's 5.798 s, with another 1.134 s in the two
`round_to_device_dtype` casts (`perf/of3t_p10optim/out/base_384.json`).

The replacement does the same operations on the same operands in the same order, into
reusable buffers. So the bar is `array_equal` and not `allclose`: this campaign is
accuracy-gated, and a lever that is free of any accuracy question is worth more than a faster
one that moves a digit. Anything short of exact equality here means the rewrite changed the
optimizer and the speed is not the point any more.

The reference below is the literal expression from the module docstring, written out here
rather than imported, so the test compares against the ALGORITHM and not against whatever
the previous revision of the same file happened to compute.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ttnn = pytest.importorskip("ttnn")

from tt_bio.main import ensure_p300_mesh_descriptor                 # noqa: E402
from tt_bio import autograd as ag                                   # noqa: E402
from tt_bio.train.optim import AdamW, round_to_device_dtype         # noqa: E402
from tt_bio.train.tensors import to_device, to_host                 # noqa: E402

LR, B1, B2, EPS, WD, CLIP = 3e-4, 0.9, 0.999, 1e-8, 0.01, 10.0

# Shapes that run large-then-small and small-then-large within one step, which is the order
# a buffer-reuse bug needs: a slot grown to the biggest parameter is then handed to a smaller
# one as a view, and a stale tail would show up as a wrong update on the parameter after it.
SHAPES = {"big": (96, 512), "small": (8, 16), "mid": (64, 64),
          "biggest": (128, 512), "tiny": (4,), "odd": (37, 13)}


@pytest.fixture(scope="module")
def dev():
    ensure_p300_mesh_descriptor()
    d = ttnn.open_device(device_id=0)
    try:
        yield d
    finally:
        ttnn.close_device(d)


def _grad_draw(rng, step):
    """Gradients including the three cases the loop branches on.

    `no_grad` never gets one, so `step()` synthesises zeros for it; `dead` gets an all-zero
    gradient, which is the confidence head's case after a batch that disables it; and the
    scale is large enough on the first step that the global norm exceeds `clip_norm` and the
    `clip != 1.0` branch fires, then small enough afterwards that it does not.
    """
    scale = 40.0 if step == 0 else 1e-4
    out = {}
    for name, shape in SHAPES.items():
        if name == "no_grad":
            continue
        out[name] = (np.zeros(shape, np.float32) if name == "tiny"
                     else (rng.standard_normal(shape) * scale).astype(np.float32))
    return out


def reference_step(master, m, v, grads, pows):
    """AdamW as `tt_bio/train/optim.py` documents it, allocating freely."""
    gnorm = math.sqrt(sum(float(g.ravel() @ g.ravel()) for g in grads.values()))
    clip = min(1.0, CLIP / gnorm) if (CLIP > 0 and gnorm > 0) else 1.0
    pows[0] *= B1
    pows[1] *= B2
    bc1, bc2 = 1.0 - pows[0], 1.0 - pows[1]
    for name, theta in master.items():
        g = grads.get(name)
        if g is None:
            g = np.zeros_like(theta)
        elif clip != 1.0:
            g = g * clip
        mm, vv = m[name], v[name]
        mm *= B1
        mm += (1.0 - B1) * g
        vv *= B2
        vv += (1.0 - B2) * (g * g)
        upd = LR * ((mm / bc1) / (np.sqrt(vv / bc2) + EPS) + WD * theta)
        theta -= upd
    return clip, gnorm


def test_the_arena_step_is_bit_identical_to_the_expression(dev):
    rng_init = np.random.default_rng(31)
    init = {n: rng_init.standard_normal(s).astype(np.float32) for n, s in SHAPES.items()}

    params = {n: ag.Tensor(to_device(a, dev, dtype=ttnn.bfloat16), requires_grad=True)
              for n, a in init.items()}
    opt = AdamW(params, lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD, clip_norm=CLIP)

    # The reference starts from the masters the optimizer built, which are the bf16 weights
    # widened, not from `init`. Comparing against `init` would be comparing two casts.
    ref_master = {n: opt.master[n].copy() for n in init}
    ref_m = {n: np.zeros_like(a) for n, a in ref_master.items()}
    ref_v = {n: np.zeros_like(a) for n, a in ref_master.items()}
    pows = [1.0, 1.0]

    rng_g = np.random.default_rng(32)
    clips = []
    for step in range(5):
        draw = _grad_draw(rng_g, step)
        for n, g in draw.items():
            params[n].grad = to_device(g, dev, dtype=ttnn.bfloat16)
        # What the optimizer will actually see, read back through the same path it uses, so
        # the reference is fed the bf16 gradient rather than the fp32 one it was drawn as.
        seen = {n: to_host(params[n].grad).astype(np.float32).reshape(ref_master[n].shape)
                for n in draw}
        clip, _ = reference_step(ref_master, ref_m, ref_v, seen, pows)
        clips.append(clip)
        opt.step()

        for n in SHAPES:
            assert np.array_equal(opt.master[n], ref_master[n]), (
                f"step {step}, {n}: {int((opt.master[n] != ref_master[n]).sum())} of "
                f"{ref_master[n].size} master elements differ; the arena rewrite changed "
                f"the optimizer")
            assert np.array_equal(opt.exp_avg[n], ref_m[n]), f"step {step}, {n}: exp_avg"
            assert np.array_equal(opt.exp_avg_sq[n], ref_v[n]), f"step {step}, {n}: exp_avg_sq"

    assert clips[0] < 1.0, "the fixture never exercised the clip branch"
    assert clips[-1] == 1.0, "the fixture only ever exercised the clip branch"

    # The card, not just the master: the write is skipped when the update rounds away, so an
    # equal master with an unequal device weight would be a missing `to_device`.
    for n, shape in SHAPES.items():
        on_card = to_host(params[n].value).astype(np.float32).reshape(shape)
        assert np.array_equal(on_card, round_to_device_dtype(ref_master[n], ttnn.bfloat16)), n


def test_the_arena_holds_one_buffer_per_slot(dev):
    """The claim the rewrite rests on: a buffer per slot, grown to the largest parameter.

    A buffer per PARAMETER would be the allocation it set out to remove, wearing a pool.
    """
    rng = np.random.default_rng(33)
    params = {n: ag.Tensor(to_device(rng.standard_normal(s).astype(np.float32), dev,
                                     dtype=ttnn.bfloat16), requires_grad=True)
              for n, s in SHAPES.items()}
    opt = AdamW(params, lr=LR, clip_norm=CLIP)
    for n, s in SHAPES.items():
        params[n].grad = to_device(rng.standard_normal(s).astype(np.float32), dev,
                                   dtype=ttnn.bfloat16)
    opt.step()

    biggest = max(int(np.prod(s)) for s in SHAPES.values())
    assert opt._arena.peak_elements == biggest
    held = sum(b.nbytes for b in opt._arena._f32.values())
    assert held <= 8 * biggest * 4, (
        f"the arena holds {held} bytes for a {biggest}-element largest parameter; that is "
        f"more slots than step() uses")


def test_the_buffered_cast_is_the_unbuffered_one():
    """`Tensor.copy_` across dtypes has to be the same conversion `.to()` runs."""
    import torch
    rng = np.random.default_rng(34)
    for scale in (1.0, 1e-38, 3e4, 2.0 ** -9):
        arr = (rng.standard_normal((64, 33)) * scale).astype(np.float32)
        plain = round_to_device_dtype(arr, ttnn.bfloat16)
        out = np.empty_like(arr)
        tmp = torch.empty(arr.shape, dtype=torch.bfloat16)
        buffered = round_to_device_dtype(arr, ttnn.bfloat16, out=out, tmp=tmp)
        assert buffered is out
        assert np.array_equal(plain, buffered), f"scale {scale}"


def test_grad_norm_does_not_leak_the_gradients_it_read(dev):
    """`grad_norm` is public. A caller who wants a scalar should not be handed 1.5 GB."""
    rng = np.random.default_rng(35)
    params = {n: ag.Tensor(to_device(rng.standard_normal(s).astype(np.float32), dev,
                                     dtype=ttnn.bfloat16), requires_grad=True)
              for n, s in SHAPES.items()}
    opt = AdamW(params, lr=LR, clip_norm=CLIP)
    for n, s in SHAPES.items():
        params[n].grad = to_device(rng.standard_normal(s).astype(np.float32), dev,
                                   dtype=ttnn.bfloat16)
    assert opt.grad_norm() > 0
    assert opt._grads == {}
    opt.step()
    assert opt._grads == {}, "step() left the whole gradient set held after it finished"
