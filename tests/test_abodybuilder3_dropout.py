"""`Dropout`'s mask is torch's, and a seed replays a run exactly. Host-only: no device needed."""
from __future__ import annotations

import torch

from tt_bio.abodybuilder3 import Dropout


def _run(seed, rate=0.1, shape=(4, 64, 128), calls=3):
    """The masks a run would upload. `Dropout.mask` is the whole of what decides replayability, so
    it is what is tested; the device multiply that consumes it is one `ops.mul` and is covered by
    the op gradcheck."""
    drop = Dropout(rate, seed)
    return [drop.mask(shape) for _ in range(calls)]


def test_mask_is_inverted_dropout_at_the_configured_rate():
    masks = _run(0, rate=0.1, shape=(8, 256, 128), calls=1)
    mask = masks[0]
    zeroed = (mask == 0).double().mean().item()
    assert 0.09 < zeroed < 0.11, zeroed
    kept = mask[mask > 0]
    # Torch scales the kept entries by 1/(1-p), so the mask's own mean is ~1 and the layer is a
    # no-op in expectation. A mask of plain 0/1 would shrink every activation by 10 %.
    assert torch.allclose(kept, torch.full_like(kept, 1.0 / 0.9))
    assert abs(mask.mean().item() - 1.0) < 0.01


def test_a_seed_replays_the_run_and_a_different_seed_does_not():
    a, b, c = _run(7), _run(7), _run(8)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not any(torch.equal(x, y) for x, y in zip(a, c))


def test_successive_calls_draw_different_masks():
    """Two sites per block and 8 blocks means 16 draws per micro-batch; one repeated mask would
    correlate the noise across sites and silently weaken the regularisation."""
    masks = _run(0, calls=4)
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert not torch.equal(masks[i], masks[j])


def test_rate_zero_is_a_no_op_and_draws_nothing():
    """Inference passes `dropout=None`, but a rate of zero must also cost nothing rather than
    uploading a mask of ones."""
    drop = Dropout(0.0, 0)
    x = torch.randn(2, 8, 16)
    assert drop(x) is x
    assert drop.calls == 0
