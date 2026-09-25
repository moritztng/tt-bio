"""Our clipping is their clipping: the two R26 divergences, as checks that can fail.

`perf/of3t_leaves/clip_equiv.py` measures both against OpenFold3's real `grad_manager`, which
needs their package installed. These are the same two properties as card-free assertions, so a
regression is caught by the suite rather than only on a host that has openfold3.

R26(1) their `_clip_grads` drops `disabled_params` BEFORE computing the global norm, and their
runner disables the confidence head whenever a sample's summed confidence weight is zero, which
`initial_training.yml` does on 4 of its 5 datasets. Norming over a set they excluded scales
every gradient in the step by a different factor.

R26(2) per-sample clipping is a different algorithm from per-batch clipping, not a different
constant: it changes the DIRECTION of the accumulated update. `per_sample_clipping: True` at
`clip_val 10.0` is their shipped default.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("ttnn")

from tt_bio.train.optim import AdamW                                 # noqa: E402

CLIP = 10.0


class _P:
    """The least a parameter needs here: a numpy `.grad`. `to_host` passes numpy through."""

    def __init__(self, g=None):
        self.grad = None if g is None else np.asarray(g, dtype=np.float32)
        self.value = None


def _opt(grads, shapes=None, clip_norm=CLIP):
    """An `AdamW` with no device behind it: `grad_norm`, `clip_coef` and the accumulator."""
    opt = AdamW.__new__(AdamW)
    opt.params = {k: _P(v) for k, v in grads.items()}
    opt.clip_norm = float(clip_norm)
    shapes = shapes or {k: np.shape(v) for k, v in grads.items()}
    opt.master = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    opt.accum, opt.participation, opt.accum_count = {}, {}, 0
    return opt


def test_a_disabled_parameter_leaves_the_global_norm():
    a = np.full(64, 0.5, np.float32)                 # norm 4
    conf = np.full(256, 1.0, np.float32)             # norm 16, and over the clip on its own
    opt = _opt({"a": a, "conf": conf})
    both = math.sqrt(float(a @ a) + float(conf @ conf))
    assert opt.grad_norm() == pytest.approx(both, rel=1e-6)
    assert opt.grad_norm({"conf"}) == pytest.approx(4.0, rel=1e-6)
    # The coefficient is what reaches every gradient in the step, so that is what is compared.
    # Including the disabled tensor clips at 10/16.49; excluding it does not clip at all.
    assert opt.clip_coef(opt.grad_norm()) == pytest.approx(CLIP / both, rel=1e-6)
    assert opt.clip_coef(opt.grad_norm({"conf"})) == 1.0


def test_a_disabled_parameter_is_not_accumulated():
    """Theirs skips it in `clip_and_accumulate` too, not only in the norm."""
    opt = _opt({"a": np.full(64, 0.5, np.float32), "conf": np.full(256, 1.0, np.float32)})
    opt.clip_and_accumulate({"conf"})
    assert sorted(opt.accum) == ["a"], sorted(opt.accum)
    assert opt.participation == {"a": 1}
    # ...and not clipped by a norm the disabled tensor dominated: 4.0 is under the clip.
    assert float(np.linalg.norm(opt.accum["a"])) == pytest.approx(4.0, rel=1e-5)


def test_per_sample_clipping_accumulates_a_different_vector_from_per_batch():
    """The direction changes. That is the whole finding, and a norm alone would hide it."""
    rng = np.random.default_rng(3)
    samples = [rng.standard_normal(64).astype(np.float32) * s for s in (0.2, 0.2, 40.0)]

    opt = _opt({"a": None}, shapes={"a": (64,)})
    for s in samples:
        opt.params["a"].grad = s.copy()
        opt.clip_and_accumulate()
        opt.params["a"].grad = None
    per_sample = opt.accum["a"]
    assert opt.participation == {"a": 3} and opt.accum_count == 3

    total = np.sum(samples, axis=0)
    batch = _opt({"a": total})
    per_batch = total * batch.clip_coef(batch.grad_norm())

    rel = float(np.linalg.norm(per_sample - per_batch) / np.linalg.norm(per_sample))
    assert rel > 1e-2, (
        f"per-sample and per-batch clipping agreed to {rel:.3e} on a batch with a 200x "
        f"outlier; they are different algorithms and this check cannot see the difference")
    cos = float(per_sample @ per_batch /
                (np.linalg.norm(per_sample) * np.linalg.norm(per_batch)))
    assert cos < 0.9999, f"the two arms differ only in length (cos {cos:.6f})"


def test_the_accumulator_is_not_clipped_a_second_time():
    """Clipping the sum after clipping each sample would be a third algorithm."""
    opt = _opt({"a": None}, shapes={"a": (64,)})
    for _ in range(3):
        opt.params["a"].grad = np.full(64, 1.0, np.float32)     # norm 8, under the clip
        opt.clip_and_accumulate()
    assert float(np.linalg.norm(opt.accum["a"])) == pytest.approx(24.0, rel=1e-5)
