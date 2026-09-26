"""The optimizer's per-step report is computed on the host, and it is the same report.

`AdamW.step` used to read every weight back from the card TWICE per step -- once before the
write and once after -- so it could report how much of the update survived the cast to the
device dtype. At OF3T's crop-384 census (3,152 parameters, 381,302,188 elements) that is
0.66 s of PCIe per step against 0.36 s for the same two casts on the host
(`perf/of3t_p10host/out/optsplit.json`). `round_to_device_dtype` replaces both reads.

That is only allowed if the host cast is BIT-IDENTICAL to what the card holds, so this file
asserts exactly that, on a real card, over the values an optimizer actually produces --
including the ones that round away, which are the common case during warmup and the whole
reason the report exists.

Three separate claims, because they fail separately:

  * the cast reproduces a `to_device` -> `to_host` round trip bit for bit;
  * a skipped write leaves the card holding the tensor the write would have built;
  * `norm(upd)` is `norm(theta - before)`, so dropping the full-census copy changes nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ttnn = pytest.importorskip("ttnn")

from tt_bio.main import ensure_p300_mesh_descriptor                 # noqa: E402
from tt_bio.train.optim import round_to_device_dtype                # noqa: E402
from tt_bio.train.tensors import to_device, to_host                 # noqa: E402


def _cases(rng):
    """Values an AdamW step produces, not a uniform draw.

    The interesting ones are the updates that vanish: `af3_lr`'s warmup puts every step of
    the first 1000 under bf16 spacing, so `theta - upd` rounding back to `theta` is the case
    the report is measuring and the case the skip fires on.
    """
    base = rng.standard_normal((64, 128)).astype(np.float32)
    return {
        "plain": base,
        "tiny_update": base - base * np.float32(1e-9),     # rounds away entirely
        "half_ulp": base * np.float32(1.0 + 2.0 ** -9),    # straddles the tie
        "denormal": (base * np.float32(1e-38)).astype(np.float32),
        "zeros": np.zeros_like(base),
        "big": (base * np.float32(3e4)).astype(np.float32),
    }


@pytest.fixture(scope="module")
def dev():
    # This opens the device directly rather than through T.get_device(), so it has to
    # apply the lone-P300 mesh-graph descriptor itself or open_device is a TT_FATAL.
    ensure_p300_mesh_descriptor()
    d = ttnn.open_device(device_id=0)
    try:
        yield d
    finally:
        ttnn.close_device(d)


def test_host_cast_matches_a_device_round_trip(dev):
    rng = np.random.default_rng(7)
    for name, arr in _cases(rng).items():
        on_card = to_host(to_device(arr, dev, dtype=ttnn.bfloat16)).astype(np.float32)
        on_host = round_to_device_dtype(arr, ttnn.bfloat16)
        assert on_host is not None, name
        assert np.array_equal(on_host.reshape(on_card.shape), on_card), (
            f"{name}: the host cast and the card disagree on "
            f"{int((on_host.reshape(on_card.shape) != on_card).sum())} of {arr.size} "
            f"elements; the per-step report cannot be computed on the host")


def test_a_skipped_write_leaves_the_card_holding_the_right_tensor(dev):
    """`step()` skips `to_device` when the update rounds away. It has to be an exactness."""
    rng = np.random.default_rng(8)
    theta = rng.standard_normal((32, 64)).astype(np.float32)
    on_card = to_device(theta, dev, dtype=ttnn.bfloat16)

    upd = theta * np.float32(1e-9)                 # under bf16 spacing everywhere
    after = theta - upd
    assert np.array_equal(round_to_device_dtype(after, ttnn.bfloat16),
                          round_to_device_dtype(theta, ttnn.bfloat16)), (
        "the fixture no longer exercises the skip")

    rewritten = to_host(to_device(after, dev, dtype=ttnn.bfloat16)).astype(np.float32)
    kept = to_host(on_card).astype(np.float32)
    assert np.array_equal(kept, rewritten), (
        "skipping the write is only safe if the tensor already on the card is the one the "
        "write would have built")


def test_norm_of_the_update_is_the_master_step():
    """`theta -= upd` is the only thing that moves the master, so `norm(upd)` is the step.

    Card-free: this is the identity that let a full-census copy per step go away (1.42 GiB
    of allocation at OF3T's census, 0.361 s against 0.032 s).
    """
    rng = np.random.default_rng(9)
    theta = rng.standard_normal(4096).astype(np.float32)
    upd = rng.standard_normal(4096).astype(np.float32) * np.float32(1e-3)
    before = theta.copy()
    theta = theta - upd
    assert float(np.linalg.norm(upd)) == pytest.approx(
        float(np.linalg.norm(theta - before)), rel=1e-6)


def test_an_unrepresentable_dtype_falls_back_instead_of_guessing():
    """`bfloat8_b` is a block format with a shared exponent; torch has no such scalar type."""
    assert round_to_device_dtype(np.zeros(8, np.float32), ttnn.bfloat8_b) is None
    arr = np.zeros(8, np.float32)
    assert round_to_device_dtype(arr, ttnn.float32) is arr
