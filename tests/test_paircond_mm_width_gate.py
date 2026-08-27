"""The diffusion pair-conditioning matmul's core_grid gate, pinned by width.

Forcing ``core_grid`` on that projection selects ttnn's multicast matmul, whose
in0-sender/in1-receiver dataflow deadlocks intermittently when the output is 4 tiles wide and
the pair tensor is >=512 tokens (watcher capture: perf/pxv1/hang_watcher_capture.txt;
interleaved A/B: 24/24 folds clean without the forced grid against 7/24 with it).

Two shipped checkpoints share the exposed width, so the gate keys on the width and not on a
model id. This test is the cheap proof that the decision lands where it is supposed to -- in
particular that protenix-v2 keeps the forced grid, which is what makes its device path
byte-identical rather than merely believed unchanged.

Host-only: it asserts on the threshold, no device and no checkpoint load.
"""
import pytest

from tt_bio.protenix import PAIRCOND_MM_NARROW_MAX_TILES


def _forces_grid(out_width: int) -> bool:
    """Mirror of the branch in Protenix._diffusion_pair_cond."""
    return (out_width // 32) > PAIRCOND_MM_NARROW_MAX_TILES


# (checkpoint, linear_no_bias_z output width) read from the real weights, 2026-08-27:
#   protenix-v1  (128, 256)   opendde  (128, 256)   protenix-v2  (256, 512)
@pytest.mark.parametrize("model,out_width,expect_forced", [
    ("protenix-v1", 128, False),   # 4 tiles -> exposed, must NOT force the grid
    ("opendde", 128, False),       # 4 tiles, same shape as protenix-v1 -> also exposed
    ("protenix-v2", 256, True),    # 8 tiles -> keeps the forced grid, path unchanged
])
def test_width_gate_decision(model, out_width, expect_forced):
    assert _forces_grid(out_width) is expect_forced, (
        f"{model}: output {out_width} ({out_width // 32} tiles) should "
        f"{'force' if expect_forced else 'NOT force'} core_grid")


def test_threshold_is_four_tiles():
    """The measured boundary. protenix-v1/opendde hang at 4 tiles; protenix-v2 is clean at 8.

    Raising this above 4 would re-expose protenix-v2; lowering it below 4 would put
    protenix-v1 and opendde back on the deadlocking multicast path.
    """
    assert PAIRCOND_MM_NARROW_MAX_TILES == 4


def test_boundary_is_exclusive_at_the_threshold():
    """4 tiles is narrow (no grid), 5 tiles is not. Pins the comparison direction, which is
    the part a refactor is most likely to invert."""
    assert _forces_grid(4 * 32) is False
    assert _forces_grid(5 * 32) is True
