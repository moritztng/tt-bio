"""`perf/bcx_stack/stack.py`: the shared block harness the PERF10/BCX rows measure through.

Card-free. The one thing pinned here is the shape guard, because the campaign lost a device
attribution table to its absence: `bcx-p10-devmap` and `bcx-p10-trimul` timed AF2 blocks at
n=275 while a BindCraft 2 round executes 288, since `tt_bio.bindcraft2.EvoformerOnDevice._pad`
rounds the token axis up to a tile and masks the padding before anything is uploaded. At 275
three fast paths that are open in production decline on `% 32`, so the harness measured a
slower program than the one that ships.
"""
import pytest

from perf.bcx_stack import stack


def test_a_ragged_token_axis_is_refused_because_the_card_never_runs_one():
    with pytest.raises(ValueError, match="not a multiple of 32"):
        stack.inputs(None, 275, 0)


def test_the_refusal_names_the_shape_the_card_would_actually_run():
    with pytest.raises(ValueError, match="executes at 288"):
        stack.inputs(None, 275, 0)


def test_the_pre_pad_host_shape_stays_reachable_on_purpose():
    """Measuring the ragged shape is legitimate; doing it by accident is what this stops.

    `ref=None` gets past the guard and dies in the embedding, which is enough to show the guard
    let it through.
    """
    with pytest.raises(Exception) as e:
        stack.inputs(None, 275, 0, ragged=True)
    assert "not a multiple of 32" not in str(e.value)
