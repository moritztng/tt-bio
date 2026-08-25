"""The ttnn.gather bound the gathered atom softmax stands on, pinned host-only.

scripts/rfd3_port/p81{,b,c}_*.py measured ttnn.gather 0.68.0 on dim 3 returning wrong data for
every tile-row after the first once the indexed axis exceeds 1920 elements: exact at NK=1920
fp32 (7.5 KB/row), 93.75 % wrong at NK=1952, 99.87 % wrong at the production [1,4,6051,6080].
The trigger is element count and not bytes -- bf16 breaks at 2048 while the wider fp32 row at
1920 is fine -- so no dtype choice reaches the page fixture's 6080.

Host-only. These pin the bound and that the model refuses rather than computing garbage, which
is the failure mode that cost this lineage a pass: the arm was built, its five invariant tests
passed, and every one of them tested a property of the INDEX rather than the output of the op.
"""
import pytest

from tt_bio.rfd3 import model as M


def test_gather_bound_is_the_measured_one():
    """1920 elements, read off perf/p81/gather_boundary.json. Not a round guess."""
    assert M._TTNN_GATHER_MAX_KEY_AXIS == 1920


def test_page_fixture_key_axis_is_past_the_bound():
    """6051 atoms tile-align to 6080, 3.2x the bound -- the arm cannot run at the page fixture."""
    assert M.align_tile(6051) == 6080
    assert M.align_tile(6051) > M._TTNN_GATHER_MAX_KEY_AXIS


def test_guard_refuses_the_page_fixture():
    with pytest.raises(RuntimeError, match="RFD3_GATHERED_SOFTMAX is unusable"):
        M._check_gather_bound(6051)


def test_guard_names_both_numbers():
    """The message has to carry the actual axis and the bound, or it is not actionable."""
    with pytest.raises(RuntimeError) as e:
        M._check_gather_bound(6051)
    assert "6080" in str(e.value) and "1920" in str(e.value)


@pytest.mark.parametrize("length", [32, 512, 1024, 1889, 1920])
def test_guard_allows_shapes_measured_exact(length):
    """Everything at or below the bound is where ttnn.gather was measured bit-exact."""
    M._check_gather_bound(length)


@pytest.mark.parametrize("length", [1921, 1952, 2048, 6051])
def test_guard_refuses_shapes_measured_wrong(length):
    with pytest.raises(RuntimeError):
        M._check_gather_bound(length)


def test_gathered_softmax_is_off_by_default():
    """Nothing here changes the shipped default."""
    assert M._GATHERED_SOFTMAX is False
