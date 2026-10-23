"""RFD4's default sample width: as many designs a device step as fit in one 512-token design's pair rows, at most 16.

Run: python3 -m pytest tests/test_rfd4_width.py (no card)
"""
import inspect

import pytest

from tt_bio.rfd4.engine import DesignEngine, auto_width
from tt_bio.sample_chunks import resolve_sample_chunk_width


@pytest.mark.parametrize("n, width", [(64, 16), (128, 16), (192, 7), (256, 4), (362, 2), (363, 1), (512, 1),
                                      (1024, 1)])
def test_auto_width_fits_one_512_token_designs_pair_rows_up_to_16(n, width):
    assert auto_width(n) == width
    assert n * n * width <= 512 * 512 or width == 1


def test_design_engine_defaults_to_auto():
    assert inspect.signature(DesignEngine).parameters["max_parallel_samples"].default == "auto"


def test_auto_width_is_rebalanced_over_the_designs_asked_for():
    # 20 designs at 128 tokens run as two lanes of 10, not 16 + 4
    assert resolve_sample_chunk_width(20, auto_width(128)) == 10
    assert resolve_sample_chunk_width(3, auto_width(128)) == 3
