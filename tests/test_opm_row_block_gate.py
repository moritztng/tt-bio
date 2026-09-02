"""OuterProductMean's row-blocking gate, and why it cannot share a token-count threshold.

OPM's unblocked result is `(I, C*D, J)` with C = D = 32, so 1024 channels against a pair
tensor's c_z of 256 -- four times the pair tensor, and refused long before one is. It used to
be row-blocked only when `I > SEQ_LEN_MORE_CHUNKING`, a threshold that bounds how many full
PAIR tensors are live and that `_apply_grid_thresholds` re-fits against DRAM. `6e1f9e77`
(2026-08-16) moved that re-fit from 608 to 1088 on a 12 GiB Galaxy part on a Boltz-2
measurement, and protenix-v2, reading the same constant, stopped row-blocking its outer
product everywhere from 640 to 1088 padded tokens.

Measured on the Galaxy 2026-09-02, protenix-v2, warm MSA, platform flags: at 1024 padded
tokens the unblocked path asks for 2 147 483 648 B and is refused with 331 MB free per bank
and a 136 MB largest block; at 1088 padded it asks 2 424 307 712 B. Both are `N*1024*N*2` to
the byte, which is what identifies the site.

Host-only: the gate is pure arithmetic over shapes, no device and no fold.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

# OuterProductMean's projected width, both sides. C*D = 1024 is what makes this tensor big.
C = D = 32
# The Galaxy value of the pair-tensor gate after the DRAM re-fit: 1024 * sqrt((12 GiB/3)/3.461
# GiB) snapped to 32. Pinned here so the test states the configuration it is about instead of
# depending on whatever a host without a device happens to compute.
GALAXY_SEQ_GATE = 1088
# padded tokens -> bytes the unblocked path would ask the allocator for, N * (C*D) * N * 2.
REFUSED = {1024: 2147483648, 1088: 2424307712}
# The largest padded width the release gate's ladder is green at, and its ask.
FOLDS = {640: 838860800}


@pytest.fixture
def galaxy(monkeypatch):
    """The 12 GiB Wormhole Galaxy configuration the numbers above were measured on."""
    monkeypatch.setattr(T, "SEQ_LEN_MORE_CHUNKING", GALAXY_SEQ_GATE)
    return GALAXY_SEQ_GATE


def unblocked_bytes(n):
    return n * C * D * n * 2


def test_measured_asks_are_reproduced_by_the_shape_arithmetic():
    """If this drifts, the site named above is no longer the one being reasoned about."""
    for n, want in {**REFUSED, **FOLDS}.items():
        assert unblocked_bytes(n) == want


def test_the_refused_widths_are_row_blocked(galaxy):
    for n in REFUSED:
        assert n <= galaxy, "these widths are BELOW the pair-tensor gate; that is the point"
        rows = T.opm_row_block(n, C, D, n)
        assert rows is not None, f"{n} padded tokens still runs unblocked"
        assert rows % 32 == 0 and rows >= 32


def test_a_blocked_run_never_asks_for_more_than_the_budget(galaxy):
    """The block exists to keep the per-block matmul under OPM_Z_BUDGET_BYTES."""
    for n in REFUSED:
        rows = T.opm_row_block(n, C, D, n)
        assert rows * C * D * n * 2 <= T.OPM_Z_BUDGET_BYTES


def test_widths_that_fold_today_are_left_alone(galaxy):
    """No perf regression by construction: a width whose ask fits keeps the single matmul."""
    for n in FOLDS:
        assert unblocked_bytes(n) <= T.OPM_ROWBLOCK_BYTES
        assert T.opm_row_block(n, C, D, n) is None


def test_the_byte_gate_is_what_catches_them_not_the_token_gate(galaxy, monkeypatch):
    """Negative control: with only the token gate, every refused width runs unblocked again.

    This is the pre-fix tree. If it does not reproduce the bug, the test above is not reading
    the gate it claims to read.
    """
    monkeypatch.setattr(T, "OPM_ROWBLOCK_BYTES", 1 << 62)
    for n in REFUSED:
        assert T.opm_row_block(n, C, D, n) is None


def test_the_token_gate_still_works_on_its_own(galaxy, monkeypatch):
    """The byte gate is added to the token gate, not substituted for it."""
    monkeypatch.setattr(T, "OPM_ROWBLOCK_BYTES", 1 << 62)
    assert T.opm_row_block(galaxy + 32, C, D, 64) is not None


def test_the_threshold_sits_inside_the_measured_bracket():
    """Between the largest ask that folds and the smallest that is refused."""
    assert max(FOLDS.values()) < T.OPM_ROWBLOCK_BYTES < min(REFUSED.values())
