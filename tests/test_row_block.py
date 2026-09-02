"""`row_block` sizes every row-blocked section, so its two call sites are pinned here.

No device: `row_block` is arithmetic, and what it decides -- how many blocks a design is
cut into -- is exactly what has to be pinned. A budget that quietly starts blocking a
128-residue design would change the op sequence under every parity fixture in the suite.
"""
import pytest

from tt_bio.rfd3.model import _ATOM_PAIR_LIVE_ROWS
from tt_bio.rfd3.tiles import TILE, align_tile
from tt_bio.tenstorrent import OPM_CHUNK_SIZE, OPM_Z_BUDGET_BYTES, row_block

WORMHOLE_BUDGET = (12 * 2 ** 30) // 4          # atom_pair_budget_bytes() with no device open


def _rows(atoms, budget=WORMHOLE_BUDGET):
    return row_block(_ATOM_PAIR_LIVE_ROWS * align_tile(atoms) * TILE * 2, budget)


def _peak(atoms, budget=WORMHOLE_BUDGET):
    return min(_rows(atoms, budget), atoms) * _ATOM_PAIR_LIVE_ROWS * align_tile(atoms) * TILE * 2


@pytest.mark.parametrize("atoms", [419, 1200, 2896, 4373, 4558, 9000, 12000])
def test_peak_stays_under_budget(atoms):
    assert _peak(atoms) <= WORMHOLE_BUDGET


@pytest.mark.parametrize("atoms", [419, 1200, 2290, 2770])
def test_small_designs_are_one_block(atoms):
    # RFD3 runs about 8.9 atoms per residue (4373 atoms at 490), so the budget puts the
    # first cut at roughly 325 residues: 128 and 256 take the unblocked path, byte for
    # byte, and only sizes at and above the old cap are blocked at all.
    assert _rows(atoms) >= atoms


def test_the_old_wormhole_cap_is_blocked():
    # 4558 atoms is the 414-residue rung that threw on a 1,334,874,112 B request.
    assert _rows(4558) < 4558


def test_a_block_is_tile_aligned_and_never_empty():
    for atoms in (1, 31, 32, 33, 100000):
        r = _rows(atoms)
        assert r >= TILE and r % TILE == 0


def test_a_bigger_part_blocks_later():
    p150a = (31875 * 2 ** 20) // 4
    assert _rows(4558, p150a) > _rows(4558, WORMHOLE_BUDGET)


# --- OuterProductMean --------------------------------------------------------------------
# The shared helper replaced an inline formula at this site, and this site is on the trunk of
# every model with an MSA. "Same arithmetic" is cheap to assert and cheap to check, so it is
# checked: the helper has to agree with the expression it replaced on every shape the site
# can present, not just on the one that motivated the refactor.


def _opm_inline(C, D, J):
    """The expression OuterProductMean carried before it called `row_block`."""
    per_row = C * D * J * 2
    return max(32, min(OPM_CHUNK_SIZE, (OPM_Z_BUDGET_BYTES // max(per_row, 1)) // 32 * 32))


@pytest.mark.parametrize("C", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("D", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("J", [64, 128, 285, 512, 992, 1536, 4096])
def test_opm_row_block_matches_the_formula_it_replaced(C, D, J):
    assert row_block(C * D * J * 2, OPM_Z_BUDGET_BYTES, cap=OPM_CHUNK_SIZE) == _opm_inline(C, D, J)
