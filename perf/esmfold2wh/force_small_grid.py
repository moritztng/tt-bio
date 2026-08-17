"""Run tt_bio.main with the small-grid (Wormhole) L1 budgets forced on.

A 13x10 Blackhole leaves SMALL_GRID_MSA_TILE_AREA at 0 and _IS_SMALL_GRID False, so the
MSA encoder allocates L*M*c in one block and a 605 aa fold with an 8192-deep MSA dies on a
5,242,880,000 B request. Forcing the small-grid branch gives the card the same budgets --
and the same bf16 SwiGLU projection (esmc.py:450-460) -- that the 72-core Galaxy runs,
which is the part the depth cap is gated to. That makes a deep-MSA fold at the cap own
length scoreable on the card that is actually free.

Applied identically to both arms of a depth A/B, so the only difference between them stays
--max_msa_seqs. This is a harness, not a code change: nothing is committed to the engine.
"""
import sys

import tt_bio.tenstorrent as _T

_orig = _T._apply_grid_thresholds


def _forced(_grid):
    _orig((8, 9))  # take the small-grid branch whatever the real grid is


_T._apply_grid_thresholds = _forced

from tt_bio.main import cli  # noqa: E402  (must import after the patch)

if __name__ == "__main__":
    sys.exit(cli())
