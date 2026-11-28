"""The row grid that keeps the confidence network to a handful of device shapes.

The confidence model runs one graph per pose, with the ligand, ligand-receptor and
ligand-atom edge sets rebuilt from a distance cutoff each time, so their row counts are a
property of the pose. A device program is compiled per tensor shape, which made every pose of
every ligand a new program: 1,023 program-cache entries after one complex and 1,838 after a
second one on the same receptor, measured on anx-glx (state/dfd/SHAPES.md).

:func:`tt_bio.diffdock.graph.grid` rounds those axes onto a fixed ladder instead. These tests
are the correctness half, and they are card-free: the dead rows a wider axis adds must change
no number, which is a property of the host convolution the device layer mirrors.
"""

from __future__ import annotations

import torch

from tt_bio.diffdock.conv import TensorProductConvLayer
from tt_bio.diffdock.graph import GRID_RATIO, grid, grid_count, pad_count


def test_grid_is_a_ladder_of_tile_multiples():
    pts = sorted({grid(n) for n in range(1, 4000)})
    assert all(p % 32 == 0 for p in pts)
    assert pts == sorted(set(pts))
    # 1 to 4,000 rows -- every count a confidence edge set reaches -- lands on a few dozen
    # shapes, where rounding to 32 alone would give 125.
    assert len(pts) < 40


def test_grid_never_loses_a_row_and_never_wastes_more_than_the_ratio():
    for n in range(1, 4000):
        g = grid(n)
        assert g >= n
        assert g <= max(32, int(n * GRID_RATIO) + 32)
        assert grid_count(n) > n          # a spare row for the pad node, as pad_count leaves
        assert grid_count(n) >= pad_count(n)


def test_small_axes_keep_todays_padding():
    """Below the first geometric step the grid is the multiple of 32 the port already used, so
    a small ligand's shapes do not move and neither does its cost."""
    for n in range(1, 320):
        assert grid(n) == ((n + 31) // 32) * 32


def _layer(seed: int = 0) -> TensorProductConvLayer:
    torch.manual_seed(seed)
    return TensorProductConvLayer("8x0e + 2x1o", "1x0e + 1x1o", "8x0e + 2x1o", 16,
                                  residual=False).double().eval()


def test_dead_rows_change_no_value():
    """A gridded axis is the same arithmetic in float64: pad edges point at the pad node and
    carry mask 0, pad nodes carry zero features, and the mean divides by the mask sum."""
    torch.manual_seed(1)
    lyr = _layer()
    n, e = 37, 211
    x = torch.randn(n, lyr.in_irreps.dim, dtype=torch.float64)
    ei = torch.randint(0, n, (2, e))
    ea = torch.randn(e, 16, dtype=torch.float64)
    sh = torch.randn(e, lyr.sh_irreps.dim, dtype=torch.float64)
    mask = torch.ones(e, dtype=torch.float64)
    with torch.no_grad():
        tight = lyr(x, ei, ea, sh, edge_mask=mask)

        n_g, e_g = grid_count(n), grid(e)
        xg = torch.nn.functional.pad(x, (0, 0, 0, n_g - n))
        eig = torch.cat([ei, torch.full((2, e_g - e), n_g - 1)], 1)
        eag = torch.nn.functional.pad(ea, (0, 0, 0, e_g - e))
        shg = torch.nn.functional.pad(sh, (0, 0, 0, e_g - e))
        mg = torch.nn.functional.pad(mask, (0, e_g - e))
        wide = lyr(xg, eig, eag, shg, edge_mask=mg)

    assert wide.shape[0] == n_g and n_g > n and e_g > e
    assert float(wide[n:].abs().max()) == 0.0          # the pad node gets nothing but zeros
    # Bit-exact at most sizes and one float64 ulp apart at the rest: the dead rows add exact
    # zeros to the scatter, which only reassociates the sum.
    dev = (wide[:n] - tight).abs().max() / tight.abs().max()
    assert float(dev) < 1e-15
