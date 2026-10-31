"""Triangle attention's fused path at any pair width: the matmul block resolver and triatt_ln's column groups.

`_mm_block_at` gives `_MM_DEFAULT` (byte-identical to no config) to a contraction of 8+ tiles that the swept table
has no row for, and leaves every tabled kt exactly as it was. `triatt_ln._plan` takes column groups from the divisors
of the core count, so a plan found at 1, 2 or 4 groups is unchanged. Op timings and parity: perf/mol1_triattn/."""
import pytest

pytest.importorskip("ttnn")
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import triatt_ln as TL  # noqa: E402


def test_tabled_keys_unchanged():
    for key, blk in T._MM_BLOCK.items():
        assert T._mm_block_at(*key) is blk
    for kt, nt in ((12, 1), (8, 1), (12, 24), (4, 1), (2, 4)):      # tabled kt, no row: stays unconfigured
        assert T._mm_block_at(kt, nt) is None


def test_untabled_wide_kt_gets_the_default(monkeypatch):
    for kt, H in ((10, 10), (14, 14)):                                # Mol-1 c_z 320 and 448
        for nt in (3 * H, H, 4 * H, 4 * H + 1):                      # qkv, gate and out, qkvg, qkvgb
            assert T._mm_block_at(kt, nt) is T._MM_DEFAULT
    assert T._mm_block_at(6, 18) is None                              # under 8 tiles the default pads K
    monkeypatch.setattr(T, "_MM_DEFAULT_WIDE", False)
    assert T._mm_block_at(10, 30) is None


@pytest.mark.parametrize("ncores", [130, 64, 56, 110])
def test_plan_groups_divide_the_grid_and_keep_old_plans(monkeypatch, ncores):
    for lcf in (2 ** 18, 2 ** 19, 700_000, 2 ** 20, 1_400_000):
        monkeypatch.setattr(TL, "_lcf", lambda dev, lcf=lcf: lcf)
        for kt, nw, tr in ((8, 33, False), (8, 33, True), (12, 49, False), (10, 41, False), (14, 57, False),
                           (14, 57, True)):
            plan = TL._plan(kt, nw, tr, None, ncores)
            old = next(((g, -(-nw // g), nb) for g in (1, 2, 4) if not ncores % g
                        for nb in ((2, 1) if tr else (0,))
                        if TL._cb_bytes(kt, -(-nw // g), nb, TL._share(g, tr, 1)) <= lcf), None)
            if old is not None:
                assert plan == old
            if plan is not None:
                assert ncores % plan[0] == 0 and plan[0] <= nw
