"""K4 negative control at the construction level: the pick must be IDENTICAL outside the band,
and identical everywhere on Wormhole.

`_sdpa_chunks_shipped` is `lru_cache`d, so a post-import flag flip needs `cache_clear()` between
arms. Without it every length reads "unchanged" and the control passes for the wrong reason.

Two arms, because K4 ships with two conditions and each one can fail on its own:

  1. The BAND. The branch is `256 < q_len <= 384 and 256 < k_len <= 384`, so every other length is
     untouched by construction. This asserts that rather than inferring it, and prints the pick the
     fold actually runs at each length so the 298 aa cell's k move is visible as a number. The
     length list covers every release-gate size-ladder rung (256/512/640/768/896/1024/1088), which
     is what makes "a green gate is no-regression evidence about K4's neighbours, and is not
     evidence about K4" a checked statement instead of a claim about the source.

  2. The GRID. K4 is scoped by `and not _IS_SMALL_GRID`, which is the Blackhole scope. Wormhole
     keeps the old k=64 pick because the band's q half looked like a win on a Wormhole op screen
     and the fold came back 0.9333x. Nothing asserted that scope until this arm: with the flag on
     and a small grid, NO length may move.
"""
import tt_bio.tenstorrent as TT

LENS = [128, 256, 288, 298, 320, 352, 384, 512, 640, 768, 896, 1024, 1088]
GATE_RUNGS = [256, 512, 640, 768, 896, 1024, 1088]


def pick(n, flag, small_grid):
    TT._SDPA_BAND_DIV_K = flag
    TT._IS_SMALL_GRID = small_grid
    TT._sdpa_chunks_shipped.cache_clear()
    return TT._sdpa_chunks_shipped(n, n)


print("=== arm 1: Blackhole (_IS_SMALL_GRID False) ===")
print("len  padded   pick_off        pick_on         moved")
moved, same = [], []
for n in LENS:
    off, on = pick(n, False, False), pick(n, True, False)
    (moved if off != on else same).append(n)
    print(f"{n:4d} {TT._padded_sdpa_len(n):6d}   {str(off):14s}  {str(on):14s}  "
          f"{'MOVED' if off != on else '-'}")
print(f"\nmoved: {moved}")
print(f"unchanged (negative control): {same}")

print("\n=== arm 2: Wormhole (_IS_SMALL_GRID True) — the scope, which nothing tested before ===")
wh_moved = [n for n in LENS if pick(n, False, True) != pick(n, True, True)]
print(f"moved on a small grid: {wh_moved}  (must be empty)")
print(f"in-band pick on WH, 320 aa: {pick(320, True, True)}  vs BH {pick(320, True, False)}")

TT._SDPA_BAND_DIV_K = False
TT._IS_SMALL_GRID = False
TT._sdpa_chunks_shipped.cache_clear()

assert 512 in same and 768 in same and 1024 in same and 256 in same, "a length outside the band moved"
assert 298 in moved and 320 in moved and 384 in moved, "the band cell did not move"
assert 288 in same and 352 in same, "288/352 must keep 64, they have no clearing divisor"
assert not [n for n in GATE_RUNGS if n in moved], (
    f"a release-gate size-ladder rung is inside K4's band: {[n for n in GATE_RUNGS if n in moved]} "
    "-- the gate would then be evidence about K4 and this control's premise is wrong")
assert not wh_moved, f"K4 is not scoped to Blackhole: {wh_moved} moved on a small grid"
print("\nCONTROL PASS: only 256 < n <= 384 with a clearing divisor moves, no gate rung is in the "
      "band, and a small grid moves nothing at all.")
