"""K4 negative control at the construction level: the pick must be IDENTICAL outside the band.

 is d, so a post-import flag flip needs  between
arms. Without it every length reads "unchanged" and the control passes for the wrong reason.

The band branch is `256 < q_len <= 384 and 256 < k_len <= 384`, so every other length is
untouched by construction. This asserts that rather than inferring it, and prints the pick the
fold actually runs at each length so the 298 aa cell's k move is visible as a number.
"""
import tt_bio.tenstorrent as TT

LENS = [128, 256, 288, 298, 320, 352, 384, 512, 640, 768, 1024]
print("len  padded   pick_off        pick_on         moved")
moved, same = [], []
for n in LENS:
    TT._SDPA_BAND_DIV_K = False
    TT._sdpa_chunks_shipped.cache_clear()
    off = TT._sdpa_chunks_shipped(n, n)
    TT._SDPA_BAND_DIV_K = True
    TT._sdpa_chunks_shipped.cache_clear()
    on = TT._sdpa_chunks_shipped(n, n)
    tag = "MOVED" if off != on else "-"
    (moved if off != on else same).append(n)
    print(f"{n:4d} {TT._padded_sdpa_len(n):6d}   {str(off):14s}  {str(on):14s}  {tag}")
TT._SDPA_BAND_DIV_K = False
print(f"\nmoved: {moved}")
print(f"unchanged (negative control): {same}")
assert 512 in same and 768 in same and 1024 in same and 256 in same, "a length outside the band moved"
assert 298 in moved and 320 in moved and 384 in moved, "the band cell did not move"
assert 288 in same and 352 in same, "288/352 must keep 64, they have no clearing divisor"
print("CONTROL PASS: only 256 < n <= 384 with a clearing divisor moves; 512/768/1024 identical.")
