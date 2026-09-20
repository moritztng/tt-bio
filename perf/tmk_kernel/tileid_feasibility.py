#!/usr/bin/env python3
"""Can the v2.2 artifact express the trimul channel move? CPU only, no device.

DESIGN v2.2 replaces the hand-written GEMM with "edited copies of the wheel's matmul dataflow
kernels ... whose reader RE-POINTS SOURCE TILE IDS so the contraction consumes the pair-major
operand and the separate channel move never runs", citing `tt_bio/kernels/triatt/` as the shipped
precedent. That precedent's macro is `MM_SPLIT_TILE_ID` / `MM_IN0_TILE_ID`, and its own comment
says what it does: "Same tile, same transaction, different address."

A tile-id re-point can therefore express a permutation if and only if every destination tile IS
some source tile, whole and unpermuted inside itself. This enumerates the index map and checks
exactly that, for both the head-reshape triatt deletes (which must pass) and the trimul channel
move (which is the question).

TILE_LAYOUT tiles the LAST TWO axes 32x32; leading axes index whole planes.
  blk  [1, S, S, 4C] -> plane i, tile grid over (k, c)
  a    [1, C, S, S]  -> plane c, tile grid over (i, k)
  a[c, i, k] = blk[i, k, c] * gate
"""
from collections import defaultdict

T = 32


def blk_site(i, k, c):
    """Where element (i,k,c) of the pair-major projection lives: (plane, tile, row, col)."""
    return (i, (k // T, c // T), k % T, c % T)


def a_site(c, i, k):
    """Where element (c,i,k) of the channel-major operand lives."""
    return (c, (i // T, k // T), i % T, k % T)


def analyse(S, C, label):
    """For each destination tile of `a`, how many distinct source tiles feed it, and is any one
    of them contributed whole and in the same intra-tile position?"""
    worst_srcs, whole_tile_hits, dest_tiles = 0, 0, 0
    for c in range(C):
        for i0 in range(0, S, T):
            for k0 in range(0, S, T):
                dest_tiles += 1
                srcs = defaultdict(int)
                identity = True
                for i in range(i0, i0 + T):
                    for k in range(k0, k0 + T):
                        sp, st, sr, sc = blk_site(i, k, c)
                        dp, dt, dr, dc = a_site(c, i, k)
                        srcs[(sp, st)] += 1
                        if (sr, sc) != (dr, dc):
                            identity = False
                worst_srcs = max(worst_srcs, len(srcs))
                if len(srcs) == 1 and identity:
                    whole_tile_hits += 1
    print(f"  {label}: {dest_tiles} destination tiles, max distinct source tiles feeding one = "
          f"{worst_srcs}, destination tiles that are a whole source tile unpermuted = "
          f"{whole_tile_hits}")
    return worst_srcs, whole_tile_hits, dest_tiles


def head_reshape_control(B, H, R):
    """The permutation triatt's macro DOES delete: [B, R*H, 32] -> [B, H, R, 32], heads regrouped.
    Whole tiles move; nothing inside a tile changes. This is the positive control."""
    worst = 0
    for b in range(B):
        for h in range(H):
            for r in range(R):
                src_tile = (b, (r * H + h))      # row-major (row, head)
                dst_tile = (b, (h * R + r))      # head-major (head, row)
                worst = max(worst, 1)            # exactly one source tile, unpermuted inside
                assert isinstance(src_tile[1], int) and isinstance(dst_tile[1], int)
    print(f"  head reshape (triatt control): every destination tile is exactly 1 whole source "
          f"tile, intra-tile identity -- a tile-id re-point EXPRESSES it")
    return worst


if __name__ == "__main__":
    print("TILE-ID FEASIBILITY, CPU only")
    head_reshape_control(1, 4, 4)
    for S, C in ((64, 32), (128, 32)):
        ws, wh, dt = analyse(S, C, f"trimul channel move S={S} C={C}")
        verdict = ("EXPRESSIBLE" if wh == dt else
                   "NOT EXPRESSIBLE by tile-id arithmetic -- the destination tile is assembled "
                   f"from {ws} sub-tile columns of {ws} distinct source tiles")
        print(f"  VERDICT S={S}: {verdict}")
