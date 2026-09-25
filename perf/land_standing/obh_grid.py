#!/usr/bin/env python3
"""What does `out_block_h = 5` cost on a grid it was not fitted on?

`_pair_proj_program_config` rounds `per_core_M` up to a multiple of 5 so that `out_block_h = 5`
divides it, and its own comment prices that on one grid: "120 of 130 cores instead of 130". The
literal 5 came from `perf/pf_matmul/proj_ab.py`, whose docstring says qb1's 13x10. qb2's p300c
is 11x10. This is the same shape of gap as the dividing-k census -- a constant fitted at one
geometry and shipped at another -- so price it at both, from the shipped functions rather than
by re-deriving the arithmetic.

Host only, no device: `_pair_proj_l1_rungs` and the per_core_M expression are pure.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tt_bio import tenstorrent as T                                    # noqa: E402

GRIDS = {"qb1 p150a 13x10": (13, 10), "qb2 p300c 11x10": (11, 10),
         "wh galaxy 8x9": (8, 9), "wh 13x10": (13, 10)}
# m_tiles for the pair track: batch * ceil(N_tok/32), per _pair_proj_config's own comment.
SHAPES = {"298 tok (gate cdk2x2)": 298, "384 tok": 384, "512 tok": 512,
          "704 tok": 704, "832 tok": 832, "1024 tok": 1024}


def per_core_M(m_tiles, num_cores, mult=5):
    q = -(-m_tiles // num_cores)                 # ceil
    return -(-q // mult) * mult                  # round up to a multiple of `mult`


print(f"{'grid':18s} {'tokens':>7} {'m_tiles':>8} {'pcM':>5} {'cores':>6} {'of':>4} "
      f"{'util':>7} {'pcM@1':>6} {'cores@1':>8} {'util@1':>7}  rungs")
rows = []
for gname, (gx, gy) in GRIDS.items():
    nc = gx * gy
    for sname, tok in SHAPES.items():
        m_tiles = tok * -(-tok // 32)            # batch rows x tile rows, per the comment
        if m_tiles < nc:
            continue
        p5 = per_core_M(m_tiles, nc, 5)
        c5 = -(-m_tiles // p5)
        p1 = per_core_M(m_tiles, nc, 1)
        c1 = -(-m_tiles // p1)
        rungs = T._pair_proj_l1_rungs(p5, 8)
        legal5 = any(h == 5 for h, _ in rungs)
        rows.append({"grid": gname, "tok": tok, "m_tiles": m_tiles, "pcM5": p5,
                     "cores5": c5, "nc": nc, "util5": c5 / nc, "pcM1": p1,
                     "cores1": c1, "util1": c1 / nc, "h5_legal": legal5})
        print(f"{gname:18s} {tok:>7} {m_tiles:>8} {p5:>5} {c5:>6} {nc:>4} "
              f"{c5 / nc:>6.1%} {p1:>6} {c1:>8} {c1 / nc:>6.1%}  "
              f"{[hw for hw in rungs[:3]]}{' h=5 ILLEGAL' if not legal5 else ''}")

worst = min(rows, key=lambda r: r["util5"])
print(f"\nworst core utilisation under the 5-rounding: {worst['util5']:.1%} "
      f"({worst['grid']}, {worst['tok']} tok, {worst['cores5']} of {worst['nc']})")
print(f"same shape without the rounding: {worst['util1']:.1%} "
      f"({worst['cores1']} of {worst['nc']})")
bad = [r for r in rows if not r["h5_legal"]]
print(f"shapes where out_block_h=5 is not even legal: {len(bad)}")
