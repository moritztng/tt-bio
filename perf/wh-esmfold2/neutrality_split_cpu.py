"""Blackhole neutrality of SPLIT_SWIGLU_SMALL_GRID once it DEFAULTS to True.

While the flag defaulted False, "Blackhole is unaffected" was true for a boring reason: nobody set
it. Now that it ships True, the claim has to rest on the guard instead, and the guard is what broke
once already (7f9335fc: the fc1 dtype line read the flag without the _IS_SMALL_GRID test and cost
Blackhole 6.1 % plus a different CIF).

So assert the two things the flag can reach -- the split predicate and the fc1 output dtype -- are
identical on a >= 110-core grid with the flag False and True, and differ only on 8x9. No device:
the one device call inside _apply_grid_thresholds is stubbed to the per-core L1 the audit measured.
"""
import sys
sys.path.insert(0, "/home/ttuser/.coworker/wt/wh-perf-esmfold2")
import ttnn
ttnn.get_max_worker_l1_unreserved_size = lambda *a, **k: 1466080
import tt_bio.tenstorrent as T
import tt_bio.esmc as EC

T.set_fast_mode(True)
SIZES = (128, 256, 298, 320, 512, 640, 768, 1024)
PAD = 32


def reachable(L):
    """(split predicate, fc1 output dtype) exactly as SwiGLUFFN evaluates them."""
    small = getattr(T, "_IS_SMALL_GRID", False)
    split = bool(EC._SPLIT_SWIGLU
                 and (EC._SPLIT_SWIGLU_SMALL_GRID or not small)
                 and L >= EC.SPLIT_SWIGLU_MIN_SEQ)
    dt = (EC._dtype(ttnn.bfloat16)
          if (EC._SPLIT_SWIGLU_SMALL_GRID and small) else EC._dtype())
    lo, hi = EC.PAIR_FFN_ROW_BLOCK_SEQ
    return (split, str(dt), bool(split and EC._PAIR_FFN_ROW_BLOCK and lo <= L <= hi))


print("shipped default SPLIT_SWIGLU_SMALL_GRID =", EC.SPLIT_SWIGLU_SMALL_GRID)
bad = []
for grid, name in [((13, 10), "qb1 13x10"), ((11, 10), "qb2 11x10"), ((8, 9), "galaxy 8x9")]:
    T.COMPUTE_GRID_MAIN = grid
    T._apply_grid_thresholds(grid)
    small = getattr(T, "_IS_SMALL_GRID", False)
    seen = {}
    for flag in (False, True):
        EC.set_split_swiglu_small_grid(flag)
        seen[flag] = {L: reachable(-(-L // PAD) * PAD) for L in SIZES}
    EC.set_split_swiglu_small_grid(EC.SPLIT_SWIGLU_SMALL_GRID)
    same = seen[False] == seen[True]
    print("%-12s cores=%3d small_grid=%-5s flag_changes_nothing=%s"
          % (name, grid[0] * grid[1], small, same))
    if not small and not same:
        bad.append(name)
    if small and same:
        bad.append(name + " (flag is inert where it is supposed to be live)")

assert not bad, "NOT NEUTRAL: " + ", ".join(bad)
print("\nOK: the flag is unreadable on every >= 110-core grid and live only on 8x9,")
print("    with the shipped default now True.")
