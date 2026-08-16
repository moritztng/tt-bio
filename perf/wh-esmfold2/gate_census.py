"""Per-size gate census for ESMFold2, on the CPU, for the three grids that matter.

Section 8 step 7 of state/wh-perf-esmfold2.md asks for what actually fired beside every wall in
the size sweep. The census half needs no card: every gate here is a pure function of the grid and
the padded length, and `_apply_grid_thresholds`' only device call is stubbed to the per-core L1 the
audit MEASURED on the Galaxy (1,466,080 B), the same substitution `neutrality_cpu.py` makes.

The thing being looked for is memory `tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa`: a gate
that silently switches off past `SEQ_LEN_MORE_CHUNKING` (608 on this machine) with nothing in the
log. Printing the census per size is how that stops being silent.
"""
import json, sys
sys.path.insert(0, "/home/ttuser/.coworker/wt/wh-perf-esmfold2")
import ttnn
ttnn.get_max_worker_l1_unreserved_size = lambda *a, **k: 1466080
import tt_bio.tenstorrent as T
import tt_bio.esmc as EC

T.set_fast_mode(True)
PAD = 32
SIZES = (128, 256, 298, 320, 512, 640, 768, 1024)
CZ = 256

def census(L):
    """What the shipped gates return at this padded length. `split` is the lever A predicate from
    esmc.SwiGLUFFN.__call__, evaluated for a 4D pair input, which is the trunk's transition."""
    small = getattr(T, "_IS_SMALL_GRID", False)
    split = bool(EC._SPLIT_SWIGLU
                 and (EC._SPLIT_SWIGLU_SMALL_GRID or not small)
                 and L >= EC.SPLIT_SWIGLU_MIN_SEQ)
    lo, hi = EC.PAIR_FFN_ROW_BLOCK_SEQ
    return {
        "trimul_l1_max_seq": T._trimul_l1_max_seq(),
        "trimul_resident": "L1" if L <= T._trimul_l1_max_seq() else "DRAM",
        "trimul_chunk_size": T._trimul_chunk_size(L, CZ),
        "pair_row_tile": T.pair_row_tile(L),
        "seq_len_more_chunking": T.SEQ_LEN_MORE_CHUNKING,
        "more_chunking_fires": bool(T.SEQ_LEN_MORE_CHUNKING and L > T.SEQ_LEN_MORE_CHUNKING),
        "small_grid_seq_tile": T.SMALL_GRID_SEQ_TILE,
        "split_fires": split,
        "row_block_fires": bool(split and EC._PAIR_FFN_ROW_BLOCK and lo <= L <= hi),
        "row_block": EC._PAIR_FFN_ROW_BLOCK,
    }

out = {"per_core_l1_B": 1466080, "pad_multiple": PAD, "sizes": list(SIZES), "grids": {}}
for grid, name in [((13, 10), "qb1 13x10"), ((11, 10), "qb2 11x10"), ((8, 9), "galaxy 8x9")]:
    T.COMPUTE_GRID_MAIN = grid
    T._apply_grid_thresholds(grid)
    small = getattr(T, "_IS_SMALL_GRID", False)
    for lever_a in ([False, True] if small else [False]):
        EC.set_split_swiglu_small_grid(lever_a)
        key = name + (" +A" if lever_a else "")
        rows = {}
        for L0 in SIZES:
            L = -(-L0 // PAD) * PAD
            rows[L0] = dict(padded=L, **census(L))
        out["grids"][key] = {"cores": grid[0] * grid[1], "small_grid": small, "rows": rows}
        EC.set_split_swiglu_small_grid(False)

hdr = "aa->pad  trimul(thr,resid,width)  pair_row_tile  morechunk  split  rowblk"
for key, g in out["grids"].items():
    print("\n=== %s  cores=%d  small_grid=%s" % (key, g["cores"], g["small_grid"]))
    print(hdr)
    for L0, r in g["rows"].items():
        print("%4d->%4d  (%4d,%-4s,%3d)  %13d  %9s  %5s  %6s" % (
            L0, r["padded"], r["trimul_l1_max_seq"], r["trimul_resident"],
            r["trimul_chunk_size"], r["pair_row_tile"], r["more_chunking_fires"],
            r["split_fires"], r["row_block_fires"]))

# The thing memory tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa is about: a gate whose value
# at 512 differs from its value at 640/768/1024 on the grid JapanFold serves.
wh = out["grids"]["galaxy 8x9 +A"]["rows"]
dark = []
for k in ("trimul_resident", "trimul_chunk_size", "split_fires", "row_block_fires", "pair_row_tile"):
    if wh[512][k] != wh[640][k] or wh[512][k] != wh[1024][k]:
        dark.append((k, wh[512][k], wh[640][k], wh[1024][k]))
print("\nGates whose value at 512 does not hold to 640/1024 on 8x9 with lever A on:")
print("  none" if not dark else "\n".join("  %s: 512=%s 640=%s 1024=%s" % d for d in dark))
out["gates_changing_above_512_wh"] = dark
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("\nwrote " + sys.argv[1])
