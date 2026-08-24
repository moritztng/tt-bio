"""p119b -- size the fix. If plan() padded the query axis UP to a multiple of Q instead of
refusing, what would it cost?

`plan()` today returns None when `nb_rows % q_block`. The alternative is to pad nb_rows up to the
next multiple of q_block, which the arm can already do -- it pads to a tile boundary the same way
and slices the extra rows off the output. That makes the arm live at every atom count, but the
blocked chain costs nb*q_block*U, so padding the query axis up inflates the cost by exactly
padded_to_Q / tile(atoms). Cheaper than dense only while that inflation times U/n_key stays < 1.

Pure arithmetic, no device, no model. Reports the inflation distribution so the next pass can price
the change instead of guessing at it.
"""
import json
import pathlib
import statistics
import sys

TILE = 32
Q = 1216
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p119/padup.json")
# Measured on R4: median union 3296 of 6080 padded keys (state doc), and the bucket picks that
# p117 observed out of sample. Used only to say where the pad-up break-even sits, not as a
# prediction for another atom count -- U is data-dependent and per-target.
R4_U_FRAC = 3296 / 6080


def tile(n):
    return -(-n // TILE) * TILE


rows = []
for a in range(256, 12001):
    base = tile(a)
    padded = -(-base // Q) * Q
    rows.append((a, base, padded, padded / base))

infl = [r[3] for r in rows]
live_today = [r for r in rows if r[1] % Q == 0]
# Break-even: pad-up beats dense while inflation * U/n_key < 1.
break_even_infl = 1.0 / R4_U_FRAC
would_win = [r for r in rows if r[3] < break_even_infl]

res = {
    "q_block": Q,
    "span_atoms": [256, 12000],
    "n_atom_counts": len(rows),
    "live_today": len(live_today),
    "live_today_fraction": round(len(live_today) / len(rows), 5),
    "padup_inflation": {
        "min": round(min(infl), 4), "median": round(statistics.median(infl), 4),
        "mean": round(statistics.fmean(infl), 4), "max": round(max(infl), 4),
        "p90": round(sorted(infl)[int(0.90 * len(infl))], 4),
    },
    "r4_u_fraction": round(R4_U_FRAC, 4),
    "break_even_inflation": round(break_even_infl, 4),
    "n_below_break_even": len(would_win),
    "fraction_below_break_even": round(len(would_win) / len(rows), 5),
    "note": ("Inflation is padded_to_Q / tile(atoms). Below ~1216 atoms every target pads to a "
             "single 1216 block, which is why the max is large. The U fraction is R4's and does "
             "not transfer; this brackets the fix, it does not price another target."),
    "examples": {str(a): {"padded_tile": b, "padded_to_Q": p, "inflation": round(i, 3)}
                 for a, b, p, i in rows if a in (1350, 2000, 3000, 4000, 5000, 6051, 8000, 10000)},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
