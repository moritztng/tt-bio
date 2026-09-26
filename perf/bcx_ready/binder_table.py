"""28 Sep binder table, verified against BindCraft 2s own two paddings rather than re-derived.

Closed form says the largest drawable binder is 32*floor((C-L)/32). This does not trust that: it
calls padmap.tokens (pad_design_chains + padded_prediction_length on real Protein objects) at
B_max and at B_max+1 and asserts the first fits the ceiling and the second overflows it.
"""
import sys, json
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/ttuser/bcx_e2e/bc2")
from padmap import tokens

def bmax(L, C):
    return 32 * ((C - L) // 32) if C > L else 0

rows = []
for C in (512, 544):
    for L in (80, 100, 129, 160, 161, 192, 200, 224, 256, 288, 289, 320, 352, 384, 416, 448, 480):
        B = bmax(L, C)
        if B <= 0:
            rows.append(dict(ceiling=C, target=L, longest=0, verified="refused"))
            print("C=%d L=%4d longest=%4d REFUSED" % (C, L, B), flush=True)
            continue
        t_fit = int(tokens(L, B))
        t_over = int(tokens(L, B + 1))
        ok = (t_fit <= C) and (t_over > C)
        rows.append(dict(ceiling=C, target=L, longest=B, tokens_at_max=t_fit,
                         tokens_at_max_plus1=t_over, verified="OK" if ok else "FAIL"))
        print("C=%d L=%4d longest=%4d tokens(B)=%4d tokens(B+1)=%4d %s"
              % (C, L, B, t_fit, t_over, rows[-1]["verified"]), flush=True)
json.dump(rows, open("/tmp/bindertable.json", "w"), indent=1)
print("WROTE", sum(1 for r in rows if r["verified"] == "FAIL"), "FAIL rows")
