#!/usr/bin/env python3
"""Deepen an existing sweep a3m so MSA depth can be swept with token count held fixed.

The customer chart says nothing about MSA depth, and depth is a first-order cost driver that the
campaign fixtures hold at 35 rows while a real ColabFold search on a 580 aa target returns
thousands (3553-11346 rows measured on 9yio's three chains). A measurement at depth 35 is therefore
a LOWER BOUND on the customer's fold, and this builds the arms that bracket it.

Rows are tiled from the 34 non-query rows. `protenix_data.py:170` dedups identical alignment rows,
so a plain tile would collapse straight back to 35 -- each copy gets one column mutated to a
distinct residue so it survives the dedup. Content is irrelevant here: the MSA stack's cost is set
by the row count and the token width, not by which homolog a row is. The query row is copied
byte-for-byte, because `seed_msa_cache` asserts it equals the target's own sequence.
"""
import sys
from pathlib import Path

FIX = Path(__file__).resolve().parents[1] / "size512" / "fixtures"
AA = "ACDEFGHIKLMNPQRSTVWY"


def build(size: int, depth: int) -> Path:
    src = (FIX / f"cdk2x2_{size}.a3m").read_text().rstrip("\n").split("\n")
    heads, rows = src[0::2], src[1::2]
    assert all(h.startswith(">") for h in heads), "a3m is not strict header/row pairs"
    out_h, out_r, seen = [heads[0]], [rows[0]], {rows[0]}
    i = 0
    while len(out_r) < depth:
        k = 1 + i % (len(rows) - 1)          # never re-tile the query row
        copy = i // (len(rows) - 1)
        r = list(rows[k])
        if copy:                              # mutate one column so the dedup keeps this row
            pos = copy % len(r)
            r[pos] = AA[(AA.find(r[pos].upper()) + copy) % len(AA)] if r[pos].upper() in AA else AA[copy % len(AA)]
        s = "".join(r)
        i += 1
        if s in seen:
            continue
        seen.add(s)
        out_h.append(f"{heads[k]}_d{copy}")
        out_r.append(s)
    assert out_r[0] == rows[0], "query row was modified"
    txt = "\n".join(f"{h}\n{r}" for h, r in zip(out_h, out_r)) + "\n"
    p = FIX / f"cdk2x2_{size}d{depth}.a3m"
    p.write_text(txt)
    (FIX / f"cdk2x2_{size}d{depth}.yaml").write_text((FIX / f"cdk2x2_{size}.yaml").read_text())
    print(f"  size={size} depth={txt.count('>')} -> {p.name}")
    return p


if __name__ == "__main__":
    size = int(sys.argv[1])
    for d in (int(x) for x in sys.argv[2:]):
        build(size, d)
