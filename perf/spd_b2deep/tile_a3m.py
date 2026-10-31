#!/usr/bin/env python3
"""Tile a deep single-chain a3m to the length of a cdk2x2_<N> fixture (CDK2 repeated to N aa).

The page fixtures carry 35 alignment rows; served Boltz-2 inputs carry up to 8192. This takes the
real colabfold CDK2 alignment (perf/capacity/cdk2_1hcl_colabfold_deep.a3m.gz, 8833 rows over 298
match columns) and repeats every row's match columns the way the fixture repeats the sequence, so
the query row equals the fixture sequence and the depth is the served one.

    python3 tile_a3m.py --length 748 --out deep_748.a3m
"""
import argparse, gzip, re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "capacity" / "cdk2_1hcl_colabfold_deep.a3m.gz"


def columns(row):
    """Match columns of an a3m row, each carrying the lowercase insertion before it."""
    return re.findall(r"[a-z]*[A-Z\-]", row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    name, rows = None, []
    for line in gzip.open(SRC, "rt"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            name = line
        elif line and name is not None:
            rows.append((name, line))
            name = None
    with open(a.out, "w") as f:
        for name, row in rows:
            cols = columns(row)
            assert len(cols) == 298, (name, len(cols))
            tiled = (cols * (-(-a.length // len(cols))))[: a.length]
            f.write(f"{name}\n{''.join(tiled)}\n")
    print(f"{a.out}: {len(rows)} rows x {a.length} columns")


if __name__ == "__main__":
    main()
