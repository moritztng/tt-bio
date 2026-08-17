#!/usr/bin/env python3
"""Cut a real ColabFold alignment down to a series of lengths, keeping its depth.

Why this exists. A residue-count ladder built from `perf/size512/build_sweep_fixtures.py`
carries a 35-row a3m, and on Wormhole that measures the wrong thing: OpenFold3's ceiling is
set by an MSA-track activation whose size goes as tokens x MSA rows, so a shallow alignment
clears sizes a served fold cannot reach. Measured 2026-08-17 on GWH02: single sequence folds
768 aa, while 614 aa with an 8138-row alignment cannot allocate.

So the instrument has to vary length with a REAL alignment attached. Cutting one is the only
option that keeps depth fixed while length moves — a fresh search per length would move both,
and no two searches return the same number of rows.

Each output pair is a yaml naming the cut query plus an a3m written under the name tt-bio's
own MSA cache uses, `sha256(query)[:16].a3m`, so the fold resolves it with
`--msa_dir <cache> --msa_cache_only`: no search, no network, no second variable.

Usage:
    build_real_msa_fixtures.py SOURCE.a3m --out-fixtures DIR --out-cache DIR 448 512 544 576
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def cut(row: str, ncols: int) -> str | None:
    """The first `ncols` match columns of an a3m row, or None if the row is shorter.

    Lowercase is an insertion and does not count toward the column budget; a trailing
    insertion run is dropped so every row ends on a match column. Same rule as
    perf/size512/build_sweep_fixtures.py, which is what the token-count sweep uses.
    """
    out: list[str] = []
    seen = 0
    for ch in row:
        if ch.islower():
            if seen == 0 or seen >= ncols:
                continue
            out.append(ch)
        else:
            if seen >= ncols:
                break
            out.append(ch)
            seen += 1
    return "".join(out) if seen == ncols else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="a3m to cut (query on line 2)")
    ap.add_argument("--out-fixtures", type=Path, required=True)
    ap.add_argument("--out-cache", type=Path, required=True)
    ap.add_argument("--prefix", default="real")
    ap.add_argument("sizes", nargs="+", type=int)
    a = ap.parse_args()

    lines = a.source.read_text().rstrip("\n").split("\n")
    heads, rows = lines[0::2], lines[1::2]
    assert all(h.startswith(">") for h in heads), "a3m is not strict header/row pairs"
    assert rows[0] == rows[0].upper(), "query row carries insertions"

    a.out_fixtures.mkdir(parents=True, exist_ok=True)
    a.out_cache.mkdir(parents=True, exist_ok=True)
    for n in a.sizes:
        query = cut(rows[0], n)
        if query is None:
            raise SystemExit(f"source query is shorter than {n} match columns")
        kept = [(h, c) for h, r in zip(heads, rows) if (c := cut(r, n)) is not None]
        key = hashlib.sha256(query.encode()).hexdigest()[:16]
        (a.out_cache / f"{key}.a3m").write_text(
            "\n".join(x for pair in kept for x in pair) + "\n")
        (a.out_fixtures / f"{a.prefix}_{n}.yaml").write_text(
            f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {query}\n")
        print(f"{a.prefix}_{n}: depth={len(kept)} cache={key}.a3m")


if __name__ == "__main__":
    main()
