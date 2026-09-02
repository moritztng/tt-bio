#!/usr/bin/env python3
"""Walk one MSA-on ladder for OpenFold3 with depth held exactly constant.

The size ceiling this feeds is a PRODUCT of tokens and alignment depth, so a ladder that
only moves tokens has to hold depth fixed to the row. Cutting or tiling an a3m by MATCH
COLUMNS does that exactly: every row keeps its identity, only its length changes, so the
row count is invariant by construction and is asserted below rather than hoped for.

The parent alignment is the deepest real ColabFold search on the Galaxy (14190 rows). Its
query is CDK2 tandem-repeated, which is what makes lengths above the parent reachable at
all: the unit is cut once and then tiled, the same construction
``perf/size512/build_sweep_fixtures.py`` uses for prot300. The folded structure is
meaningless and this fixture must NOT be used to score parity (see
`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`). What it measures is whether
tokens x depth allocates, and that depends on shape, not on biology.

    python3 perf/ceiling_of3/build_msa_rungs.py --a3m parent.a3m --unit 298 \
        --out-msa msacache_deep --out-yaml msafix --prefix deep --sizes 608 640 672 704
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

YAML = "sequences:\n  - protein:\n      id: A\n      sequence: {seq}\n"


def cut(row: str, ncols: int) -> str:
    """First ``ncols`` match columns of an a3m row.

    Lowercase is an insertion and does not count toward a column; a trailing insertion run
    is dropped so every row ends on a match column and rows concatenate cleanly.
    """
    out, seen = [], 0
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
    if seen != ncols:
        raise ValueError(f"row has {seen} match columns, needed {ncols}")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a3m", required=True, type=Path)
    ap.add_argument("--unit", type=int, required=True,
                    help="match columns of the tandem unit to cut the parent down to")
    ap.add_argument("--out-msa", required=True, type=Path)
    ap.add_argument("--out-yaml", required=True, type=Path)
    ap.add_argument("--prefix", default="deep")
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    a = ap.parse_args()

    lines = a.a3m.read_text().rstrip("\n").split("\n")
    heads, rows = lines[0::2], lines[1::2]
    if not all(h.startswith(">") for h in heads):
        raise SystemExit("a3m is not strict header/row pairs")
    depth = len(rows)
    unit = [cut(r, a.unit) for r in rows]
    a.out_msa.mkdir(parents=True, exist_ok=True)
    a.out_yaml.mkdir(parents=True, exist_ok=True)

    for L in a.sizes:
        reps = -(-L // a.unit)
        built = [cut(r * reps, L) for r in unit]
        # Depth is the whole point of the construction, so it is asserted, not assumed.
        assert len(built) == depth, f"{L}: depth moved {depth} -> {len(built)}"
        query = built[0]
        assert query == query.upper() and len(query) == L, f"{L}: query malformed"
        key = hashlib.sha256(query.encode()).hexdigest()[:16]
        out = "\n".join(f"{h}\n{r}" for h, r in zip(heads, built)) + "\n"
        (a.out_msa / f"{key}.a3m").write_text(out)
        (a.out_yaml / f"{a.prefix}_{L}.yaml").write_text(YAML.format(seq=query))
        print(f"{a.prefix}_{L}: depth={depth} key={key} reps={reps}")


if __name__ == "__main__":
    main()
