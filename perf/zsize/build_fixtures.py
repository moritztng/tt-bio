#!/usr/bin/env python3
"""Extend the size512 sweep fixtures to the sizes this leg needs, byte-compatibly.

`perf/size512/build_sweep_fixtures.py` tandem-doubles CDK2 (298 aa, depth 35) and cuts to six
lengths, which caps it at 596 match columns. The boundary this leg has to bracket on qb1 sits at
506/507 and the brief also asks for 640, so the query is repeated `ceil(L/298)` times instead of
exactly twice. At every L <= 596 that is the same string the doubling produced, and the check below
asserts it: all six existing fixtures have to come out byte-for-byte identical or nothing is
written. Depth stays exactly 35 at every size, so token count remains the only variable.

The repeated chain is not a real protein and its structure is meaningless. This leg measures
whether a fold completes and which capacity gate decides it, not structure quality.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_A3M = ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m"
SRC_YAML = ROOT / "examples/prot300.yaml"
OUT = ROOT / "perf/size512/fixtures"
SIZES = [298, 320, 352, 384, 400, 416, 417, 432, 448, 464, 480, 496, 506, 507, 512, 576, 640]
EXISTING = [298, 320, 352, 384, 448, 512]


def cut(row: str, ncols: int) -> str:
    """First `ncols` match columns of an a3m row. Lowercase is an insertion and does not count; a
    trailing insertion run is dropped so every row ends on a match column."""
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
    assert seen == ncols, "row has %d match columns, needed %d" % (seen, ncols)
    return "".join(out)


def main():
    lines = SRC_A3M.read_text().rstrip("\n").split("\n")
    heads, rows = lines[0::2], lines[1::2]
    assert all(h.startswith(">") for h in heads), "a3m is not strict header/row pairs"
    query = rows[0]
    assert query == query.upper(), "query row carries insertions"
    yaml_src = SRC_YAML.read_text()
    OUT.mkdir(parents=True, exist_ok=True)

    made, kept = [], []
    for L in SIZES:
        reps = max(2, -(-L // len(query)))
        seq = (query * reps)[:L]
        assert len(seq) == L, "L=%d exceeds %d repeats of the query" % (L, reps)
        a3m = "\n".join("%s\n%s" % (h, cut(r * reps, L)) for h, r in zip(heads, rows)) + "\n"
        a3m_rows = a3m.split("\n")
        assert a3m_rows[1] == seq, "L=%d: a3m query row != target sequence" % L
        assert a3m.count(">") == 35, "L=%d: depth drifted to %d" % (L, a3m.count(">"))
        fa, fy = OUT / ("cdk2x2_%d.a3m" % L), OUT / ("cdk2x2_%d.yaml" % L)
        head, tail = yaml_src.split("sequence: ", 1)
        y = head + "sequence: " + seq + "\n"
        if L in EXISTING and fa.exists():
            same = fa.read_text() == a3m and fy.read_text() == y
            kept.append((L, same))
            assert same, ("L=%d does not reproduce the committed size512 fixture -- refusing to "
                          "write, the two sweeps would not be comparable" % L)
            continue
        fa.write_text(a3m)
        fy.write_text(y)
        made.append((L, reps))

    print("reproduces the committed size512 fixtures:", all(s for _, s in kept),
          "(" + ", ".join(str(L) for L, _ in kept) + ")")
    for L, reps in made:
        print("  wrote L=%4d  (query x%d)" % (L, reps))
    return 0 if all(s for _, s in kept) else 1


sys.exit(main())
