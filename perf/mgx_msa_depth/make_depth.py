#!/usr/bin/env python3
"""Depth fixtures: one protein chain of N tokens with an alignment of exactly D distinct rows.

    make_depth.py SRC.a3m N D OUTDIR

SRC is a real alignment at least N columns wide (the mgx-bigalloc cdk2x2_1536_d14190 rung: CDK2
tiled to 1536, 14190 rows). Insertions (lowercase) are dropped and every row is cut to its first
N columns. Rows past the source's depth are source rows with 3 % of their residues substituted
from a seeded RNG, so they are distinct (protenix and OF3 drop duplicate rows, and a duplicated
fixture would fold shallower than its name) and stay alignment-like. The msa path is written
absolute: tt_bio resolves a relative one against the cwd, and a missed path silently queries the
MSA server at whatever depth it returns.
"""
import random
import sys
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"


def rows(src):
    seqs, cur = [], []
    for line in open(src):
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        else:
            cur.append("".join(c for c in line.strip() if not c.islower()))
    if cur:
        seqs.append("".join(cur))
    return seqs


def main():
    src, n, d, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    seqs = [s[:n] for s in rows(src)]
    query, body = seqs[0], seqs[1:]
    assert len(query) == n, f"source is {len(query)} columns, need {n}"
    seen, keep = {query}, [query]
    for s in body:
        if s not in seen:
            seen.add(s)
            keep.append(s)
    rng, i = random.Random(0), 0
    while len(keep) < d:
        base = list(body[i % len(body)])
        i += 1
        pos = [j for j, c in enumerate(base) if c != "-"]
        for j in rng.sample(pos, max(1, round(0.03 * len(pos)))):
            base[j] = rng.choice(AA)
        s = "".join(base)
        if s not in seen:
            seen.add(s)
            keep.append(s)
    keep = keep[:d]
    out.mkdir(parents=True, exist_ok=True)
    stem = f"cdk2_{n}_d{d}"
    a3m = (out / f"{stem}.a3m").resolve()
    a3m.write_text("".join(f">{k}\n{s}\n" for k, s in enumerate(keep)))
    (out / f"{stem}.yaml").write_text(
        f"version: 1\n# CDK2 tiled to {n} residues, apo, one chain, {d} distinct alignment rows.\n"
        f"# perf/mgx_msa_depth/make_depth.py from {Path(src).name}.\nsequences:\n"
        f"  - protein:\n      id: A\n      sequence: {query}\n      msa: {a3m}\n")
    print(stem, len(keep))


if __name__ == "__main__":
    main()
