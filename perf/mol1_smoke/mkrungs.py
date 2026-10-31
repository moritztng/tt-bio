"""Size-ladder fixtures for Mol-1: one protein chain of N residues with a deep alignment.

    python perf/mol1_smoke/mkrungs.py <non_pairing.a3m> <out_dir> <depth> <N> [<N> ...]

The query and every alignment row are tiled column-wise to N residues, so the fold carries the
same depth at every rung and only the token axis grows (the wall moves with tokens x rows).
Insertions (lowercase) are dropped before tiling; an alignment row then has exactly the query's
length. Rows beyond `depth` are not written: the engine pools at most 16384 rows anyway.
Writes <out_dir>/tile_<N>_d<depth>.json (tt-bio json input) and its .a3m.
"""
import json
import sys
from pathlib import Path


def read_a3m(path):
    names, seqs = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            names.append(line[1:])
            seqs.append("")
        elif line.strip():
            seqs[-1] += "".join(c for c in line.strip() if not c.islower())
    return names, seqs


def tile(s, n):
    return (s * (n // len(s) + 1))[:n]


def main():
    src, out, depth, sizes = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), map(int, sys.argv[4:])
    names, seqs = read_a3m(src)
    L = len(seqs[0])
    keep = [(h, s) for h, s in zip(names, seqs) if len(s) == L][:depth]
    out.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        stem = f"tile_{n}_d{len(keep)}"
        a3m = out / f"{stem}.a3m"
        a3m.write_text("".join(f">{h}\n{tile(s, n)}\n" for h, s in keep))
        job = [{"name": stem, "sequences": [{"proteinChain": {
            "sequence": tile(keep[0][1], n), "count": 1, "id": ["A"],
            "unpairedMsaPath": str(a3m.resolve())}}]}]
        (out / f"{stem}.json").write_text(json.dumps(job))
        print(out / f"{stem}.json", len(keep), n)


if __name__ == "__main__":
    main()
