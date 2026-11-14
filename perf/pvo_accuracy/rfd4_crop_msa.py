"""Slice a cached a3m down to a contiguous crop of its query, and publish it under the crop's own
cache key.

Why this exists: RFD4 designs its binder against a CROP of the target (here 500 of 9TH6 chain A's
547 residues), so the crop's sequence hashes differently from the full chain and misses the MSA
cache PFM built. Missing the cache is not an error -- OF3 falls back to a one-row query-only a3m --
so the refold silently folds a 500-residue viral antigen from a single sequence and returns
overall_plddt ~33 and a 29 A refold-vs-design backbone RMSD. Every design then fails every pass
criterion, and the resulting 0 % success rate measures the oracle, not the designs.

The crop is a contiguous substring of the cached query, so its alignment is the cached one with the
columns outside the crop removed. That is the alignment, not a new search: no network, no MMseqs2,
and the same bytes PFM's MSAs are built from.

    python3 rfd4_crop_msa.py --cache ~/pvo-accuracy-data/msa/cache \
        --crop <seq> --out <dir>

Writes `<out>/<seq_hash(crop)>.a3m` in the shared-cache layout `resolve_openfold3_msas` reads
(`<msa_dir>/<hash>.a3m`), so pointing a refold's --msa-dir at `<out>` gives the target its MSA and
leaves the designed binder on the one-row fallback -- target MSA, binder single sequence, which is
the standard binder-refolding protocol and what AF2-IG's initial-guess arm approximates.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys


def seq_hash(seq: str) -> str:
    """tt_bio/cache.py:31, copied rather than imported: this script must run off a card."""
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def read_a3m(path: pathlib.Path) -> list[tuple[str, str]]:
    out, name, buf = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(buf)))
            name, buf = line, []
        elif name is not None:
            buf.append(line.strip())
    if name is not None:
        out.append((name, "".join(buf)))
    return out


def slice_row(seq: str, start: int, end: int) -> str:
    """The a3m row restricted to query columns [start, end).

    In a3m, uppercase and '-' consume a query column; lowercase is an insertion sitting between the
    previous column and the next one. An insertion is kept only when the column it precedes is
    inside the crop, so the crop's own deletion features survive and the flanks contribute nothing.
    """
    kept, pending, col = [], [], 0
    for ch in seq:
        if ch.islower():
            pending.append(ch)
            continue
        if start <= col < end:
            kept.extend(pending)
            kept.append(ch)
        pending = []
        col += 1
    return "".join(kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="directory of <hash>.a3m written by a full search")
    ap.add_argument("--crop", required=True, help="the cropped sequence, or @<file> holding it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    crop = args.crop
    if crop.startswith("@"):
        crop = pathlib.Path(crop[1:]).read_text().strip()
    crop = "".join(crop.split())

    hit = None
    for path in sorted(pathlib.Path(args.cache).glob("*.a3m")):
        rows = read_a3m(path)
        if not rows:
            continue
        query = rows[0][1]
        if crop in query:
            hit = (path, rows, query.index(crop))
            break
    if hit is None:
        print(f"no cached a3m in {args.cache} has this crop as a contiguous substring of its query",
              file=sys.stderr)
        return 2
    path, rows, start = hit
    end = start + len(crop)

    out_rows = []
    for name, seq in rows:
        sliced = slice_row(seq, start, end)
        # A row that aligns nowhere inside the crop carries no information about it. Upstream's
        # readers tolerate such a row, but it inflates the depth figure a report would quote.
        if sliced.replace("-", "").replace(".", "") == "":
            continue
        out_rows.append((name, sliced))

    if out_rows[0][1] != crop:
        print(f"the sliced query is not the crop: {out_rows[0][1][:60]}... vs {crop[:60]}...",
              file=sys.stderr)
        return 3

    out_dir = pathlib.Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{seq_hash(crop)}.a3m"
    dest.write_text("".join(f"{n}\n{s}\n" for n, s in out_rows))
    print(f"source {path.name} query {len(rows[0][1])} depth {len(rows)}")
    print(f"crop at column {start}..{end} length {len(crop)}")
    print(f"wrote {dest} depth {len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
