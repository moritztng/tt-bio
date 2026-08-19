"""Slice an a3m alignment to a crop of its query, and repoint a PXDesign yaml at the result.

    python perf/pxdesign/slice_a3m.py --msa-dir <full_chain_msa> --yaml lacz_512.yaml \
        --out-root /work/msa_sliced

`pxdesign prepare-msa` reads `target.file` and searches the FULL chain sequence: it never looks at
`target.chains.<id>.crop`. So on a cropped target it injects an MSA whose query is the whole chain
-- here a 1023-residue query against a 128-residue target. Rather than run four separate searches
(four different alignments, four different depths, one more confound in a ladder whose whole point
is to vary one thing), slice the single full-chain alignment to each crop. Same homologs, same
search, restricted to the crop's columns.

a3m column semantics: in a non-query row, uppercase and '-' each consume one query column,
lowercase is an insertion that belongs to the gap before the next query column. So a row is sliced
by walking it, counting query columns, and keeping the characters whose column falls in range plus
the insertions sitting inside that range.
"""

import argparse
import pathlib
import re


def read_a3m(path):
    name, seq, out = None, [], []
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(seq)))
            name, seq = line, []
        elif line.strip():
            seq.append(line.strip())
    if name is not None:
        out.append((name, "".join(seq)))
    return out


def slice_row(row, i0, i1):
    """Keep query columns [i0, i1) (0-based). Insertions are kept when they sit inside the range."""
    col, keep, pending = 0, [], []
    for ch in row:
        if ch.islower():
            pending.append(ch)
            continue
        if i0 <= col < i1:
            keep.extend(pending)
            keep.append(ch)
        pending = []
        col += 1
    return "".join(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msa-dir", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()

    text = pathlib.Path(a.yaml).read_text()
    # prepare-msa rewrites the yaml through yaml.safe_dump, which turns the inline
    # crop: ["143-910"] into a block list, so both spellings have to be accepted
    m = re.search(r'crop:\s*(?:\[\s*"?(\d+)-(\d+)"?\s*\]|\n\s*-\s*"?(\d+)-(\d+)"?)', text)
    if not m:
        raise SystemExit("no crop range in %s" % a.yaml)
    g = [x for x in m.groups() if x is not None]
    start, end = int(g[0]), int(g[1])
    i0, i1 = start - 1, end                       # label_seq_id is 1-based and sequential

    out = pathlib.Path(a.out_root) / pathlib.Path(a.yaml).stem
    out.mkdir(parents=True, exist_ok=True)
    depths = {}
    for fn in ("pairing.a3m", "non_pairing.a3m"):
        src = pathlib.Path(a.msa_dir) / fn
        if not src.exists():
            continue
        rows = read_a3m(src)
        q_len = len(rows[0][1])
        if i1 > q_len:
            raise SystemExit("crop %d-%d exceeds query length %d in %s"
                             % (start, end, q_len, src))
        sliced, kept = [], 0
        for name, seq in rows:
            s = slice_row(seq, i0, i1)
            # a row that is all gaps over the crop carries no information about it
            if kept and set(s.upper()) <= {"-"}:
                continue
            sliced.append((name, s))
            kept += 1
        (out / fn).write_text("".join("%s\n%s\n" % (n, s) for n, s in sliced))
        depths[fn] = {"rows_in": len(rows), "rows_out": len(sliced),
                      "query_len_in": q_len,
                      "query_len_out": len([c for c in sliced[0][1] if not c.islower()])}

    if re.search(r"\n\s+msa:\s*\S+", text):
        new = re.sub(r"(\n\s+msa:\s*)\S+", lambda mm: mm.group(1) + str(out.resolve()), text)
    else:
        new = text.replace("binder_length:", "      msa: %s\nbinder_length:" % out.resolve())
    pathlib.Path(a.yaml).write_text(new)
    print("%s crop %d-%d -> %s" % (pathlib.Path(a.yaml).name, start, end, out))
    for fn, d in depths.items():
        print("   %-16s rows %d -> %d, query %d -> %d"
              % (fn, d["rows_in"], d["rows_out"], d["query_len_in"], d["query_len_out"]))


if __name__ == "__main__":
    main()
