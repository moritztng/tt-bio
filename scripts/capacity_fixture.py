"""Build a capacity-gate fixture: one protein chain at a target token count, at real MSA depth.

WHY A GENERATED FIXTURE. The bar is 1504 tokens and no fixture in the tree reaches it. Running a
fresh alignment search per gate run would put minutes of network wall-clock and an unreproducible
alignment in front of a check whose whole point is being cheap enough that nobody skips it. So the
fixture is derived, offline, from one committed source: CDK2 (PDB 1HCL, 298 aa) with its full
ColabFold alignment, 8833 rows, in perf/capacity/. The chain is tandem-repeated to the target
length and every aligned row is repeated with it, so depth is exactly the source's at every size
and the token count is the only thing that varies.

The repeated chain is not a real protein and its structure is meaningless. That is the correct
trade here: this fixture answers "does the shape allocate and the pipeline complete", never "is
the answer right". Correctness lives in scripts/full_parity_gate.py against real targets.

WHY DEPTH IS NOT OPTIONAL. For the OF3-family models the failing tensor scales with tokens x rows:
at 14190 rows OpenFold3 folds 576 and dies at 614, while single-sequence it folds 768 in 301 s. A
1504-token single-sequence pass proves nothing a user hits, so the deep source is the default and
--depth is a named reduction the gate has to write into its own output.

The row count in the FILE is not the row count the model sees: `_parse_a3m_to_msa` deduplicates by
sequence, so a tiled alignment whose rows are copies of each other arrives as a handful. Tandem
repetition of DISTINCT rows keeps them distinct (8832 of the source's 8833 rows are unique), and
`effective_depth` reports the deduplicated count so the gate records what reached the model.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEEP_A3M = ROOT / "perf" / "capacity" / "cdk2_1hcl_colabfold_deep.a3m.gz"
SRC_YAML = ROOT / "examples" / "prot300.yaml"
#: Match columns in the source alignment, i.e. the tandem repeat unit.
UNIT = 298


def cut(row: str, ncols: int) -> str:
    """First `ncols` match columns of an a3m row.

    Lowercase is an insertion and does not count towards the column budget; a trailing insertion
    run is dropped so every row ends on a match column. Shared with
    perf/size512/build_sweep_fixtures.py, which cuts the same source to the perf ladder's rungs.
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
    assert seen == ncols, f"row has {seen} match columns, needed {ncols}"
    return "".join(out)


def read_a3m(path: Path) -> tuple[list[str], list[str]]:
    text = gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text()
    lines = text.rstrip("\n").split("\n")
    heads, rows = lines[0::2], lines[1::2]
    assert all(h.startswith(">") for h in heads), f"{path.name} is not strict header/row pairs"
    assert rows[0] == rows[0].upper(), f"{path.name}: query row carries insertions"
    return heads, rows


def build(residues: int, out_dir: Path, *, depth: int | None = None,
          source: Path = DEEP_A3M) -> dict:
    """Write `<out_dir>/cap_<residues>[_d<depth>].{yaml,a3m}`. Returns what was actually built."""
    heads, rows = read_a3m(source)
    if depth is not None:
        heads, rows = heads[:depth], rows[:depth]
    reps = -(-residues // UNIT)
    seq = (rows[0] * reps)[:residues]
    assert len(seq) == residues, f"source is {len(rows[0])} cols, cannot reach {residues}"
    kept = [cut(r * reps, residues) for r in rows]
    a3m = "\n".join(f"{h}\n{r}" for h, r in zip(heads, kept)) + "\n"
    assert kept[0] == seq, "a3m query row does not match the fixture sequence"

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cap_{residues}" + (f"_d{depth}" if depth is not None else "")
    a3m_path = out_dir / f"{stem}.a3m"
    a3m_path.write_text(a3m)
    head, _ = SRC_YAML.read_text().split("sequence: ", 1)
    yaml_path = out_dir / f"{stem}.yaml"
    yaml_path.write_text(head + "sequence: " + seq + "\n")
    return {
        "yaml": yaml_path, "a3m": a3m_path, "residues": residues, "repeats": reps,
        # Deduplicated over the rows AS WRITTEN, not the source rows: cutting can collapse two
        # rows that differed only outside the kept columns.
        "file_depth": len(kept), "effective_depth": len(set(kept)),
        "source": source.name,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("residues", type=int, nargs="+")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=None,
                    help="truncate the alignment to this many rows (a NAMED coverage reduction)")
    a = ap.parse_args(argv)
    for n in a.residues:
        r = build(n, a.out_dir, depth=a.depth)
        print(f"  {r['yaml'].name}: {r['residues']} residues, {r['repeats']} tandem copies, "
              f"{r['file_depth']} rows in file, {r['effective_depth']} after dedup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
