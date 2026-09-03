"""Cut a cdk2x2 rung of N residues from the 1024 fixture.

Every existing rung is the same 298-aa CDK2 pattern tiled and truncated to N *query columns*, so a
new rung is a column prefix of the 1024 one. a3m lowercase characters are insertions and do not
consume a column, which is why this is not a character cut. `--verify` regenerates the committed
512/576/640/768/896/960 rungs and compares byte-for-byte; that is the proof the rule is the rule.
"""
import os
import sys
from pathlib import Path

# The committed rungs this script cuts from. TT_BIO_RUNG_SRC relocates it so the script can
# be run from a scratch directory on a remote box without a repo checkout around it.
SRC = Path(os.environ.get("TT_BIO_RUNG_SRC")
           or Path(__file__).resolve().parents[1] / "size512" / "fixtures")
HDR = ("version: 1\n"
       "# CDK2 (PDB 1HCL) tiled to {n} residues, apo, one chain. Cut from cdk2x2_1024 by\n"
       "# perf/ceilings/make_rung.py, the same 298-aa pattern every other rung uses.\n"
       "sequences:\n  - protein:\n      id: A\n      sequence: {seq}\n      msa: {msa}\n")


def _column_prefix(row, n):
    out, cols = [], 0
    for ch in row:
        if ch.islower():          # an insertion, not a query column
            out.append(ch)
            continue
        if cols >= n:
            break
        out.append(ch)
        cols += 1
    return "".join(out)


# Substituted into a match column to make a tiled copy distinct. Any residue works; the
# rotation only has to differ from the character already there.
_SUBS = "ACDEFGHIKLMNPQRSTVWY"


def _make_distinct(seq, seen):
    """Return `seq` mutated in as few match columns as it takes to be unseen.

    `tt_bio.protenix_data._parse_a3m_to_msa` DEDUPS identical alignment rows, keying on the
    full row string. So a row that merely repeats an earlier one contributes nothing: it is
    dropped before the model sees it. This walks match columns left to right, substituting a
    different residue in each, until the row is one the parser will keep.
    """
    cols = [i for i, ch in enumerate(seq) if not ch.islower()]
    out = list(seq)
    for i in cols:
        for sub in _SUBS:
            if sub == out[i]:
                continue
            out[i] = sub
            cand = "".join(out)
            if cand not in seen:
                return cand
    raise SystemExit("cannot make row distinct")


def deepen(rows, depth):
    """Tile the alignment's non-query rows up to `depth` DISTINCT rows in total.

    A ceiling is a shape, and the MSA track's shape is (depth x columns), so a tiled alignment
    prices depth as a real one does. It is NOT a biological MSA and nothing about the structure
    it folds to means anything -- use it to find where the engine stops, never to judge an output.

    Every emitted row is distinct because the parser deduplicates. A plain tiling does not
    survive that: giving each copy a unique HEADER is not enough, since the dedup key is the
    sequence. The first version of this function tiled rows verbatim and a 2048-row file reached
    the model as 35 rows, silently, while every run returned bit-identical confidence and the
    deepest one ran FASTEST. Measure depth with `--verify-depth`, never assume it.
    """
    head, body = rows[:2], rows[2:]
    if depth <= 1 or not body:
        return head
    pairs = [body[i:i + 2] for i in range(0, len(body), 2)]
    seen = {rows[1]}
    out = list(head)
    i = 0
    while len(out) // 2 < depth:
        name, seq = pairs[i % len(pairs)]
        if seq in seen:
            seq = _make_distinct(seq, seen)
        seen.add(seq)
        out += [f"{name}\tcopy{i // len(pairs)}" if i >= len(pairs) else name, seq]
        i += 1
    return out


def unique_rows(a3m):
    """Rows the parser will actually keep: the dedup rule of `_parse_a3m_to_msa`, applied here
    so the generator can prove its own output without importing torch."""
    lines = a3m.splitlines()
    return len({lines[i + 1] for i in range(0, len(lines), 2)})


def cut(n, out_dir, depth=None):
    rows = (SRC / "cdk2x2_1024.a3m").read_text().splitlines()
    if n > len(_column_prefix(rows[1], 10**9)):
        raise SystemExit(f"{n} columns is past the 1024 fixture")
    # Cut columns BEFORE deepening. The other order lets the column cut truncate away the very
    # substitution that made a row distinct, collapsing the depth again for small n.
    rows = [rows[i] if i % 2 == 0 else _column_prefix(rows[i], n) for i in range(len(rows))]
    if depth:
        rows = deepen(rows, depth)
    a3m = "".join(f"{rows[i]}\n{rows[i + 1]}\n" for i in range(0, len(rows), 2))
    seq = rows[1]
    out_dir = Path(out_dir)
    tag = f"cdk2x2_{n}" if not depth else f"cdk2x2_{n}_d{depth}"
    (out_dir / f"{tag}.a3m").write_text(a3m)
    (out_dir / f"{tag}.yaml").write_text(
        HDR.format(n=n, seq=seq, msa=out_dir / f"{tag}.a3m"))
    return seq


if __name__ == "__main__":
    if sys.argv[1] == "--verify":
        import tempfile
        import yaml
        bad = 0
        with tempfile.TemporaryDirectory() as td:
            for n in (512, 576, 640, 768, 896, 960):
                got = cut(n, td)
                want_seq = yaml.safe_load(
                    (SRC / f"cdk2x2_{n}.yaml").read_text())["sequences"][0]["protein"]["sequence"]
                ok_a3m = (Path(td) / f"cdk2x2_{n}.a3m").read_text() == \
                    (SRC / f"cdk2x2_{n}.a3m").read_text()
                bad += not (ok_a3m and got == want_seq)
                print(f"{n}: a3m {'OK' if ok_a3m else 'MISMATCH'} "
                      f"seq {'OK' if got == want_seq else 'MISMATCH'}")
        sys.exit(1 if bad else 0)
    if sys.argv[1] == "--verify-depth":
        # A depth fixture is only a depth fixture if the parser KEEPS the rows. Counting lines
        # in the file measures the generator, not the model's input; this counts what survives
        # dedup, which is what reaches the MSA track.
        import tempfile
        bad = 0
        with tempfile.TemporaryDirectory() as td:
            for n in (576, 992):
                for d in (128, 512, 2048):
                    cut(n, td, d)
                    got = unique_rows((Path(td) / f"cdk2x2_{n}_d{d}.a3m").read_text())
                    bad += got != d
                    print(f"{n} aa depth {d}: parser keeps {got} rows "
                          f"{'OK' if got == d else 'MISMATCH'}")
        sys.exit(1 if bad else 0)
    out = sys.argv[1]
    depth = None
    rest = sys.argv[2:]
    if rest and rest[0].startswith("--depth="):
        depth = int(rest.pop(0).split("=")[1])
    for n in map(int, rest):
        cut(n, out, depth)
        print(f"wrote cdk2x2_{n}{f'_d{depth}' if depth else ''} ({n} aa, "
              f"depth {depth or 35}) in {out}")
