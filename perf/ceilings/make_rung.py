"""Cut a cdk2x2 rung of N residues from the 1024 fixture.

Every existing rung is the same 298-aa CDK2 pattern tiled and truncated to N *query columns*, so a
new rung is a column prefix of the 1024 one. a3m lowercase characters are insertions and do not
consume a column, which is why this is not a character cut. `--verify` regenerates the committed
512/576/640/768/896/960 rungs and compares byte-for-byte; that is the proof the rule is the rule.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "size512" / "fixtures"
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


def cut(n, out_dir):
    rows = (SRC / "cdk2x2_1024.a3m").read_text().splitlines()
    if n > len(_column_prefix(rows[1], 10**9)):
        raise SystemExit(f"{n} columns is past the 1024 fixture")
    a3m = "".join(f"{rows[i]}\n{_column_prefix(rows[i + 1], n)}\n" for i in range(0, len(rows), 2))
    seq = _column_prefix(rows[1], n)
    out_dir = Path(out_dir)
    (out_dir / f"cdk2x2_{n}.a3m").write_text(a3m)
    (out_dir / f"cdk2x2_{n}.yaml").write_text(
        HDR.format(n=n, seq=seq, msa=out_dir / f"cdk2x2_{n}.a3m"))
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
    out = sys.argv[1]
    for n in map(int, sys.argv[2:]):
        cut(n, out)
        print(f"wrote cdk2x2_{n} ({n} aa) in {out}")
