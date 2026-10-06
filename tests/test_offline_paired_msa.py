"""An offline paired search must pair: one colabfold_search record for the complex, split back
into one row-aligned a3m per chain. Host-only, with a stand-in colabfold_search."""
import os
import stat
import sys
from pathlib import Path

import pytest

from tt_bio import main as m

A, B = "GIVEQCCTSICSLYQLENYCN", "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"
# What colabfold 1.6.3's msa_to_str writes for a two-chain complex in --pair-mode paired:
# the length line, then each row's chains concatenated (headers tab-joined).
PAIRED = (f"#{len(A)},{len(B)}\t1,1\n>101\t102\n{A}{B}\n"
          ">UniRef100_X\t51\t0.9\tUniRef100_Y\t62\t0.8\n"
          f"{'-' * 3}{A[3:10]}kk{A[10:]}{B[:5]}aa{B[5:]}\n"
          f">UniRef100_Z\t40\tUniRef100_W\t30\n{A}{'-' * len(B)}\n")


def test_split_keeps_rows_aligned_and_insertions_with_their_chain():
    a, b = m.split_paired_a3m(PAIRED, [len(A), len(B)])
    ra, rb = a.splitlines()[1::2], b.splitlines()[1::2]
    assert ra == [A, "---" + A[3:10] + "kk" + A[10:], A]
    assert rb == [B, B[:5] + "aa" + B[5:], "-" * len(B)]
    for r, n in ((ra, len(A)), (rb, len(B))):
        assert all(sum(c == "-" or c.isupper() for c in x) == n for x in r)


def test_split_refuses_a_row_longer_than_the_chains():
    with pytest.raises(ValueError):
        m.split_paired_a3m(f"#2,2\t1,1\n>q\nAAAAA\n", [2, 2])


def _stub(tmp_path, log):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    stub = bin_ / "colabfold_search"
    stub.write_text(f"""#!{sys.executable}
import sys, pathlib
fasta, db, out = sys.argv[1:4]
pathlib.Path({str(log)!r}).write_text(" ".join(sys.argv[1:]) + "\\n" + open(fasta).read())
pathlib.Path(out, "complex.a3m").write_text({PAIRED!r})
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return bin_


def test_pair_true_searches_one_complex_record(tmp_path, monkeypatch):
    log = tmp_path / "argv"
    monkeypatch.setenv("PATH", f"{_stub(tmp_path, log)}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(m, "_find_mmseqs", lambda _c: None)
    out = tmp_path / "pdir"
    out.mkdir()
    m.compute_msa_offline({"ha": A, "hb": B}, "t", out, str(tmp_path), pair=True)
    argv, fasta = log.read_text().split("\n", 1)
    assert fasta == f">complex\n{A}:{B}\n"
    assert "--pair-mode paired" in argv
    rows = [(out / f"{h}.a3m").read_text().splitlines()[1::2] for h in ("ha", "hb")]
    assert len(rows[0]) == len(rows[1]) == 3 and rows[0][0] == A and rows[1][0] == B
