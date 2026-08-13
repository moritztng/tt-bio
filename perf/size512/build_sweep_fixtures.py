#!/usr/bin/env python3
"""Build the token-count sweep from the campaign fixture, keeping MSA depth constant.

There is no 512 aa monomer in the campaign fixtures, and the two alternatives were rejected:
a fresh ColabFold search puts a network dependency and an unreproducible alignment in the middle
of a leg whose product is a boolean, and the AbAg complexes are multimers that `seed_msa_cache`
refuses and whose MSA pairing would be a second variable.

So the sweep is prot300 (CDK2, 298 aa, 35 sequences) tiled `k = ceil(L/298)` times and cut to each
target length. Repeating a query and repeating every aligned row gives a well-formed a3m, so depth
stays exactly 35 at every size and token count is the only variable -- the same contract
`tt_baseline.py` states for its own two targets. L=298 reproduces the original fixture, which is
the self-check. k was 2 while the sweep topped out at 512; 576 and 640 already ship built at k=2
and k=3, and 768/1024 need k=3 and k=4. Deriving k from L rather than hard-coding it reproduces
every committed fixture byte-for-byte (verified for 298/512/576/640) and extends the family without
a second convention.

The tiled chain is not a real protein and its structure is meaningless. That is fine: this leg
measures where a capacity fit test flips and what the flip costs, not structure quality. pLDDT is
used only as a run-integrity check that both arms folded the same thing. The corollary matters for
parity: a non-bit-exact change cannot be scored on this fixture above ~298 aa, because the free
hinges between tiles saturate RMSD -- score those on cdk2x2_298, which is a real single domain.
"""
import argparse, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_A3M = ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m"
SRC_YAML = ROOT / "examples/prot300.yaml"
OUT = Path(__file__).resolve().parent / "fixtures"
SIZES = [128, 256, 298, 320, 352, 384, 448, 512, 768, 1024]


def cut(row: str, ncols: int) -> str:
    """First `ncols` match columns of an a3m row. Lowercase is an insertion and does not count;
    a trailing insertion run is dropped so every row ends on a match column."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]

    lines = SRC_A3M.read_text().rstrip("\n").split("\n")
    heads, rows = lines[0::2], lines[1::2]
    assert all(h.startswith(">") for h in heads), "a3m is not strict header/row pairs"
    query = rows[0]
    assert query == query.upper(), "query row carries insertions"
    OUT.mkdir(parents=True, exist_ok=True)
    yaml_src = SRC_YAML.read_text()
    made = []
    for L in sizes:
        k = -(-L // len(query))
        seq = (query * k)[:L]
        a3m = "\n".join(f"{h}\n{cut(r * k, L)}" for h, r in zip(heads, rows)) + "\n"
        a3m_rows = a3m.split("\n")
        assert a3m_rows[1] == seq, f"L={L}: a3m query row != target sequence"
        (OUT / f"cdk2x2_{L}.a3m").write_text(a3m)
        head, tail = yaml_src.split("sequence: ", 1)
        (OUT / f"cdk2x2_{L}.yaml").write_text(head + "sequence: " + seq + "\n")
        made.append((L, k, len(seq), a3m.count(">"),
                     hashlib.sha256((OUT / f"cdk2x2_{L}.a3m").read_bytes()).hexdigest(),
                     hashlib.sha256((OUT / f"cdk2x2_{L}.yaml").read_bytes()).hexdigest()))
    # self-check: L=298 must reproduce the committed fixture
    ref = "\n".join(f"{h}\n{r}" for h, r in zip(heads, rows)) + "\n"
    ok = True
    if 298 in sizes:
        got = (OUT / "cdk2x2_298.a3m").read_text()
        ok = got == ref
        print("L=298 reproduces prot300.a3m:", ok)
    for L, k, n, d, sa, sy in made:
        print(f"  L={L:5d}  k={k}  seq={n:5d}  depth={d}  a3m sha256 {sa}  yaml sha256 {sy}")
    return 0 if ok else 1


sys.exit(main())
