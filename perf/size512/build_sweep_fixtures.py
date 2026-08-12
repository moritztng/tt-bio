#!/usr/bin/env python3
"""Build the token-count sweep from the campaign fixture, keeping MSA depth constant.

There is no 512 aa monomer in the campaign fixtures, and the two alternatives were rejected:
a fresh ColabFold search puts a network dependency and an unreproducible alignment in the middle
of a leg whose product is a boolean, and the AbAg complexes are multimers that `seed_msa_cache`
refuses and whose MSA pairing would be a second variable.

So the sweep is prot300 (CDK2, 298 aa, 35 sequences) tandem-repeated to at least the target
length and cut to it. Repeating a query and repeating every aligned row the same number of times
gives a well-formed a3m, so depth stays exactly 35 at every size and token count is the only
variable -- the same contract `tt_baseline.py` states for its own two targets. L=298 reproduces
the original fixture, which is the self-check.

The repeat count is ceil(L / 298), so 128 and 256 are one copy cut short, 320-512 are two, 768 is
three and 1024 is four. Before 2026-08-13 this was hard-coded to two copies and capped the family
at 596 aa.

The doubled chain is not a real protein and its structure is meaningless. That is fine: this leg
measures where a capacity fit test flips and what the flip costs, not structure quality. pLDDT is
used only as a run-integrity check that both arms folded the same thing.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_A3M = ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m"
SRC_YAML = ROOT / "examples/prot300.yaml"
OUT = Path(__file__).resolve().parent / "fixtures"
SIZES = [298, 320, 352, 384, 448, 512]


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


def main(sizes):
    lines = SRC_A3M.read_text().rstrip("\n").split("\n")
    heads, rows = lines[0::2], lines[1::2]
    assert all(h.startswith(">") for h in heads), "a3m is not strict header/row pairs"
    query = rows[0]
    assert query == query.upper(), "query row carries insertions"
    OUT.mkdir(parents=True, exist_ok=True)
    yaml_src = SRC_YAML.read_text()
    made = []
    for L in sizes:
        reps = max(1, -(-L // len(query)))
        seq = (query * reps)[:L]
        a3m = "\n".join(f"{h}\n{cut(r * reps, L)}" for h, r in zip(heads, rows)) + "\n"
        a3m_rows = a3m.split("\n")
        assert a3m_rows[1] == seq, f"L={L}: a3m query row != target sequence"
        (OUT / f"cdk2x2_{L}.a3m").write_text(a3m)
        head, tail = yaml_src.split("sequence: ", 1)
        (OUT / f"cdk2x2_{L}.yaml").write_text(head + "sequence: " + seq + "\n")
        made.append((L, len(seq), a3m.count(">")))
    for L, n, d in made:
        print(f"  L={L:4d}  seq={n:4d}  depth={d}")
    # self-check: L=298 must reproduce the committed fixture
    if 298 in sizes:
        ref = "\n".join(f"{h}\n{r}" for h, r in zip(heads, rows)) + "\n"
        got = (OUT / "cdk2x2_298.a3m").read_text()
        print("L=298 reproduces prot300.a3m:", got == ref)
        return 0 if got == ref else 1
    return 0


sys.exit(main([int(x) for x in sys.argv[1:]] or SIZES))
