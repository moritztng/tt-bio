"""Build the Blackhole ESMFold2-Fast coverage rungs.

Rungs are TOKEN counts. Every chain is protein with no ligand, so tokens equal residues.
The sequences are the same two CDK2 tandem chains the boltz2 and esmfold2 Blackhole
coverage rungs used, so all three models 1536 rungs are the same fixture and the numbers
are comparable. They are lifted out of a3m files only because that is where those exact
sequences live on this host; esmfold2-fast has no MSA encoder and the alignments are not
used. `seq_hash(query) == filename` is checked so the sequence is identified rather than
assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

WT = "/home/ttuser/.coworker/wt/cov-unproven-esmfold2fast-bhp150a"
sys.path.insert(0, WT)
from tt_bio.cache import seq_hash

HERE = Path(__file__).parent

CHAINS = {
    1024: Path("/home/ttuser/esmfold2wh_msa/cache/d4bb492258e7af30.a3m"),
    512: Path("/home/ttuser/esmfold2wh_msa/cache/4e5c2b391bde62d4.a3m"),
}

#: rung -> chain lengths. 1024 is the control: it is what esmfold2-fast is PROVEN at on the
#: serving hardware, so a 1536 result is only attributable once the same card folds it.
RUNGS = {"1024": [1024], "1536": [1024, 512]}


def query(p: Path) -> str:
    with p.open() as fh:
        fh.readline()
        return fh.readline().strip()


def main() -> int:
    seqs = {}
    for n, a3m in CHAINS.items():
        h, s = a3m.stem, query(a3m)
        if len(s) != n:
            raise SystemExit(f"{h}: query is {len(s)} aa, expected {n}")
        if seq_hash(s) != h:
            raise SystemExit(f"{h}: seq_hash is {seq_hash(s)}, so this is not this sequence")
        seqs[n] = s
        print(f"chain {n} aa  hash {h}")

    for name, chains in RUNGS.items():
        body = ["sequences:"]
        for i, n in enumerate(chains):
            body += ["  - protein:", f"      id: {chr(ord(chr(65)) + i)}", f"      sequence: {seqs[n]}"]
        (HERE / "inputs" / f"e2f_{name}.yaml").write_text("\n".join(body) + "\n")
        print(f"rung {name}: {sum(chains)} tokens, chains {chains}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
