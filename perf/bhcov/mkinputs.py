"""Build the Blackhole boltz2 coverage rungs and prove the alignment depth that reaches the model.

The rungs are token counts, not residue counts, and the two are only equal because every chain
here is protein with no ligand. Each chain reuses an MSA the engine itself wrote, so the cache
key is not asserted -- `seq_hash` is recomputed from the sequence and checked against the file
name, which is the only thing that makes "this fold ran at depth N" a measurement rather than a
hope. Depth is then read back through the engine's own `parse_a3m` at the shipped
`--max_msa_seqs 8192`, because an a3m with 14721 `>` records can still arrive as 35 rows after
dedup.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/home/ttuser/.coworker/wt/cov-unproven-boltz2-bhp150a")
from tt_bio.cache import seq_hash
from tt_bio.data.parse import parse_a3m

SRC = Path("/home/ttuser/esmfold2wh_msa/cache")
HERE = Path(__file__).parent
MSA = HERE / "msa"
MAX_MSA = 8192

#: (chain length, the a3m the engine wrote for it). Both are CDK2 tandem tilings from the
#: esmfold2 Wormhole ladder, so the sequences are chimeric -- fine for capacity and for
#: backbone continuity, useless for pLDDT.
CHAINS = {1024: "d4bb492258e7af30", 512: "4e5c2b391bde62d4"}

#: rung tokens -> chain lengths. 1024 is the control: it is what boltz2 is PROVEN at on
#: Wormhole, so a 1536 result is only attributable once the same card folds it.
RUNGS = {1024: [1024], 1536: [1024, 512]}


def query(p: Path) -> str:
    with p.open() as fh:
        fh.readline()
        return fh.readline().strip()


def main() -> int:
    seqs, depths = {}, {}
    for n, h in CHAINS.items():
        a3m = SRC / f"{h}.a3m"
        s = query(a3m)
        if len(s) != n:
            raise SystemExit(f"{h}: query is {len(s)} aa, expected {n}")
        if seq_hash(s) != h:
            raise SystemExit(f"{h}: seq_hash is {seq_hash(s)}, so this cache entry is not this sequence")
        dst = MSA / f"{h}.a3m"
        if not dst.exists():
            dst.write_bytes(a3m.read_bytes())
        rows = sum(1 for line in a3m.open() if line.startswith(">"))
        reached = len(parse_a3m(dst, None, MAX_MSA).sequences)
        seqs[n], depths[n] = s, (rows, reached)
        print(f"chain {n} aa  hash {h}  a3m {rows} records -> {reached} rows reach the model")

    for tokens, chains in RUNGS.items():
        if sum(chains) != tokens:
            raise SystemExit(f"rung {tokens}: chains sum to {sum(chains)}")
        body = ["sequences:"]
        for i, n in enumerate(chains):
            body += [f"  - protein:", f"      id: {chr(ord('A') + i)}", f"      sequence: {seqs[n]}"]
        (HERE / "inputs" / f"b2_{tokens}.yaml").write_text("\n".join(body) + "\n")
        print(f"rung {tokens}: chains {chains}, depth {[depths[n][1] for n in chains]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
