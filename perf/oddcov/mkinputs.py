"""Build the Blackhole OpenDDE coverage rungs and prove the alignment depth that reaches the model.

Same fixture chains as the boltz2/esmfold2/boltzgen Blackhole coverage rows (perf/bhcov), so the
four models 1536 rungs are comparable: chain A is a 1024 aa CDK2 tandem tiling and chain B its
512 aa prefix. Every chain is protein with no ligand, so the RESIDUE token axis equals the residue
count. OpenDDE carries a second axis on top -- the structural token axis Ns = 2*n_res - n_gly --
which is computed here rather than guessed, because it is the axis the refiner runs on and the
refiner is where this model froze at this size.

The MSA each chain reuses is one the engine itself wrote. The cache key is not asserted: seq_hash
is recomputed from the sequence and checked against the file name, and the depth is then read back
through the engines own parse_a3m at the shipped --max_msa_seqs 8192, because an a3m with 14721 >
records can still arrive as 35 rows after dedup.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from tt_bio.cache import seq_hash
from tt_bio.data.parse import parse_a3m

MSA = HERE / "msa"
MAX_MSA = 8192

CHAINS = {
    1024: Path("/home/ttuser/esmfold2wh_msa/cache/d4bb492258e7af30.a3m"),
    512: Path("/home/ttuser/esmfold2wh_msa/cache/4e5c2b391bde62d4.a3m"),
    686: Path("/home/ttuser/widek_msa/941f47a0ea869880.a3m"),
}

#: rung name -> chain lengths. 1024 is the control: it is the size tt_bio/size_limits.py records
#: this model folding on this arch, so a 1536 result is only attributable once the same card and
#: the same engine fold it. 686 is human lactoferrin with its own real alignment, and it is here
#: because a chimera cannot tell a clash the fixture caused from one the hardware caused.
RUNGS = {"1024": [1024], "1536": [1024, 512], "real_686": [686]}


def query(p: Path) -> str:
    with p.open() as fh:
        fh.readline()
        return fh.readline().strip()


def main() -> int:
    seqs, depths = {}, {}
    for n, a3m in CHAINS.items():
        h, s = a3m.stem, query(a3m)
        if len(s) != n:
            raise SystemExit(f"{h}: query is {len(s)} aa, expected {n}")
        if seq_hash(s) != h:
            raise SystemExit(f"{h}: seq_hash is {seq_hash(s)}, so this cache entry is not this sequence")
        dst = MSA / f"{h}.a3m"
        dst.parent.mkdir(exist_ok=True)
        if not dst.exists():
            dst.write_bytes(a3m.read_bytes())
        rows = sum(1 for line in a3m.open() if line.startswith(">"))
        reached = len(parse_a3m(dst, None, MAX_MSA).sequences)
        seqs[n], depths[n] = s, reached
        print(f"chain {n} aa  hash {h}  a3m {rows} records -> {reached} rows reach the model")

    for name, chains in RUNGS.items():
        body = ["sequences:"]
        for i, n in enumerate(chains):
            body += ["  - protein:", f"      id: {chr(ord(chr(65)) + i)}", f"      sequence: {seqs[n]}"]
        (HERE / "inputs" / f"odd_{name}.yaml").write_text("\n".join(body) + "\n")
        ns = sum(2 * n - seqs[n].count("G") for n in chains)
        print(f"rung {name}: {sum(chains)} residue tokens, structural tokens Ns={ns}, "
              f"chains {chains}, depth {[depths[n] for n in chains]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
