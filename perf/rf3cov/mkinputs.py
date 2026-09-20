"""Build the Blackhole RoseTTAFold3 coverage rungs and stage the alignments they fold at.

The bar for bh-p150a is 1536 TOKENS. Every rung here is protein-only with no ligand, so a rung's
token count equals its residue count -- stated rather than assumed, because the two part company
the moment a ligand appears and `collect.py` reads the real number back out of the run.

Each chain reuses an a3m the engine itself wrote. `seq_hash` is recomputed from the sequence and
checked against the cache file name, because "this fold ran at depth N" is only a measurement if
the file the engine picks up is the file this script staged. The chains are the same three the
boltz2 (`bhcov`), protenix (`ptxcov`) and openbind (`obcov`) Blackhole ladders walked, so an rf3
rung is directly comparable with theirs instead of being its own private fixture.

Depth is NOT capped and must not be. `main.py` builds `msa_cap` only when the user passed
--max_msa_seqs, and rf3 is on the branch that folds the resolved alignment whole, so the served
configuration is no cap at all. rf3 then builds its own msa_stack, so the depth reaching the
model is not the record count of the a3m; `collect.py` reads `msa_depth` out of `results.json`,
which is `f["msa_stack"].shape[1]`, the shape the model was handed.
"""
from __future__ import annotations

import sys
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

from tt_bio.cache import seq_hash

HERE = Path(__file__).parent
MSA = HERE / "msa"

#: chain length -> the a3m the engine itself wrote for that sequence. 1024 and 512 are CDK2
#: tandem tilings: chimeric, which is fine for capacity and for backbone continuity and useless
#: for pLDDT. 686 is human lactoferrin with its own real alignment, and it is here because a
#: chimera cannot tell a clash the fixture caused from a clash the hardware caused.
CHAINS = {
    1024: Path("/home/ttuser/esmfold2wh_msa/cache/d4bb492258e7af30.a3m"),
    512: Path("/home/ttuser/esmfold2wh_msa/cache/4e5c2b391bde62d4.a3m"),
    686: Path("/home/ttuser/widek_msa/941f47a0ea869880.a3m"),
}

#: rung name -> chain lengths. 1024 is the control: it is what rf3 is PROVEN at on this board
#: (size_evidence, 2026-09-07, 14190 rows), so a 1536 result is only attributable once the same
#: card folds the proven rung through this harness at this depth. 2048 is above the bar and it
#: is what turns "1536 is the bar" into "1536 is the bar and the wall is elsewhere" -- a bar
#: cleared with no failing size above it cannot say which of the two it is. Every rung is a
#: multiple of 32, so the token bucket cannot move any of them.
RUNGS = {
    "1024": [1024],
    "1536": [1024, 512],
    "real_686": [686],
    "2048": [1024, 512, 512],
}


def query(p: Path) -> str:
    with p.open() as fh:
        fh.readline()
        return fh.readline().strip()


def main() -> int:
    seqs = {}
    MSA.mkdir(exist_ok=True)
    for n, a3m in CHAINS.items():
        h, s = a3m.stem, query(a3m)
        if len(s) != n:
            raise SystemExit(f"{h}: query is {len(s)} aa, expected {n}")
        if seq_hash(s) != h:
            raise SystemExit(f"{h}: seq_hash is {seq_hash(s)}, so this cache entry is not this sequence")
        dst = MSA / f"{h}.a3m"
        if not dst.exists():
            dst.write_bytes(a3m.read_bytes())
        seqs[n] = s
        records = sum(1 for line in a3m.open() if line.startswith(">"))
        print(f"chain {n} aa  hash {h}  a3m {records} records staged")

    (HERE / "inputs").mkdir(exist_ok=True)
    for name, chains in RUNGS.items():
        body = ["sequences:"]
        for i, n in enumerate(chains):
            body += ["  - protein:", f"      id: {chr(ord('A') + i)}", f"      sequence: {seqs[n]}"]
        (HERE / "inputs" / f"rf3_{name}.yaml").write_text("\n".join(body) + "\n")
        print(f"rung {name}: {sum(chains)} residues = {sum(chains)} tokens nominal, chains {chains}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
