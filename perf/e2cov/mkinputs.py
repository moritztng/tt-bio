"""Build the Blackhole ESMFold2 coverage rungs and prove the alignment depth that reaches
the model.

Rungs are TOKEN counts. Every chain here is protein with no ligand, so tokens equal residues.
Each chain reuses an a3m the engine itself wrote, and the cache key is recomputed from the
sequence rather than trusted: `seq_hash(query) == filename` is the only thing that makes
"this fold ran at depth N" a measurement.

Depth is then read back through the model's OWN featurizer, not off the a3m. An a3m with
14721 `>` records arrives as far fewer rows: `MSA.from_a3m` caps at --max_msa_seqs (8192 is
esmfold2's shipped default) and `construct_paired_msa` then merges the chains by taxonomy,
so the number that matters is the first dimension of the `msa` tensor the model is handed.
"""
from __future__ import annotations

import sys
from pathlib import Path

WT = "/home/ttuser/.coworker/wt/cov-unproven-esmfold2-bhp150a"
sys.path.insert(0, WT)
from tt_bio.cache import seq_hash

HERE = Path(__file__).parent
MSA = HERE / "msa"
#: esmfold2's shipped --max_msa_seqs. On Blackhole `msa_depth_cap` is a no-op, so this is
#: the depth the engine uses; on Wormhole it would be narrowed by L*M.
MAX_MSA = 8192

#: chain length -> the a3m the engine itself wrote for that sequence. Both are CDK2 tandem
#: tilings from the esmfold2 Wormhole ladder, and they are the same two chains the boltz2
#: Blackhole coverage rung used, so the two models' 1536 rungs are the same fixture.
CHAINS = {
    1024: Path("/home/ttuser/esmfold2wh_msa/cache/d4bb492258e7af30.a3m"),
    512: Path("/home/ttuser/esmfold2wh_msa/cache/4e5c2b391bde62d4.a3m"),
}

#: rung -> chain lengths. 1024 is the control: it is what esmfold2 is PROVEN at on the
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
            raise SystemExit(f"{h}: seq_hash is {seq_hash(s)}, so this cache entry is not this sequence")
        dst = MSA / f"{h}.a3m"
        dst.parent.mkdir(exist_ok=True)
        if not dst.exists():
            dst.write_bytes(a3m.read_bytes())
        rows = sum(1 for line in a3m.open() if line.startswith(">"))
        seqs[n] = s
        print(f"chain {n} aa  hash {h}  a3m {rows} records")

    for name, chains in RUNGS.items():
        body = ["sequences:"]
        for i, n in enumerate(chains):
            body += ["  - protein:", f"      id: {chr(ord('A') + i)}", f"      sequence: {seqs[n]}"]
        (HERE / "inputs" / f"e2_{name}.yaml").write_text("\n".join(body) + "\n")
        print(f"rung {name}: {sum(chains)} tokens, chains {chains}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
