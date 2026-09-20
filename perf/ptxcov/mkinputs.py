"""Build the Blackhole protenix-v2 coverage rungs and stage the alignments they fold at.

The rungs are token counts, not residue counts, and the two are only equal because every chain
here is protein with no ligand. Each chain reuses an a3m the engine itself wrote; `seq_hash` is
recomputed from the sequence and checked against the cache file name, because "this fold ran at
depth N" is only a measurement if the file the engine picks up is the file this script staged.

Depth is NOT capped here and must not be. protenix-v2 leaves `--max_msa_seqs` unapplied at its
default (worker.py::_build_chain_specs reads `msa_cap`, which is None unless the user passes the
flag), and the platform never passes it. So the served configuration for this model is the whole
resolved alignment, and capping to 8192 to match boltz2 would measure a configuration no user is
in. `depth.py` reads back what actually reaches the model.
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
#: tandem tilings reused from the boltz2 bhcov ladder: chimeric, which is fine for capacity and
#: for backbone continuity and useless for pLDDT. 686 is human lactoferrin with its own real
#: alignment, and it is here because a chimera cannot tell a clash the fixture caused from a
#: clash the hardware caused.
CHAINS = {
    1024: Path("/home/ttuser/esmfold2wh_msa/cache/d4bb492258e7af30.a3m"),
    512: Path("/home/ttuser/esmfold2wh_msa/cache/4e5c2b391bde62d4.a3m"),
    686: Path("/home/ttuser/widek_msa/941f47a0ea869880.a3m"),
}

#: rung name -> chain lengths. 1024 is the control: it is what protenix-v2 is PROVEN at on
#: Wormhole, so a 1536 result is only attributable once the same card folds the proven rung.
RUNGS = {"1024": [1024], "1536": [1024, 512], "real_686": [686]}


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
        print(f"chain {n} aa  hash {h}  a3m {sum(1 for l in a3m.open() if l.startswith(chr(62)))} records staged")

    for name, chains in RUNGS.items():
        body = ["sequences:"]
        for i, n in enumerate(chains):
            body += ["  - protein:", f"      id: {chr(ord(chr(65)) + i)}", f"      sequence: {seqs[n]}"]
        (HERE / "inputs" / f"ptx_{name}.yaml").write_text("\n".join(body) + "\n")
        print(f"rung {name}: {sum(chains)} tokens, chains {chains}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
