"""Build the Blackhole OpenFold3 coverage rungs and stage the alignments they fold at.

Adapted from perf/obcov (openbind), because openbind is an OpenFold3 delta and the two share
the token bucketing, the MSA track and the diffusion transformer this coverage bar stresses.
What is NOT shared is the depth reaching the model: openbind dedups its main MSA on its own
checkpoint spec and openfold3 does not, so every depth here is read back off the tensor the
model was handed rather than carried over from that ladder.

The bar is 1536 TOKENS. An apo rung's token count equals its residue count; a holo rung's does
not, because a ligand's heavy atoms are tokens the trunk pays for and are nowhere in the residue
count. openfold3 is the general folding model of the family, so the apo rung is the headline and
the holo rung is the probe on the axis openbind actually died on.

Each chain reuses an a3m the engine itself wrote; `seq_hash` is recomputed from the sequence and
checked against the cache file name, because "this fold ran at depth N" is only a measurement if
the file the engine picks up is the file this script staged.

Depth is NOT capped and must not be. main.py builds `msa_cap` only when the user passes
--max_msa_seqs, and the platform never passes it for the OF3 family, so the served configuration
is the whole resolved alignment.
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
#: tandem tilings reused from the boltz2 bhcov, protenix ptxcov and openbind obcov ladders:
#: chimeric, which is fine for capacity and for backbone continuity and useless for pLDDT.
#: 686 is human lactoferrin with its own real alignment, and it is here because a chimera cannot
#: tell a clash the fixture caused from a clash the hardware caused.
CHAINS = {
    1024: Path("/home/ttuser/esmfold2wh_msa/cache/d4bb492258e7af30.a3m"),
    512: Path("/home/ttuser/esmfold2wh_msa/cache/4e5c2b391bde62d4.a3m"),
    686: Path("/home/ttuser/widek_msa/941f47a0ea869880.a3m"),
}

#: The ligand the holo rung carries. STU is the 35-heavy-atom CCD component the Wormhole
#: openbind ladder walked with (`size_limits.py`, ladder_ligand_atoms=35), so the holo rung here
#: is comparable to that ladder rather than to a ligand nobody measured.
LIGAND_CCD = "STU"
LIGAND_ATOMS = 35

#: rung name -> (chain lengths, ligand?). 1024 is the control: it is what openfold3 is PROVEN at
#: on this board, so a 1536 result is only attributable once the same harness folds the proven
#: rung. 1571_holo is deliberately ABOVE the bar, 1536 residues plus 35 ligand atoms, and 2048
#: is above it as well, which is what turns "1536 is the bar" into "1536 is the bar and the wall
#: is elsewhere": a bar cleared with no failing size above it cannot say which of the two it is.
RUNGS = {
    "1024": ([1024], False),
    "1536": ([1024, 512], False),
    "real_686": ([686], False),
    "1571_holo": ([1024, 512], True),
    "2048": ([1024, 512, 512], False),
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
    for name, (chains, ligand) in RUNGS.items():
        body = ["sequences:"]
        for i, n in enumerate(chains):
            body += ["  - protein:", f"      id: {chr(65 + i)}", f"      sequence: {seqs[n]}"]
        if ligand:
            body += ["  - ligand:", f"      id: {chr(65 + len(chains))}",
                     f"      ccd: {LIGAND_CCD}"]
        (HERE / "inputs" / f"of3_{name}.yaml").write_text("\n".join(body) + "\n")
        tokens = sum(chains) + (LIGAND_ATOMS if ligand else 0)
        lig = f" + {LIGAND_ATOMS} ligand atoms" if ligand else ""
        print(f"rung {name}: {sum(chains)} residues{lig} = {tokens} tokens nominal, "
              f"chains {chains}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
