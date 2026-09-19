"""Build the Blackhole OpenBind-0 coverage rungs and stage the alignments they fold at.

The bar is 1536 TOKENS and openbind is a binding model, so the two axes are kept apart on
purpose. An apo rung's token count equals its residue count; a holo rung's does not, because a
ligand's heavy atoms are tokens the trunk pays for and are nowhere in the residue count. That
is the exact confusion `tt_bio/size_limits.py` records openbind dying on, so the ligand rung
here names its own token arithmetic rather than inheriting the apo one.

Each chain reuses an a3m the engine itself wrote; `seq_hash` is recomputed from the sequence and
checked against the cache file name, because "this fold ran at depth N" is only a measurement if
the file the engine picks up is the file this script staged.

Depth is NOT capped and must not be. The OF3 family leaves `--max_msa_seqs` unapplied at its
default (main.py builds `msa_cap` only when the user passed the flag) and the platform never
passes it, so the served configuration is the whole resolved alignment. Openbind then dedups its
main MSA on its own checkpoint spec, so the depth that reaches the model is not the record count
of the a3m and not openfold3's either. `collect.py` reads it back out of `results.json`
(`feats["msa"].shape[0]`), which is the shape the model was handed.
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
#: tandem tilings reused from the boltz2 bhcov and protenix ptxcov ladders: chimeric, which is
#: fine for capacity and for backbone continuity and useless for pLDDT. 686 is human lactoferrin
#: with its own real alignment, and it is here because a chimera cannot tell a clash the fixture
#: caused from a clash the hardware caused.
CHAINS = {
    1024: Path("/home/ttuser/esmfold2wh_msa/cache/d4bb492258e7af30.a3m"),
    512: Path("/home/ttuser/esmfold2wh_msa/cache/4e5c2b391bde62d4.a3m"),
    686: Path("/home/ttuser/widek_msa/941f47a0ea869880.a3m"),
}

#: The ligand every holo rung carries. STU is the 35-heavy-atom CCD component the Wormhole
#: openbind ladder walked with (`size_limits.py`, ladder_ligand_atoms=35), so the holo rung here
#: is comparable to that ladder rather than to a ligand nobody measured.
LIGAND_CCD = "STU"
LIGAND_ATOMS = 35

#: rung name -> (chain lengths, ligand?). 1024 is the control: it is what openbind is PROVEN at
#: on this board, so a 1536 result is only attributable once the same card folds the proven rung
#: with this harness. 1571_holo is deliberately ABOVE the bar -- 1536 residues plus 35 ligand
#: atoms -- because openbind's recorded Wormhole wall is on tokens and a holo probe is the only
#: thing that tests the axis it actually died on.
RUNGS = {
    "1024": ([1024], False),
    "1536": ([1024, 512], False),
    "real_686": ([686], False),
    "1571_holo": ([1024, 512], True),
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
            body += ["  - protein:", f"      id: {chr(ord('A') + i)}", f"      sequence: {seqs[n]}"]
        if ligand:
            body += ["  - ligand:", f"      id: {chr(ord('A') + len(chains))}",
                     f"      ccd: {LIGAND_CCD}"]
        (HERE / "inputs" / f"ob_{name}.yaml").write_text("\n".join(body) + "\n")
        tokens = sum(chains) + (LIGAND_ATOMS if ligand else 0)
        print(f"rung {name}: {sum(chains)} residues"
              f"{f' + {LIGAND_ATOMS} ligand atoms' if ligand else ''} = {tokens} tokens nominal, "
              f"chains {chains}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
