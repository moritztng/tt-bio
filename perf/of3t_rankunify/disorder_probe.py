"""Which targets exercise the AF3 RASA `disorder` term at all, measured on experimental
structures instead of on folds.

`of3t-confhead` measured disorder = 0.0 on every sample of every seed of 1UBQ and named that
a property of a compact 76-residue fold rather than of the rule. A whole term of the ranking
rule has therefore never been exercised. Folding candidate targets to find out which ones do
exercise it costs an hour each; the term reads only coordinates, so the ground-truth structure
answers the same question in seconds and picks the fold set on evidence.

Transcribes `openfold3_fold._disorder_score` exactly: per chain, per-residue SASA over max
per-residue SASA (Sander scale), clipped to [0, 1], smoothed with a 25-residue reflect-padded
box, then the fraction of windows above 0.581.
"""
import sys, pathlib
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tt_bio._vendor.openfold3.core.data.resources.residues import RESIDUE_SASA_SCALES

SCALE = RESIDUE_SASA_SCALES["Sander"]


def disorder(array) -> tuple[float, int, int]:
    """Returns (disorder fraction, n residues scored, n chains)."""
    protein = array[struc.filter_amino_acids(array)]
    protein = protein[~protein.hetero] if hasattr(protein, "hetero") else protein
    values = []
    nchain = 0
    for chain in struc.chain_iter(protein):
        if chain.array_length() < 3:
            continue
        nchain += 1
        atom_sasa = struc.sasa(chain, vdw_radii="ProtOr")
        residue_sasa = struc.apply_residue_wise(chain, atom_sasa, np.nansum)
        _, names = struc.get_residues(chain)
        maximum = np.array([SCALE.get(n, 113.0) for n in names])
        rasa = np.clip(residue_sasa / maximum, 0, 1)
        half = 12
        smoothed = np.convolve(
            np.pad(rasa, (half, half), mode="reflect"), np.ones(25), mode="valid") / 25
        values.extend(smoothed)
    v = np.asarray(values)
    return (float(np.mean(v > 0.581)) if len(v) else 0.0, len(v), nchain)


def load(p: pathlib.Path):
    if p.suffix == ".pdb":
        f = pdb.PDBFile.read(str(p))
        a = pdb.get_structure(f, model=1)
    else:
        f = pdbx.CIFFile.read(str(p))
        a = pdbx.get_structure(f, model=1)
    return a[a.element != "H"]


if __name__ == "__main__":
    for p in sorted(pathlib.Path(sys.argv[1]).iterdir()):
        if p.suffix not in (".cif", ".pdb"):
            continue
        try:
            a = load(p)
            d, n, c = disorder(a)
            print(f"{p.name:22s} chains={c:2d} residues={n:5d} disorder={d:.4f}")
        except Exception as e:
            print(f"{p.name:22s} FAILED {type(e).__name__}: {e}")
