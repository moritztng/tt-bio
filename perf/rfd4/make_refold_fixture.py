"""Build a cached refold for one design, in the layout upstream's oracle reads.

Upstream's `OracleMetric.compute` skips the fold entirely when the confidence JSON for every
expected save_id is already on disk, and then reads the refolded CIF beside it. That is the only
seam at which the refolding half of `rfd4-evaluate` can be exercised without AF3: drop a refold and
its confidences into `<out_dir>/evaluation_samples/<example_id>/` and upstream computes every
refolding metric and pass criterion from them.

The refold written here is the design itself, rotated, translated and jittered by a fixed seed, so
the backbone RMSD it produces is a number both implementations must agree on rather than zero. The
confidences come from `tt_bio.rfd4.oracle`, so the file is in the shape the TT oracle will write.

    python3 perf/rfd4/make_refold_fixture.py <design.cif> <out_dir> [--sigma 0.9]
        [--flip-equivalent] [--swap-chains A,B]

`--flip-equivalent` hands every chemically equivalent pair (ASP OD1/OD2, the PHE ring edges, ...)
the other's coordinate, as a real refold does about half the time; `--swap-chains` exchanges two
identical chains. Both are invisible to a model and must be undone by symmetry resolution before
any RMSD, so they grade that step: without it the all-atom numbers move by Angstroms.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from biotite.structure import filter_amino_acids
from biotite.structure.io.pdbx import CIFFile, get_structure

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_bio.rfd4.evaluate import example_id as compute_example_id  # noqa: E402
from tt_bio.rfd4.oracle import extract_confidence_metrics, summary_confidences  # noqa: E402

SEED = 20251021


def _rotation(rng):
    """A random rotation, via the QR of a Gaussian matrix with a positive-determinant fix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def perturbed(atoms, sigma):
    """The design moved into a different frame and jittered: a refold-shaped structure."""
    rng = np.random.default_rng(SEED)
    out = atoms.copy()
    centroid = out.coord.mean(axis=0)
    out.coord = (out.coord - centroid) @ _rotation(rng).T + centroid + np.array([12.0, -7.5, 3.25])
    out.coord = out.coord + rng.normal(scale=sigma, size=out.coord.shape)
    return out


EQUIVALENT = {
    "ARG": [("NH1", "NH2")],
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "LEU": [("CD1", "CD2")],
    "VAL": [("CG1", "CG2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
}


def flip_equivalent(atoms):
    """Swap the coordinates of each equivalent atom pair, residue by residue."""
    out = atoms.copy()
    key = list(zip(atoms.chain_id, atoms.res_id, atoms.atom_name))
    index = {k: i for i, k in enumerate(key)}
    flipped = 0
    for (chain, res, name), i in index.items():
        for a, b in EQUIVALENT.get(atoms.res_name[i], []):
            j = index.get((chain, res, b))
            if name == a and j is not None:
                out.coord[[i, j]] = out.coord[[j, i]]
                flipped += 1
    return out, flipped


def swap_chains(atoms, first, second):
    """Exchange the coordinates of two chains with the same atoms in the same order."""
    out = atoms.copy()
    a, b = atoms.chain_id == first, atoms.chain_id == second
    assert (atoms.atom_name[a] == atoms.atom_name[b]).all(), "chains differ, nothing to swap"
    out.coord[a], out.coord[b] = atoms.coord[b], atoms.coord[a]
    return out


def confidences(atoms, chain_of_interest):
    """A plausible AF3 summary for this design, reduced by the same code the TT oracle uses."""
    rng = np.random.default_rng(SEED + 1)
    ca = atoms[filter_amino_acids(atoms) & (atoms.atom_name == "CA")]
    n = ca.array_length()
    chains = [f"{c}_1" for c in ca.chain_id]
    plddt = np.clip(rng.normal(0.88, 0.06, n), 0.0, 1.0)
    pae = np.abs(rng.normal(0.0, 1.0, (n, n))) + 0.4
    np.fill_diagonal(pae, 0.2)
    order_labels = list(dict.fromkeys(chains))
    k = len(order_labels)
    summary, order = summary_confidences(
        plddt=plddt,
        pae=pae,
        chains=chains,
        ptm=0.84,
        iptm=0.79 if k > 1 else None,
        chain_ptm=[0.84] * k,
        chain_pair_iptm=np.full((k, k), 0.79) if k > 1 else None,
        ranking_score=0.81,
        fraction_disordered=0.03,
        has_clash=0.0,
    )
    metrics = extract_confidence_metrics(summary, chain_of_interest, order)
    metrics["chain_of_interest"] = chain_of_interest
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--inputs", type=Path, default=None, help="INPUTS root the example_id hashes against")
    ap.add_argument("--sigma", type=float, default=0.9, help="per-coordinate jitter, Angstrom")
    ap.add_argument("--oracle-name", default="af3")
    ap.add_argument("--chain-of-interest", default=None)
    ap.add_argument("--flip-equivalent", action="store_true")
    ap.add_argument("--swap-chains", default=None, help="two chain ids, e.g. A,B")
    args = ap.parse_args()

    inputs = args.inputs or args.design.parent
    eid = compute_example_id(args.design, inputs)
    atoms = get_structure(CIFFile.read(str(args.design)), model=1, include_bonds=True)
    coi = args.chain_of_interest or f"{sorted(set(atoms.chain_id))[0]}_1"

    design_dir = args.out_dir / "evaluation_samples" / eid
    design_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{eid}_{args.oracle_name}"

    # The refold keeps every column and conditioning loop of the design and changes only the
    # coordinates, which is what upstream's own oracle writes: it builds the refold from the
    # design's annotated atom array (`af3_oracle.py:733-752`). A refold written bare instead
    # makes `is_chain_of_interest` ambiguous on the refold side and takes down every metric that
    # reads it (`utils/chain.py:24`) -- the whole oracle row with them.
    refolded = CIFFile.read(str(args.design))
    moved = perturbed(atoms, args.sigma)
    if args.flip_equivalent:
        moved, flipped = flip_equivalent(moved)
        print(f"flipped {flipped} equivalent pairs", file=sys.stderr)
    if args.swap_chains:
        moved = swap_chains(moved, *args.swap_chains.split(","))
    for axis, column in enumerate(("Cartn_x", "Cartn_y", "Cartn_z")):
        refolded.block["atom_site"][column] = np.array([f"{v:.3f}" for v in moved.coord[:, axis]])
    refolded.write(str(design_dir / f"{stem}_refolded.cif"))
    (design_dir / f"{stem}_confidences.json").write_text(json.dumps(confidences(atoms, coi), indent=2))
    print(eid)
    print(design_dir / f"{stem}_refolded.cif")


if __name__ == "__main__":
    main()
