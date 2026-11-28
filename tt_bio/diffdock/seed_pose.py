"""The `--no_model` baseline: a randomized RDKit seed conformer, no network at all.

Upstream ships this (`evaluate.py --no_model`, "return seed conformer without running
model") as the null against which a sampled pose is judged. It needs no checkpoint and no
device, which also makes it the path that lets `tt-bio dock` run end to end today: the whole
surface -- input forms, pocket, seed, ranking, output layout, failure isolation -- is
exercised by a real run before any model work lands.

The randomization mirrors upstream's ``randomize_position``: a uniform random rotation about
the ligand's own centre, then a translation. Without a pocket the translation is Gaussian
with std ``initial_noise_std * tr_sigma_max`` about the receptor centre, which is upstream's
prior; with ``--pocket`` it is a uniform point in a ``--pocket_radius`` ball about the pocket
centre, which is what pocket knowledge buys.
"""

from __future__ import annotations

import numpy as np

#: Upstream's translation prior width for the v1.1 score model (model_parameters.yml
#: tr_sigma_max). Pinned here because the seed path loads no checkpoint to read it from.
TR_SIGMA_MAX = 19.0


def load_ligand(description: str):
    """SMILES or a molecule file, in upstream's detection order (file first, then SMILES)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from pathlib import Path

    path = Path(description).expanduser()
    mol = None
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".sdf":
            mol = next(iter(Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)), None)
        elif suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=True)
        elif suffix == ".mol":
            mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=True)
        elif suffix == ".pdb":
            mol = Chem.MolFromPDBFile(str(path), sanitize=True, removeHs=True)
        elif suffix in (".smi", ".smiles"):
            mol = Chem.MolFromSmiles(path.read_text().split()[0])
        if mol is None:
            raise ValueError(f"{path.name}: rdkit could not read a molecule from this file")
    else:
        mol = Chem.MolFromSmiles(description)
        if mol is None:
            raise ValueError(f"could not read the ligand as a file or as SMILES: {description!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
        raise ValueError("rdkit could not embed a 3D conformer for this ligand")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    return Chem.RemoveHs(mol)


def receptor_centre(path) -> np.ndarray:
    """The CA centroid of a PDB/mmCIF receptor, the centre upstream's prior is about."""
    from pathlib import Path

    path = Path(path)
    coords: list[tuple[float, float, float]] = []
    if path.suffix.lower() in (".cif", ".mmcif"):
        import gemmi

        st = gemmi.read_structure(str(path))
        for chain in st[0]:
            for res in chain:
                atom = res.find_atom("CA", "*")
                if atom is not None:
                    coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
    else:
        for line in path.read_text().splitlines():
            if line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip() == "CA":
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not coords:
        raise ValueError(f"{path.name}: no CA atoms found, so this is not a receptor structure")
    return np.asarray(coords, dtype=float).mean(axis=0)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniform rotation (Shoemake's quaternion sampling), as upstream's Rotation.random."""
    u1, u2, u3 = rng.random(3)
    q = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                  np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3),
                  np.sqrt(u1) * np.cos(2 * np.pi * u3)])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def seed_poses(mol, centre: np.ndarray, options, *, n: int) -> np.ndarray:
    """``n`` randomized copies of the seed conformer: (n, atoms, 3)."""
    rng = np.random.default_rng(0 if options.seed is None else options.seed)
    base = mol.GetConformer().GetPositions()
    base = base - base.mean(axis=0)
    out = np.empty((n, base.shape[0], 3))
    for i in range(n):
        rotated = base @ _random_rotation(rng).T
        if options.pocket is not None:
            # A uniform point in the pocket ball, not a Gaussian: pocket knowledge is a
            # hard constraint on where the ligand may start (evaluate.py --pocket_cutoff).
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            offset = np.asarray(options.pocket) + direction * options.pocket_radius * rng.random() ** (1 / 3)
        else:
            offset = centre + rng.normal(scale=options.initial_noise_std * TR_SIGMA_MAX, size=3)
        out[i] = rotated + offset
    return out
