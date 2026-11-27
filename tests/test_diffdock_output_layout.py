"""What a docking run leaves on disk, graded against upstream's own code.

Screening pipelines parse these filenames, so the layout is part of the interface:

    <out_dir>/<name>/rank1.sdf
    <out_dir>/<name>/rank{N}_confidence{X.XX}.sdf
    <out_dir>/<name>/rank{N}_reverseprocess.pdb     # --save_trajectory

The reference is upstream's source, not a restatement of it. `write_mol_with_coords` and
`PDBFile` are lifted out of the DiffDock checkout at `~/scratch/dfd/upstream` and run here:
`datasets/process_mols.py` imports `torch_cluster` at module level and will not import in
this env, so that one function is pulled out with `ast`; `utils/visualise.py` imports only
rdkit, numpy and torch and is loaded whole, by path. Without the checkout the tests skip.

The rollout is short on purpose. These tests grade filenames, frame counts and coordinate
records, none of which depend on the pose being any good, and a truncated rollout is never
used here to grade accuracy (D-dfd-orch-10's corollary).
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from conftest import DIFFDOCK_RECEPTOR as RECEPTOR, dock_on_host_torch, sdf_coords

UPSTREAM = Path.home() / "scratch/dfd/upstream"

SAMPLES = 4
ACTUAL_STEPS = 3


def _need(path: Path, what: str):
    if not path.exists():
        pytest.skip(f"no {what} at {path}")


def _upstream_write_mol_with_coords():
    """Upstream's `write_mol_with_coords`, executed from its own source text."""
    _need(UPSTREAM / "datasets/process_mols.py", "upstream checkout")
    import numpy as np
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    tree = ast.parse((UPSTREAM / "datasets/process_mols.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "write_mol_with_coords":
            ns = {"Chem": Chem, "Point3D": Point3D, "np": np}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<upstream>", "exec"), ns)
            return ns["write_mol_with_coords"]
    pytest.skip("upstream has no write_mol_with_coords")


def _upstream_pdbfile():
    """Upstream's `PDBFile`, loaded from `utils/visualise.py` by path."""
    path = UPSTREAM / "utils/visualise.py"
    _need(path, "upstream checkout")
    spec = importlib.util.spec_from_file_location("dfd_upstream_visualise", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PDBFile


def _pdb_models(text: str):
    """`[[(x, y, z), ...], ...]`, one list per MODEL record."""
    frames, current = [], None
    for line in text.splitlines():
        if line.startswith("MODEL"):
            current = []
        elif line.startswith("ENDMDL"):
            frames.append(current)
            current = None
        elif current is not None and line.startswith(("ATOM", "HETATM")):
            current.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return frames


# -- the layout -------------------------------------------------------------------------

def test_filenames_and_coords_match_upstream(tmp_path, monkeypatch):
    """Every pose file upstream would write, with upstream's bytes inside it.

    Three things are graded. The *names* are built here with upstream's own f-strings from
    `inference.py:286-290`, from the confidences the run reported, so a change to the
    rounding or to the rank-1 duplicate shows up as a missing file. The *bytes* are compared
    against upstream's `write_mol_with_coords` run on the molecule read back out of our own
    file, which is what catches a writer that drops a conformer or writes V3000. And the
    *frame* is checked against the sampler's own output: upstream adds `original_center`
    back before writing, so every written pose must be some sampled pose plus one shared
    constant, and that constant must be a point inside the input receptor.
    """
    import numpy as np

    upstream_write = _upstream_write_mol_with_coords()
    rollout: list = []
    out = dock_on_host_torch(tmp_path / "results", monkeypatch, samples=SAMPLES,
                             actual_steps=ACTUAL_STEPS, rollout=rollout)

    write_dir = out / "cx"
    assert write_dir.is_dir(), "one directory per complex, named by the complex"
    assert not (out / "failures.csv").exists(), "a single-complex run that worked writes no failures.csv"

    ranking = json.loads((write_dir / "confidence.json").read_text())
    assert [row["rank"] for row in ranking] == list(range(1, SAMPLES + 1))
    confidences = [row["confidence"] for row in ranking]
    assert all(c is not None for c in confidences)
    assert confidences == sorted(confidences, reverse=True)

    # Upstream's own two lines, inference.py:286-290.
    expected = {f"rank{rank + 1}_confidence{confidences[rank]:.2f}.sdf"
                for rank in range(SAMPLES)}
    expected.add("rank1.sdf")
    assert len(expected) == SAMPLES + 1, "two poses rounded to the same filename; pick another seed"
    assert {p.name for p in write_dir.glob("*.sdf")} == expected

    # Rank 1 is written twice, bare and suffixed, byte for byte.
    suffixed = write_dir / f"rank1_confidence{confidences[0]:.2f}.sdf"
    assert (write_dir / "rank1.sdf").read_bytes() == suffixed.read_bytes()

    mol, coords = sdf_coords(write_dir / "rank1.sdf")
    assert all(a.GetSymbol() != "H" for a in mol.GetAtoms()), \
        "the released score model sets remove_hs, so the written pose is heavy-atom only"

    # Upstream's writer, same molecule, same coordinates, same bytes.
    reference = tmp_path / "upstream_rank1.sdf"
    upstream_write(mol, np.asarray(coords, dtype=np.float64), str(reference))
    assert reference.read_text() == (write_dir / "rank1.sdf").read_text()

    # Back in the input frame. Every written pose is one sampled pose plus original_center:
    # the offset between them has to be constant over the molecule, the same for every rank,
    # and the rank -> sample map has to be a permutation.
    written = np.asarray([sdf_coords(write_dir / f"rank{r + 1}_confidence"
                                      f"{confidences[r]:.2f}.sdf")[1]
                          for r in range(SAMPLES)])
    sampled = np.asarray(rollout[0])
    assert sampled.shape == written.shape, "the sampler produced a different set of poses"

    offsets = written[:, None] - sampled[None, :]                  # [rank, sample, atom, 3]
    rigid = offsets.max(axis=2) - offsets.min(axis=2) < 1e-3       # [rank, sample, 3]
    match = rigid.all(axis=-1)
    assert match.sum(axis=1).tolist() == [1] * SAMPLES, \
        "a written pose is not any sampled pose shifted by a constant"
    assert sorted(match.argmax(axis=1).tolist()) == list(range(SAMPLES)), \
        "the ranking is not a permutation of the sampled poses"

    centre = offsets[0, match[0].argmax()].mean(axis=0)
    for rank in range(SAMPLES):
        assert offsets[rank, match[rank].argmax()].mean(axis=0) == pytest.approx(centre, abs=1e-3), \
            "the poses were not all shifted by the same original_center"
    lo, hi = _receptor_box()
    assert np.all(centre > lo) and np.all(centre < hi), \
        f"the shift {centre} is not a point in the receptor {lo}..{hi}: that is not original_center"


def _receptor_box():
    import numpy as np

    pos = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
           for l in RECEPTOR.read_text().splitlines() if l.startswith("ATOM")]
    pos = np.asarray(pos)
    return pos.min(axis=0), pos.max(axis=0)


def test_visualisation_frames_written(tmp_path, monkeypatch):
    """`--save_trajectory` (upstream `--save_visualisation`) writes one PDB per ranked pose.

    What is graded: one file per pose named by its *rank*, one MODEL per reverse-diffusion
    frame, the last frame equal to that rank's written pose -- which is the assertion that
    fails if the trajectories are not reordered alongside the poses, upstream's
    `visualization_list[batch_idx]` indexed by `re_order` -- and ATOM records identical to
    what upstream's `PDBFile` emits for the same coordinates.

    The frame count is ours and differs from upstream by design: upstream's file opens with
    two prologue models (the input conformer and the crystal ligand position) before the
    reverse process starts, and `tt-bio dock` writes the reverse process only, `--steps + 1`
    models from the randomised start.
    """
    import numpy as np

    PDBFile = _upstream_pdbfile()
    out = dock_on_host_torch(tmp_path / "results", monkeypatch, samples=SAMPLES,
                             actual_steps=ACTUAL_STEPS, save_trajectory=True)
    write_dir = out / "cx"

    names = sorted(p.name for p in write_dir.glob("*_reverseprocess.pdb"))
    assert names == sorted(f"rank{r}_reverseprocess.pdb" for r in range(1, SAMPLES + 1))

    confidences = [row["confidence"] for row in
                   json.loads((write_dir / "confidence.json").read_text())]

    for rank in range(1, SAMPLES + 1):
        frames = _pdb_models((write_dir / f"rank{rank}_reverseprocess.pdb").read_text())
        assert len(frames) == ACTUAL_STEPS + 1, \
            "one model per denoising step plus the randomised start"
        n_atoms = {len(f) for f in frames}
        assert len(n_atoms) == 1, "every frame is the same molecule"

        mol, coords = sdf_coords(
            write_dir / f"rank{rank}_confidence{confidences[rank - 1]:.2f}.sdf")
        assert np.allclose(frames[-1], coords, atol=1e-3), \
            f"rank {rank}'s last frame is not rank {rank}'s pose: the trajectories were not reordered"
        assert not np.allclose(frames[0], coords, atol=1e-3), \
            "the first frame is the randomised start, not the final pose"

        # Upstream's PDBFile, same molecule, same coordinates, same ATOM records.
        pdb = PDBFile(mol)
        pdb.add(np.asarray(coords, dtype=np.float64), 0, 0)
        reference = _pdb_models(pdb.write())
        assert len(reference) == 1
        assert np.allclose(reference[0], frames[-1], atol=1e-3), \
            "our MODEL records are not what upstream's PDBFile writes for these coordinates"

    assert not list((out / "cx").glob("*.pdb.gz"))


def test_no_visualisation_without_the_flag(tmp_path, monkeypatch):
    out = dock_on_host_torch(tmp_path / "results", monkeypatch, samples=2, actual_steps=2)
    assert not list((out / "cx").glob("*_reverseprocess.pdb"))
