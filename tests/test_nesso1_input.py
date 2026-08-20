"""Host input pipeline for Nesso-1: the two things that broke, pinned.

Cheap: no card, no checkpoint, no 413 MB ccd.pkl. The committed tyr48 fixture ships the
21 standard-AA mols the featurizer actually reads.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "scripts/nesso1_port/parity_artifacts/tyr48"
YAML = FIXTURE / "tyr48.yaml"
CCD = FIXTURE / "standard_aa_mols.pkl"

pytestmark = pytest.mark.skipif(
    not YAML.exists() or not CCD.exists(), reason="tyr48 fixture not present"
)


def _preprocess(out_dir, num_workers):
    from tt_bio.nesso1_input import preprocess, resolve_paths

    paths = resolve_paths(out_dir)
    manifest, failed = preprocess([YAML], paths, CCD, num_workers=num_workers)
    return paths, manifest, failed


def test_num_workers_zero_parses_inline(tmp_path):
    """Upstream dies here: 0 goes straight into ProcessPoolExecutor and raises
    ValueError: max_workers must be greater than 0. Everywhere else in the ecosystem 0
    means no worker processes, so ours parses inline."""
    paths, manifest, failed = _preprocess(tmp_path, 0)
    assert failed == []
    assert [r.id for r in manifest.records] == ["tyr48"]
    assert (paths.structures_dir / "tyr48.npz").exists()
    assert (paths.mol_dir / "tyr48__B.pkl").exists()


def test_inline_and_pooled_agree_on_everything_but_the_conformer(tmp_path):
    """The inline path is the same code as the pooled one, and the record it writes is
    byte-identical. The ligand conformer is NOT: ETKDG draws its embedding seed from the
    process RNG state, so a worker process and the parent produce coordinates that differ
    (first atoms agree, later ones do not). Protein coordinates come from the CCD and are
    stable. So num_workers changes the model input for any SMILES ligand -- one more face
    of the featurization-samples trap, alongside center_random_augmentation. Anything
    comparing numbers across runs has to commit the conformer, which is why the parity
    fixtures ship rdkit_conformers/ rather than regenerating it."""
    inline, _, _ = _preprocess(tmp_path / "inline", 0)
    pooled, _, _ = _preprocess(tmp_path / "pooled", 2)
    for rel in ("records/tyr48.json", "manifest.json"):
        assert (inline.processed / rel).read_bytes() == (pooled.processed / rel).read_bytes(), rel
    a = np.load(inline.structures_dir / "tyr48.npz")
    b = np.load(pooled.structures_dir / "tyr48.npz")
    assert sorted(a.files) == sorted(b.files)
    moved = []
    for k in a.files:
        if a[k].dtype.names and "coords" in a[k].dtype.names:
            assert a[k].shape == b[k].shape, k  # same atoms in the same order
            if not np.array_equal(a[k], b[k]):
                moved.append(k)
            continue
        assert np.array_equal(a[k], b[k]), k
    assert moved, "expected the conformer coordinates to move across the process boundary"


def test_ccd_atom_names_survive_unpickling():
    """RDKit drops atom properties when UNPICKLING too, unless SetDefaultPickleProperties
    ran first. tt_bio.nesso1_input sets it at import; without that every standard-residue
    mol comes back nameless and process_atom_features dies with KeyError: name. It only
    bites when the ccd load happens before the vendored featurizer module is imported,
    which is what makes it an import-order bug. Subprocess, so the process-global RDKit
    setting is not already on from another test."""
    code = "\n".join([
        "import sys; sys.path.insert(0, %r)" % str(REPO),
        "from pathlib import Path",
        "import tt_bio.nesso1_input  # noqa: F401 - sets the RDKit pickle option",
        "from tt_bio._vendor.nesso.data.yaml_input import load_ccd_mol_dict",
        "mols = load_ccd_mol_dict(Path(%r))" % str(CCD),
        "names = [a.GetProp('name') for a in mols['TYR'].GetAtoms() if a.GetAtomicNum() != 1]",
        "print(' '.join(sorted(names)))",
    ])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr[-2000:]
    names = out.stdout.strip().splitlines()[-1].split()
    assert "CA" in names and "OH" in names, names
