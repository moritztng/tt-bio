"""Host input pipeline for Nesso-1: the two things that broke, pinned.

Cheap: no card, no checkpoint, no 413 MB ccd.pkl. The committed tyr48 fixture ships the
21 standard-AA mols the featurizer actually reads.
"""
from __future__ import annotations

import subprocess
import sys
import warnings
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


@pytest.fixture(autouse=True)
def _private_cache(tmp_path, monkeypatch):
    """CcdMols writes its sidecar under the cache root; keep it out of the user's."""
    monkeypatch.setenv("TT_BIO_CACHE", str(tmp_path / "cache"))


def _mol_key(m):
    from rdkit import Chem

    return (Chem.MolToMolBlock(m), [sorted(a.GetPropsAsDict().items()) for a in m.GetAtoms()])


def test_ccd_sidecar_serves_the_same_mols_as_the_full_pickle(tmp_path):
    """build_dataset used to unpickle all of ccd.pkl a second time to keep 21 mols, and
    preprocess unpickled it once more: 10 s of a 20 s featurisation. CcdMols serves the
    standard residues from a derived sidecar. They must be the mols the full load gives,
    on the first (sidecar written) and second (sidecar read) construction alike, and a
    code outside the standard set must still come from the full pickle."""
    import pickle

    import tt_bio.nesso1_input as NI
    from tt_bio._vendor.nesso.data.inference import STANDARD_AA
    from tt_bio._vendor.nesso.data.yaml_input import load_ccd_mol_dict

    full = load_ccd_mol_dict(CCD)
    extra = tmp_path / "ccd.pkl"
    with extra.open("wb") as f:
        pickle.dump({**full, "XYZ": full["ALA"]}, f)
    for writes_sidecar in (True, False):
        NI.CcdMols._full.clear()
        mols = NI.CcdMols(extra)
        std = mols.standard()
        assert sorted(std) == sorted(k for k in STANDARD_AA if k in full)
        assert all(_mol_key(std[k]) == _mol_key(full[k]) for k in std)
        assert bool(NI.CcdMols._full) == writes_sidecar, "only writing the sidecar reads ccd.pkl"
    assert "XYZ" in mols and _mol_key(mols["XYZ"]) == _mol_key(full["ALA"])
    assert "NOPE" not in mols


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


def test_given_esm_path_is_used_and_a_missing_one_raises(tmp_path):
    """An ``esm:`` path only takes effect if it lands in the featurizer's esm dir under
    md5(sequence) before the encoder runs. prepare() collected the path and dropped it, so
    the 650M encoder recomputed the embedding and the supplied file was silently ignored.
    Upstream symlinks and swallows every failure, which makes a typo'd path a silently
    different model input; here it raises."""
    import hashlib

    from tt_bio.nesso1_input import link_given_esm

    seq = "MENFQKVEKIGEGTYGVVYKA"
    mid = hashlib.md5(seq.encode("utf-8")).hexdigest()
    src = tmp_path / "given.safetensors"
    src.write_bytes(b"not really a tensor, but link_given_esm never reads it")
    esm_dir = tmp_path / "esm"
    esm_dir.mkdir()

    assert link_given_esm({mid: str(src)}, esm_dir) == 1
    dst = esm_dir / f"{mid}.safetensors"
    assert dst.exists() and dst.read_bytes() == src.read_bytes()
    assert link_given_esm({mid: str(src)}, esm_dir) == 0  # already there, left alone

    with pytest.raises(FileNotFoundError):
        link_given_esm({mid: str(tmp_path / "nope.safetensors")}, tmp_path / "esm2")


def test_msa_key_is_ignored_out_loud():
    """The README tells users the affinity YAML is the Boltz-2 one, and every
    examples/affinity_*.yaml carries ``msa:``. Nesso-1 never reads an alignment, so the
    vendored parser drops the key — it used to do it in silence, the same way ``esm:``
    was collected and dropped."""
    from tt_bio.nesso1_input import warn_ignored_protein_keys

    with pytest.warns(UserWarning, match="msa"):
        assert warn_ignored_protein_keys([REPO / "examples/affinity_fkg.yaml"]) == ["msa"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_ignored_protein_keys([REPO / "examples/affinity.yaml"]) == []


def test_find_ccd_without_download_names_where_it_looked(tmp_path, monkeypatch):
    """The gate's precondition wants a verdict, not 413 MB, so download=False keeps the
    old behaviour: raise, and name every root searched plus the flags that override it."""
    from tt_bio.nesso1_input import find_ccd

    monkeypatch.delenv("NESSO_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HOME", str(tmp_path))  # so ~/.cache/huggingface is empty too
    with pytest.raises(FileNotFoundError, match="--ccd"):
        find_ccd(tmp_path / "empty", download=False)


def test_esm_embedding_is_computed_once_per_sequence_across_runs(tmp_path, monkeypatch):
    """A screen re-runs the same target with new ligands. The ESM-2 embedding is a pure
    function of the sequence, so the second run must link the stored one, byte for byte,
    instead of loading the 650M encoder again."""
    import torch
    from safetensors.torch import load_file

    import tt_bio._vendor.nesso.data.esm as esm
    from tt_bio.nesso1_input import run_esm

    calls = []
    monkeypatch.setattr(esm, "setup_esm_model", lambda *a, **k: (None, None))
    monkeypatch.setattr(esm, "extract_esm_embedding",
                        lambda seq, m, t: calls.append(seq) or torch.randn(1, len(seq) + 2, 4))
    seqs = {"m1": "MENFQK"}
    assert run_esm(seqs, (tmp_path / "run1").mkdir() or tmp_path / "run1") == 1
    assert run_esm(seqs, (tmp_path / "run2").mkdir() or tmp_path / "run2") == 0
    assert calls == ["MENFQK"]
    a, b = (load_file(tmp_path / r / "m1.safetensors")["embeddings"] for r in ("run1", "run2"))
    assert torch.equal(a, b)
