"""`--seed` makes a BoltzGen design reproducible.

The design and refold data loaders used to hand the featurizer
``np.random.default_rng(None)``, an OS-entropy generator no seed reached, so the same
spec featurized differently on every run even under ``--seed``. This drives the real
design loader's ``get_sample`` twice under one seed and asserts identical features.

CPU-only; skips when the bundled CCD mol library is absent.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

_MOL_DIR = Path(os.path.expanduser("~/.boltz/mols"))
pytestmark = pytest.mark.skipif(
    not (_MOL_DIR / "ALA.pkl").exists(), reason="needs bundled CCD mol library (~/.boltz/mols)")

# A fixed target plus a binder whose length is itself a draw.
_SPEC = """entities:
  - protein: {id: A, sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ}
  - protein: {id: B, sequence: 20..40}
"""


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    from tt_bio.boltzgen.data.featurizer import Featurizer
    from tt_bio.boltzgen.data.tokenizer import Tokenizer
    from tt_bio.boltzgen.task.predict.data_from_yaml import Dataset, PredictionDataset
    from tt_bio.data.mol import load_canonicals

    spec = tmp_path_factory.mktemp("spec") / "binder.yaml"
    spec.write_text(_SPEC)
    return PredictionDataset(
        Dataset(yaml_path=str(spec), tokenizer=Tokenizer(atomize_modified_residues=False),
                featurizer=Featurizer()),
        canonicals=load_canonicals(_MOL_DIR), moldir=_MOL_DIR,
        atom14=True, disulfide_on=True)


def _sample(dataset, seed):
    from tt_bio.runtime import seed_everything

    if seed is not None:
        seed_everything(seed)
    return dataset[0]


def _same(a: dict, b: dict) -> bool:
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor):
            if x.shape != y.shape or not torch.equal(x, y):
                return False
        elif isinstance(x, np.ndarray):
            if x.shape != y.shape or not np.array_equal(x, y):
                return False
    return True


def test_same_seed_same_features(dataset):
    assert _same(_sample(dataset, 7), _sample(dataset, 7))


def test_other_seed_or_no_seed_draws_fresh(dataset):
    base = _sample(dataset, 7)
    assert not _same(base, _sample(dataset, 8))
    # Unseeded stays a fresh draw: it continues the stream rather than restarting it.
    assert not _same(_sample(dataset, None), _sample(dataset, None))
