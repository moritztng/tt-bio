"""The RFD4-Proteina training batch: upstream's crops, loaded with numpy alone.

`perf/r4t/build_fixture.py` runs upstream's own pipeline (`build_design_train_pipeline` at the
release pretraining settings: crop 256, spatial crops, random rotation, the three loss-weight
predicates, centroid conditioning) over upstream's published `test_dataset` and collates with
upstream's `collate_batch` at ``fixed_dimensions={I: 256}``. What it writes is the fixture this
module reads: the arrays the objective and the network actually consume, and nothing else, so
the tests need neither upstream nor atomworks nor 2 TB of PDB.

The clean sample is two modalities. ``bb_ca`` is the fixture's own geometry -- upstream's
``coords[:, :, 1, :] / 10``, the CA trace in nm (`proteina.py:745`). ``local_latents`` is the
frozen autoencoder's ``encode(batch)["z_latent"]``, so it needs the AE: pass ``encode=`` and
get upstream's target, or pass nothing and get a :class:`KeyError` naming the AE rather than a
quiet zero. A zero latent is a different training target, not a missing feature, and upstream
warns and returns zeros only when it has no autoencoder at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

__all__ = ["FIXTURE", "FIXTURE_KEYS", "LATENT_PLACEHOLDER", "load", "clean_samples"]

#: Shipped with the package; see `perf/r4t/build_fixture.py` for how it was built.
FIXTURE = Path(__file__).with_name("fixtures") / "rfd4_train_16.npz"

#: Exactly what the objective, the featuriser and the parity harness read, in upstream's own
#: names (`datapipes/schema/proteina.py::SPECS`, plus `padding_mask` from the collate).
FIXTURE_KEYS = (
    "padding_mask", "is_in_context_conditioning", "supports_local_latent", "is_resolved_mask",
    "coord_cond_mask", "molecule_iid", "chain_iid", "residue_type", "token_type",
    "insertion_source_idx", "tetrahedral_vertices", "loss_weight",
    "condition_sequence_1_residue", "coords", "atom37_mask",
)

#: The release autoencoder's latent width (`hero_pretrain.yaml` -> AE checkpoint, 32).
LATENT_PLACEHOLDER = 32


def load(path=FIXTURE) -> dict:
    """The fixture as a batch dict: upstream's keys, plus ``exploration_k`` and the ids."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it with `DATA_PATH=<dir> perf/r4t/build_fixture.py "
            f"--out {path}` in an environment that has upstream RFD4-Proteina and its "
            f"test_dataset (scripts/download_assets.sh --with-test-dataset)")
    with np.load(path, allow_pickle=False) as z:
        batch = {k: z[k] for k in z.files}
    batch["exploration_k"] = int(batch["exploration_k"])
    return batch


def clean_samples(batch: dict, encode: Optional[Callable] = None,
                  latent_dim: int = LATENT_PLACEHOLDER) -> dict:
    """``x_1`` per modality: the CA trace in nm, and the frozen AE's latents.

    ``encode`` is the autoencoder's ``encode(batch) -> [b, n, latent]``. Without it the latent
    modality cannot be built and this refuses, because training against zeros instead is a
    silently different objective (LESSONS-OF3T T1: never substitute a zero for a value you do
    not have).
    """
    ca = np.asarray(batch["coords"], np.float64)[:, :, 1, :] / 10.0
    if encode is None:
        raise KeyError(
            "local_latents is the frozen autoencoder's encode(batch)['z_latent'] and no "
            "encoder was given. Pass encode=<ae.encode>; the fixture cannot carry the latents "
            "because they are a function of the AE checkpoint, and zeros are a different "
            "training target rather than a missing one")
    z = np.asarray(encode(batch), np.float64)
    if z.shape[:2] != ca.shape[:2] or z.shape[-1] != latent_dim:
        raise ValueError(f"encoder returned {z.shape}, expected {ca.shape[:2] + (latent_dim,)}")
    return {"bb_ca": ca, "local_latents": z}
