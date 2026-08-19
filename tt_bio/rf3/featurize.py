"""RF3 host featurizer.

Turns a user input (JSON / CIF / PDB) into the 35-key ``f`` dict the on-device RF3
trunk and diffusion module consume, plus the ground-truth, confidence and sampler
tensors that ride alongside it.

The AF3 transform pipeline itself is upstream code, vendored under
``tt_bio._vendor.atomworks`` and ``tt_bio._vendor.rf3`` rather than reimplemented:
it is host-side, it is where every capability lives (nucleic acids, ligands,
covalent modifications, chirality perception, MSA pairing, templating), and
reimplementing it would buy nothing but a chance to diverge.

Upstream builds this pipeline from a config stored inside the checkpoint. That
config resolves to a flat kwargs dict, reproduced here as ``PIPELINE_CONFIG`` so
tt-bio needs neither hydra nor the checkpoint to featurize. It is verified against
committed reference captures by ``scripts/rf3_port/parity_gate.py``.

Two facts that bite if you touch this:

- Seed ``random``, ``numpy`` AND ``torch``. RDKit's ETKDG conformer embedding reads
  the python/numpy RNG, so a torch-only seed leaves ``f["ref_pos"]`` different run
  to run. :func:`seed_everything` does all three.
- ``f["cyclic_asym_ids"]`` is a plain list, not a tensor. The model reads it.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

# Resolved from `cfg.datasets.val.af3_validation.dataset.transform` merged with
# RF3InferenceEngine's transform_overrides, checkpoint rf3_foundry_01_24_latest_remapped.
# Values that are engine arguments rather than checkpoint constants are named in
# DEFAULTS below and can be overridden per call.
PIPELINE_CONFIG: dict[str, Any] = {
    "is_inference": True,
    "add_residue_is_paired_feature": True,
    "allowed_chain_types_for_conditioning": None,
    "crop_size": None,
    "fallback_conformer_to_input_coords": True,
    "max_atoms_in_crop": None,
    "n_msa": 1024,
    "p_dropout_atom_level_embeddings": 0.0,
    "p_dropout_ref_conf": 0.0,
    "p_give_non_polymer_ref_conf": 0.0,
    "p_give_polymer_ref_conf": 0.0,
    "p_unconditional": 1.0,
    "protein_msa_dirs": [],
    "raise_if_missing_msa_for_protein_of_length_n": None,
    "return_atom_array": True,
    "rna_msa_dirs": [],
    "run_confidence_head": True,
    "take_first_chiral_subordering": False,
    "template_noise_scales": {"atomized": 1e-05, "not_atomized": 1e-05},
    "undesired_res_names": [],
    "use_element_for_atom_names_of_atomized_tokens": True,
}

#: Engine-level knobs, exposed so the CLI can set them.
DEFAULTS = {"n_recycles": 10, "diffusion_batch_size": 5, "seed": 42}


def seed_everything(seed: int) -> None:
    """Seed every RNG the pipeline draws from.

    RDKit conformer embedding uses the python and numpy RNGs, not torch's, so
    seeding torch alone leaves the reference conformer non-reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_pipeline(
    n_recycles: int = DEFAULTS["n_recycles"],
    diffusion_batch_size: int = DEFAULTS["diffusion_batch_size"],
    **overrides: Any,
):
    """Build the AF3 transform pipeline with RF3's inference configuration."""
    from tt_bio._vendor.rf3.data.pipelines import build_af3_transform_pipeline

    cfg = dict(PIPELINE_CONFIG)
    cfg["n_recycles"] = n_recycles
    cfg["diffusion_batch_size"] = diffusion_batch_size
    cfg.update(overrides)
    return build_af3_transform_pipeline(**cfg)


def read_inputs(
    path: str | os.PathLike,
    *,
    template_selection: list[str] | None = None,
    ground_truth_conformer_selection: list[str] | None = None,
    cyclic_chains: list[str] | None = None,
) -> list:
    """Parse a JSON / CIF / PDB input (or a directory of them) into input specs.

    ``template_selection`` and ``ground_truth_conformer_selection`` given here
    override any selection carried in the file itself.
    """
    from tt_bio._vendor.rf3.utils.inference import prepare_inference_inputs_from_paths

    specs = prepare_inference_inputs_from_paths(
        inputs=os.path.abspath(path),
        existing_outputs_dir=None,
        sharding_pattern=None,
        template_selection=template_selection,
        ground_truth_conformer_selection=ground_truth_conformer_selection,
        add_missing_atoms=True,
    )
    if cyclic_chains:
        for spec in specs:
            spec.cyclic_chains = cyclic_chains
    return specs


def featurize(
    path: str | os.PathLike,
    *,
    n_recycles: int = DEFAULTS["n_recycles"],
    diffusion_batch_size: int = DEFAULTS["diffusion_batch_size"],
    seed: int = DEFAULTS["seed"],
    template_selection: list[str] | None = None,
    ground_truth_conformer_selection: list[str] | None = None,
    cyclic_chains: list[str] | None = None,
    pipeline=None,
) -> list[dict]:
    """Featurize every example in ``path``.

    Returns one pipeline-output dict per example, each with ``feats`` (the 35-key
    ``f``), ``ground_truth``, ``confidence_feats``, ``atom_array``, and the sampler
    inputs ``t`` / ``noise`` / ``coord_atom_lvl_to_be_noised``.

    Pass ``pipeline`` to reuse one across calls; building it is not free.
    """
    if pipeline is None:
        pipeline = build_pipeline(n_recycles, diffusion_batch_size)
    specs = read_inputs(
        path,
        template_selection=template_selection,
        ground_truth_conformer_selection=ground_truth_conformer_selection,
        cyclic_chains=cyclic_chains,
    )
    out = []
    for spec in specs:
        seed_everything(seed)
        out.append(pipeline(spec.to_pipeline_input()))
    return out
