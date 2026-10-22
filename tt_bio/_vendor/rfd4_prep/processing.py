# Copyright 2026 The HuggingFace Team. All rights reserved.
# RFD4-derived preprocessing: Copyright (c) 2026, Institute for Protein Design,
# University of Washington (BSD-3-Clause). See THIRD_PARTY_LICENSES.txt.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""AtomWorks adapter: annotated CIF -> prepared tensors for `RFD4ProteinaPipeline`, and generated tensors -> CIF."""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def prepare_cif(path: str | Path, *, seed: int, device: torch.device, path_length_cutoff: int = 64):
    """Return a single-example feature batch and its atom mapping, without changing the caller's RNG state.

    `device` is where the returned tensors live; pass the pipeline's device so no copy is needed at call time.
    """
    try:
        from tt_bio._vendor.atomworks.io.parser import parse
    except ImportError as error:
        raise ImportError(
            "Annotated CIF inputs require the optional AtomWorks package. Install atomworks[ml]; RFD4 is not required."
        ) from error
    # Import before parsing to register the model's condition and provenance annotations.
    from ._cif.datapipes_pipeline import build_design_validation_pipeline
    from ._cif.datapipes_schema_proteina import SPECS
    from ._cif.utils_constants import LOCAL_LATENT_RESIDUES
    from ._cif.utils_parse_presets import ANNOTATED_CIF

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            np.random.seed(seed % 2**32)
            random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            parsed = parse(path, config=ANNOTATED_CIF)
            data = {"atom_array": parsed["assemblies"]["1"][0], "example_id": path.stem}
            # The RFD4 datapipe itself, in its inference mode: there is one pipeline here, not a
            # second one maintained alongside it. `build_design_validation_pipeline` is
            # `build_design_train_pipeline(is_inference=True)`.
            #
            # `na_ss_conditioning` turns on nucleic-acid secondary-structure featurization. RFD4
            # gates that on the metrics it is about to compute, and its own inference default is
            # no tags at all -- but an annotated CIF can carry NA secondary-structure conditioning
            # in the file, and without this it is parsed and then silently discarded, leaving the
            # model's `na_ss_state` and `na_ss_pair` inputs at their defaults. This is RFD4's own
            # switch, set the way RFD4's `ss_conditioned` config sets it, not a flag invented here.
            pipeline = build_design_validation_pipeline(
                metrics_tags=["na_ss_conditioning"],
                residue_names_supporting_local_latents=list(LOCAL_LATENT_RESIDUES),
                path_length_search_cutoff=path_length_cutoff,
                expandable_segment_rng=np.random.default_rng(seed),
            )
            data = pipeline(data)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    # Match the released schema, including padding of auxiliary fields and absent conditions.
    tensors = {
        key: torch.as_tensor(value, dtype=SPECS[key][0] if key in SPECS else None)
        for key, value in data["feats"].items()
        if isinstance(value, (torch.Tensor, np.ndarray))
    }
    length = max(
        tensor.shape[axis]
        for key, tensor in tensors.items()
        for axis, dimension in enumerate(SPECS[key][2].split() if key in SPECS else ["I"])
        if dimension == "I" and axis < tensor.ndim
    )
    features = {}
    for key, tensor in tensors.items():
        dtype, default, dimensions = SPECS.get(key, (tensor.dtype, 0, "I"))
        axes = dimensions.split()
        padding = []
        for axis in reversed(range(tensor.ndim)):
            padding.extend((0, length - tensor.shape[axis] if axis < len(axes) and axes[axis] == "I" else 0))
        if any(padding):
            tensor = torch.nn.functional.pad(tensor, padding, value=default)
        features[key] = tensor.unsqueeze(0).to(device)
    for key, (dtype, default, dimensions) in SPECS.items():
        if key not in features:
            shape = (1, *(length if dimension == "I" else int(dimension) for dimension in dimensions.split()))
            features[key] = torch.full(shape, default, dtype=dtype, device=device)
    features["padding_mask"] = torch.ones((1, length), dtype=torch.bool, device=device)
    features["coords_nm"] = features["coords"] / 10.0
    features["coord_mask"] = features["atom37_mask"]
    return features, data["atom_array"]


def write_cif(output, features, atom_array, path: str | Path, row: int = 0) -> str:
    """Write the generated atoms of an `RFD4ProteinaPipelineOutput` to a new annotated CIF.

    Conditioning annotations and provenance are preserved and predicted unindexed-motif insertion mappings are added.
    `features` and `atom_array` are the values returned by `prepare_cif`. Coordinates are converted from nanometers to
    angstroms. For a batched output (the features repeated for several designs), `row` selects the design to write.
    An existing `path` is never overwritten.
    """
    from ._cif.transforms_unindexing import annotate_unindexed_insertion
    from ._cif.utils_structure_export import process_single_sample_to_atom_array, write_generated_structure_cif

    path = Path(path)
    sample = process_single_sample_to_atom_array(
        coords=output.coords_nm[row].detach().cpu() * 10,
        residue_type=output.residue_type[row].detach().cpu(),
        padding_mask=features["padding_mask"][row].detach().cpu(),
        atom_mask=output.atom_mask[row].detach().cpu(),
        input_atom_array=atom_array,
        insertion_logits=output.insertion_logits[row].detach().cpu(),
    )
    array = sample["full_array_with_unindexed"]
    if array is None:
        array = sample["generated_atom_array"]
    array = annotate_unindexed_insertion(array, sample["predicted_insertion_mapping"])
    # Publish a complete file without overwriting any existing output, including a racing writer.
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".rfd4-cif-") as temporary:
        staged = Path(temporary) / path.name
        write_generated_structure_cif(array, staged)
        os.link(staged, path)
    return str(path)
