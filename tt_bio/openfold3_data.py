"""OpenFold3 host-side featurization: query JSON -> model-ready feature dict.

Thin driver around the vendored `tt_bio._vendor.openfold3` data pipeline
(see NOTICE), replicating `InferenceDataset.create_all_features` from the
reference `openfold3.core.data.framework.single_datasets.inference` without the
Lightning `Dataset`/`DataModule` scaffolding that class ships with (dataset
registry, DDP world-size padding, batch collation) -- none of which applies to
featurizing one query at a time.
"""

from __future__ import annotations

import torch
from biotite.structure.io.pdbx import CIFFile

from tt_bio._vendor.openfold3.core.config.msa_pipeline_configs import (
    MsaSampleProcessorInputInference,
)
from tt_bio._vendor.openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_of3,
)
from tt_bio._vendor.openfold3.core.data.pipelines.featurization.msa import (
    MsaFeaturizerOF3,
    MsaFeaturizerOF3Config,
)
from tt_bio._vendor.openfold3.core.data.pipelines.featurization.structure import (
    featurize_structure_of3,
)
from tt_bio._vendor.openfold3.core.data.pipelines.featurization.template import (
    featurize_template_structures_of3,
)
from tt_bio._vendor.openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from tt_bio._vendor.openfold3.core.data.pipelines.sample_processing.msa import (
    MsaSampleProcessorInference,
)
from tt_bio._vendor.openfold3.core.data.pipelines.sample_processing.template import (
    process_template_structures_of3,
)
from tt_bio._vendor.openfold3.core.data.primitives.structure.component import (
    BiotiteCCDWrapper,
)
from tt_bio._vendor.openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from tt_bio._vendor.openfold3.core.data.primitives.structure.tokenization import (
    add_token_positions,
    get_token_count,
    tokenize_atom_array,
)
from tt_bio._vendor.openfold3.projects.of3_all_atom.config.dataset_config_components import (
    MSASettings,
    TemplateSettings,
)
from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
    Query,
)


def _get_structure_with_ref_mols(query: Query):
    atom_array, processed_reference_molecules = structure_with_ref_mols_from_query(
        query=query
    )
    tokenize_atom_array(atom_array)
    add_token_positions(atom_array)
    return atom_array, processed_reference_molecules


def build_openfold3_features(
    query: Query,
    *,
    msa_settings: MSASettings | None = None,
    template_settings: TemplateSettings | None = None,
    ccd_file_path: str | None = None,
) -> dict[str, torch.Tensor]:
    """Featurizes a single OpenFold3 `Query` into a model-ready feature dict.

    Mirrors the reference `InferenceDataset.create_all_features` offline path: no
    MSA search tools or template structure directories are invoked -- MSA/template
    file paths already resolved on the `Query.chains` are read directly, and
    everything else falls back to the single-sequence / dummy-template features
    the model was trained to accept when none are provided.
    """
    msa_settings = msa_settings or MSASettings()
    template_settings = template_settings or TemplateSettings()
    template_preprocessor_settings = TemplatePreprocessorSettings(mode="predict")
    ccd = (
        CIFFile.read(ccd_file_path) if ccd_file_path is not None else BiotiteCCDWrapper()
    )

    atom_array, processed_reference_molecules = _get_structure_with_ref_mols(query)
    n_tokens = get_token_count(atom_array)

    features: dict = {"atom_array": atom_array}

    structure_features = featurize_structure_of3(
        atom_array=atom_array,
        n_tokens=n_tokens,
        is_gt=False,
        add_perm_features=False,
    )
    reference_conformer_features = featurize_reference_conformers_of3(
        processed_ref_mol_list=processed_reference_molecules,
        add_ref_space_uid_to_perm=False,
    )
    features.update(structure_features | reference_conformer_features)

    msa_sample_processor = MsaSampleProcessorInference(config=msa_settings)
    msa_featurizer = MsaFeaturizerOF3(
        config=MsaFeaturizerOF3Config(
            max_rows=msa_settings.max_rows,
            max_rows_paired=msa_settings.max_rows_paired,
            subsample_with_bands=msa_settings.subsample_with_bands,
        )
    )
    msa_input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        inference_query=query
    )
    msa_array_collection = msa_sample_processor(input=msa_input)
    features.update(
        msa_featurizer(
            atom_array=atom_array,
            msa_array_collection=msa_array_collection,
            n_tokens=n_tokens,
        )
    )

    assembly_data = {
        chain_id: {
            "template_ids": chain.template_entry_chain_ids,
            "cache_entry_file_path": chain.template_alignment_file_path,
        }
        for chain in query.chains
        for chain_id in chain.chain_ids
    }
    template_slice_collection = process_template_structures_of3(
        atom_array=atom_array,
        n_templates=template_settings.n_templates,
        take_top_k=template_settings.take_top_k,
        min_n_tokens_per_chain=template_settings.min_n_tokens_per_chain,
        template_cache_directory=None,
        assembly_data=assembly_data,
        template_structures_directory=template_preprocessor_settings.structure_directory,
        template_structure_array_directory=template_preprocessor_settings.structure_array_directory,
        template_file_format=template_preprocessor_settings.structure_file_format,
        ccd=ccd,
    )
    features.update(
        featurize_template_structures_of3(
            atom_array=atom_array,
            template_slice_collection=template_slice_collection,
            n_templates=template_settings.n_templates,
            n_tokens=n_tokens,
            min_bin=template_settings.distogram.min_bin,
            max_bin=template_settings.distogram.max_bin,
            n_bins=template_settings.distogram.n_bins,
        )
    )

    return features
