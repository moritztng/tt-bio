"""OpenFold3 host-side featurization: query JSON -> model-ready feature dict.

Thin driver around the vendored `tt_bio._vendor.openfold3` data pipeline
(see NOTICE), replicating `InferenceDataset.create_all_features` from the
reference `openfold3.core.data.framework.single_datasets.inference` without the
Lightning `Dataset`/`DataModule` scaffolding that class ships with (dataset
registry, DDP world-size padding, batch collation) -- none of which applies to
featurizing one query at a time.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
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


def resolve_openfold3_msas(
    query: Query,
    msa_dir: str | Path,
    *,
    target_id: str,
    msa_db_path: str | None = None,
    use_envdb: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    api_key: str | None = None,
    msa_endpoint: str | None = None,
    fetch: bool = True,
) -> Query:
    """Attach cached or freshly searched unpaired MSAs to an OF3 query.

    This is the same sequence-hash cache and search stage used by Protenix-v2,
    OpenDDE, and ESMFold2. Existing ``main_msa_file_paths`` are preserved.
    ``fetch=False`` attaches from the cache only (no search) — the
    single-sequence / staged-fixture mode, matching Protenix-v2's "cached a3m
    is always used, search only when a source is configured" semantics.
    """
    from tt_bio.main import _generate_esmfold2_a3m

    msa_dir = Path(msa_dir).expanduser()
    msa_dir.mkdir(parents=True, exist_ok=True)
    needed: dict[str, str] = {}
    paths: dict[int, Path] = {}
    for i, chain in enumerate(query.chains):
        if chain.molecule_type.name != "PROTEIN" or chain.main_msa_file_paths:
            continue
        seq = chain.sequence or ""
        seq_hash = hashlib.sha256(seq.encode()).hexdigest()[:16]
        path = msa_dir / f"{seq_hash}.a3m"
        paths[i] = path
        if fetch and not path.exists():
            needed[seq_hash] = seq
    if needed:
        _generate_esmfold2_a3m(
            needed, target_id, msa_dir, msa_db_path, use_envdb,
            msa_server_url, msa_pairing_strategy, msa_server_username,
            msa_server_password, api_key, msa_endpoint=msa_endpoint,
        )
    paths = {i: p for i, p in paths.items() if p.exists()}
    for i, path in paths.items():
        # OF3 filters direct MSA files by canonical source basename. Keep the shared
        # hash cache unchanged and expose the same bytes under its ColabFold source name.
        of3_path = msa_dir / "of3" / path.stem / "colabfold_main.a3m"
        of3_path.parent.mkdir(parents=True, exist_ok=True)
        if not of3_path.exists():
            try:
                os.link(path, of3_path)
            except FileExistsError:
                pass  # a concurrent worker linked it first
            except OSError:
                try:
                    shutil.copyfile(path, of3_path)
                except shutil.SameFileError:
                    pass  # lost the race: of3_path is already a link to path
        query.chains[i].main_msa_file_paths = [of3_path]
    return query


def augment_openfold3_msas_with_query_sequence(
    query: Query,
    msa_dir: str | Path,
) -> Query:
    """Give every MSA-less protein/RNA chain a one-row alignment holding just
    its own sequence — upstream's no-MSA mode
    (``augment_main_msa_with_query_sequence``, colabfold_msa_server.py), which
    leaves the MSA stack ON rather than disabling it.

    A one-row MSA and no MSA are different inputs to the model: on ubiquitin
    (76 aa) the one-row arm folds at 0.84 A CA-RMSD against the crystal
    structure where the MSA-stack-off arm folds at 2.34 A, outside upstream's
    own 1.8 A single-sequence ceiling. Chains that already carry an alignment
    (YAML ``msa:`` key, or a cached a3m attached by ``resolve_openfold3_msas``)
    are kept as-is, matching upstream, which only fills chains whose
    ``main_msa_file_paths`` are unset.

    The dummy a3m lives under ``<msa_dir>/of3/dummy/`` and NOT in the shared
    hash-named cache at ``<msa_dir>/<hash>.a3m``: a one-row alignment must
    never satisfy a later run's real-MSA cache lookup. Content is the exact
    bytes upstream writes (``">query\\n" + sequence``, no trailing newline);
    the write is tmp-file + rename so concurrent workers never expose a
    partial file.
    """
    msa_dir = Path(msa_dir).expanduser()
    for chain in query.chains:
        if chain.molecule_type.name not in ("PROTEIN", "RNA"):
            continue
        if chain.main_msa_file_paths:
            continue
        seq_hash = hashlib.sha256(chain.sequence.encode()).hexdigest()[:16]
        a3m = msa_dir / "of3" / "dummy" / seq_hash / "colabfold_main.a3m"
        if not a3m.exists():
            a3m.parent.mkdir(parents=True, exist_ok=True)
            tmp = a3m.parent / f".{a3m.name}.{os.getpid()}.tmp"
            tmp.write_text(">query\n" + chain.sequence)
            os.replace(tmp, a3m)
        chain.main_msa_file_paths = [a3m]
    return query


def make_openfold3_msa_features(
    features: dict[str, torch.Tensor], *, max_sequences: int = 1024, seed: int = 0
) -> torch.Tensor:
    """Build the 34-channel OF3 MSA input with deterministic sampling."""
    msa_feat = torch.cat(
        [
            features["msa"],
            features["has_deletion"].unsqueeze(-1),
            features["deletion_value"].unsqueeze(-1),
        ],
        dim=-1,
    )
    msa_mask = features["msa_mask"]
    valid = torch.nonzero(msa_mask.sum(dim=-1) > 0, as_tuple=False).flatten()
    if valid.numel() > max_sequences:
        generator = torch.Generator(device=valid.device).manual_seed(seed)
        valid = valid[torch.randperm(valid.numel(), generator=generator)[:max_sequences]]
    return msa_feat.index_select(0, valid)


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
    template_structures_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Featurizes a single OpenFold3 `Query` into a model-ready feature dict.

    Mirrors the reference `InferenceDataset.create_all_features` offline path: no
    MSA search tools or template structure directories are invoked -- MSA/template
    file paths already resolved on the `Query.chains` are read directly, and
    everything else falls back to the single-sequence / dummy-template features
    the model was trained to accept when none are provided.

    Defaults follow the reference inference configuration
    (`InferenceDatasetConfigKwargs`, dataset_configs.py:368): `subsample_main=False`
    (the bare `MSASettings` default of `True` randomly subsamples the main MSA -- a
    training augmentation) and `take_top_k=True`. The default parse order/caps are
    extended with `cfdb_hits` (cap 100000000, per docs/source/precomputed_msa_how_to.md)
    so the OpenFold3 S3 benchmark MSA directories, which ship `cfdb_hits.a3m`, parse.
    """
    if msa_settings is None:
        msa_settings = MSASettings(subsample_main=False)
        if "cfdb_hits" not in msa_settings.max_seq_counts:
            msa_settings.max_seq_counts["cfdb_hits"] = 100000000
            msa_settings.aln_order.insert(
                msa_settings.aln_order.index("uniref90_hits") + 1, "cfdb_hits"
            )
    template_settings = template_settings or TemplateSettings(take_top_k=True)
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

    # Templates: the cache npz holds per-entry ALIGNMENTS only (index/release_date/
    # idx_map); coordinates come from raw <pdb_id>.cif files in
    # template_structures_directory. The vendored loader silently falls back to all-zero
    # dummy templates when ids or directories are missing (its training-time path) — a
    # REQUESTED template that cannot load must be a loud error, not a template-free fold.
    assembly_data: dict[str, dict] = {}
    templates_requested = False
    needed_pdb_ids: set[str] = set()
    for chain in query.chains:
        tmpl_path = chain.template_alignment_file_path
        template_ids = chain.template_entry_chain_ids
        if tmpl_path is not None:
            templates_requested = True
            tmpl_path = Path(str(tmpl_path)).expanduser()
            if not template_ids:
                # CLI case: only the npz was given. Its keys are the entry ids, in the
                # same e-value order upstream wrote them (verified against the benchmark
                # query JSONs, e.g. 7XI5).
                with np.load(str(tmpl_path), allow_pickle=True) as _npz:
                    template_ids = list(_npz.keys())
            needed_pdb_ids |= {str(tid).split("_")[0] for tid in template_ids}
        for chain_id in chain.chain_ids:
            assembly_data[chain_id] = {
                "template_ids": template_ids,
                "cache_entry_file_path": tmpl_path,
            }
    if templates_requested:
        if template_structures_directory is None:
            raise RuntimeError(
                "templates were requested but no template_structures_directory was "
                "provided — template coordinates come from raw <pdb_id>.cif files; "
                "the cache npz holds alignments only.")
        ts_dir = Path(template_structures_directory).expanduser()
        missing = sorted(p for p in needed_pdb_ids
                         if not (ts_dir / f"{p}.cif").exists())
        if missing:
            raise RuntimeError(
                f"template structure(s) missing under {ts_dir}: "
                + ", ".join(f"{p}.cif" for p in missing)
                + ". Fetch with e.g. `curl -o <pdb>.cif "
                  "https://files.rcsb.org/download/<PDB>.cif` (uppercase pdb id).")
        # non-None unlocks the vendored inference cache branch (the npz itself is read
        # via cache_entry_file_path; this directory is only the gate)
        template_cache_directory = ts_dir
    else:
        ts_dir = None
        template_cache_directory = None
    template_slice_collection = process_template_structures_of3(
        atom_array=atom_array,
        n_templates=template_settings.n_templates,
        take_top_k=template_settings.take_top_k,
        min_n_tokens_per_chain=template_settings.min_n_tokens_per_chain,
        template_cache_directory=template_cache_directory,
        assembly_data=assembly_data,
        template_structures_directory=ts_dir,
        template_structure_array_directory=None,
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
