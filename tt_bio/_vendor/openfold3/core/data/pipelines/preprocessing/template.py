# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocessing pipelines for template data ran before training/evaluation.

tt-bio note: trimmed to just `TemplatePreprocessorSettings` (the config section
`InferenceDataset` reads). The full file also implements the PDB/S3 template-cache
*build* pipeline (fetch, precache, multiprocessing) — training/offline-corpus tooling
that a single query-JSON inference call never exercises, so it was dropped here rather
than pulling in `func_timeout`/`boto3`/`tqdm` for dead code. See NOTICE.
"""

import os
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, model_validator
from pydantic import ConfigDict as PydanticConfigDict

from tt_bio._vendor.openfold3.core.config.config_utils import _convert_molecule_type, _ensure_list
from tt_bio._vendor.openfold3.core.data.resources.residues import MoleculeType
from pathlib import Path


class TemplatePreprocessorSettings(BaseModel):
    """Settings for template preprocessing.

    See AF3 SI Section 2.4. for details on some of these settings.

    Attributes:
        mode (Literal["train", "inference"]):
            Whether templates are preprocessed for training or inference.
        moltypes (list[MoleculeType]):
            List of molecule types to preprocess templates for.
        max_sequences_parse (int):
            Maximum number of align sequences to parse from the template alignments
            before filtering.
        max_seq_id (float | None):
            Maximum allowed sequence identity of the template relative to the query for
            the template to pass filtering.
        min_align (float | None):
            Minimum required alignment coverage of the the query by the template for it
            to pass filtering.
        min_len (int | None):
            Minimum required number of aligned template residues for the template to
            pass filtering.
        max_release_date (str | None):
            Maximum allowed release date of the template structure for it to pass
            filtering.
        min_release_date_diff (int | None):
            Minimum number of days required between the query and template release dates
            for the template to pass filtering. Equivalently, the minimum number of days
            that a template structure needed to have been released before the query
            structure it is provided for as a template.
        max_templates (int):
            Maximum number of valid templates to keep per query chain after filtering.
        min_f_resolved (float):
            Minimum fraction of resolved residues (n resolved / n total) needed for a
            template to be considered valid. NOTE that this is only used if the template
            structure arrays and template precache entries are computed separately from
            the main template cache entries.
        fetch_missing_template_structures (bool):
            Whether to fetch missing template structures from the PDB. Requires internet
            access.
        create_precache (bool):
            Whether to cache of the template structure data (release date and sequence
            information) for template filtering.
        preparse_structures (bool):
            Whether to preparse the template structures into per-chain AtomArray .npz
            files for faster subsequent online template processing.
        n_processes (int):
            Number of processes to use template preprocessing.
        chunksize (int):
            Number of tasks per worker in multiprocessing.
        preprocess_timeout (int):
                Maximum time in seconds allowed for preprocessing templates for a
                single query chain. Defaults to 60.
        structure_directory (DirectoryPath):
            Directory containing raw template structures or where template structures
            are to be downloaded.
        structure_file_format (str):
            File format of the template structures. One of "cif", "pdb".
        precache_directory (DirectoryPath | None):
            Directory containing precomputed template structure pre-caches or where new
            ones are to be saved.
        structure_array_directory (DirectoryPath | None):
            Directory containing preparsed template structures or where new ones will be
            saved.
        cache_directory (DirectoryPath | None):
            Directory containing template cache entry .npz files or where new ones will
            be saved.
        ccd_file_path (FilePath | None):
            Path to the Chemical Component Dictionary file. Only required if
            `preparse_structures` is True.
    """

    model_config = PydanticConfigDict(extra="forbid")
    mode: Literal["train", "predict"] = "predict"
    moltypes: Annotated[
        list[MoleculeType],
        BeforeValidator(lambda v: _convert_molecule_type(_ensure_list(v))),
    ] = [MoleculeType.PROTEIN]
    max_sequences_parse: int = 200
    max_seq_id: float | None = None
    min_align: float | None = None
    min_len: int | None = None
    max_release_date: datetime | None = None
    min_release_date_diff: int | None = None
    max_templates: int = 20
    cif_direct_min_score: float = 0.1
    min_f_resolved: float = 0.1

    fetch_missing_structures: bool = True
    create_precache: bool = False
    preparse_structures: bool = False
    create_logs: bool = False
    n_processes: int = 1
    chunksize: int = 1
    preprocess_timeout: int = 60

    structure_directory: Path | None = None
    structure_file_format: str = "cif"
    output_directory: Path | None = None

    precache_directory: Path | None = None
    structure_array_directory: Path | None = None
    cache_directory: Path | None = None
    log_directory: Path | None = None

    ccd_file_path: Path | None = None

    @model_validator(mode="after")
    def _prepare_output_directories(self) -> "TemplatePreprocessorSettings":
        # TODO: add .pdb support
        if self.structure_file_format not in ["cif", "npz"]:
            raise NotImplementedError(
                f"structure_file_format {self.structure_file_format} was provided but "
                "currently, only cif and npz file format is supported for template "
                "structure preprocessing due to metadata requirements of the template "
                "pipeline."
            )

        if self.output_directory is None:
            from tt_bio._vendor.openfold3.core.data.tools.utils import get_of3_tmpdir

            self.output_directory = get_of3_tmpdir("template_data")

        base = self.output_directory

        # only set these if the user did not give them explicitly
        self.structure_directory = self.structure_directory or (
            base / "template_structures"
        )
        self.cache_directory = self.cache_directory or (base / "template_cache")
        if self.create_precache:
            self.precache_directory = self.precache_directory or (
                base / "template_precache"
            )
        if self.preparse_structures:
            self.structure_array_directory = self.structure_array_directory or (
                base / "template_structure_arrays"
            )
        if self.create_logs:
            self.log_directory = self.log_directory or (base / "template_logs")

        for d in (
            base,
            self.output_directory,
            self.structure_directory,
            self.cache_directory,
            self.precache_directory,
            self.structure_array_directory,
            self.log_directory,
        ):
            if d is not None:
                os.makedirs(d, exist_ok=True)

        return self
