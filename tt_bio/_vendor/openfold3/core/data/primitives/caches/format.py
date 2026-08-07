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

"""tt-bio note: trimmed to `DatasetChainData`/`DatasetReferenceMoleculeData` — the only
symbols the inference-time featurization pipeline references (generic type-hint
bases). The full file also defines the LMDB-backed training/validation dataset-cache
dataclasses (PDB-weighted, disordered, clustered, ...), which pull in `lmdb` for a
training-only cache format an offline query-JSON call never touches. See NOTICE.
"""

from dataclasses import dataclass


@dataclass
class DatasetChainData:
    """Central class for chain-wise data that can be used for general type-hinting."""

    pass


@dataclass
class DatasetReferenceMoleculeData:
    """Fields that every Dataset format's reference molecule data should have."""

    conformer_gen_strategy: str
    fallback_conformer_pdb_id: str | None
    canonical_smiles: str
    set_fallback_to_nan: bool
