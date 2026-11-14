# Vendored from Mol-1 (https://github.com/molence/Mol-1) @ e6e64866ee89640a91cef7ee777a85f1d8be03bb, Apache-2.0.
# Copyright 2024 ByteDance and/or its affiliates; modifications Copyright 2026 Molence Bio.
# Regenerate with scripts/mol1_io/vendor_templates.py; do not edit by hand.
# Copyright 2024 ByteDance and/or its affiliates.
# Modifications Copyright 2026 Molence Bio
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
#
# This file has been modified by Molence Bio
# Based on Protenix: https://github.com/bytedance/Protenix
# Upstream revision: bd54a05d047b8925a241056f36d619700604068a
# Upstream file: runner/rna_msa_search.py
# Changes: Missing RNA databases are reported with their public source.

import functools
import multiprocessing
import os
import shutil
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Optional, Union

from .constants_full import RNA_CHAIN
from .msa_utils import RawMsa
from .search import (
    DatabaseConfig,
    Jackhmmer,
    JackhmmerConfig,
    MsaTool,
    Nhmmer,
    NhmmerConfig,
    RunConfig,
)
from .helpers import get_logger

logger = get_logger(__name__)


def get_msa_tool(
    msa_tool_config: Union[JackhmmerConfig, NhmmerConfig],
) -> MsaTool:
    """
    Returns the requested MSA tool based on the provided configuration.

    Args:
        msa_tool_config: Configuration for Jackhmmer or Nhmmer.

    Returns:
        An instance of MsaTool (Jackhmmer or Nhmmer).

    Raises:
        ValueError: If the configuration type is unknown.
    """
    if isinstance(msa_tool_config, JackhmmerConfig):
        return Jackhmmer(
            binary_path=msa_tool_config.binary_path,
            database_path=msa_tool_config.database_config.path,
            n_cpu=msa_tool_config.n_cpu,
            n_iter=msa_tool_config.n_iter,
            e_value=msa_tool_config.e_value,
            z_value=msa_tool_config.z_value,
            max_sequences=msa_tool_config.max_sequences,
        )
    elif isinstance(msa_tool_config, NhmmerConfig):
        return Nhmmer(
            binary_path=msa_tool_config.binary_path,
            hmmalign_binary_path=msa_tool_config.hmmalign_binary_path,
            hmmbuild_binary_path=msa_tool_config.hmmbuild_binary_path,
            database_path=msa_tool_config.database_config.path,
            n_cpu=msa_tool_config.n_cpu,
            e_value=msa_tool_config.e_value,
            max_sequences=msa_tool_config.max_sequences,
            alphabet=msa_tool_config.alphabet,
        )
    else:
        raise ValueError(f"Unknown MSA tool: {msa_tool_config}.")


def get_msa(
    target_sequence: str,
    run_config: RunConfig,
    chain_poly_type: str,
    deduplicate: bool = False,
) -> RawMsa:
    """
    Computes the MSA for a given query sequence.

    Args:
        target_sequence: The target amino-acid or nucleotide sequence.
        run_config: MSA run configuration.
        chain_poly_type: The type of chain for which to get an MSA.
        deduplicate: If True, the MSA sequences will be deduplicated in the input
            order. Lowercase letters (insertions) are ignored when deduplicating.

    Returns:
        Aligned RawMsa object.
    """
    return RawMsa.from_a3m(
        query=target_sequence,
        ctype=chain_poly_type,
        a3m=get_msa_tool(run_config.config).query(target_sequence).a3m,
        depth_limit=run_config.crop_size,
        dedup=deduplicate,
    )


# Cache to avoid re-running the Nhmmer for the same sequence in homomers.
@functools.cache
def _get_rna_msa(
    sequence: str,
    nt_rna_msa_config: RunConfig,
    rfam_msa_config: RunConfig,
    rnacentral_msa_config: RunConfig,
) -> RawMsa:
    """
    Processes a single RNA chain by running multiple MSA tools in parallel.

    Args:
        sequence: The RNA sequence.
        nt_rna_msa_config: Config for NT-RNA database search.
        rfam_msa_config: Config for Rfam database search.
        rnacentral_msa_config: Config for RNAcentral database search.

    Returns:
        Merged RawMsa object.
    """
    logger.info("Getting RNA MSAs for sequence %s", sequence)
    rna_msa_start_time = time.time()

    # Run various MSA tools in parallel. Use a ThreadPoolExecutor because
    # they're not blocked by the GIL, as they're sub-shelled out.
    with futures.ThreadPoolExecutor() as executor:
        # Currently only Rfam search is enabled in the provided logic
        rfam_msa_future = executor.submit(
            get_msa,
            target_sequence=sequence,
            run_config=rfam_msa_config,
            chain_poly_type=RNA_CHAIN,
        )
        rna_central_msa_future = executor.submit(
            get_msa,
            target_sequence=sequence,
            run_config=rnacentral_msa_config,
            chain_poly_type=RNA_CHAIN,
        )
        nt_rna_msa_future = executor.submit(
            get_msa,
            target_sequence=sequence,
            run_config=nt_rna_msa_config,
            chain_poly_type=RNA_CHAIN,
        )

    rfam_msa = rfam_msa_future.result()
    rna_central_msa = rna_central_msa_future.result()
    nt_rna_msa = nt_rna_msa_future.result()

    rna_msa = RawMsa.merge(
        msas=[rfam_msa, rna_central_msa, nt_rna_msa],
        deduplicate=True,
    )

    logger.info(
        "Getting RNA MSAs took %.2f seconds for sequence %s, found %d unpaired"
        " sequences",
        time.time() - rna_msa_start_time,
        sequence,
        rna_msa.depth,
    )
    return rna_msa


# The clustered RNA databases AlphaFold 3 distributes, zstd-compressed, under the
# file names below plus ".zst".
RNA_DATABASE_SOURCE = "https://storage.googleapis.com/alphafold-databases/v3.0"

# (sources key, file name under $MOL_ROOT_DIR/search_database, CLI option)
RNA_DATABASES = (
    ("nt_rna_search_database",
     "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta", "--ntrna_database_path"),
    ("rfam_search_database",
     "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta", "--rfam_database_path"),
    ("rnacentral_search_database",
     "rnacentral_active_seq_id_90_cov_80_linclust.fasta", "--rna_central_database_path"),
)


def resolve_rna_databases(
    ntrna_database_path: Optional[str],
    rfam_database_path: Optional[str],
    rna_central_database_path: Optional[str],
) -> tuple[str, str, str]:
    """Return the paths of the three RNA search databases, checking all of them.

    A path left as None defaults to ``$MOL_ROOT_DIR/search_database/<file>``.
    The release archives do not include these databases, so every missing one
    is reported at once, with where to get it.

    Raises:
        FileNotFoundError: If a database is missing.
    """
    root = Path(os.environ.get("MOL_ROOT_DIR", str(Path.home()))) / "search_database"
    given = (ntrna_database_path, rfam_database_path, rna_central_database_path)
    paths = [
        path if path is not None else (root / filename).as_posix()
        for path, (_, filename, _) in zip(given, RNA_DATABASES)
    ]
    missing = [
        (path, spec) for path, spec in zip(paths, RNA_DATABASES)
        if not os.path.exists(path)
    ]
    if not missing:
        return tuple(paths)
    lines = [
        f"  {path}\n    from {RNA_DATABASE_SOURCE}/{filename}.zst "
        f"(decompress with `zstd -d`), or pass {option}"
        for path, (_, filename, option) in missing
    ]
    raise FileNotFoundError(
        "RNA MSA search needs databases that are missing:\n"
        + "\n".join(lines)
        + "\nOr predict without RNA MSAs (--use_rna_msa false)."
    )


def run_rna_msa_search(
    rna_seq_for_msa_search: Optional[str] = None,
    rna_result_path: Optional[str] = None,
    rna_seq_id: str = "example",
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
) -> None:
    """
    Run RNA MSA search using nhmmer against multiple databases.

    Args:
        rna_seq_for_msa_search (Optional[str]): RNA sequence to search.
        rna_result_path (Optional[str]): Output path for MSA results.
        rna_seq_id (str): Identifier for the RNA sequence.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        ntrna_database_path (Optional[str]): NT-RNA database path.
        rfam_database_path (Optional[str]): Rfam database path.
        rna_central_database_path (Optional[str]): RNAcentral database path.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.
    """
    assert rna_result_path is not None, "output path should not be None"
    assert rna_seq_for_msa_search is not None, "RNA sequence should not be None"

    if nhmmer_binary_path is None:
        nhmmer_binary_path = shutil.which("nhmmer")
        if nhmmer_binary_path is None:
            raise AssertionError(
                "nhmmer binary path should not be None. You can install "
                "hmmer using: apt install hmmer or conda install -c bioconda hmmer"
            )
    else:
        if not os.path.exists(nhmmer_binary_path):
            raise AssertionError(
                f"nhmmer binary path {nhmmer_binary_path} does not exist"
            )

    if hmmalign_binary_path is None:
        hmmalign_binary_path = shutil.which("hmmalign")
        if hmmalign_binary_path is None:
            raise AssertionError(
                "hmmalign binary path should not be None. You can install "
                "hmmer using: apt install hmmer or conda install -c bioconda hmmer"
            )
    else:
        if not os.path.exists(hmmalign_binary_path):
            raise AssertionError(
                f"hmmalign binary path {hmmalign_binary_path} does not exist"
            )

    if hmmbuild_binary_path is None:
        hmmbuild_binary_path = shutil.which("hmmbuild")
        if hmmbuild_binary_path is None:
            raise AssertionError(
                "hmmbuild binary path should not be None. You can install "
                "hmmer using: apt install hmmer or conda install -c bioconda hmmer"
            )
    else:
        if not os.path.exists(hmmbuild_binary_path):
            raise AssertionError(
                f"hmmbuild binary path {hmmbuild_binary_path} does not exist"
            )

    if nhmmer_n_cpu is None:
        nhmmer_n_cpu = min(multiprocessing.cpu_count(), 8)

    ntrna_database_path, rfam_database_path, rna_central_database_path = (
        resolve_rna_databases(
            ntrna_database_path, rfam_database_path, rna_central_database_path
        )
    )

    logger.info("RNA MSA search start!")
    rna_msa_start_time = time.time()

    nt_rna_msa_config = RunConfig(
        config=NhmmerConfig(
            binary_path=nhmmer_binary_path,
            hmmalign_binary_path=hmmalign_binary_path,
            hmmbuild_binary_path=hmmbuild_binary_path,
            database_config=DatabaseConfig(
                name="nt_rna",
                path=ntrna_database_path,
            ),
            n_cpu=nhmmer_n_cpu,
            e_value=1e-3,
            alphabet="rna",
            max_sequences=10_000,
        ),
        chain_poly_type=RNA_CHAIN,
        crop_size=None,
    )

    rfam_msa_config = RunConfig(
        config=NhmmerConfig(
            binary_path=nhmmer_binary_path,
            hmmalign_binary_path=hmmalign_binary_path,
            hmmbuild_binary_path=hmmbuild_binary_path,
            database_config=DatabaseConfig(
                name="rfam_rna",
                path=rfam_database_path,
            ),
            n_cpu=nhmmer_n_cpu,
            e_value=1e-3,
            alphabet="rna",
            max_sequences=10_000,
        ),
        chain_poly_type=RNA_CHAIN,
        crop_size=None,
    )

    rnacentral_msa_config = RunConfig(
        config=NhmmerConfig(
            binary_path=nhmmer_binary_path,
            hmmalign_binary_path=hmmalign_binary_path,
            hmmbuild_binary_path=hmmbuild_binary_path,
            database_config=DatabaseConfig(
                name="rna_central_rna",
                path=rna_central_database_path,
            ),
            n_cpu=nhmmer_n_cpu,
            e_value=1e-3,
            alphabet="rna",
            max_sequences=10_000,
        ),
        chain_poly_type=RNA_CHAIN,
        crop_size=None,
    )

    rna_msa = _get_rna_msa(
        sequence=rna_seq_for_msa_search,
        nt_rna_msa_config=nt_rna_msa_config,
        rfam_msa_config=rfam_msa_config,
        rnacentral_msa_config=rnacentral_msa_config,
    )
    rna_msa_a3m = rna_msa.to_a3m()
    os.makedirs(f"{rna_result_path}/{rna_seq_id}", exist_ok=True)
    with open(f"{rna_result_path}/{rna_seq_id}/rna_msa.a3m", "w") as f:
        f.write(rna_msa_a3m)
    rna_msa_end_time = time.time()
    logger.info(
        f"RNA MSA search done!, using {(rna_msa_end_time - rna_msa_start_time):.2f}s"
    )


def update_rna_msa_info(
    json_data: list[dict[str, Any]],
    out_dir: str,
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
) -> bool:
    """
    Update RNA MSA information in the JSON data.
    If unpairedMsaPath is missing, it performs an RNA MSA search.

    Args:
        json_data (list[dict[str, Any]]): The input JSON data.
        out_dir (str): Directory to save RNA MSA results.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        ntrna_database_path (Optional[str]): NT-RNA database path.
        rfam_database_path (Optional[str]): Rfam database path.
        rna_central_database_path (Optional[str]): RNAcentral database path.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.

    Returns:
        bool: True if any RNA MSA information was updated.
    """
    actual_updated = False
    for task_idx, infer_data in enumerate(json_data):
        task_name = infer_data.get("name", f"task_{task_idx}")
        for sequence_idx, sequence in enumerate(infer_data["sequences"]):
            if "rnaSequence" in sequence:
                rna_chain = sequence["rnaSequence"]
                rna_sequence = rna_chain.get("sequence", "")
                if not rna_sequence:
                    continue

                if "unpairedMsaPath" in rna_chain and os.path.exists(
                    rna_chain["unpairedMsaPath"]
                ):
                    continue

                logger.info(
                    f"Running RNA MSA search for task {task_name}, rna_sequence: {rna_sequence}"
                )
                rna_output_dir = os.path.join(
                    out_dir, task_name, "rna_msa", str(sequence_idx)
                )
                run_rna_msa_search(
                    rna_seq_for_msa_search=rna_sequence,
                    rna_result_path=os.path.dirname(rna_output_dir),
                    rna_seq_id=os.path.basename(rna_output_dir),
                    nhmmer_binary_path=nhmmer_binary_path,
                    hmmalign_binary_path=hmmalign_binary_path,
                    hmmbuild_binary_path=hmmbuild_binary_path,
                    ntrna_database_path=ntrna_database_path,
                    rfam_database_path=rfam_database_path,
                    rna_central_database_path=rna_central_database_path,
                    nhmmer_n_cpu=nhmmer_n_cpu,
                )
                rna_msa_path = os.path.join(rna_output_dir, "rna_msa.a3m")
                if os.path.exists(rna_msa_path):
                    rna_chain["unpairedMsaPath"] = rna_msa_path
                    actual_updated = True
    return actual_updated


if __name__ == "__main__":
    example_seq = (
        "GGCGCGUUAACAAAGCGGUUAUGUAGCGGAUUGCAAAUCCGUCUAGUCCGGUUCGACUCCGGAACGCGCCUCCA"
    )
    run_rna_msa_search(
        rna_seq_for_msa_search=example_seq,
        rna_result_path="./output",
        rna_seq_id="rna_seq_1",
    )
