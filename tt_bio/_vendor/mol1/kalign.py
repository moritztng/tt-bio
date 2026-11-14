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
# Upstream file: protenix/data/tools/kalign.py

import os
import shutil
import subprocess
import time
from contextlib import nullcontext
from typing import Sequence

from .tools_common import parse_kalign_a3m, tmpdir_manager, to_a3m
from .helpers import get_logger

logger = get_logger(__name__)


class KalignError(RuntimeError):
    """Kalign is missing or does not run: an environment problem, not a bad hit."""


def find_kalign() -> str:
    """Return the ``kalign`` on ``PATH``, which aligns each template hit to its query.

    Raises:
        KalignError: If there is none, saying how to install it.
    """
    path = shutil.which("kalign")
    if path is None:
        raise KalignError(
            "Templates need the kalign binary, and none is on PATH. Install it "
            "with `apt-get install kalign` on Debian or Ubuntu, or with "
            "`conda install -c conda-forge -c bioconda kalign3=3.3.5` in a conda "
            "environment, or put an existing kalign on PATH."
        )
    return path


class Kalign:
    """
    Python wrapper of the Kalign binary on ``PATH``.
    Adapted from openfold.data.tools.kalign

    Raises:
      KalignError: If no Kalign binary is on PATH.
    """

    def __init__(self):
        self.binary_path = find_kalign()

    def align(self, sequences: Sequence[str]) -> Sequence[str]:
        """Aligns the sequences and returns the alignment in A3M string.

        Args:
          sequences: A list of query sequence strings. The sequences have to be at
            least 6 residues long (Kalign requires this). Note that the order in
            which you give the sequences might alter the output slightly as
            different alignment tree might get constructed.

        Returns:
          A list of strings with the aligned sequences in a3m format.

        Raises:
          KalignError: If Kalign cannot be run or exits with an error.
          ValueError: If any of the sequences is less than 6 residues long.
        """

        for s in sequences:
            if len(s) < 6:
                raise ValueError(
                    "Kalign requires all sequences to be at least 6 "
                    f"residues long. Got {s} ({len(s)} residues)."
                )

        with tmpdir_manager() as query_tmp_dir:
            input_fasta_path = os.path.join(query_tmp_dir, "input.fasta")
            output_a3m_path = os.path.join(query_tmp_dir, "output.a3m")

            logger.debug(
                "Kalign tmpdir=%s input_fasta=%s output_a3m=%s",
                query_tmp_dir,
                input_fasta_path,
                output_a3m_path,
            )

            write_st = time.perf_counter()
            with open(input_fasta_path, "w") as f:
                f.write(to_a3m(sequences))
            write_seconds = time.perf_counter() - write_st

            cmd = [
                self.binary_path,
                "-i",
                input_fasta_path,
                "-o",
                output_a3m_path,
                "-format",
                "fasta",
            ]

            popen_st = time.perf_counter()
            try:
                # kalign reads sequences from a stdin that is not a terminal,
                # so an inherited open pipe would make it wait forever.
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except OSError as error:
                raise KalignError(f"Could not run {self.binary_path}: {error}") from error

            with nullcontext():
                _, stderr = process.communicate()
                retcode = process.wait()
            popen_seconds = time.perf_counter() - popen_st

            if retcode:
                logger.debug(
                    "Kalign timings write=%.4fs run=%.4fs read=%.4fs",
                    write_seconds,
                    popen_seconds,
                    0.0,
                )
                raise KalignError(
                    f"{self.binary_path} exited with status {retcode}:\n"
                    f"{stderr.decode('utf-8', 'replace')}"
                )

            read_st = time.perf_counter()
            with open(output_a3m_path) as f:
                a3m = f.read()
            read_seconds = time.perf_counter() - read_st

            logger.debug(
                "Kalign timings write=%.4fs run=%.4fs read=%.4fs",
                write_seconds,
                popen_seconds,
                read_seconds,
            )
            return parse_kalign_a3m(a3m)
