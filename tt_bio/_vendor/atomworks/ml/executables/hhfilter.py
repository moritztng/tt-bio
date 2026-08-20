"""HHfilter executable wrapper for MSA filtering."""

import logging
import os
import subprocess
from os import PathLike

from tt_bio._vendor.atomworks.ml.executables import Executable, ExecutableError

logger = logging.getLogger(__name__)


class HHFilter(Executable):
    """
    Executable wrapper for the HHFilter program from HH-suite.

    HHfilter is used to filter multiple sequence alignments (MSAs) by maximum
    pairwise sequence identity, minimum coverage, and other criteria to reduce
    MSA size while maintaining diversity.

    Example:
        ```python
        hhfilter = HHFilter.get_or_initialize()
        version = hhfilter.get_version()
        bin_path = hhfilter.get_bin_path()

        # Run hhfilter with parameters
        result = hhfilter.run_command("-i", "input.a3m", "-o", "output.a3m", "-maxseq", "1000", "-id", "90", "-cov", "50")
        ```
    """

    name = "hhfilter"
    required_verification_text = ("HHfilter", "Filter an alignment", "hhfilter -i", "-o")
    version_cmd = "--help"  # HHfilter shows version in help output

    @classmethod
    def initialize(cls, bin_path: PathLike | None = None, *args, **kwargs) -> "HHFilter":
        """Initialize HHfilter executable.

        Args:
            bin_path: Path to hhfilter executable. If None, attempts to find using HHFILTER_PATH env variable.

        Returns:
            Initialized HHFilter executable.

        Raises:
            ExecutableError: If executable not found or invalid.
        """
        if bin_path is None:
            bin_path = cls._infer_bin_path_from_env_var()
        return super().initialize(bin_path, *args, **kwargs)

    @staticmethod
    def _infer_bin_path_from_env_var() -> PathLike:
        """Get the path to the hhfilter executable from environment variables."""
        hhfilter_path = os.environ.get("HHFILTER_PATH")
        if hhfilter_path is not None and os.path.isfile(hhfilter_path) and os.access(hhfilter_path, os.X_OK):
            return hhfilter_path

        raise ExecutableError(
            "No `bin_path` provided and `HHFILTER_PATH` environment variable not set.\n"
            "Please set the `HHFILTER_PATH` environment variable to the path of the hhfilter executable "
            "or provide a `bin_path` to the `HHFilter` constructor: "
            "`HHFilter(bin_path='/path/to/hhfilter')`."
        )

    @classmethod
    def run_command(cls, *args: str) -> subprocess.CompletedProcess:
        """Run hhfilter with the specified arguments.

        Args:
            *args: Command line arguments to pass to hhfilter.

        Returns:
            CompletedProcess instance with command output.

        Raises:
            ExecutableError: If HHFilter not initialized.
            subprocess.CalledProcessError: If the command fails.
        """
        if not cls._is_initialized:
            raise ExecutableError("HHFilter not initialized. Run `HHFilter.initialize(...)` first.")

        cmd = [cls._bin_path, *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=True)
