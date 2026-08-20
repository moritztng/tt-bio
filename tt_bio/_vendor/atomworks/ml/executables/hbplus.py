"""HBplus executable wrapper for hydrogen bond identification in protein structures."""

import logging
import os
from collections.abc import Iterable
from os import PathLike

from tt_bio._vendor.atomworks.ml.executables import Executable, ExecutableError

logger = logging.getLogger(__name__)


class HBplus(Executable):
    """
    Executable wrapper for HBplus, a program for identifying hydrogen bonds in protein structures.

    HBplus identifies and calculates hydrogen bonds in protein structures. It outputs
    hydrogen bond information in fixed-width format (.hb2 files).

    Note: HBplus does not support version or help commands, so verification is minimal.

    Example:
        ```python
        # Initialize automatically from HBPLUS_PATH environment variable
        hbplus = HBplus.get_or_initialize()

        # Or initialize directly with a path
        HBplus.initialize("/path/to/hbplus")

        # Get the executable path for use in subprocess calls
        bin_path = hbplus.get_bin_path()
        ```
    """

    name: str = "hbplus"
    version_cmd: str = ""  # HBplus doesn't support version command
    verification_cmd: str = ""  # HBplus doesn't support help command
    required_verification_text: Iterable[str] = None  # No verification possible

    @classmethod
    def initialize(cls, bin_path: PathLike | None = None, *args, **kwargs) -> "HBplus":
        """Initialize HBplus executable.

        Args:
            bin_path: Path to hbplus executable. If None, attempts to find using HBPLUS_PATH env variable.

        Returns:
            Initialized HBplus executable.

        Raises:
            ExecutableError: If executable not found or invalid.
        """
        if bin_path is None:
            bin_path = cls._infer_bin_path_from_env_var()
        return super().initialize(bin_path, *args, **kwargs)

    @staticmethod
    def _infer_bin_path_from_env_var() -> PathLike:
        """Get the path to the hbplus executable from environment variables."""
        hbplus_path = os.environ.get("HBPLUS_PATH")
        if hbplus_path is not None and os.path.isfile(hbplus_path) and os.access(hbplus_path, os.X_OK):
            return hbplus_path

        raise ExecutableError(
            "No `bin_path` provided and `HBPLUS_PATH` environment variable not set.\n"
            "Please set the `HBPLUS_PATH` environment variable to the path of the HBplus executable "
            "or provide a `bin_path` when initializing: "
            "`HBplus.initialize(bin_path='/path/to/hbplus')`.\n"
            "On the DIGS, the path should be: `/projects/ml/hbplus`"
        )

    @classmethod
    def _setup(cls, bin_path: PathLike) -> None:
        """
        Setup for HBplus executable.

        HBplus is an older program that doesn't follow modern CLI conventions,
        so we only verify the file exists and is executable.
        """
        # The basic checks (file exists, is executable) are done by assert_is_valid_executable_path
        # No additional setup needed for HBplus
        pass
