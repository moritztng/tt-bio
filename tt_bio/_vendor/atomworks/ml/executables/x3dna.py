import logging
import os
from os import PathLike

from tt_bio._vendor.atomworks.ml.executables import Executable, ExecutableError

logger = logging.getLogger(__name__)


class X3DNAFiber(Executable):
    """Executable wrapper for the x3dna-fiber program from the 3DNA package.

    This class manages the x3dna-fiber executable, which is used to generate B-form conformation
    DNA fibers (i.e. linear duplexes).


    Example:
        ```python
        fiber = X3DNAFiber.get_or_initialize("/path/to/x3dna-v2.4/bin/fiber")
        version = fiber.get_version()
        bin_path = fiber.get_bin_path()
        ```
    """

    name = "x3dna-fiber"
    required_verification_text = ("fiber", "3DNA", "SYNOPSIS", "DESCRIPTION")

    @classmethod
    def initialize(cls, bin_path: PathLike | None = None, *args, **kwargs) -> "X3DNAFiber":
        if bin_path is None:
            bin_path = cls._infer_bin_path_from_env_var()
        return super().initialize(bin_path, *args, **kwargs)

    @staticmethod
    def _infer_bin_path_from_env_var() -> PathLike:
        x3dna_path = os.environ.get("X3DNA")
        if x3dna_path is not None:
            bin_path = os.path.join(x3dna_path, "bin", "fiber")
            return bin_path
        raise ExecutableError(
            "No `bin_path` provided and `X3DNA` environment variable not set.\n"
            "Please set the `X3DNA` environment variable to the root directory of the X3DNA installation "
            "or provide a `bin_path` to the `X3DNAFiber` constructor: "
            "`X3DNAFiber(bin_path='/path/to/fiber')`."
        )

    @classmethod
    def _setup(cls, bin_path: PathLike) -> None:
        """Sets up the X3DNA environment by setting the X3DNA environment variable.

        The X3DNA environment variable must point to the root directory of the X3DNA installation,
        which is typically two levels up from the executable location.

        Args:
            - bin_path (PathLike): Path to the x3dna-fiber executable.
        """
        # ... set the X3DNA environment variable
        x3dna_path = os.path.dirname(os.path.dirname(bin_path))
        logger.info(f"Setting environment variable X3DNA={x3dna_path}")
        os.environ["X3DNA"] = x3dna_path
