import os
import re
import subprocess
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Iterable
from os import PathLike

__all__ = ["Executable", "ExecutableError", "get_executable", "list_executables"]

_EXECUTABLES = {}
"""
Container to hold all executable instances.

Do not write to this variable directly; instead, use the `get_executable` function.
"""


class ExecutableError(Exception):
    """
    Exception raised for errors related to executable configuration or validation.
    """

    pass


class Executable: ...


def get_executable(name: str) -> Executable:
    """
    Retrieves an initialized executable instance by name.

    Args:
        - name (str): Name of the executable to retrieve.

    Returns:
        - Executable: The initialized executable instance.

    Raises:
        - ExecutableError: If no executable with the given name exists.
    """
    if name not in _EXECUTABLES:
        raise ExecutableError(
            f"Executable '{name}' not found. Initialize it first with the appropriate `Executable` subclass."
        )
    return _EXECUTABLES[name]


def list_executables() -> dict[str, str]:
    """
    List all initialized executables.

    Returns:
        - dict[str, str]: A dictionary of executable names and their string representations.
    """
    return {name: str(executable) for name, executable in _EXECUTABLES.items()}


def _get_executable_from_cls(cls: object) -> Executable | None:
    """
    Get the executable from the given class.
    """
    for executable in _EXECUTABLES.values():
        # NOTE: Use '__name__' instead of 'isinstance' to avoid metaclass issues
        #  by only going 1 level deep in the inheritance hierarchy
        if executable.__name__ == cls.__name__:
            return executable
    return None


def get_name(bin_path: PathLike) -> str:
    """
    Get the name of the executable at the given path.
    """
    return os.path.basename(bin_path)


def get_version(bin_path: PathLike, version_cmd: str = "--version") -> str:
    """
    Extracts the version string from an executable's version command output.

    Runs the executable with the specified version command and attempts to parse a semantic version
    string from the output using regex pattern matching.

    Args:
        - bin_path (PathLike): Path to the executable binary.
        - version_cmd (str, optional): Command line argument to get version info. Defaults to "--version".

    Returns:
        - str: Version string in semantic versioning format (e.g., "1.2.3" or "2.0.0-beta").
            If no version string is found, returns "unknown".
    """
    bin_path = os.path.abspath(bin_path)
    result = subprocess.run([bin_path, *version_cmd.split()], capture_output=True, text=True)
    output = result.stdout + result.stderr
    # Find matches for version string containing major, minor, and optional patch version
    match = re.search(r"v?(?:\d+\.)+\d+(?:-[a-zA-Z0-9]+)?", output)
    if match is None:
        return "unknown"
    version_string = match.group(0)
    # Strip leading 'v' if present
    return version_string.lstrip("v")


def assert_is_valid_executable_path(
    bin_path: PathLike, required_verification_text: Iterable[str] | None = None, verification_cmd: str = "--help"
) -> None:
    """
    Assert that the given path is a valid executable and that it satisfies the given verification requirements.

    Args:
        path (str): The path to the executable.
        required_verification_text (Iterable[str] | None): The text that must be present in the executable's help text.
        verification_args (list[str]): The arguments to pass to the executable for verification. By default,
            the `--help` argument is used.
    """
    bin_path = os.path.abspath(bin_path)
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Executable not found at {bin_path}")
    if not os.access(bin_path, os.X_OK):
        raise PermissionError(f"Executable lacks execute permissions: {bin_path}\nPlease run: chmod +x {bin_path}")

    # Verify executable functionality
    if required_verification_text:
        if isinstance(required_verification_text, str):
            required_verification_text = [required_verification_text]

        result = subprocess.run([bin_path, *verification_cmd.split()], capture_output=True, text=True)
        output = result.stdout + result.stderr
        for indicator in required_verification_text:
            if indicator not in output:
                raise ExecutableError(
                    f"Executable help text missing required content: {indicator}\n"
                    f"Output received:\n{output}\n"
                    "Please verify the installation is complete and properly configured."
                )


class ExecutableMeta(ABCMeta):
    """
    Metaclass for Executable classes.
    """

    def __repr__(cls) -> str:
        """
        Returns a string representation of the executable.
        """
        if not cls._is_initialized:
            name = cls.__name__
            raise ExecutableError(
                f"Executable {name} not initialized. Run `{name}(...)` once first to initialize the executable."
            )
        memory_location = hex(id(cls))
        return f"{cls.name} (version: {cls.get_version()}, bin_path: {cls.get_bin_path()}, id: {memory_location})"


class Executable(ABC, metaclass=ExecutableMeta):
    """
    Abstract base class for managing external executable programs with version control and validation.

    This class provides a singleton pattern implementation for executable management, ensuring only one instance
    of each executable type exists. It handles executable path validation, version checking, and basic setup.

    Initialize executables using the `initialize()` or `reinitialize()` class methods rather than instantiation.

    Attributes:
        - name (str): Name of the executable. If not set, defaults to the basename of the executable path.
        - version_cmd (str): Command line argument to get version information. Defaults to "--version".
        - verification_cmd (str): Command line argument to verify executable functionality. Defaults to "--help".
        - required_verification_text (Iterable[str]): Required text in verification command output to validate executable.
        - _bin_path (PathLike): Path to the executable binary (internal use).
        - _version (str): Cached version string of the executable (internal use).
        - _is_initialized (bool): Flag indicating if the executable has been initialized (internal use).

    Example:
        ```python
        class MyExecutable(Executable):
            name = "my-tool"
            required_verification_text = ["Expected Output", "Version"]

            @classmethod
            def _setup(cls, bin_path):
                # Custom setup logic if there is any (e.g. setting environment variables)
                pass


        # Initialize the executable
        exe = MyExecutable.initialize("/path/to/binary")
        version = exe.get_version()
        ```
    """

    # Set these in subclasses
    name: str = None
    version_cmd: str = "--version"
    verification_cmd: str = "--help"
    required_verification_text: Iterable[str] = None
    # Do not set these in subclasses
    _bin_path: PathLike = None
    _version: str = None
    _is_initialized: bool = False

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            f"'{cls.__name__}' should not be instantiated directly. "
            f"Use '{cls.__name__}.initialize()' or '{cls.__name__}.reinitialize()' instead."
        )

    @classmethod
    def get_or_initialize(cls, bin_path: PathLike | None = None, *args, **kwargs) -> "Executable":
        """
        Get the executable instance if it has already been initialized, or initialize it if it hasn't.
        """
        if cls._is_initialized:
            return cls
        return cls.initialize(bin_path, *args, **kwargs)

    @classmethod
    def initialize(cls, bin_path: PathLike, *args, **kwargs) -> "Executable":
        """
        Initialize the executable if it hasn't been initialized before.

        Args:
            - bin_path (PathLike): Path to the executable binary.
            - *args: Additional arguments passed to _setup.
            - **kwargs: Additional keyword arguments passed to _setup.

        Returns:
            - Executable: The initialized executable class.

        Raises:
            - ExecutableError: If the executable is already initialized.
        """
        if cls._is_initialized:
            raise ExecutableError(
                f"Executable '{cls.__name__}' is already initialized. "
                f"Use '{cls.__name__}.reinitialize()' to reinitialize with a new path."
            )
        return cls._do_initialize(bin_path, *args, **kwargs)

    @classmethod
    def reinitialize(cls, bin_path: PathLike, *args, **kwargs) -> "Executable":
        """
        Reinitialize the executable with a new path, even if it was previously initialized.

        Args:
            - bin_path (PathLike): New path to the executable binary.
            - *args: Additional arguments passed to _setup.
            - **kwargs: Additional keyword arguments passed to _setup.

        Returns:
            - Executable: The reinitialized executable class.
        """
        if cls._is_initialized:
            del _EXECUTABLES[cls.name]
            cls._is_initialized = False
        return cls._do_initialize(bin_path, *args, **kwargs)

    @classmethod
    def _do_initialize(cls, bin_path: PathLike, *args, **kwargs) -> "Executable":
        """Internal method to handle the actual initialization logic."""
        bin_path = os.path.abspath(bin_path)
        name = cls.name or get_name(bin_path)

        cls._setup(bin_path, *args, **kwargs)
        assert_is_valid_executable_path(bin_path, cls.required_verification_text, cls.verification_cmd)
        version = get_version(bin_path, cls.version_cmd)

        cls._is_initialized = True
        cls._bin_path = bin_path
        cls._version = version
        cls.name = name
        _EXECUTABLES[cls.name] = cls
        return cls

    @classmethod
    def get_bin_path(cls) -> PathLike:
        """
        Retrieves the absolute path to the executable binary.

        Returns:
            - PathLike: Absolute path to the executable binary.

        Raises:
            - ExecutableError: If the executable has not been initialized.
        """
        if not cls._is_initialized:
            raise ExecutableError(
                "Executable not initialized. Run `Executable(...)` once first to initialize the executable."
            )
        return cls._bin_path

    @classmethod
    def get_version(cls) -> str:
        """
        Retrieves the version string of the executable.

        Returns:
            - str: Version string in semantic versioning format (e.g., "1.2.3" or "2.0.0-beta").

        Raises:
            - ExecutableError: If the executable has not been initialized.
        """
        if not cls._is_initialized:
            raise ExecutableError(
                "Executable not initialized. Run `Executable(...)` once first to initialize the executable."
            )
        return cls._version

    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if the executable has been initialized.
        """
        return cls._is_initialized

    @classmethod
    @abstractmethod
    def _setup(cls, bin_path: PathLike, *args, **kwargs) -> None:
        """
        Performs custom setup logic for the executable implementation.

        This method should be implemented by subclasses to handle any specific initialization
        requirements, such as setting environment variables or validating dependencies.

        Args:
            - bin_path (PathLike): Absolute path to the executable binary.
            - *args: Additional positional arguments passed to the executable constructor.
            - **kwargs: Additional keyword arguments passed to the executable constructor.
        """
        pass

    def __repr__(self) -> str:
        """Returns the string representation using the metaclass's __repr__ method."""
        return ExecutableMeta.__repr__(self.__class__)
