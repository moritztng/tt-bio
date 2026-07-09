from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tt-bio")
except PackageNotFoundError:  # running from a source tree, not an installed dist
    __version__ = "0+unknown"
