from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tt-bio")
except PackageNotFoundError:  # running from a source tree, not an installed dist
    __version__ = "0+unknown"

# $TT_BIO_CACHE relocates every weight, both the flat checkpoints and the Hugging Face
# hub cache. The hub reads its cache path from the environment at import time, so the
# default has to be in place before anything imports huggingface_hub. A user's own
# HF_HOME/HF_HUB_CACHE is left alone.
from tt_bio.weights import configure_hf_cache as _configure_hf_cache  # noqa: E402

_configure_hf_cache()
