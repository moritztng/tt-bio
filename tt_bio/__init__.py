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

# glibc gives every fold thread its own 64 MB malloc arena, and each arena is another anonymous
# mapping the kernel has to unmap when the process exits -- on the bounded per-CPU system_wq,
# where a long teardown is the documented host livelock (see tt_bio.runtime). The cap has to be
# in place before torch starts its thread pools, so it goes here, at the first tt_bio import
# every process reaches, CLI and spawned worker alike. TT_BIO_MALLOC_ARENAS=0 leaves glibc's
# default (8 * cores) alone.
from tt_bio.envflags import env_int as _env_int  # noqa: E402
from tt_bio.runtime import cap_malloc_arenas as _cap_malloc_arenas  # noqa: E402

_cap_malloc_arenas(_env_int("TT_BIO_MALLOC_ARENAS", 2))
