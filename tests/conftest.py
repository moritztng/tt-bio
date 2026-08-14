"""Shared test helpers."""
import os
from pathlib import Path

# The pinned Hugging Face snapshot the BoltzGen guards used to hardcode. Kept as a
# fallback for machines provisioned before tt-bio downloaded its own weights.
_HF_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--boltzgen--boltzgen-1"
    / "snapshots/c1be29e1f82ffcc72264f64b993c43fb4e0d17f0"
)


def boltzgen_checkpoint(filename: str, env_var: str | None = None) -> Path:
    """Resolve a BoltzGen checkpoint the way the shipped CLI does.

    tt_bio/boltzgen/cli/boltzgen.py fetches `huggingface:moritztng/boltzgen:<file>`
    into `$BOLTZ_CACHE/boltzgen/` (default `~/.boltz/boltzgen/`). The guards here
    used to point at a pinned snapshot of a different repo (`boltzgen/boltzgen-1`),
    which nothing populates, so every BoltzGen device test skipped on a machine
    tt-bio had set up for itself.
    """
    if env_var and os.environ.get(env_var):
        return Path(os.environ[env_var])
    cache = Path(os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz")))
    shipped = cache / "boltzgen" / filename
    return shipped if shipped.exists() else _HF_SNAPSHOT / filename
