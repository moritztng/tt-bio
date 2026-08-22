"""Shared test helpers."""
import os
import subprocess
from pathlib import Path


def git_tracked(repo, *args):
    """Paths git tracks under *repo*, or None when *repo* is not a work tree.

    Three test files ask git what is tracked. All three used `check=True`, and
    `git ls-files` exits 128 outside a work tree, so each of them turned a
    perfectly ordinary environment into a CalledProcessError. Two of the three
    raise it at collection time, which takes down the whole session rather than
    one test. That environment is not exotic: a release gate runs the suite
    against a `git archive` export, and so does anyone testing an unpacked
    sdist. Returning None lets each caller decide -- skip, if the question only
    means something in a checkout, or fall back to walking the tree.
    """
    try:
        out = subprocess.run(["git", "ls-files", *args], cwd=repo,
                             capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    sep = "\0" if "-z" in args else "\n"
    return [x for x in out.stdout.decode().split(sep) if x]

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
