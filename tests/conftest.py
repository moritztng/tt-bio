"""Shared test helpers."""
import os
import subprocess
from pathlib import Path

import pytest


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


# The device-selection environment is process-global, and tt_bio.device_lease reads it
# live on every open. A test that sets TT_VISIBLE_DEVICES and then *pops* it instead of
# putting the caller's value back leaves the rest of the session unpinned, and an unpinned
# open brings up every card on the box -- which the card grant then refuses.
#
# On a one-card host that is invisible: unpinned and granted are the same set. On qb1
# (four cards) under the one-card grant a gate leg actually gets, it is 78 failures in
# test_tenstorrent.py and test_esmc.py, all of them the leak rather than the code, and all
# of them a long way from the test that caused it. Restoring here rather than in each test
# keeps a new test from reintroducing it.
_DEVICE_ENV = ("TT_VISIBLE_DEVICES", "TT_BIO_LEASE_CARDS", "TT_BIO_LEASE_HOLDER",
               "TT_BIO_LOGICAL_DEVICE_ID")


@pytest.fixture(autouse=True)
def _restore_device_env():
    before = {k: os.environ.get(k) for k in _DEVICE_ENV}
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
