"""Shared test helpers."""
import glob
import os
import subprocess
import sys
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


# get_device() caches a module-global handle and holds an EXCLUSIVE flock on the card for the
# life of the process that opened it. A test file that folds in-process therefore makes the
# pytest parent the lease holder, and every later test that shells out to run a device fold
# blocks on that lease until it times out -- test_protenix_confidence opening the device in
# process is enough to take the four protenix fold legs after it down with it, each one
# reported as a failure a long way from the file that caused it.
#
# cleanup() closes the chip, releases the lease, and bumps device_generation(), which every
# module-level device-tensor cache is already required to key on. So the next module that wants
# a card opens a fresh one and pays only the open.
@pytest.fixture(scope="module", autouse=True)
def _release_device_after_module():
    yield
    tt = sys.modules.get("tt_bio.tenstorrent")
    if tt is not None and getattr(tt, "_device", None) is not None:
        tt.cleanup()


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


# --- tests that need a physical TT card ---------------------------------------------
#
# Mark them `@pytest.mark.device`, or `pytestmark = pytest.mark.device` for a whole file.
#
# The guard answers THREE questions, not two. "A card is present" and "this process may
# open one" are different, and the helper this replaced conflated them: it required
# TT_VISIBLE_DEVICES to be set and non-empty, so on a host that HAS a card but runs pytest
# unpinned every device test skipped silently while the card sat right there. A quiet
# no-run is worse than a loud failure, so that case is not a skip:
#
#   no card on this host                   -> skip.  Nothing to open.
#   TT_VISIBLE_DEVICES set but empty       -> skip.  The caller declared a CPU-only run.
#   card present, TT_VISIBLE_DEVICES pinned-> run.
#   card present, TT_VISIBLE_DEVICES unset -> refuse the session, loudly.
#
# The last line is the point. ttnn brings up every chip it can SEE, not just the one it
# computes on (tt_bio/device_lease.py), so an unpinned pytest on a QuietBox takes all four
# cards out from under whoever else is using them -- which happened twice in two days on
# the fleet. pytest will not guess which of those two the caller meant; it says what to type.
RUN, SKIP, REFUSE = "run", "skip", "refuse"

#: Present cards, not the driver directory: with the module loaded and no card seated,
#: /dev/tenstorrent exists and is empty. Same expression device_lease.physical_cards uses.
_CARD_NODES = "/dev/tenstorrent/[0-9]*"


def device_verdict(env=None):
    """``(verdict, reason)`` for a test that opens a TT card. See the note above."""
    env = os.environ if env is None else env
    pin = env.get("TT_VISIBLE_DEVICES")
    if pin is not None and not pin.strip():
        return SKIP, "TT_VISIBLE_DEVICES is set empty: a deliberate CPU-only run"
    if not glob.glob(_CARD_NODES):
        return SKIP, "no TT card on this host (/dev/tenstorrent/ has no card node)"
    if pin is None:
        return REFUSE, (
            "this host has TT cards and TT_VISIBLE_DEVICES is unset, so pytest would open "
            "EVERY card on the box -- ttnn brings up the whole visible set, not just the "
            "chip it computes on. Say which you meant:\n"
            "  TT_VISIBLE_DEVICES=<card> python3 -m pytest ...   run the device tests on that card\n"
            "  TT_VISIBLE_DEVICES= python3 -m pytest ...         skip them and run the rest"
        )
    return RUN, None


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "device: needs a physical TT card; skipped or refused by device_verdict()")


def pytest_collection_modifyitems(config, items):
    verdict, reason = device_verdict()
    if verdict == RUN:
        return
    marked = [i for i in items if i.get_closest_marker("device")]
    if not marked:
        return                      # nothing selected wants a card, so the pin is nobody's business
    if verdict == REFUSE:
        raise pytest.UsageError(reason)
    skip = pytest.mark.skip(reason=reason)
    for item in marked:
        item.add_marker(skip)


#: ttnn's abort when the cluster comes up with no chips. This is what a device test that
#: nobody marked dies of on a card-free host.
_NO_CHIPS = "num_chips > 0"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Backstop: an UNMARKED device test reads as a skip on a card-free host, not a failure.

    The marker is what makes a device test skip early, before it takes a card lease or forks
    a child, so marking is still the job. But a test file is added to this repo most weeks and
    a device test is not always obvious from its imports -- `tt_bio.tenstorrent.PairformerModule`
    opens a card without the word `get_device` appearing anywhere. Left alone, one unmarked file
    puts the whole suite back to unreadable, which is the thing this exists to stop.

    Narrow on purpose: it fires only on ttnn's own no-chips abort, and only when the verdict
    above already says there is no card to open. In that state a death inside the cluster
    bring-up cannot be a defect in the code under test. On a card host the hook is inert, so a
    real device failure stays a failure.
    """
    out = yield
    rep = out.get_result()
    if rep.when not in ("setup", "call") or not rep.failed:
        return
    if item.get_closest_marker("device") is not None:
        return
    if device_verdict()[0] != SKIP or _NO_CHIPS not in str(rep.longrepr):
        return
    rep.outcome = "skipped"
    rep.longrepr = (str(item.path), (item.location[1] or 0) + 1,
                    "Skipped: opened a TT card with none present -- mark it @pytest.mark.device "
                    "so it skips before it takes a lease")
