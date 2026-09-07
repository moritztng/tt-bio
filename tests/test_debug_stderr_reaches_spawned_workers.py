"""`--debug` has to turn the stderr filter off in the WORKER, not just in the CLI process.

The filter forwards fd-level stderr through a pipe drained by a forked child, and that child
holds PR_SET_PDEATHSIG SIGKILL. So when a worker dies, the child is killed with whatever is
still in the pipe -- the worker's traceback. The CLI only ever prints "the worker's own
traceback above says why", which then points at nothing.

The escape hatch was `"--debug" not in sys.argv`, evaluated at import in every process. A
multiprocessing spawn re-execs python with `-c from multiprocessing.spawn import spawn_main`,
so the worker's argv does not carry `--debug` and the filter reinstalled itself in exactly the
process whose stderr was asked for. OpenDDE's 992-residue deep-MSA failure hit this twice, on
two different trees, and could not be attributed either time.

Host-only: no device, no weights. The subprocess sets argv before importing so the import-time
decision is the thing under test.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tt_bio.main import _stderr_filter_wanted

REPO = Path(__file__).resolve().parent.parent


def _import_with(argv_tail: str, env_extra: dict[str, str]) -> str:
    """Import tt_bio.main in a fresh process and report TT_BIO_DEBUG_STDERR after import."""
    code = textwrap.dedent(f"""
        import os, sys
        sys.argv = ['tt-bio', 'predict', 'x.yaml'] + {argv_tail!r}.split()
        import tt_bio.main            # noqa: F401  -- the import IS the thing under test
        print('ENV=' + (os.environ.get('TT_BIO_DEBUG_STDERR') or ''))
    """)
    env = {**os.environ, "TT_VISIBLE_DEVICES": "", "TT_BIO_DEBUG_STDERR": ""}
    env.pop("TT_BIO_DEBUG_STDERR")
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("ENV="))
    return line[len("ENV="):]


def test_a_spawned_worker_inherits_the_decision_not_the_argv():
    """The regression. `--debug` on the CLI must leave a mark a spawned child can read.

    A spawn inherits the environment and not the argv, so the environment is the only channel
    that survives. Without this the assertion below is empty and the worker refilters.
    """
    assert _import_with("--debug", {}) == "1"


def test_without_debug_nothing_is_marked():
    """The negative control: the mark must come from `--debug`, not from importing at all."""
    assert _import_with("", {}) == ""


def test_the_env_alone_suppresses_the_filter_for_a_worker_with_no_debug_argv():
    """This is the worker's own situation: env set by the parent, argv without `--debug`."""
    assert not _stderr_filter_wanted(["tt-bio", "predict"], {"TT_BIO_DEBUG_STDERR": "1"})


def test_argv_alone_still_suppresses_it_for_the_cli_process():
    assert not _stderr_filter_wanted(["tt-bio", "predict", "--debug"], {})


def test_a_plain_run_still_gets_the_filter():
    """Or the nanobind leak spam this filter exists to drop comes back for every user."""
    assert _stderr_filter_wanted(["tt-bio", "predict"], {})


@pytest.mark.parametrize("blank", ["", None])
def test_a_blank_env_value_is_not_a_mark(blank):
    """An exported-but-empty variable must not silently disable the filter for everyone."""
    env = {} if blank is None else {"TT_BIO_DEBUG_STDERR": blank}
    assert _stderr_filter_wanted(["tt-bio", "predict"], env)
