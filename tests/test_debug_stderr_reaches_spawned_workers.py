"""`--debug` has to keep the worker's stderr whole, not just the CLI process's.

tt-bio drops the C-level stderr of a process's last steps at exit (nanobind's leak report, see
`runtime.end_after_teardown`), and `--debug` keeps it. Until 2026-10 the same switch turned off a
forked stderr filter whose child was killed with the worker, taking the worker's traceback with
it: OpenDDE's 992-residue deep-MSA failure hit that twice and could not be attributed either time.

A multiprocessing spawn re-execs python with `-c from multiprocessing.spawn import spawn_main`,
so the worker's argv does not carry `--debug`. An argv test would quiet exactly the process whose
stderr was asked for; the environment is the channel a spawn inherits.

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

from tt_bio.runtime import quiet_teardown_wanted

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


def test_the_env_alone_keeps_stderr_for_a_worker_with_no_debug_argv():
    """This is the worker's own situation: env set by the parent, argv without `--debug`."""
    assert not quiet_teardown_wanted({"TT_BIO_DEBUG_STDERR": "1"})


def test_a_plain_run_is_quieted():
    """Or the nanobind leak spam comes back at the end of every user's run."""
    assert quiet_teardown_wanted({})


@pytest.mark.parametrize("blank", ["", None])
def test_a_blank_env_value_is_not_a_mark(blank):
    """An exported-but-empty variable must not silently bring the spam back for everyone."""
    env = {} if blank is None else {"TT_BIO_DEBUG_STDERR": blank}
    assert quiet_teardown_wanted(env)
