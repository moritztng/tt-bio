"""`tt-bio --version` prints the installed version and exits, without starting anything.

Two things can rot here. The flag can drift away from `tt_bio.__version__`, which is what a
bug report quotes. And it can get re-implemented as a plain flag read inside the group
callback, which arms the orphan guard and then fails outright, because a click group with no
subcommand exits 2 with "Missing command". Both are checked below; the second one was
confirmed by writing that implementation and watching these tests go red.
"""

import pytest

click = pytest.importorskip("click")
from click.testing import CliRunner

import tt_bio
from tt_bio import main


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_package_version(flag):
    result = CliRunner().invoke(main.cli, [flag])
    assert result.exit_code == 0, result.output
    assert result.output == f"tt-bio, version {tt_bio.__version__}\n"


def test_version_flag_skips_the_group_setup(monkeypatch):
    """click answers the flag while parsing, so the group callback never runs.

    arm_orphan_guard is the visible marker: it is the one thing the callback does that a
    caller can observe.
    """
    from tt_bio import device_lease

    def boom():
        raise AssertionError("the group callback ran before --version was handled")

    monkeypatch.setattr(device_lease, "arm_orphan_guard", boom)

    result = CliRunner().invoke(main.cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("tt-bio, version ")
