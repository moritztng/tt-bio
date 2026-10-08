"""Two token axes are now the comparison this row needs, so a card job's artifacts have to be named
by binder length as well as by card. They were not: a second campaign at another length overwrote
the first one's log and clock, and refused to start at all because the project folder from the
first was still sitting at the shared path.

The first attempt at the rename wrote `chip$CARD_l$LEN`, which bash reads as the variable
`CARD_l` -- not `$CARD` followed by `_l`. `set -u` would have caught that one at launch, but only
after a turn had gone to the trouble of arming the job, so it is cheaper to catch it here.
"""

import pathlib
import re

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "perf" / "bci_accept"
CARD_JOBS = ["card_campaign.sh", "card_ab.sh"]
ASSIGNMENT = re.compile(r"^(OUT|CLK|PROJ)=(\S+)$", re.M)


def assignments(name):
    return dict(ASSIGNMENT.findall((SCRIPTS / name).read_text()))


@pytest.mark.parametrize("name", CARD_JOBS)
def test_every_artifact_path_is_keyed_by_card_and_by_length(name):
    found = assignments(name)
    assert found, f"{name} has no OUT/CLK/PROJ assignment to check"
    for var, value in found.items():
        assert "${CARD}" in value, f"{name}: {var}={value} does not vary with the card"
        assert "${LEN}" in value, f"{name}: {var}={value} does not vary with the binder length"


@pytest.mark.parametrize("name", CARD_JOBS)
def test_no_path_uses_the_brace_less_form_bash_reads_as_another_name(name):
    text = (SCRIPTS / name).read_text()
    bad = re.findall(r"\$(?:CARD|LEN)[A-Za-z0-9_]", text)
    assert not bad, f"{name}: {bad} -- bash reads these as longer variable names"


@pytest.mark.parametrize("name", CARD_JOBS)
def test_the_two_card_jobs_do_not_collide_on_one_path(name):
    other = [n for n in CARD_JOBS if n != name]
    mine = set(assignments(name).values())
    for o in other:
        assert not (mine & set(assignments(o).values())), f"{name} and {o} share an artifact path"
