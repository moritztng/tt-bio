"""The AICLK sampler has to read the board that ran the fold, and on qb1 that is not the card
number. state/bci/CHIPS.md: "qb1 logical card N (TT_VISIBLE_DEVICES) is NOT node N: 0 = node 1,
1 = node 2, 2 = node 3, 3 = node 0."

The first card runs of this row sampled tenstorrent!$CARD and logged a steady 1350 MHz. That was
node 1, held by another campaign's chain; the fold was on node 2. A wrong board answers with a
plausible clock rather than an error, so nothing about the log looks wrong -- which is the whole
reason this mapping is pinned by a test instead of by a comment.
"""

import pathlib
import subprocess

import pytest

PROFILE = pathlib.Path(__file__).resolve().parents[1] / "perf" / "bci_accept" / "host_profile.sh"


def profile(host, card, **env):
    """Source host_profile.sh as the scripts do and report what it resolved."""
    script = f'CARD={card}; . "{PROFILE}"; echo "$CLK_NODE|$LOCK|$LOCK_WAIT|$WT"'
    done = subprocess.run(
        ["bash", "-c", script],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "BCI_HOST_NAME": host, **env},
    )
    assert done.returncode == 0, done.stderr
    clk_node, lock, lock_wait, wt = done.stdout.strip().split("|")
    return {"clk_node": clk_node, "lock": lock, "lock_wait": lock_wait, "wt": wt}


@pytest.mark.parametrize("card,node", [("0", "1"), ("1", "2"), ("2", "3"), ("3", "0")])
def test_qb1_samples_the_node_the_card_actually_is(card, node):
    assert profile("tt-quietbox", card)["clk_node"] == node


@pytest.mark.parametrize("card", ["0", "1", "2", "3"])
def test_qb2_card_and_node_are_the_same_number(card):
    assert profile("tt-quietbox2", card)["clk_node"] == card


def test_qb1_rejects_a_card_it_has_no_mapping_for():
    done = subprocess.run(
        ["bash", "-c", f'CARD=7; . "{PROFILE}"; echo "$CLK_NODE"'],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "BCI_HOST_NAME": "tt-quietbox"},
    )
    assert done.returncode != 0
    assert "no qb1 node mapping" in done.stderr


def test_an_unknown_host_refuses_rather_than_guessing():
    done = subprocess.run(
        ["bash", "-c", f'CARD=1; . "{PROFILE}"'],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "BCI_HOST_NAME": "some-other-box"},
    )
    assert done.returncode != 0
    assert "no BCI host profile" in done.stderr


def test_the_lock_wait_is_short_by_default_and_overridable():
    assert profile("tt-quietbox", "1")["lock_wait"] == "60"
    assert profile("tt-quietbox", "1", BCI_LOCK_WAIT="7200")["lock_wait"] == "7200"


def test_the_per_card_lock_is_per_card():
    locks = {profile("tt-quietbox", c)["lock"] for c in "0123"}
    assert len(locks) == 4


def test_a_private_clone_overrides_the_default_tree():
    assert profile("tt-quietbox", "1", BCI_WT="/home/ttuser/bci_accept_qb1_ab")["wt"] == \
        "/home/ttuser/bci_accept_qb1_ab"
