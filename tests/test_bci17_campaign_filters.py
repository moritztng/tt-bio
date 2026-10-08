"""A campaign run from a hand-built settings dict has NO filters, and says nothing about it.

`campaign.run_campaign_arm` calls `build_design_settings(settings)` on whatever it is handed
(campaign.py:136); BindCraft 2's own entry point `read_settings` puts `load_settings` in front of
that, and `load_settings` is what lays the dict over DEFAULT_SETTINGS. Skip it and `filters` is
absent, `build_filters` returns `{}`, and `trajectory.py:120` turns the empty set into `None`:
every redesign candidate is written with `outcome=passed` and an empty `failed_filters`, and the
stages stop gating rounds.

That is what the first #17 paired comparison measured. Nine candidates across three on-card
trajectories were all "accepted" -- at i_pTM 0.05 to 0.10, four of them with no interface residues
at all -- while the default thresholds the reporter is actually graded against are i_pTM >= 0.7
and Interface_Residues >= 7. An acceptance count taken without them is not the reporter's
measurement in either arm.
"""

import pytest

from bindcraft.filters import build_filters
from bindcraft.settings import load_settings

RAW = {
    "targets": [{"name": "hPDL1", "target_path": "/nonexistent/hPDL1.pdb",
                 "chains": "A", "hotspots": "54,56,66,115"}],
    "binder_lengths": [60, 60],
    "max_trajectories": 3,
    "number_of_final_designs": 3,
    "design_dropout": True,
    "trajectory_only": False,
    "design_models": 1,
    "campaign_seed": 42,
    "screen_steps": 50, "refine_steps": 25, "anneal_steps": 45, "harden_steps": 5,
    "mutate_steps": 0,
    "project_folder": "/tmp/does-not-matter",
}

# The thresholds #17 and #21 are actually graded on, from settings/core/default.json.
GRADED_ON = {"i_pTM": (0.7, True), "Interface_Residues": (7, True), "Target_pLDDT": None,
             "Unbound_Binder_pLDDT": (0.8, True), "i_pAE": (0.35, False)}


def test_the_hand_built_dict_on_its_own_builds_no_filters():
    assert build_filters(RAW.get("filters") or {}) == {}


def test_load_settings_restores_them():
    merged = load_settings(RAW)
    assert build_filters(merged.get("filters") or {}), "load_settings left the filters empty"


@pytest.mark.parametrize("name", sorted(GRADED_ON))
def test_the_metrics_this_issue_is_graded_on_are_present_after_the_merge(name):
    assert name in (load_settings(RAW).get("filters") or {})


@pytest.mark.parametrize("name,expected", [(n, v) for n, v in GRADED_ON.items() if v])
def test_the_thresholds_are_the_defaults_not_something_weaker(name, expected):
    entry = load_settings(RAW)["filters"][name]
    threshold, higher = expected
    assert entry["threshold"] == threshold
    assert bool(entry.get("higher")) is higher


def test_the_merge_does_not_quietly_drop_what_the_caller_asked_for():
    """A merge that restored the defaults by discarding the overrides would be no better."""
    merged = load_settings(RAW)
    for key in ("binder_lengths", "max_trajectories", "campaign_seed", "screen_steps",
                "harden_steps", "design_dropout", "project_folder"):
        assert merged[key] == RAW[key], key
