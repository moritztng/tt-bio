"""A campaign's accepted set has to be reproducible from its seed -- including the names.

B2P's charter asks that "the accepted set is reproducible from a seed". BindCraft 2 answers the
substance of that itself: `max_trajectories` is in `EXCLUDED_SETTING_NAMES`, so rerunning a seed
under a smaller budget designs the same binders. What it cannot defend against is a caller that
writes the budget into a setting it DOES hash. `perf/bcx_p10_campaign/campaign_run.py` did
exactly that -- `binder_lengths` as a list scaled to the budget -- and the soak's own two
Blackhole runs designed an identical first binder (same sequence, i_pTM 0.85, pLDDT 0.88,
i_pAE 0.19, and the same rejections at anneal and mutate) under two different names, so a diff
of the accepted sets would have reported nothing in common when everything was.

The first test runs anywhere, including a host with no BindCraft 2; the second needs it and
skips without it.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = ROOT / "perf" / "bcx_p10_campaign" / "campaign_run.py"


def test_the_harness_asks_for_one_length_not_one_per_trajectory():
    """The override the harness builds must not carry the budget into the design identity."""
    line = [l for l in HARNESS.read_text().splitlines() if "binder_lengths=[" in l]
    assert line, "the harness no longer sets binder_lengths; this test is pinning nothing"
    assert not any("* budget" in l or "] * budget" in l or "'] * budget" in l for l in line), line
    assert any(re.search(r"binder_lengths=\[\{args\.binder\}\]", l) for l in line), line


def test_the_same_seed_under_two_budgets_designs_the_same_identity():
    identity = pytest.importorskip("bindcraft.design_identity",
                                   reason="BindCraft 2 is not importable on this host")
    from bindcraft.settings import parse_setting_overrides, read_settings
    examples = pathlib.Path(identity.__file__).resolve().parents[1] / "examples" / "pdl1.json"
    if not examples.is_file():
        pytest.skip("BindCraft 2's example settings are not on this host")

    def hashed(budget, lengths):
        overrides = [f"campaign_seed=100", f"max_trajectories={budget}",
                     f"binder_lengths={lengths}", "project_folder=/tmp/identity-probe",
                     "number_of_final_designs=500"]
        settings = read_settings(str(examples), parse_setting_overrides(overrides))
        return identity.design_hash(settings)[0]

    assert hashed(24, "[146]") == hashed(3, "[146]")
    # ... and the shape the harness used to write is what broke it, so state that too.
    assert hashed(24, "[" + ",".join(["146"] * 24) + "]") != hashed(3, "[146,146,146]")
    assert identity.is_excluded_setting("max_trajectories")
    assert not identity.is_excluded_setting("binder_lengths")
