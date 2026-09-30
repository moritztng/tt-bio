"""When a BindCraft 2 campaign stops, and whether that is what the researcher asked for.

A campaign has two stop conditions and they answer different questions. `number_of_final_designs`
is what the researcher wants -- enough accepted designs -- and `max_trajectories` is what they are
willing to spend getting there. Whichever comes first ends the campaign, and `claim_trajectory()`
is the single place both are read: it hands out the next trajectory number or `None`, under the
project folder's lock, so N interleaved trajectories on one card and N worker processes on one
folder see one decision.

These are card-free characterisation tests against the real accounting. The soak's question is
not "does it stop" but "does it stop where a researcher would predict", and the two places it
does not are named in the test names below: a campaign can deliver MORE designs than were asked
for, and a budget of zero is taken literally.
"""
import pytest

campaign_output = pytest.importorskip(
    "bindcraft.campaign_output",
    reason="BindCraft 2 is not importable here; run with BCX_BC2 on PYTHONPATH")

CampaignProgress = campaign_output.CampaignProgress


def progress(tmp_path, *, designs, budget):
    return CampaignProgress(str(tmp_path), requested_designs=designs, max_trajectories=budget)


def test_the_designs_asked_for_end_the_campaign_before_the_budget(tmp_path):
    """Enough accepted designs stops it, with budget left over. This is the common case: a
    researcher asks for 2 designs and 100 trajectories and expects to pay for as few as it
    takes."""
    campaign = progress(tmp_path, designs=2, budget=100)
    assert campaign.claim_trajectory() == (1, 0)
    campaign.record_accepted_design()
    assert campaign.claim_trajectory() == (2, 1), "one design short, so it carries on"
    campaign.record_accepted_design()
    assert campaign.claim_trajectory() is None, "the second design ends it at trajectory 2 of 100"
    assert campaign.campaign_status() == (2, 2)


def test_the_budget_ends_a_campaign_whose_filters_accept_nothing(tmp_path):
    """The other way out, and the one a hard target takes: nothing passes, so the spend is what
    stops it rather than an endless run."""
    campaign = progress(tmp_path, designs=500, budget=4)
    numbers = [campaign.claim_trajectory() for _ in range(4)]
    assert [claim[0] for claim in numbers] == [1, 2, 3, 4]
    assert campaign.claim_trajectory() is None
    assert campaign.campaign_status() == (0, 4)


def test_a_campaign_can_deliver_more_designs_than_were_asked_for(tmp_path):
    """**Surprising, and it is upstream's design, not a defect.**

    The stop condition is read when a trajectory is CLAIMED, and an interleaved campaign has N
    trajectories in flight at once. Each one that finishes can accept a design, so the last claim
    granted while the count was short can push the total past the request: three arms claim, all
    three accept, and a researcher who asked for two gets three.

    That is the right trade -- throwing away a finished, filter-passing design to hit a number
    exactly would be worse -- but it means `number_of_final_designs` is a floor, not a quota, and
    the accepted table is the count to trust. The soak asserts it so a change upstream is visible
    here, and so the number is not reported as a promise.
    """
    campaign = progress(tmp_path, designs=2, budget=100)
    claimed = [campaign.claim_trajectory() for _ in range(3)]
    assert [claim[0] for claim in claimed] == [1, 2, 3], "three arms in flight, none accepted yet"
    for _ in range(3):
        campaign.record_accepted_design()
    accepted, _trajectories = campaign.campaign_status()
    assert accepted == 3 > 2, "asked for 2, delivered 3"
    assert campaign.claim_trajectory() is None, "and it stops at the next claim, not before"


def test_a_budget_of_zero_is_unbounded_not_none(tmp_path):
    """**The trap.** `max_trajectories=0` reads like "do not design" and means "no limit".

    Measured against the real accounting: 0 and `None` both grant trajectory 1, 2, 3, ... with no
    end, while -1 grants nothing. So a researcher who writes 0 meaning "none", with designs still
    requested, gets a campaign that runs until the filters pass enough designs -- on a hard target,
    never -- holding a leased chip the whole time. Nothing in the campaign said so, which is why
    `bindcraft2.stop_conditions` now says it before a card is opened.
    """
    campaign = progress(tmp_path, designs=500, budget=0)
    assert [claim[0] for claim in (campaign.claim_trajectory() for _ in range(5))] == [1, 2, 3, 4, 5]

    unset = progress(tmp_path.parent / "unset", designs=500, budget=None)
    assert unset.claim_trajectory() == (1, 0), "None behaves the same as 0"

    negative = progress(tmp_path.parent / "negative", designs=500, budget=-1)
    assert negative.claim_trajectory() is None, "and a negative budget designs nothing at all"


def test_asking_for_no_designs_runs_nothing_either(tmp_path):
    campaign = progress(tmp_path, designs=0, budget=100)
    assert campaign.claim_trajectory() is None


def test_the_two_conditions_are_read_under_one_lock_so_arms_agree(tmp_path):
    """The last trajectory of the budget is granted exactly once, however many arms ask.

    Two arms reaching the end of the budget together must not both be told to go: that is the
    overshoot that charges a researcher for a trajectory past the budget they set. The count is
    incremented under the project folder's lock inside the same call that reads it.
    """
    campaign = progress(tmp_path, designs=500, budget=5)
    same_folder = progress(tmp_path, designs=500, budget=5)
    granted = []
    for _ in range(4):
        granted.append(campaign.claim_trajectory())
    granted.append(same_folder.claim_trajectory())
    assert sorted(claim[0] for claim in granted) == [1, 2, 3, 4, 5]
    assert campaign.claim_trajectory() is None and same_folder.claim_trajectory() is None


def test_a_campaign_that_met_its_request_stays_stopped_when_it_is_resumed(tmp_path):
    """Restarting a finished campaign must not spend another trajectory on it."""
    campaign = progress(tmp_path, designs=1, budget=100)
    campaign.claim_trajectory()
    campaign.record_accepted_design()
    assert campaign.claim_trajectory() is None
    assert progress(tmp_path, designs=1, budget=100).claim_trajectory() is None, (
        "a resumed campaign that already has the designs asked for does not start another "
        "trajectory")


class TestWhatTheCampaignSaysBeforeItOpensACard:
    """`stop_conditions` turns the accounting above into the line a researcher reads first."""

    def line(self, **settings):
        from tt_bio import bindcraft2
        return bindcraft2.stop_conditions(settings)

    def test_both_conditions_are_named_with_their_numbers(self):
        line = self.line(number_of_final_designs=4, max_trajectories=24)
        assert "4 accepted designs" in line and "24 trajectories spent" in line
        assert "floor, not a quota" in line

    def test_an_unbounded_budget_says_so_and_says_it_holds_the_chip(self):
        for budget in (0, None):
            line = self.line(number_of_final_designs=500, max_trajectories=budget)
            assert "no trajectory budget" in line, budget
            assert "unbounded, not zero" in line and "holds this chip" in line
            assert "500 designs" in line

    def test_a_negative_budget_says_the_folder_will_be_empty(self):
        line = self.line(number_of_final_designs=500, max_trajectories=-2)
        assert "designs NOTHING" in line and "reads like a crash" in line

    def test_asking_for_no_designs_says_so(self):
        assert "accepts NOTHING" in self.line(number_of_final_designs=0, max_trajectories=24)

    def test_a_trajectory_only_run_is_upstreams_line_not_ours(self):
        assert self.line(trajectory_only=True, max_trajectories=4) == ""

    def test_a_malformed_setting_is_left_to_preflight(self):
        assert self.line(number_of_final_designs="lots", max_trajectories=24) == ""

    def test_the_singular_reads_as_english(self):
        line = self.line(number_of_final_designs=1, max_trajectories=1)
        assert "1 accepted design," in line and "1 trajectory spent" in line

    def test_run_campaign_prints_it_before_a_card_is_opened(self, monkeypatch, tmp_path, capsys):
        """It has to be on the shipped path, not only callable: a line nothing prints is not a
        line a researcher reads."""
        from tt_bio import bindcraft2
        monkeypatch.setattr(bindcraft2.bcinputs, "refuse_unusable_inputs", lambda settings: None)
        monkeypatch.setattr(bindcraft2, "design_tokens", lambda settings: 288)
        pytest.importorskip("bindcraft.campaign")
        from bindcraft import campaign
        monkeypatch.setattr(campaign, "run_campaign", lambda *a, **kw: 0)
        bindcraft2.run_campaign({"number_of_final_designs": 3, "max_trajectories": 9},
                                str(tmp_path), trajectories_per_card=1)
        assert "3 accepted designs" in capsys.readouterr().out
