"""Reward aggregation and the beam-search cost model (tt_bio.rfd4.search).

The recipe is RFD4's released `thinking_medium`: 60 sampling steps, 180 trunk passes, front
loaded (`configs/recycle_strategy/medium/front_loaded.yaml` @ 2db07bb4). The shipped beam example
(`configs/inference_experiment/inference_beam_example.yaml`) branches 2 ways from a beam of 3 and
scores with an oracle at steps 20, 40, 50 and 60.
"""
import math

import pytest

from tt_bio.rfd4 import search

THINKING_MEDIUM = [[5, 6], [10, 5], [5, 4], [10, 3], [20, 2], [10, 1]]
NSTEPS = 60
ORACLE_CKPT = {"metrics": ["oracle"], "rewards": {"oracle.plddt_coi": {"weight": 1.0, "clip": {"min": 50.0}}}}


def test_thinking_medium_is_180_front_loaded_passes():
    depths = search.recycle_depths(THINKING_MEDIUM, NSTEPS)
    assert len(depths) == NSTEPS and sum(depths) == 180
    assert sum(depths[:20]) == 100 and sum(depths[20:40]) == 50
    assert sum(depths[40:50]) == 20 and sum(depths[50:]) == 10


def test_shipped_beam_example_cost():
    cost = search.beam_cost(
        checkpoints={str(s): ORACLE_CKPT for s in (20, 40, 50, 60)},
        n_branch=2, beam_width=3, n_recycle_schedule=THINKING_MEDIUM, nsteps=NSTEPS)
    # 2 * [(100+80) + (50+30) + (20+10) + (10+0)]
    assert cost["trunk_passes_per_design"] == 600
    assert cost["trunk_passes_per_input"] == 1800
    assert cost["oracle_folds_per_design"] == 8
    assert cost["oracle_folds_per_input"] == 24
    assert cost["trunk_ratio_vs_plain"] == pytest.approx(600 / 180)
    assert cost["oracle_breakeven_trunk_passes"] == pytest.approx(75.0)


def test_class_defaults_cost_twice_the_example_per_design():
    cost = search.beam_cost(
        checkpoints={str(s): ORACLE_CKPT for s in (20, 40, 50, 60)},
        n_branch=4, beam_width=4, n_recycle_schedule=THINKING_MEDIUM, nsteps=NSTEPS)
    assert cost["trunk_passes_per_design"] == 1200
    assert cost["oracle_folds_per_design"] == 16
    # Same split between model and oracle: the breakeven depends on the schedule, not the widths.
    assert cost["oracle_breakeven_trunk_passes"] == pytest.approx(75.0)


def test_clean_prediction_checkpoint_skips_the_rollout_and_the_oracle():
    """Upstream's PPI recipe: a cheap novelty gate at 20, one oracle checkpoint at 40."""
    cost = search.beam_cost(
        checkpoints={
            "20": {"metrics": ["novelty"], "use_clean_prediction": True,
                   "rewards": {"novelty.is_novel": {"weight": 1.0}}},
            "40": {"metrics": ["oracle_ppi"], "rewards": {"oracle_ppi.min_ipae_coi": {"weight": -1.0}}},
        },
        n_branch=2, beam_width=2, n_recycle_schedule=THINKING_MEDIUM, nsteps=NSTEPS)
    assert cost["trunk_passes_per_design"] == 2 * (100 + (50 + 30))
    assert cost["oracle_folds_per_design"] == 2


def test_per_checkpoint_n_branch_overrides_the_default():
    cost = search.beam_cost(
        checkpoints={"20": dict(ORACLE_CKPT, n_branch=4), "60": ORACLE_CKPT},
        n_branch=1, beam_width=1, n_recycle_schedule=THINKING_MEDIUM, nsteps=NSTEPS)
    assert cost["trunk_passes_per_design"] == 4 * (100 + 80) + 1 * (80 + 0)
    assert cost["oracle_folds_per_design"] == 5


def test_a_checkpoint_past_the_schedule_is_rejected():
    with pytest.raises(ValueError, match="exceeds nsteps"):
        search.beam_cost(checkpoints={"61": ORACLE_CKPT}, n_branch=2, beam_width=2,
                         n_recycle_schedule=THINKING_MEDIUM, nsteps=NSTEPS)


@pytest.mark.parametrize("bad", [{}, {0: {}}, {-1: {}}, {"20": {"nonsense": 1}},
                                 {"20": {"use_clean_prediction": "yes"}}, {"20": {"n_branch": 0}},
                                 {"20": {"metrics": "oracle"}}])
def test_checkpoint_schema_rejects(bad):
    with pytest.raises(ValueError):
        search.parse_checkpoints(bad)


def test_duplicate_step_keys_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        search.parse_checkpoints({20: ORACLE_CKPT, "20": ORACLE_CKPT})


def test_reward_is_weighted_clipped_and_normalised():
    model = search.RewardModel({
        "oracle.plddt_coi": {"weight": 1.0, "clip": {"min": 50.0}, "normalize_by": 100.0},
        "oracle_ppi.min_ipae_coi": {"weight": -1.0},
    })
    assert model.aggregate_metrics_output(
        {"oracle.plddt_coi": 80.0, "oracle_ppi.min_ipae_coi": 0.5}) == pytest.approx(0.8 - 0.5)
    # Clipped from below, so a hopeless candidate cannot drag the sum arbitrarily.
    assert model.aggregate_metrics_output(
        {"oracle.plddt_coi": 10.0, "oracle_ppi.min_ipae_coi": 0.0}) == pytest.approx(0.5)


def test_a_missing_weighted_key_is_minus_inf():
    """A failed oracle fold must drop its candidate, not rank it."""
    model = search.RewardModel({"oracle.plddt_coi": {"weight": 1.0}})
    assert model.aggregate_metrics_output({}) == -math.inf
    assert model.aggregate_metrics_output({"oracle.ptm_coi": 90.0}) == -math.inf


def test_a_zero_weight_key_is_not_required():
    model = search.RewardModel({"oracle.plddt_coi": {"weight": 0.0}})
    assert model.aggregate_metrics_output({}) == 0.0


@pytest.mark.parametrize("bad", [{"noprefix": {"weight": 1}}, {"a.b": {"weight": float("nan")}},
                                 {"a.b": {"normalize_by": 0}}, {"a.b": {"weight": 1, "junk": 2}},
                                 {"a.b": {"clip": {"min": 2, "max": 1}}}, {"a.b": {"clip": {}}}])
def test_reward_schema_rejects(bad):
    with pytest.raises(ValueError):
        search.RewardModel(bad)
