"""Beam-search configuration and what it costs, ported from RFD4-Proteina @ 2db07bb4.

Three pieces that need no model and no device, so they are settled before the port of the search
loop itself: the checkpoint schema (`rfproteina/search/checkpoints.py`), the reward aggregation
(`rfproteina/rewards/reward_model.py`) and the cost accounting that tells you what a beam-searched
design will actually run.

The cost model matters as much as the algorithm here. Beam search multiplies the model's own work
and adds an oracle fold per candidate, and RFD4's released recipe is front-loaded -- `thinking_medium`
spends 100 of its 180 trunk passes in the first 20 of 60 steps -- so an early checkpoint costs five
times a late one and a flat per-step count is wrong by a factor of three.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

#: The reward key `RewardModel` publishes its own aggregate under.
SEARCH_REWARD_KEY = "search.reward"

#: Metric names upstream's shipped configs give to oracle providers
#: (`configs/metrics/refolding_oracles/*.yaml`). Only these cost a fold.
ORACLE_METRIC_NAMES = frozenset({"oracle", "oracle_ppi", "oracle_dna", "oracle_apo", "oracle_off_target"})

_CHECKPOINT_KEYS = {"metrics", "rewards", "use_clean_prediction", "continuation_criteria", "n_branch"}


def parse_checkpoints(checkpoints: Mapping) -> dict[int, dict]:
    """Validate and sort the scoring checkpoints; step zero is implicit and is never scored."""
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError("search.checkpoints must be a nonempty mapping of positive integer steps.")
    parsed: dict[int, dict] = {}
    for step, config in checkpoints.items():
        if isinstance(step, str) and step.isascii() and step.isdigit():
            step = int(step)
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError(f"Checkpoint steps must be positive integers; got {step!r}.")
        if step in parsed:
            raise ValueError(f"Duplicate checkpoint {step}; quote YAML step keys before applying dotted overrides.")
        if not isinstance(config, Mapping) or set(config) - _CHECKPOINT_KEYS:
            raise ValueError(f"Checkpoint {step} accepts {', '.join(sorted(_CHECKPOINT_KEYS))}.")
        if not isinstance(config.get("use_clean_prediction", False), bool):
            raise ValueError(f"Checkpoint {step} use_clean_prediction must be a bool.")
        if "n_branch" in config:
            count = config["n_branch"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"Checkpoint {step} n_branch must be a positive integer.")
        names = config.get("metrics", [])
        if not isinstance(names, list) or not all(isinstance(name, str) and name for name in names):
            raise ValueError(f"Checkpoint {step} metrics must be a list of metric names.")
        parsed[step] = dict(config)
    return dict(sorted(parsed.items()))


class RewardModel:
    """One checkpoint's weighted sum over named metric outputs.

    Each `metric.output` key carries a weight, an optional clip and an optional divisor. A weighted
    key the metrics did not produce returns -inf, which is how a failed oracle fold removes its
    candidate from the beam instead of ranking it by accident.
    """

    def __init__(self, rewards: Mapping) -> None:
        weights, clips = {}, {}
        if not isinstance(rewards, Mapping):
            raise ValueError("Checkpoint rewards must map metric.output to reward terms.")
        for key, term in rewards.items():
            if not isinstance(key, str) or "." not in key:
                raise ValueError(f"Reward key must be metric.output; got {key!r}.")
            if not isinstance(term, Mapping) or set(term) - {"weight", "clip", "normalize_by"}:
                raise ValueError(f"Reward {key!r} accepts weight, clip, and normalize_by.")
            weight, scale = term.get("weight", 1.0), term.get("normalize_by", 1.0)
            if any(isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
                   for v in (weight, scale)) or scale <= 0:
                raise ValueError(f"Reward {key!r} requires a finite weight and positive finite normalize_by.")
            weights[key] = weight / scale
            if "clip" in term:
                bounds = term["clip"]
                if not isinstance(bounds, Mapping) or set(bounds) - {"min", "max"} or not bounds:
                    raise ValueError(f"Reward {key!r} clip accepts min and/or max.")
                if any(isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
                       for v in bounds.values()):
                    raise ValueError(f"Reward {key!r} clip bounds must be finite numbers.")
                if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
                    raise ValueError(f"Reward {key!r} clip min must not exceed max.")
                clips[key] = dict(bounds)
        self.weights = weights
        self.clip_values = clips

    def _clip(self, key: str, value: float) -> float:
        bounds = self.clip_values.get(key)
        if bounds is None:
            return float(value)
        clipped = float(value)
        if (lo := bounds.get("min")) is not None:
            clipped = max(clipped, lo)
        if (hi := bounds.get("max")) is not None:
            clipped = min(clipped, hi)
        return clipped

    def aggregate_metrics_output(self, metrics_output: dict) -> float:
        """The scalar reward for one candidate. Never mutates `metrics_output`."""
        reward = 0.0
        for key, weight in self.weights.items():
            if not weight:
                continue
            if key not in metrics_output:
                return float("-inf")
            reward += weight * self._clip(key, metrics_output[key])
        return reward


def recycle_depths(n_recycle_schedule: list, nsteps: int) -> tuple[int, ...]:
    """Expand a `[[width, depth], ...]` schedule into one trunk-pass count per sampling step."""
    depths = tuple(depth for width, depth in n_recycle_schedule for _ in range(width))
    if len(depths) != nsteps:
        raise ValueError(f"recycle schedule expands to {len(depths)} steps; expected {nsteps}")
    return depths


def beam_cost(*, checkpoints: Mapping, n_branch: int, beam_width: int,
              n_recycle_schedule: list, nsteps: int,
              oracle_metric_names=ORACLE_METRIC_NAMES) -> dict:
    """What one input and one final design cost under a beam-search config.

    A checkpoint branches `beam_width * n_branch` candidates over its own interval, then -- unless
    it scores the clean prediction -- rolls each candidate out to the end of the schedule to be
    scored, and throws that rollout away for everything it prunes. Trunk passes, not steps, are
    the unit: the recycle schedule weights each step.

    Returns per-input and per-design trunk passes and oracle folds, and the ratio against plain
    generation. Per design the beam width cancels: B members yield B designs.
    """
    parsed = parse_checkpoints(checkpoints)
    depths = recycle_depths(n_recycle_schedule, nsteps)
    if max(parsed) > nsteps:
        raise ValueError(f"Last checkpoint {max(parsed)} exceeds nsteps={nsteps}.")

    passes = lambda a, b: sum(depths[a:b])  # noqa: E731 -- trunk passes over a step interval
    per_member = 0
    oracle_calls = 0
    previous = 0
    for step, config in parsed.items():
        branch = config.get("n_branch", n_branch)
        clean = config.get("use_clean_prediction", False)
        per_member += branch * (passes(previous, step) + (0 if clean else passes(step, nsteps)))
        if not clean and set(config.get("metrics", [])) & set(oracle_metric_names):
            oracle_calls += branch
        previous = step

    plain = passes(0, nsteps)
    return {
        "trunk_passes_per_input": per_member * beam_width,
        "trunk_passes_per_design": per_member,
        "oracle_folds_per_input": oracle_calls * beam_width,
        "oracle_folds_per_design": oracle_calls,
        "plain_trunk_passes_per_design": plain,
        "trunk_ratio_vs_plain": per_member / plain,
        # One oracle fold costing more than this many trunk passes makes the search oracle-bound.
        "oracle_breakeven_trunk_passes": per_member / oracle_calls if oracle_calls else math.inf,
    }
