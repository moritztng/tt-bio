"""The beam-search strategy itself, ported from RFD4-Proteina @ 2db07bb4.

`rfproteina/search/beam_search.py:86` branches each beam member into `n_branch` independent
segments, scores them at a checkpoint, and keeps the best `beam_width` per input. The surface here
is upstream's (`state/rfd4/BEAM.md` section 1): the same constructor, the same
`generate(model, batch)`, the same stable-sort tie-break, and the same -inf on a missing weighted
key, which is how a failed oracle fold removes a candidate instead of ranking it first.

What is NOT upstream's is the model seam. Upstream reaches into a Hydra-built flow-matching object;
here the loop talks to a `FlowModel` protocol of six methods, so the search can be tested, costed
and graded before `rfd4-trunk` lands the trunk, and so the same loop drives the torch reference and
the TT model without a branch inside it. Scoring goes through `MetricProvider`, which is where the
oracle seam (`tt_bio/rfd4/oracle.py`) plugs in.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tt_bio.rfd4.search import ORACLE_METRIC_NAMES, SEARCH_REWARD_KEY, RewardModel, parse_checkpoints

DESIGN_METRIC_NAMES = frozenset({"designability", "novelty", "diversity", "secondary_structure",
                                 "radius_of_gyration", "hbond", "clash"})


@runtime_checkable
class FlowModel(Protocol):
    """What the search needs of a generative model. One method per thing the loop actually does."""

    nsteps: int

    def initial_state(self, batch: Any, n_per_input: int) -> Any:
        """`n_per_input` noise states per input, interleaved [i0,i1,..,i0,i1,..]."""

    def expand(self, state: Any, factor: int) -> Any:
        """Each member repeated `factor` times, so the branches start from the same point."""

    def select(self, state: Any, indices: Sequence[int]) -> Any:
        """A new state holding exactly these members, in this order."""

    def partial_simulation(self, state: Any, start_step: int, end_step: int) -> Any:
        """Integrate members from `start_step` to `end_step`. Independent per member."""

    def clean_prediction(self, state: Any, step: int) -> Any:
        """The step's own x_1_hat, with no rollout and no further sampling."""

    def decode(self, state: Any) -> list[Any]:
        """One decoded candidate (atom array / structure) per member."""


class MetricProvider(Protocol):
    """A named scorer. Outputs are flattened as `{name}.{output}`, which is what rewards key on."""

    name: str

    def score(self, candidates: list[Any], batch: Any) -> list[dict]: ...


@dataclass
class Candidate:
    orig_batch_idx: int
    member: int                       # index into the branched state
    metrics: dict = field(default_factory=dict)
    reward: float = float("-inf")
    rollout: Any = None               # the decoded rollout, kept only if it may be returned


def _passes(criteria: Mapping | None, metrics: Mapping) -> bool:
    """Upstream's threshold dicts: `{key: {min: x, max: y}}`, all of which must hold."""
    if not criteria:
        return True
    for key, bounds in criteria.items():
        if key not in metrics:
            return False
        value = metrics[key]
        if (lo := bounds.get("min")) is not None and value < lo:
            return False
        if (hi := bounds.get("max")) is not None and value > hi:
            return False
    return True


class BeamSearchStrategy:
    """Branch, score, prune. `checkpoints` selects it; without them generation is vanilla."""

    def __init__(self, checkpoints: Mapping, providers: Mapping[str, MetricProvider] | None = None,
                 n_branch: int = 4, beam_width: int = 4, save_rollouts: bool = False,
                 save_criteria: Mapping | None = None, save_states: bool = False,
                 resume_from: str | None = None, resume_seed: int | None = None) -> None:
        if not isinstance(n_branch, int) or isinstance(n_branch, bool) or n_branch <= 0:
            raise ValueError("n_branch must be a positive integer.")
        if not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width <= 0:
            raise ValueError("beam_width must be a positive integer.")
        self.checkpoints = parse_checkpoints(checkpoints)
        self.providers = dict(providers or {})
        self.n_branch = n_branch
        self.beam_width = beam_width
        self.save_rollouts = save_rollouts
        self.save_criteria = dict(save_criteria) if save_criteria else None
        self.save_states = save_states
        self.resume_from = resume_from
        self.resume_seed = resume_seed
        self.rewards = {step: RewardModel(config.get("rewards", {}))
                        for step, config in self.checkpoints.items()}
        self.configure_metrics()

    # --- config time -----------------------------------------------------------------------
    def configure_metrics(self) -> None:
        """Upstream refuses at config time, not mid-run (`beam_search.py:148`)."""
        for step, config in self.checkpoints.items():
            names = set(config.get("metrics", []))
            clean = config.get("use_clean_prediction", False)
            if clean and not names <= DESIGN_METRIC_NAMES:
                raise ValueError(f"Checkpoint {step} scores a clean prediction, so it can only list "
                                 f"design metrics; got {sorted(names - DESIGN_METRIC_NAMES)}.")
            for key in self.rewards[step].weights:
                provider = key.split(".", 1)[0]
                if provider not in names:
                    raise ValueError(f"Checkpoint {step} weights {key!r} but does not list metric "
                                     f"{provider!r}.")
            for name in names:
                if name not in self.providers:
                    raise ValueError(f"Checkpoint {step} lists metric {name!r}, which has no "
                                     f"provider. A refolding metric needs its oracle.")

    def oracle_checkpoints(self) -> list[int]:
        return [step for step, config in self.checkpoints.items()
                if set(config.get("metrics", [])) & ORACLE_METRIC_NAMES]

    # --- the loop --------------------------------------------------------------------------
    def generate(self, model: FlowModel, batch: Any, n_inputs: int = 1,
                 nsamples: int = 1) -> list[dict]:
        state, members, previous, saved = self._start(model, batch, n_inputs, nsamples)
        live = set(members)
        extra: list[dict] = []
        kept: list[Candidate] = []

        for step, config in self.checkpoints.items():
            if not live:
                break
            branch = int(config.get("n_branch", self.n_branch))
            branched = model.expand(state, branch)
            branched_members = [m for m in members for _ in range(branch)]
            branched = model.partial_simulation(branched, previous, step)

            cands = [Candidate(orig_batch_idx=m, member=i)
                     for i, m in enumerate(branched_members) if m in live]
            self._score(model, batch, branched, cands, step, config)

            keep: list[Candidate] = []
            for input_idx in sorted(live):
                ranked = sorted((c for c in cands if c.orig_batch_idx == input_idx),
                                key=lambda c: -c.reward)          # stable, as upstream's argsort
                survivors = ranked[:self.beam_width]
                if not _beam_continues(config.get("continuation_criteria"), survivors):
                    live.discard(input_idx)
                    continue
                keep += survivors
                extra += self._saved_rollouts(ranked[self.beam_width:])

            state = model.select(branched, [c.member for c in keep])
            members = [c.orig_batch_idx for c in keep]
            kept = keep
            previous = step
            if self.save_states:
                saved = self._write_state(step, members, keep)

        return self._finish(model, state, members, previous, extra, kept)

    def _start(self, model, batch, n_inputs, nsamples):
        if self.resume_from is not None:
            resumed = self._read_state(self.resume_from)
            return resumed["state"], resumed["members"], resumed["step"], self.resume_from
        members = [i for _ in range(self.beam_width * nsamples) for i in range(n_inputs)]
        return model.initial_state(batch, self.beam_width * nsamples), members, 0, None

    def _score(self, model, batch, branched, cands, step, config) -> None:
        if not cands:
            return
        indices = [c.member for c in cands]
        sub = model.select(branched, indices)
        if config.get("use_clean_prediction", False):
            scored = model.decode(model.clean_prediction(sub, step))
        else:
            scored = model.decode(model.partial_simulation(sub, step, model.nsteps))
            for cand, rollout in zip(cands, scored):
                cand.rollout = rollout
        for name in config.get("metrics", []):
            provider = self.providers[name]
            for cand, row in zip(cands, provider.score(scored, batch)):
                cand.metrics.update(row)
        reward_model = self.rewards[step]
        for cand in cands:
            cand.reward = reward_model.aggregate_metrics_output(cand.metrics)
            cand.metrics[SEARCH_REWARD_KEY] = cand.reward

    def _saved_rollouts(self, dropped: list[Candidate]) -> list[dict]:
        """A non-surviving rollout becomes a design only if it was asked for and it qualifies."""
        if not self.save_rollouts:
            return []
        return [{"generated_atom_array": c.rollout, "reward": c.reward,
                 "orig_batch_idx": c.orig_batch_idx, "metrics": dict(c.metrics), "from_beam": False}
                for c in dropped if c.rollout is not None and _passes(self.save_criteria, c.metrics)]

    def _finish(self, model, state, members, previous, extra, kept) -> list[dict]:
        """Survivors become designs. A rollout-scored survivor was already rolled out to the end of
        the schedule at the last checkpoint, and that rollout is the design: re-simulating it would
        double the sampling cost of every returned design and, with a stochastic sampler, hand back
        a structure different from the one the reward was computed on."""
        rollouts = [c.rollout for c in kept]
        if previous < model.nsteps and all(r is not None for r in rollouts) and rollouts:
            decoded, rewards = rollouts, [c.reward for c in kept]
            metrics = [dict(c.metrics) for c in kept]
        else:
            if previous < model.nsteps:
                state = model.partial_simulation(state, previous, model.nsteps)
            decoded = model.decode(state)
            rewards = [c.reward for c in kept] if kept else [None] * len(members)
            metrics = [dict(c.metrics) for c in kept] if kept else [{}] * len(members)
        designs = [{"generated_atom_array": array, "reward": reward, "orig_batch_idx": idx,
                    "metrics": row, "from_beam": True}
                   for array, idx, reward, row in zip(decoded, members, rewards, metrics)]
        return designs + extra

    # --- checkpointing ---------------------------------------------------------------------
    def _write_state(self, step: int, members: list[int], keep: list[Candidate]) -> str:
        import torch

        path = Path(f"beam_state_step{step}.pt")
        torch.save({"step": step, "members": members, "beam_width": self.beam_width,
                    "rewards": [c.reward for c in keep], "seed": self.resume_seed}, path)
        return str(path)

    def _read_state(self, path: str) -> dict:
        import torch

        saved = torch.load(path, weights_only=False)
        if "state" not in saved:
            raise ValueError(f"{path} carries no sampler state; a resumed beam needs the noise "
                             f"states, not only the bookkeeping.")
        return saved


def _beam_continues(criteria: Mapping | None, survivors: list[Candidate]) -> bool:
    """An input is dropped when its WHOLE beam fails the checkpoint (`beam_search.py:465`)."""
    if not survivors:
        return False
    return any(_passes(criteria, c.metrics) for c in survivors)
