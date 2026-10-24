"""The beam-search loop: branching, the stable tie-break, pruning and the saved rollouts.

The model is a toy whose state is one float per member, so the test is about control flow -- how
many candidates get branched, which survive, what is scored and what is folded -- rather than about
a trunk. That is the point of the `FlowModel` seam: the loop is gradeable before `rfd4-trunk` lands.
"""
import pytest

from tt_bio.rfd4.beam import BeamSearchStrategy, FlowModel
from tt_bio.rfd4.search import SEARCH_REWARD_KEY, RewardModel


class ToyModel:
    """State is a list of floats. A segment adds the steps it integrates, so a decoded candidate
    tells you exactly which path produced it."""

    nsteps = 60

    def __init__(self):
        self.rollout_steps = 0
        self.clean_calls = 0

    def initial_state(self, batch, n_per_input):
        return [0.0] * (n_per_input * len(batch))

    def expand(self, state, factor):
        return [x for x in state for _ in range(factor)]

    def select(self, state, indices):
        return [state[i] for i in indices]

    def partial_simulation(self, state, start_step, end_step):
        if end_step == self.nsteps:
            self.rollout_steps += (end_step - start_step) * len(state)
        return [x + (end_step - start_step) for x in state]

    def clean_prediction(self, state, step):
        self.clean_calls += 1
        return list(state)

    def decode(self, state):
        return [{"value": x, "i": i} for i, x in enumerate(state)]


class Provider:
    """Scores a candidate by position in the batch, so rankings in the test are chosen, not random."""

    def __init__(self, name, scores=None, key="plddt_coi"):
        self.name = name
        self.scores = scores
        self.key = key
        self.calls = 0

    def score(self, candidates, batch):
        self.calls += 1
        if self.scores is None:
            return [{f"{self.name}.{self.key}": float(c["value"])} for c in candidates]
        return [dict(row) for row in self.scores[:len(candidates)]]


def strategy(**kw):
    kw.setdefault("checkpoints", {30: {"metrics": ["oracle"],
                                       "rewards": {"oracle.plddt_coi": {"weight": 1.0}}}})
    kw.setdefault("providers", {"oracle": Provider("oracle")})
    kw.setdefault("n_branch", 2)
    kw.setdefault("beam_width", 2)
    return BeamSearchStrategy(**kw)


def test_the_model_seam_is_the_protocol_the_loop_declares():
    assert isinstance(ToyModel(), FlowModel)


def test_branch_score_prune_counts():
    oracle = Provider("oracle")
    s = strategy(providers={"oracle": oracle})
    model = ToyModel()
    out = s.generate(model, batch=["x"], n_inputs=1)

    assert oracle.calls == 1
    # beam 2 x branch 2 = 4 candidates scored, 2 survive, 2 designs out
    assert len(out) == 2
    assert all(d["from_beam"] and d["orig_batch_idx"] == 0 for d in out)
    # 4 candidates rolled out from 30 to 60, and nothing more: a surviving candidate's rollout
    # IS its design, so finishing the beam re-simulates nothing
    assert model.rollout_steps == 4 * 30
    assert all(d["reward"] is not None for d in out)
    assert model.clean_calls == 0


def test_a_missing_weighted_key_removes_the_candidate_rather_than_ranking_it():
    """The oracle fails on two of four candidates; the survivors are the two that scored."""
    scores = [{"oracle.plddt_coi": 70.0}, {}, {}, {"oracle.plddt_coi": 40.0}]
    s = strategy(providers={"oracle": Provider("oracle", scores=scores)})
    out = s.generate(ToyModel(), batch=["x"], n_inputs=1, nsamples=1)
    assert len(out) == 2


BAD_VALUES = [float("nan"), float("inf"), float("-inf"), "n/a", None, True]


@pytest.mark.parametrize("bad", BAD_VALUES, ids=repr)
@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2), (2, 0, 3, 1)])
def test_a_non_finite_metric_makes_the_candidate_invalid_and_ranks_it_last(bad, order):
    """A NaN reward compares unordered, so before this the survivors followed insertion order; an
    inf ranked first. Whatever the order the candidates arrive in, the survivors are the two best
    valid ones, best first, and the invalid ones carry a reason naming the key and the value."""
    rows = [{"oracle.plddt_coi": v} for v in (bad, 30.0, bad, 50.0)]
    s = strategy(providers={"oracle": Provider("oracle", scores=[rows[i] for i in order])},
                 save_rollouts=True)
    out = s.generate(ToyModel(), batch=["x"], n_inputs=1)
    beam = [d for d in out if d["from_beam"]]
    assert [d["reward"] for d in beam] == [50.0, 30.0]
    assert all(d["invalid_reason"] is None for d in beam)
    dropped = [d for d in out if not d["from_beam"]]
    assert len(dropped) == 2 and all(d["reward"] == float("-inf") for d in dropped)
    assert all(d["invalid_reason"] == f"oracle.plddt_coi={bad!r}" for d in dropped)


def test_an_invalid_reward_names_every_offending_key():
    reward = RewardModel({"oracle.plddt_coi": {"weight": 1.0}, "oracle.ptm": {"weight": 2.0},
                          "oracle.ipae": {"weight": -1.0}, "oracle.unused": {"weight": 0.0}})
    assert reward.score({"oracle.plddt_coi": 80.0, "oracle.ptm": float("nan"), "oracle.unused": None}) == (
        float("-inf"), "oracle.ipae missing, oracle.ptm=nan")
    assert reward.score({"oracle.plddt_coi": 1e308, "oracle.ptm": 1e308, "oracle.ipae": 0}) == (
        float("-inf"), "the weighted sum is inf")
    assert reward.score({"oracle.plddt_coi": 80.0, "oracle.ptm": 0.5, "oracle.ipae": 3}) == (78.0, None)


def test_the_tie_break_is_stable_so_selection_is_deterministic():
    scores = [{"oracle.plddt_coi": 50.0}] * 4
    s = strategy(providers={"oracle": Provider("oracle", scores=scores)},
                 save_rollouts=True)
    out = s.generate(ToyModel(), batch=["x"], n_inputs=1)
    beam = [d for d in out if d["from_beam"]]
    dropped = [d for d in out if not d["from_beam"]]
    assert len(beam) == 2 and len(dropped) == 2
    # equal rewards: the first two candidates survive, both times
    assert [d["generated_atom_array"]["i"] for d in dropped] == [2, 3]


def test_clean_prediction_scores_without_a_rollout():
    s = strategy(checkpoints={20: {"metrics": ["novelty"], "use_clean_prediction": True,
                                   "rewards": {"novelty.is_novel": {"weight": 1.0}}}},
                 providers={"novelty": Provider("novelty", key="is_novel")})
    model = ToyModel()
    s.generate(model, batch=["x"], n_inputs=1)
    assert model.clean_calls == 1
    assert model.rollout_steps == 2 * 40      # only the two survivors finish, after the checkpoint


def test_continuation_criteria_drops_a_whole_input():
    scores = [{"oracle.plddt_coi": 10.0}] * 4
    s = strategy(checkpoints={30: {"metrics": ["oracle"],
                                   "rewards": {"oracle.plddt_coi": {"weight": 1.0}},
                                   "continuation_criteria": {"oracle.plddt_coi": {"min": 50.0}}}},
                 providers={"oracle": Provider("oracle", scores=scores)})
    assert s.generate(ToyModel(), batch=["x"], n_inputs=1) == []


def test_save_criteria_filters_the_dropped_rollouts():
    scores = [{"oracle.plddt_coi": v} for v in (90.0, 80.0, 70.0, 10.0)]
    s = strategy(providers={"oracle": Provider("oracle", scores=scores)},
                 save_rollouts=True, save_criteria={"oracle.plddt_coi": {"min": 50.0}})
    out = s.generate(ToyModel(), batch=["x"], n_inputs=1)
    dropped = [d for d in out if not d["from_beam"]]
    assert len(out) == 3 and len(dropped) == 1        # the 10.0 rollout is not kept
    assert dropped[0]["reward"] == 70.0
    assert dropped[0]["metrics"][SEARCH_REWARD_KEY] == 70.0


def test_rollouts_are_not_returned_unless_asked_for():
    s = strategy()
    assert all(d["from_beam"] for d in s.generate(ToyModel(), batch=["x"], n_inputs=1))


def test_two_checkpoints_run_in_order_and_the_beam_narrows_each_time():
    oracle = Provider("oracle")
    s = strategy(checkpoints={40: {"metrics": ["oracle"],
                                   "rewards": {"oracle.plddt_coi": {"weight": 1.0}}},
                              20: {"metrics": ["oracle"],
                                   "rewards": {"oracle.plddt_coi": {"weight": 1.0}}}},
                 providers={"oracle": oracle})
    assert list(s.checkpoints) == [20, 40]
    out = s.generate(ToyModel(), batch=["x"], n_inputs=1)
    assert oracle.calls == 2 and len(out) == 2


def test_each_input_keeps_its_own_beam():
    s = strategy()
    out = s.generate(ToyModel(), batch=["x", "y"], n_inputs=2)
    assert sorted(d["orig_batch_idx"] for d in out) == [0, 0, 1, 1]


def test_oracle_checkpoints_are_the_ones_that_cost_a_fold():
    s = strategy(checkpoints={20: {"metrics": ["novelty"], "use_clean_prediction": True,
                                   "rewards": {"novelty.is_novel": {"weight": 1.0}}},
                              40: {"metrics": ["oracle_ppi"],
                                   "rewards": {"oracle_ppi.min_ipae_coi": {"weight": -1.0}}}},
                 providers={"novelty": Provider("novelty", key="is_novel"),
                            "oracle_ppi": Provider("oracle_ppi", key="min_ipae_coi")})
    assert s.oracle_checkpoints() == [40]


@pytest.mark.parametrize("checkpoints, providers, message", [
    ({20: {"metrics": ["oracle"], "use_clean_prediction": True,
           "rewards": {"oracle.plddt_coi": {"weight": 1.0}}}},
     {"oracle": Provider("oracle")}, "design metrics"),
    ({20: {"metrics": ["novelty"], "rewards": {"oracle.plddt_coi": {"weight": 1.0}}}},
     {"novelty": Provider("novelty"), "oracle": Provider("oracle")}, "does not list metric"),
    ({20: {"metrics": ["oracle"], "rewards": {"oracle.plddt_coi": {"weight": 1.0}}}},
     {}, "no provider"),
])
def test_configure_metrics_refuses_at_config_time(checkpoints, providers, message):
    with pytest.raises(ValueError, match=message):
        BeamSearchStrategy(checkpoints=checkpoints, providers=providers)


@pytest.mark.parametrize("kw", [{"n_branch": 0}, {"beam_width": -1}, {"n_branch": True}])
def test_the_beam_shape_must_be_positive_integers(kw):
    with pytest.raises(ValueError):
        strategy(**kw)


def test_the_oracle_seam_plugs_into_the_loop_end_to_end():
    """Decoded candidate -> OracleRequest -> AF2-IG adapter -> reward keys -> beam. The trunk is
    stubbed, so this is the wiring of the whole chain, which is what the seam exists for."""
    from tt_bio.rfd4 import oracle_af2ig as adapter
    from tt_bio.rfd4.beam import OracleProvider
    from tt_bio.rfd4.oracle import OracleRequest
    from tests.test_rfd4_oracle_af2ig import _StubTrunk, _pdb

    structure = _pdb()
    oracle = adapter.AF2IGOracle(params="unused", name="oracle_ppi", recycles=0)
    oracle.model = _StubTrunk(10)
    provider = OracleProvider(oracle=oracle,
                              render=lambda cand, i: OracleRequest(structure=structure,
                                                                   example_id=f"cand{i}",
                                                                   chain_of_interest="B"))
    assert provider.name == "oracle_ppi"

    s = BeamSearchStrategy(
        checkpoints={30: {"metrics": ["oracle_ppi"],
                          "rewards": {"oracle_ppi.min_ipae_coi": {"weight": -1.0}}}},
        providers={"oracle_ppi": provider}, n_branch=2, beam_width=2)
    out = s.generate(ToyModel(), batch=["x"], n_inputs=1)

    assert len(out) == 2
    assert oracle.model.calls == 4            # one trunk pass per candidate, recycles=0
    for design in out:
        assert design["metrics"]["oracle_ppi.min_ipae_coi"] > 0
        assert design["reward"] == -design["metrics"]["oracle_ppi.min_ipae_coi"]


def test_a_failed_oracle_candidate_is_dropped_by_the_loop_not_ranked():
    from tt_bio.rfd4.beam import OracleProvider
    from tt_bio.rfd4.oracle import OracleRequest

    class AlwaysFails:
        name = "oracle"

        def setup(self): pass

        def teardown(self): pass

        def fold_batch(self, jobs): return [{} for _ in jobs]

    provider = OracleProvider(oracle=AlwaysFails(),
                              render=lambda c, i: OracleRequest(structure="", example_id=str(i)))
    s = BeamSearchStrategy(checkpoints={30: {"metrics": ["oracle"],
                                             "rewards": {"oracle.plddt_coi": {"weight": 1.0}}}},
                           providers={"oracle": provider}, n_branch=2, beam_width=2)
    out = s.generate(ToyModel(), batch=["x"], n_inputs=1)
    assert all(d["reward"] == float("-inf") for d in out)
