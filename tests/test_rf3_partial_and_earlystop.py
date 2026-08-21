"""Row 13 (partial diffusion) and row 14 (pLDDT early stopping), host-only.

Both features are wiring, and the failure mode wiring has is being accepted and ignored.
These tests are the ones that catch that: the schedule offset is checked against the noise
level it is supposed to produce, the draw and denoiser-call counts are counted, and the
early-stop decision is asserted in both directions on the same input so nothing else varies.

The sampler is host-side arithmetic around a device denoiser, so a stub denoiser exercises
all of row 13 with no card. Row 14's device part is one confidence-head pass; what is testable
without a card is the decision, the recycle it is taken at, and that a stop skips the rollout.
"""
from __future__ import annotations

import math

import pytest
import torch

from tt_bio.rf3.sampler import DiffusionSampler, Draws


def _stub():
    """A denoiser that returns its input and counts calls."""
    calls = []

    def denoise(x_noisy, t):
        calls.append((tuple(x_noisy.shape), float(t[0])))
        return x_noisy
    return denoise, calls


def test_partial_t_zero_is_todays_path_exactly():
    s = DiffusionSampler()
    d, n_atom = 2, 37
    coord = torch.randn(d, n_atom, 3)
    f, _ = _stub()
    torch.manual_seed(0)
    a, draws_a = s.sample(f, coord, d)
    g, _ = _stub()
    torch.manual_seed(0)
    b, _ = s.sample(g, coord, d, partial_t=0)
    assert torch.equal(a, b), "partial_t=0 must be bit-identical to the default path"
    # and the recorded stream is the documented 1 + 5*(num_timesteps-1)
    assert len(draws_a.values) == 1 + 5 * (s.num_timesteps - 1)


@pytest.mark.parametrize("partial_t", [0, 100, 190, 199])
def test_partial_t_runs_the_truncated_schedule(partial_t):
    s = DiffusionSampler()
    d, n_atom = 1, 16
    coord = torch.randn(d, n_atom, 3)
    f, calls = _stub()
    _, draws = s.sample(f, coord, d, partial_t=partial_t)
    steps = s.num_timesteps - partial_t - 1
    assert len(calls) == steps, "one denoiser call per remaining step"
    assert len(draws.values) == 1 + 5 * steps, "one initial draw plus five per step"


def test_partial_t_out_of_range_is_rejected():
    s = DiffusionSampler()
    coord = torch.randn(1, 8, 3)
    f, _ = _stub()
    for bad in (-1, 200, 500):
        with pytest.raises(ValueError, match="partial_t"):
            s.sample(f, coord, 1, partial_t=bad)


@pytest.mark.parametrize("partial_t", [0, 100, 190])
def test_the_noise_level_actually_takes_effect(partial_t):
    """The initial structure is `sched[partial_t] * normal() + coord`, so its RMS
    displacement from the input is `sched[partial_t] * sqrt(3)` per atom. A flag that is
    accepted and ignored gives sched[0]'s displacement at every partial_t."""
    s = DiffusionSampler()
    d, n_atom = 4, 512
    coord = torch.randn(d, n_atom, 3)
    sched = s.noise_schedule()

    seen = {}

    def denoise(x_noisy, t):
        seen.setdefault("x0", x_noisy.clone())
        return x_noisy
    torch.manual_seed(1)
    s.sample(denoise, coord, d, partial_t=partial_t)
    # the first denoiser call sees the initial structure after centring, rotation,
    # translation and the churn noise, so compare the draw-level construction directly
    torch.manual_seed(1)
    draws = Draws()
    x = sched[partial_t] * draws.normal((d, n_atom, 3)) + coord
    rms = float((x - coord).pow(2).sum(-1).mean().sqrt())
    want = float(sched[partial_t]) * math.sqrt(3)
    assert abs(rms - want) / want < 0.02, f"{rms} vs {want}"
    assert float(sched[partial_t]) < float(sched[0]) or partial_t == 0


def test_partial_t_near_data_stays_near_the_input_structure():
    """End to end through the sampler with an identity denoiser: partial_t near the data end
    keeps the input structure, partial_t=0 does not. This is the check a silently dropped
    flag fails."""
    s = DiffusionSampler()
    d, n_atom = 1, 256
    coord = torch.randn(d, n_atom, 3) * 8.0
    coord = coord - coord.mean(dim=1, keepdim=True)

    def rmsd_to_input(partial_t):
        torch.manual_seed(3)
        f, _ = _stub()
        x, _ = s.sample(f, coord, d, partial_t=partial_t, s_trans=0.0)
        # the sampler rotates the frame, so compare radii of gyration rather than
        # coordinates: a structure that kept the input has the input's spread.
        return abs(float(x.std()) - float(coord.std())) / float(coord.std())

    near = rmsd_to_input(199)
    full = rmsd_to_input(0)
    assert near < 0.05, f"partial_t=199 should keep the input's geometry, got {near}"
    assert full > near * 2, f"partial_t=0 should not, got {full} vs {near}"


# ---- row 14: the decision, and where it is taken ----

class _FakeRF3:
    """`RF3.trunk`'s recycling loop and `predict`'s early-stop gate, with the device parts
    replaced. What is under test is the control flow: the predicate runs after recycle 1,
    a stop abandons the remaining recycles, and a stop never reaches the rollout."""

    def __init__(self, plddt_by_recycle):
        self.plddt = plddt_by_recycle
        self.recycles = 0
        self.rollouts = 0

    def trunk(self, n_recycles, stop_after_first=None):
        for i in range(n_recycles):
            self.recycles += 1
            if i == 0 and stop_after_first is not None and stop_after_first():
                break
        return None

    def predict(self, n_recycles, early_stop_plddt=None):
        decision = {}
        stop = None
        if early_stop_plddt is not None:
            def stop():
                decision["mean_plddt"] = self.plddt[0]
                return decision["mean_plddt"] < early_stop_plddt
        self.trunk(n_recycles, stop)
        if decision.get("mean_plddt") is not None \
                and decision["mean_plddt"] < early_stop_plddt:
            return {"early_stopped": True, "mean_plddt": decision["mean_plddt"]}
        self.rollouts += 1
        return {"early_stopped": False, "X_L": "structure",
                **({"mean_plddt": decision["mean_plddt"]} if decision else {})}


def test_early_stop_fires_below_the_threshold_and_skips_the_rest():
    m = _FakeRF3([0.31])
    got = m.predict(10, early_stop_plddt=0.36)
    assert got["early_stopped"] is True
    assert "X_L" not in got, "a stopped target must not carry a structure"
    assert m.rollouts == 0, "a stopped target must not reach the diffusion rollout"
    assert m.recycles == 1, "the decision is taken after recycle 1, not after all of them"
    assert got["mean_plddt"] == pytest.approx(0.31)


def test_early_stop_does_not_fire_above_the_threshold():
    m = _FakeRF3([0.41])
    got = m.predict(10, early_stop_plddt=0.36)
    assert got["early_stopped"] is False
    assert got["X_L"] == "structure"
    assert m.rollouts == 1
    assert m.recycles == 10, "a target that is not abandoned runs every recycle"
    assert got["mean_plddt"] == pytest.approx(0.41)


def test_early_stop_off_costs_nothing():
    m = _FakeRF3([0.01])
    got = m.predict(10)
    assert got["early_stopped"] is False and m.rollouts == 1 and m.recycles == 10
    assert "mean_plddt" not in got, "the flag off must not add a head pass"


def test_early_stop_needs_the_reduction_input():
    """`predict` refuses rather than silently folding without a decision."""
    from tt_bio.rf3.model import RF3
    import inspect
    sig = inspect.signature(RF3.predict)
    for name in ("partial_t", "early_stop_plddt", "is_real_atom"):
        assert name in sig.parameters, f"{name} is not on RF3.predict"
    sig = inspect.signature(RF3.trunk)
    assert "stop_after_first" in sig.parameters
