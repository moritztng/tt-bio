"""Metric gradient guidance, against the one differentiable metric upstream ships.

`contract_ca.sq_norm` is the summed squared CA norm, and upstream's shipped config gives it a
negative weight over the first 10 % of the trajectory, which contracts the chain towards the
origin. Guidance maximizes the weighted objective, so the sign of the weight is the whole
behaviour, and the test that matters is that a negative weight makes the designs smaller and a
positive one makes them bigger, on the same seed.
"""
import torch

from tt_bio.rfd4 import guidance as G
from tt_bio.rfd4 import sampler as S
from tests.test_rfd4_sampler import Net, groups, masks, short


def run(spec, *, guidance=None, n_designs=3, n_tokens=48, seed=4):
    return S.generate(spec, Net(), n_designs=n_designs, n_tokens=n_tokens, seed=seed,
                      emit_masks=masks(n_designs, n_tokens),
                      groups=groups(n_designs, n_tokens), guidance=guidance)


def shipped_terms(w=-0.1):
    """`configs/guidance/contract_ca.yaml`: a window over t in [0, 0.1] on the backbone."""
    return {"contract_ca.sq_norm": {
        S.BB_CA: lambda t: G.window(t, w=w, t_start=0.0, t_end=0.1)}}


def test_the_shipped_schedules_are_upstreams():
    assert G.constant(0.4, w=-0.1) == G.constant(0.9, w=-0.1) == -0.1
    assert G.window(0.05, w=-0.1, t_start=0.0, t_end=0.1) == -0.1
    assert G.window(0.5, w=-0.1, t_start=0.0, t_end=0.1) == 0.0


def test_the_gradient_of_the_summed_squared_norm_is_twice_the_coordinates():
    fn = G.gradient_guidance({"contract_ca.sq_norm": {S.BB_CA: lambda t: G.constant(t, w=1.0)}})
    x = torch.randn(2, 5, 3)
    delta = fn(x_1_pred={S.BB_CA: x}, t={S.BB_CA: 0.5}, step=0)
    assert torch.allclose(delta[S.BB_CA], 2 * x)


def test_a_term_outside_its_window_contributes_nothing():
    fn = G.gradient_guidance(shipped_terms())
    assert fn(x_1_pred={S.BB_CA: torch.randn(2, 5, 3)}, t={S.BB_CA: 0.5}, step=30) == {}


def test_a_modality_no_term_names_is_untouched():
    fn = G.gradient_guidance(shipped_terms())
    x = {S.BB_CA: torch.randn(2, 5, 3), S.LOCAL_LATENTS: torch.randn(2, 5, 32)}
    delta = fn(x_1_pred=x, t={S.BB_CA: 0.05, S.LOCAL_LATENTS: 0.05}, step=0)
    assert set(delta) == {S.BB_CA}


def test_an_emit_mask_keeps_guidance_off_the_tokens_the_flow_does_not_move():
    emit = masks(2, 6)[S.BB_CA]
    fn = G.gradient_guidance({"contract_ca.sq_norm": {S.BB_CA: lambda t: G.constant(t, w=1.0)}},
                             emit_masks={S.BB_CA: emit})
    delta = fn(x_1_pred={S.BB_CA: torch.ones(2, 6, 3)}, t={S.BB_CA: 0.5}, step=0)
    assert torch.all(delta[S.BB_CA][~emit] == 0) and torch.all(delta[S.BB_CA][emit] == 2.0)


def test_an_unknown_metric_is_named_not_ignored():
    try:
        G.gradient_guidance({"not_a_metric.value": {S.BB_CA: lambda t: 1.0}})
    except ValueError as e:
        assert "not_a_metric.value" in str(e)
    else:
        raise AssertionError("an unknown guidance term has to be refused")


def _radius(out):
    return float(out.x[S.BB_CA].pow(2).sum(-1).sqrt().mean())


def test_a_negative_weight_contracts_the_design_and_a_positive_one_expands_it():
    """The same seed and the same network field, three runs: the sign of the weight is the only
    difference, so the ordering of the radii is the behaviour guidance is supposed to have."""
    spec = short(24)
    plain = _radius(run(spec))
    contracted = _radius(run(spec, guidance=G.gradient_guidance(shipped_terms(-0.1))))
    expanded = _radius(run(spec, guidance=G.gradient_guidance(shipped_terms(+0.1))))
    assert contracted < plain < expanded


def test_guidance_does_not_run_when_it_is_not_asked_for():
    """A run with a term whose schedule is zero everywhere is the unguided run, bit for bit."""
    spec = short(24)
    off = G.gradient_guidance({"contract_ca.sq_norm": {S.BB_CA: lambda t: G.constant(t, w=0.0)}})
    assert torch.equal(run(spec).x[S.BB_CA], run(spec, guidance=off).x[S.BB_CA])
