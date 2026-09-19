"""The starting weights, and the two defects that let a leg train nothing for 1,388 steps.

The first `base-loss` leg ran on an all-zero model. Every gate the campaign had stayed green,
because each of them compares a device forward against the reference built the same zero way,
and a zero model agrees with a zero model to every bit. What these tests read instead is the
property no forward comparison can see: that the starting weights are not degenerate, and that
every parameter the model holds is one the optimizer can reach.
"""

from __future__ import annotations

import math

import pytest
import torch

from tt_bio.abodybuilder3_reference import ABB3Config, ABB3StructureModule
from tt_bio.af2_reference import Linear
from tt_bio.train.abb3_init import (SOFTPLUS_INVERSE_1, TRUNC_NORMAL_STD_2SIGMA, initialise_)


@pytest.fixture(scope="module")
def initialised():
    return initialise_(ABB3StructureModule(ABB3Config(use_plddt=False)), 0)


def test_construction_alone_gives_an_all_zero_model():
    """The defect itself, pinned as behaviour rather than described in a comment.

    `Linear` allocating zeros is correct and deliberate: every inference path loads a state
    dict over it. What was wrong was `initial_state_dict` returning that allocation. If this
    ever stops being true the docstrings in `abb3_init` and `af2_reference.Linear` are stale.
    """
    raw = ABB3StructureModule(ABB3Config(use_plddt=False)).state_dict()
    nonzero = [k for k, v in raw.items() if torch.any(v != 0)]
    assert all(k.endswith("layer_norm.weight") or "layer_norm" in k for k in nonzero), nonzero
    assert not torch.any(raw["ipa_layers.0.linear_q.weight"] != 0)


def test_every_parameter_is_initialised(initialised):
    """`initialise_` refuses to return with a parameter left at its NaN fill."""
    for name, p in initialised.named_parameters():
        assert not torch.isnan(p).any(), name


def test_an_uninitialised_parameter_is_named_rather_than_shipped():
    """The negative control for the coverage check: a parameter nothing initialises must fail.

    This is the shape of the original defect -- a tensor that nobody draws a value for -- and
    the check has to break on it, not on something adjacent.
    """
    model = ABB3StructureModule(ABB3Config(use_plddt=False))
    model.register_parameter("unloved", torch.nn.Parameter(torch.zeros(4)))
    with pytest.raises(RuntimeError, match="never initialised"):
        initialise_(model, 0)


def test_no_weight_matrix_starts_at_zero_unless_upstream_says_final(initialised):
    """The property whose absence cost the leg: a zero weight passes no gradient backwards.

    `final` and `gating` sites are zero on purpose, upstream included, and they are exempt by
    the name recorded at their own call site rather than by a list kept here.
    """
    zero_but_should_not_be = []
    for name, module in initialised.named_modules():
        if not isinstance(module, Linear):
            continue
        if module.init in ("final", "gating"):
            assert not torch.any(module.weight != 0), f"{name} is {module.init}, expected zeros"
            continue
        if not torch.any(module.weight != 0):
            zero_but_should_not_be.append(name)
    assert not zero_but_should_not_be


def test_fan_in_scaling_matches_upstreams_formula(initialised):
    """`trunc_normal_init_`: std is sqrt(scale / fan_in) / truncnorm.std(-2, 2).

    Read off the drawn weights rather than off the code that drew them, at 3 % tolerance, which
    is what a fan-in of 23 supports on 2,944 samples.
    """
    for name, scale in [("linear_in_node", 1.0), ("ipa_layers.0.linear_q", 1.0),
                        ("transition_layers.0.layers.0.linear_1", 2.0),
                        ("angle_resnet_layers.0.layers.0.linear_2", 2.0)]:
        w = dict(initialised.named_modules())[name].weight
        want = math.sqrt(scale / w.shape[-1]) / TRUNC_NORMAL_STD_2SIGMA
        got = float(w.detach().float().std())
        # The draw is truncated at +-2 sigma, so the realised std is `want * truncnorm.std`.
        assert abs(got - want * TRUNC_NORMAL_STD_2SIGMA) < 0.03 * want, (name, got, want)


def test_the_truncation_constant_is_scipys():
    """`TRUNC_NORMAL_STD_2SIGMA` is a number in our tree and a formula in upstream's."""
    truncnorm = pytest.importorskip("scipy.stats").truncnorm
    assert abs(truncnorm.std(a=-2, b=2) - TRUNC_NORMAL_STD_2SIGMA) < 1e-12


def test_ipa_head_weights_start_at_softplus_inverse_one(initialised):
    """Upstream `ipa_point_weights_init_`, not the zero the allocation leaves behind."""
    for layer in initialised.ipa_layers:
        assert torch.allclose(layer.head_weights,
                              torch.full_like(layer.head_weights, SOFTPLUS_INVERSE_1))
    assert abs(math.log1p(math.exp(SOFTPLUS_INVERSE_1)) - 1.0) < 1e-12


def test_the_seed_is_the_whole_of_the_randomness():
    """Both data-parallel ranks build the model from the seed instead of broadcasting rank 0's.

    Before `initialise_` this passed for the wrong reason: nothing was random, so every seed
    gave the same all-zero model. So the test has to assert BOTH directions.
    """
    a = initialise_(ABB3StructureModule(ABB3Config(use_plddt=False)), 7).state_dict()
    b = initialise_(ABB3StructureModule(ABB3Config(use_plddt=False)), 7).state_dict()
    c = initialise_(ABB3StructureModule(ABB3Config(use_plddt=False)), 8).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_the_released_parameter_count_is_unchanged(initialised):
    """7,111,515 for `base-loss`, which is the count the config's docstring pins."""
    assert sum(p.numel() for p in initialised.parameters()) == 7_111_515


def test_repro_initial_state_dict_draws_the_weights():
    """The call site that was wrong. It is the one the run uses, so it is the one under test."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_repro_under_test", root / "scripts" / "abb3_port" / "repro.py")
    repro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repro)
    sd = repro.initial_state_dict(ABB3Config(use_plddt=False), 0)
    assert torch.any(sd["ipa_layers.0.linear_q.weight"] != 0)
    assert torch.any(sd["linear_in_node.weight"] != 0)
