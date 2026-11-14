"""The custom-loss hook: it reaches both paths, it restores BindCraft 2, and it refuses a term
whose gradient cannot move a design.

The first test needs nothing but tt-bio. The rest need BindCraft 2 and jax and skip without them;
on pc that is `/home/moritz/bcx_hostcut_venv/bin/python3` with
`PYTHONPATH=<worktree>:/home/moritz/bcx_shipped/bc2`.

The calibration test is the important one. A screen that refuses a correct term is worse than no
screen, so `test_the_screen_refuses_none_of_bindcraft_2s_own_terms` runs all of them through it:
nine are inactive on a synthetic design for honest reasons (an empty flag mask, a hinge on its
flat side, a parameter nothing set) and that has to stay a warning rather than a refusal.
"""
import functools
import inspect
import re
import warnings

import pytest


def test_the_hook_is_reachable_from_the_one_module_a_user_imports():
    """A user imports `tt_bio.bindcraft2`; the loss surface has to be on it."""
    pytest.importorskip("torch", reason="tt_bio.bindcraft2 needs torch")
    from tt_bio import bindcraft2

    for name in ("loss_terms", "loss_term", "terms", "check_gradient", "synthetic_design",
                 "INTERMEDIATES", "DeadGradient", "LossTermError", "UnknownTerm"):
        assert hasattr(bindcraft2, name), name


@pytest.fixture(scope="module")
def loss_module():
    pytest.importorskip("jax", reason="jax is not on this interpreter")
    return pytest.importorskip("bindcraft.loss", reason="BindCraft 2 is not importable here")


@pytest.fixture(scope="module")
def design():
    from tt_bio import bindcraft2_loss

    return bindcraft2_loss.synthetic_design()


def test_the_documented_metrics_are_the_ones_the_losses_actually_read(loss_module):
    """`INTERMEDIATES` is a promise to a user, so it must not drift from BindCraft 2's heads."""
    from tt_bio import bindcraft2_loss

    read = set(re.findall(r"metrics\[['\"]([a-z_0-9]+)['\"]\]",
                          inspect.getsource(loss_module)))
    documented = {key for key, (source, *_) in bindcraft2_loss.INTERMEDIATES.items()
                  if source == "metrics"}
    assert read <= documented, f"undocumented metrics a term can read: {sorted(read - documented)}"
    assert documented - read == set(), f"documented but never read: {sorted(documented - read)}"


def test_an_added_term_lands_in_the_dict_both_paths_consume(loss_module, design):
    """`build_design_losses` is the single per-trajectory builder; the gradient path and
    `weighted_design_loss` both read what it returns."""
    from tt_bio import bindcraft2_loss

    def term(protein_states, predictions, prediction_state="complex"):
        return 1 - predictions[prediction_state].metrics["iptm"]

    settings = {"weights_plddt_loss": 0.1, "weights_iptm_loss": 0.05}
    baseline = loss_module.build_design_losses(settings, {"complex": 1.0}, 32, 0)
    assert "probe" not in baseline

    with bindcraft2_loss.loss_terms(add={"probe": (term, 0.7)},
                                   weight={"plddt_loss": 0.25}, design=design):
        inside = loss_module.build_design_losses(settings, {"complex": 1.0}, 32, 0)
    assert inside["probe"].weight == pytest.approx(0.7)
    assert inside["plddt_loss"].weight == pytest.approx(0.25)
    assert inside["probe"].function(*design) == pytest.approx(
        float(1 - design[1]["complex"].metrics["iptm"]), rel=1e-6)

    after = loss_module.build_design_losses(settings, {"complex": 1.0}, 32, 0)
    assert set(after) == set(baseline)
    assert {n: e.weight for n, e in after.items()} == {n: e.weight for n, e in baseline.items()}


def test_a_zero_weight_switches_a_term_off_and_the_settings_file_is_untouched(loss_module, design):
    from tt_bio import bindcraft2_loss

    settings = {"weights_plddt_loss": 0.1, "weights_iptm_loss": 0.05}
    frozen = dict(settings)
    with bindcraft2_loss.loss_terms(weight={"plddt_loss": 0.0}, design=design):
        inside = loss_module.build_design_losses(settings, {"complex": 1.0}, 32, 0)
    assert "plddt_loss" not in inside and "iptm_loss" in inside
    assert settings == frozen, "the hook edited the caller's settings dict"


def test_a_replacement_keeps_the_name_the_weight_and_the_target_weighting(loss_module, design):
    from tt_bio import bindcraft2_loss

    def softer(protein_states, predictions, prediction_state="binder_alone", chain="binder"):
        return predictions[prediction_state].metrics["plddt"].mean() ** 2

    original = loss_module.REGISTERED_LOSSES["plddt_loss"]
    weighting = loss_module.LOSS_TARGET_WEIGHTING["plddt_loss"]
    with bindcraft2_loss.loss_terms(replace={"plddt_loss": softer}, design=design):
        assert loss_module.REGISTERED_LOSSES["plddt_loss"] is softer
        assert loss_module.LOSS_TARGET_WEIGHTING["plddt_loss"] == weighting
        built = loss_module.build_design_losses({"weights_plddt_loss": 0.1},
                                                {"complex": 1.0}, 32, 0)
        assert built["plddt_loss"].weight == pytest.approx(0.1)
    assert loss_module.REGISTERED_LOSSES["plddt_loss"] is original


def test_the_registry_and_build_losses_are_restored_even_when_the_body_raises(loss_module, design):
    from tt_bio import bindcraft2_loss

    def term(protein_states, predictions, prediction_state="complex"):
        return 1 - predictions[prediction_state].metrics["ptm"]

    build = loss_module.build_losses
    registered = dict(loss_module.REGISTERED_LOSSES)
    with pytest.raises(RuntimeError, match="the body"):
        with bindcraft2_loss.loss_terms(add={"probe": (term, 1.0)}, design=design):
            raise RuntimeError("the body")
    assert loss_module.build_losses is build
    assert loss_module.REGISTERED_LOSSES == registered
    assert "probe" not in loss_module.LOSS_TARGET_WEIGHTING


def test_a_term_with_no_gradient_is_refused_by_name(loss_module, design):
    """The worst outcome this hook could have is a wrong gradient that runs."""
    import jax.numpy as jnp
    from tt_bio import bindcraft2_loss

    def argmax_term(protein_states, predictions, prediction_state="complex"):
        return jnp.argmax(predictions[prediction_state].metrics["distogram"], -1).mean() * 1.0

    def threshold_term(protein_states, predictions, prediction_state="complex"):
        return (predictions[prediction_state].metrics["pae"] > 10.0).mean() * 1.0

    for name, term in (("argmax", argmax_term), ("threshold", threshold_term)):
        with pytest.raises(bindcraft2_loss.DeadGradient, match="exactly zero"):
            with bindcraft2_loss.loss_terms(add={name: (term, 1.0)}, design=design):
                pass
        assert name not in loss_module.REGISTERED_LOSSES

    with bindcraft2_loss.loss_terms(add={"argmax": (argmax_term, 1.0)}, check=False,
                                    design=design):
        assert "argmax" in loss_module.REGISTERED_LOSSES, "check=False must still install it"


def test_a_nan_gradient_under_a_finite_value_is_refused(loss_module, design):
    import jax.numpy as jnp
    from tt_bio import bindcraft2_loss

    def sqrt_at_zero(protein_states, predictions, prediction_state="complex"):
        plddt = predictions[prediction_state].metrics["plddt"]
        return jnp.sqrt(jnp.sum((plddt - plddt) ** 2))

    with pytest.raises(bindcraft2_loss.DeadGradient, match="non-finite"):
        with bindcraft2_loss.loss_terms(add={"sqrt0": (sqrt_at_zero, 1.0)}, design=design):
            pass


def test_what_a_term_returns_is_checked_before_anything_differentiates_it(loss_module, design):
    import jax.numpy as jnp
    from tt_bio import bindcraft2_loss

    def vector(protein_states, predictions, prediction_state="complex"):
        return predictions[prediction_state].metrics["plddt"]

    def infinite(protein_states, predictions, prediction_state="complex"):
        return jnp.log(predictions[prediction_state].metrics["plddt"].min() * 0.0)

    with pytest.raises(bindcraft2_loss.LossTermError, match="must return a scalar"):
        with bindcraft2_loss.loss_terms(add={"vector": (vector, 1.0)}, design=design):
            pass
    with pytest.raises(bindcraft2_loss.LossTermError, match="non-finite term"):
        with bindcraft2_loss.loss_terms(add={"infinite": (infinite, 1.0)}, design=design):
            pass


@pytest.mark.parametrize("kwargs,error,message", [
    ({"weight": {"iptm_los": 1.0}}, "UnknownTerm", "did you mean"),
    ({"replace": {"plddt": lambda a, b: 0.0}}, "UnknownTerm", "did you mean"),
    ({"add": {"iptm_loss": lambda a, b: 0.0}}, "LossTermError", "already exists"),
])
def test_a_name_that_is_not_a_term_is_refused_with_the_near_misses(loss_module, kwargs, error,
                                                                   message):
    from tt_bio import bindcraft2_loss

    with pytest.raises(getattr(bindcraft2_loss, error), match=message):
        with bindcraft2_loss.loss_terms(**kwargs):
            pass


def test_a_signature_bindcraft_2_cannot_bind_is_refused(loss_module, design):
    from tt_bio import bindcraft2_loss

    def no_default(protein_states, predictions, cutoff):
        return predictions["complex"].metrics["ptm"] * cutoff

    def one_argument(protein_states):
        return 0.0

    def different_states(protein_states, predictions, chain="binder"):
        return 1 - predictions["complex"].metrics["ptm"]

    with pytest.raises(bindcraft2_loss.TermSignature, match="no default"):
        with bindcraft2_loss.loss_terms(add={"a": (no_default, 1.0)}, design=design):
            pass
    with pytest.raises(bindcraft2_loss.TermSignature, match="mandatory"):
        with bindcraft2_loss.loss_terms(add={"b": (one_argument, 1.0)}, design=design):
            pass
    # plddt_loss takes prediction_state, so a replacement without it would change the fan-out.
    with pytest.raises(bindcraft2_loss.TermSignature, match="prediction_state"):
        with bindcraft2_loss.loss_terms(replace={"plddt_loss": different_states}, design=design):
            pass


def test_the_screen_refuses_none_of_bindcraft_2s_own_terms(loss_module, design):
    """Calibration. If this ever refuses one, the screen is wrong, not the term."""
    from tt_bio import bindcraft2_loss

    refused, inactive = [], []
    for name, function in sorted(loss_module.REGISTERED_LOSSES.items()):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                bindcraft2_loss._screen(name, functools.partial(function), design=design)
            except bindcraft2_loss.LossTermError as refusal:
                refused.append((name, str(refusal)[:80]))
                continue
        if any("inactive" in str(w.message) for w in caught):
            inactive.append(name)
    assert refused == [], f"the screen refuses BindCraft 2's own terms: {refused}"
    assert len(loss_module.REGISTERED_LOSSES) - len(inactive) >= 25, (
        f"only {len(loss_module.REGISTERED_LOSSES) - len(inactive)} terms are active on the "
        "synthetic design; it has gone degenerate and the screen is no longer screening")


def test_a_float64_term_grades_to_float64_and_a_float32_one_does_not(loss_module):
    """The per-op bar is set by the term's own forward dtype, not by the inputs."""
    import jax
    import jax.numpy as jnp
    from tt_bio import bindcraft2_loss

    def plain(protein_states, predictions, prediction_state="complex"):
        metrics = predictions[prediction_state].metrics
        contact = jax.nn.softmax(metrics["distogram"], axis=-1)[..., :32].sum(-1)
        return (1 - metrics["plddt"].mean()) + 0.01 * metrics["pae"].mean() - contact.mean()

    def through_a_helper(protein_states, predictions, prediction_state="complex",
                         binder="binder"):
        coordinates, _ = loss_module.chain_atom_coordinates(
            predictions[prediction_state].protein_complex[binder])
        return jnp.sqrt(jnp.mean(jnp.square(
            loss_module.pairwise_atom_distances(coordinates, coordinates) - 8.0)) + 1e-8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows, worst, dtype = bindcraft2_loss.check_gradient(plain, probes=4)
        _, helper_worst, helper_dtype = bindcraft2_loss.check_gradient(through_a_helper, probes=4)

    assert dtype == jnp.float64 and worst < 1e-7, (dtype, worst)
    assert helper_dtype == jnp.float32 and helper_worst < 1e-3, (helper_dtype, helper_worst)
    graded = [row for row in rows if row.probes]
    assert graded, "nothing was probed; the analytic gradient came out zero everywhere"
    assert all(row.gradient > 0 for row in graded)
    # jax_enable_x64 is process-wide, so the grade has to put it back.
    assert not jax.config.jax_enable_x64
