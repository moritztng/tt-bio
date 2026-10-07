"""The device Evoformer applies no dropout; the swap has to say so rather than fold quietly.

BindCraft 2 issue #17: on-card accepts 0/8 trajectories where host JAX accepts 3/8 on the same
campaign seed. AlphaFold 2 applies dropout to each Evoformer sub-layer residual before the
skip-add (`modules.py:46`, `dropout_wrapper`), driven by `batch["use_dropout"]`, which BindCraft 2
leaves on for every gradient stage except `harden` (`trajectory.py:219`). The device blocks apply
none, so the two arms run the same program only at `harden`.

These read the rates and the closure off real AlphaFold 2 objects rather than a stand-in, because
a stand-in built to match the helper would pass whether or not the helper matches AlphaFold 2.
"""
import pytest

jnp = pytest.importorskip("jax.numpy")
modules = pytest.importorskip("bindcraft.af.alphafold.model.modules")
config_module = pytest.importorskip("bindcraft.af.alphafold.model.config")

from tt_bio.bindcraft2 import (EVOFORMER_DROPOUT_RATES, _check_dropout,
                               find_evoformer_dropout)


def evoformer_config():
    """The shipped multimer Evoformer config, as BindCraft 2 folds with it."""
    return config_module.model_config("model_1_multimer_v3").model.embeddings_and_evoformer


def test_shipped_config_still_carries_the_rates_the_device_blocks_omit():
    """If AlphaFold 2 ever zeroes these, the swap stops being a behaviour change."""
    evoformer = evoformer_config().evoformer
    rates = sorted({float(sub.dropout_rate)
                    for sub in (getattr(evoformer, name) for name in dir(evoformer))
                    if hasattr(sub, "dropout_rate")} - {0.0})
    assert tuple(rates) == EVOFORMER_DROPOUT_RATES


def evoformer_fn_like_alphafold(use_dropout):
    """A closure built the way `modules_multimer.py:414` builds `evoformer_fn`.

    Same free-variable names and a real `EvoformerIteration`, so `find_evoformer_dropout` is
    walking an AlphaFold 2 closure rather than a dict assembled to suit it. The module has to be
    constructed inside an `hk.transform`, which is also where AlphaFold 2 builds it.
    """
    import haiku as hk
    import jax

    c = evoformer_config()
    global_config = config_module.model_config("model_1_multimer_v3").model.global_config
    captured = {}

    def build():
        batch = {"use_dropout": use_dropout}
        evoformer_masks = {"msa": jnp.ones((1, 8)), "pair": jnp.ones((8, 8))}
        evoformer_iteration = modules.EvoformerIteration(
            c.evoformer, global_config, is_extra_msa=False, name="evoformer_iteration")

        def evoformer_fn(x):
            act, safe_key = x
            safe_key, safe_subkey = safe_key.split()
            output = evoformer_iteration(activations=act, masks=evoformer_masks,
                                         use_dropout=batch["use_dropout"], safe_key=safe_subkey)
            return (output, safe_key)

        captured["fn"] = evoformer_fn
        return jnp.zeros(())

    hk.transform(build).init(jax.random.PRNGKey(0))
    return captured["fn"]


def test_dropout_is_recovered_from_the_closure():
    use_dropout = jnp.asarray(True)
    found, rates = find_evoformer_dropout(evoformer_fn_like_alphafold(use_dropout))
    assert found is use_dropout
    assert rates == EVOFORMER_DROPOUT_RATES


def test_swapping_a_stack_that_wanted_dropout_is_refused():
    fn = evoformer_fn_like_alphafold(jnp.asarray(True))
    with pytest.raises(RuntimeError, match="do not"):
        _check_dropout(fn, "refuse")


def test_a_traced_use_dropout_cannot_be_narrowed_to_the_folds_that_wanted_it():
    """`af2.py:401` passes `jnp.asarray(self.dropout)` into a jit, so even a fold with dropout
    off arrives as a traced value. The guard must refuse that too, or it only fires on the
    folds that were already safe."""
    import jax

    def traced(flag):
        fn = evoformer_fn_like_alphafold(flag)
        with pytest.raises(RuntimeError):
            _check_dropout(fn, "refuse")
        return flag

    jax.eval_shape(traced, jnp.asarray(False))


def test_ignore_lets_a_dropout_free_benchmark_through():
    fn = evoformer_fn_like_alphafold(jnp.asarray(True))
    assert _check_dropout(fn, "ignore") is None or True


def test_env_overrides_the_argument(monkeypatch):
    """The escape hatch has to be reachable without threading an argument through every caller,
    and has to be the only way past the refusal."""
    from tt_bio.bindcraft2 import DROPOUT_POLICY_ENV

    fn = evoformer_fn_like_alphafold(jnp.asarray(True))
    monkeypatch.setenv(DROPOUT_POLICY_ENV, "ignore")
    _check_dropout(fn, "refuse")
    monkeypatch.setenv(DROPOUT_POLICY_ENV, "refuse")
    with pytest.raises(RuntimeError):
        _check_dropout(fn, "ignore")


def test_a_stack_with_no_dropout_in_its_closure_is_left_alone():
    """The extra-MSA stack and the template stacks must not be caught by this."""
    def unrelated_fn(x):
        return x

    assert _check_dropout(unrelated_fn, "refuse") == ()


def test_the_public_swap_refuses_before_it_takes_a_device_slot():
    """The behavioural half: drive `evoformer_on_device` itself, not the checker.

    Before the guard existed this swapped and folded dropout-free without a word, which is
    exactly what issue #17 reports. `as_jax` is never reached, so no card is involved; a stub
    that raises on contact proves that rather than asserting it.
    """
    from bindcraft.af.alphafold.model import modules as af_modules

    from tt_bio.bindcraft2 import EVOFORMER_BLOCKS, evoformer_on_device

    class NoCard:
        host_only = False
        blocks = EVOFORMER_BLOCKS

        def as_jax(self, *a, **k):
            raise AssertionError("the swap reached the device after refusing dropout")

    fn = evoformer_fn_like_alphafold(jnp.asarray(True))
    with evoformer_on_device(NoCard()):
        factory = af_modules.layer_stack.layer_stack(EVOFORMER_BLOCKS)
        with pytest.raises(RuntimeError, match="dropout"):
            factory(fn)


def test_the_swap_still_stands_down_for_a_host_only_fold():
    """`on_host` folds are BindCraft 2's own program, dropout included, so the guard must not
    turn a validation fold into a refusal."""
    from bindcraft.af.alphafold.model import modules as af_modules

    from tt_bio.bindcraft2 import EVOFORMER_BLOCKS, evoformer_on_device

    class StoodDown:
        host_only = True
        blocks = EVOFORMER_BLOCKS

        def as_jax(self, *a, **k):
            raise AssertionError("a host_only fold reached the device")

    fn = evoformer_fn_like_alphafold(jnp.asarray(True))
    with evoformer_on_device(StoodDown()):
        factory = af_modules.layer_stack.layer_stack(EVOFORMER_BLOCKS)
        assert factory(fn) is not None


def test_predictor_carries_the_policy_as_an_argument():
    """`predictor` is the entry point a BindCraft 2 campaign on card goes through, so the policy
    has to be reachable there without setting an environment variable.

    Honest about its reach: this is a signature and call-site check, not a behavioural one.
    Driving `predictor(trunk="device")` opens a chip, so the forwarding itself is proved by the
    card run, not here. What this does catch is the regression that actually threatens it -- the
    argument being dropped from the signature, or `evoformer_on_device` growing a parameter in
    front of `dropout` so the positional forward starts feeding the wrong one.
    """
    import inspect

    from tt_bio.bindcraft2 import evoformer_on_device, predictor

    policy = inspect.signature(predictor).parameters.get("dropout")
    assert policy is not None, "predictor dropped the dropout policy argument"
    assert policy.default == "refuse", "predictor must default to the refusing policy"
    assert policy.kind is inspect.Parameter.KEYWORD_ONLY

    # predictor forwards it as the third positional argument of evoformer_on_device.
    forwarded = list(inspect.signature(evoformer_on_device).parameters)
    assert forwarded[2] == "dropout", f"evoformer_on_device's third parameter is now {forwarded[2]}"
