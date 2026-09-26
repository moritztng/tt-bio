"""One BindCraft 2 gradient step through `tt_bio.bindcraft2` on a real card.

The card-free half of the packaging question is in `test_bindcraft2.py`. This is the other half:
the same entry point with `trunk="device"` differentiates AlphaFold 2 through a sequence with the
48 Evoformer blocks on a Tenstorrent chip, and the splice's own call counters say the card ran
them rather than JAX.

Run it pinned, which the session guard in `conftest.py` enforces:

    TT_VISIBLE_DEVICES=2 PYTHONPATH=/path/to/BindCraft2 python3 -m pytest \\
        tests/test_bindcraft2_hw.py -s
"""
import pathlib
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from perf import clocksample  # noqa: E402
from tt_bio import bindcraft2  # noqa: E402

pytestmark = pytest.mark.device


def _design_state(settings_file):
    import jax
    from bindcraft.preflight import cleaned_campaign_settings
    from bindcraft.settings import build_design_settings, parse_setting_overrides, read_settings
    from bindcraft.trajectory import initialize_design_trajectory

    settings = cleaned_campaign_settings(read_settings(
        str(settings_file), parse_setting_overrides(["binder_lengths=[60]", "campaign_seed=0"])))
    design_settings = build_design_settings(settings)
    protein_states, _, losses = initialize_design_trajectory(
        design_settings, jax.random.PRNGKey(design_settings.seed))
    return protein_states, losses


def _pdl1_draw():
    """BindCraft 2's shipped PD-L1 draw, or a skip naming what is missing."""
    bindcraft = pytest.importorskip("bindcraft", reason="BindCraft 2 is not on sys.path")
    from tt_bio import weights
    params = weights.resolve("af2-params")
    if params is None or not (pathlib.Path(params) / "params_model_1_ptm.npz").exists():
        pytest.skip("no AlphaFold 2 parameters; run `tt-bio weights --download af2ig`")
    settings_file = pathlib.Path(bindcraft.__file__).resolve().parents[1] / "examples/pdl1.json"
    if not settings_file.exists():
        pytest.skip(f"BindCraft 2 has no {settings_file}")
    protein_states, losses = _design_state(settings_file)
    return params, protein_states, losses


def _exact_counters():
    """Every live exact counter, flattened and prefixed, as one snapshot."""
    from tt_bio import autograd
    return {f"sm_{k}": v for k, v in autograd.EXACT_SOFTMAX_STATS.items()} | \
           {f"ln_{k}": v for k, v in autograd.EXACT_LAYER_NORM_STATS.items()}


def _moved(before):
    now = _exact_counters()
    return {k: now[k] - before[k] for k in now}


def test_one_gradient_step_with_the_evoformer_on_card(capsys):
    params, protein_states, losses = _pdl1_draw()

    # A number without a clock is not a measurement, and on Blackhole the AI clock sets the step
    # time. Sampled on a thread DURING the step, never read once before it.
    with clocksample.during(period=1.0) as clock:
        with bindcraft2.predictor(trunk="device", checkpoints=params) as build:
            model = build(presets="model_1_ptm", data_dir=str(params), max_cache_size=1,
                          num_recycle=1, length_bucket_size=32)
            started = time.time()
            predictions, gradients, loss = model.sequence_gradients(protein_states, losses)
            elapsed = time.time() - started
            calls = dict(build.evoformer.calls)
            live = build.evoformer.live_tapes()

    assert calls["taped"] >= 1 and calls["backward"] >= 1, calls
    assert live == 0, f"{live} tapes still on card after the backward"
    gradient = np.asarray(next(iter(sorted(gradients.items())))[1])
    assert np.isfinite(gradient).all()
    assert (gradient != 0).any()
    assert np.isfinite(float(loss))

    tokens = sum(len(p) for p in next(iter(predictions.values())).protein_complex.values())
    with capsys.disabled():
        print(f"\ntokens {tokens} padded to {bindcraft2._pad32(tokens)}, calls {calls}, "
              f"step {elapsed:.1f} s (first call: weight load and JAX compile included), "
              f"loss {float(loss):.4f}, gradient {tuple(gradient.shape)}")
        print(clock.line())
        print(f"host load/core during the step: "
              f"{sum(clock.load) / max(len(clock.load), 1):.2f}")


def test_the_extra_msa_swap_completes_a_backward_on_card(capsys):
    """`predictor(extra_msa=True)` through a full gradient step, which is where it used to die.

    The swap landed reachable and broken: `_Trunk.extra_msa` hoisted the outer-product constant
    out of the loop and let the checkpoint closure capture it, `_residual` freed it in the
    forward, and `autograd._recompute` re-entered the same closure in the backward against a dead
    buffer. `ttnn.typecast` raised TT_THROW "Buffer is not allocated" on every backward, on the
    default `recompute=True` path.

    Nothing card-free saw it, because the forward-only `_primal` arm uses each constant once.
    Nor did the on-card forward. Only a real backward does, which is why this test drives the
    whole step rather than the stack.
    """
    params, protein_states, losses = _pdl1_draw()

    with clocksample.during(period=1.0) as clock:
        with bindcraft2.predictor(trunk="device", checkpoints=params, extra_msa=True) as build:
            model = build(presets="model_1_ptm", data_dir=str(params), max_cache_size=1,
                          num_recycle=1, length_bucket_size=32)
            started = time.time()
            predictions, gradients, loss = model.sequence_gradients(protein_states, losses)
            elapsed = time.time() - started
            calls = dict(build.extra_msa.calls)
            swapped, live = list(build.extra_msa.swapped), build.extra_msa.live_tapes()

    assert swapped == [4], f"the extra-MSA stack was not swapped out: {swapped}"
    assert calls["taped"] >= 1, calls
    assert calls["backward"] >= 1, f"the card never ran the extra-MSA backward: {calls}"
    assert live == 0, f"{live} extra-MSA tapes still on card after the backward"
    gradient = np.asarray(next(iter(sorted(gradients.items())))[1])
    assert np.isfinite(gradient).all()
    assert (gradient != 0).any()
    assert np.isfinite(float(loss))

    tokens = sum(len(p) for p in next(iter(predictions.values())).protein_complex.values())
    with capsys.disabled():
        print(f"\ntokens {tokens} padded to {bindcraft2._pad32(tokens)}, "
              f"extra_msa calls {calls}, step {elapsed:.1f} s "
              f"(first call: weight load and JAX compile included), loss {float(loss):.4f}")
        print(clock.line())
        print(f"host load/core during the step: "
              f"{sum(clock.load) / max(len(clock.load), 1):.2f}")


def test_the_exact_instrument_reaches_a_real_backward_on_card(capsys):
    """`predictor(exact=True)` drives softmax and layer norm through host float64, and the
    default does not.

    `exact` ships off, so `exact=True` is a lever with no user traffic. The card-free tests in
    `test_bindcraft2.py` assert on `exact_training_ops()` and `build.exact`, which is the switch
    being SET. `extra_msa=True` passed exactly that class of test for two weeks while raising
    `TT_THROW "Buffer is not allocated"` on every backward, because nothing walked the backward.

    So this grades both arms on the live counters around a whole `sequence_gradients` instead.
    `EXACT_LAYER_NORM_STATS["bw"]` only moves inside a backward, so it is the assertion that a
    forward-only test cannot fake. The default arm has to read exactly zero on every counter in
    the same process, or the two arms have not been shown to separate.

    The instrument is expensive by construction: expect the exact arm to take roughly 25x the
    default arm's gradient call (479.59 s against 19.285 s at n=192 on qb2 card 1 at AICLK 1350,
    `perf/bcx_exact/ROUND_AB.json`). The walls are printed rather than asserted -- one round per
    arm settles reachability, not cost.
    """
    from tt_bio import autograd

    params, protein_states, losses = _pdl1_draw()
    # One pool across both arms: the weights load on first use and the pool outlives the
    # predictor scope, so the exact arm does not pay a second 910 MB load.
    pool = bindcraft2.TrunkPool(str(params), resident=1)
    arms = {}

    for name, kwargs in (("default", {}), ("exact", {"exact": True})):
        before = _exact_counters()
        with clocksample.during(period=1.0) as clock:
            with bindcraft2.predictor(trunk="device", checkpoints=pool, **kwargs) as build:
                armed = list(autograd.exact_training_ops())
                model = build(presets="model_1_ptm", data_dir=str(params), max_cache_size=1,
                              num_recycle=1, length_bucket_size=32)
                started = time.time()
                predictions, gradients, loss = model.sequence_gradients(protein_states, losses)
                elapsed = time.time() - started
                calls, live = dict(build.evoformer.calls), build.evoformer.live_tapes()
        arms[name] = {
            "moved": _moved(before), "armed": armed, "exact": build.exact, "calls": calls,
            "live": live, "seconds": elapsed, "loss": float(loss),
            "gradient": np.asarray(next(iter(sorted(gradients.items())))[1]),
            "tokens": sum(len(p) for p in
                          next(iter(predictions.values())).protein_complex.values()),
            "clock": clock.line(), "load": sum(clock.load) / max(len(clock.load), 1)}

    # Both arms ran a real step: the card taped a forward and ran a backward, the tapes came
    # back off the card, and the gradient that came out is usable.
    for name, arm in arms.items():
        assert arm["calls"]["taped"] >= 1 and arm["calls"]["backward"] >= 1, (name, arm["calls"])
        assert arm["live"] == 0, f"{name}: {arm['live']} tapes still on card"
        assert np.isfinite(arm["gradient"]).all(), name
        assert (arm["gradient"] != 0).any(), f"{name}: the gradient is all zero"
        assert np.isfinite(arm["loss"]), name

    off = arms["default"]
    assert off["exact"] is False, "predictor() no longer defaults the exact instrument off"
    assert off["armed"] == [], off["armed"]
    ran = {k: v for k, v in off["moved"].items() if v}
    assert not ran, f"the default arm ran the host float64 instrument: {ran}"

    on = arms["exact"]
    assert on["exact"] is True
    assert on["armed"] == list(autograd.EXACT_TRAINING_OPS), on["armed"]
    assert on["moved"]["sm_verb"] > 0, on["moved"]
    assert on["moved"]["ln_verb"] > 0, on["moved"]
    assert on["moved"]["ln_bw"] > 0, f"exact=True never reached the backward: {on['moved']}"
    assert on["moved"]["ln_elements"] > 0, on["moved"]

    with capsys.disabled():
        print(f"\ntokens {on['tokens']} padded to {bindcraft2._pad32(on['tokens'])}")
        for name, arm in arms.items():
            print(f"  {name:<7} exact={arm['exact']!s:<5} step {arm['seconds']:.1f} s "
                  f"loss {arm['loss']:.4f} counters {arm['moved']}")
            print(f"          {arm['clock']}  host load/core {arm['load']:.2f}")
