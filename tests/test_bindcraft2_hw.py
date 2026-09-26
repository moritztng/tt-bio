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


def test_one_gradient_step_with_the_evoformer_on_card(capsys):
    bindcraft = pytest.importorskip("bindcraft", reason="BindCraft 2 is not on sys.path")
    from tt_bio import weights
    params = weights.resolve("af2-params")
    if params is None or not (pathlib.Path(params) / "params_model_1_ptm.npz").exists():
        pytest.skip("no AlphaFold 2 parameters; run `tt-bio weights --download af2ig`")
    settings_file = pathlib.Path(bindcraft.__file__).resolve().parents[1] / "examples/pdl1.json"
    if not settings_file.exists():
        pytest.skip(f"BindCraft 2 has no {settings_file}")

    protein_states, losses = _design_state(settings_file)

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
    bindcraft = pytest.importorskip("bindcraft", reason="BindCraft 2 is not on sys.path")
    from tt_bio import weights
    params = weights.resolve("af2-params")
    if params is None or not (pathlib.Path(params) / "params_model_1_ptm.npz").exists():
        pytest.skip("no AlphaFold 2 parameters; run `tt-bio weights --download af2ig`")
    settings_file = pathlib.Path(bindcraft.__file__).resolve().parents[1] / "examples/pdl1.json"
    if not settings_file.exists():
        pytest.skip(f"BindCraft 2 has no {settings_file}")

    protein_states, losses = _design_state(settings_file)

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
