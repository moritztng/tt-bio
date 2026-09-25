"""`tt_bio.bindcraft2`: BindCraft 2's design loop against tt-bio's AlphaFold 2 trunk.

The device leg lives in `test_bindcraft2_hw.py`. Everything here runs without a card, and the
gradient-step test runs BindCraft 2's own trunk (`trunk="jax"`), which is the control arm a
device result is read against.
"""
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch

from tt_bio import bindcraft2

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _child_env():
    """The environment for a fresh process, with this repo's `perf/` off the path.

    The packaging question this module exists to answer is whether a user can reach the backend
    from `tt_bio` alone, so the measurement harness must not be importable in the child.
    """
    env = dict(os.environ)
    entries = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
               if p and "perf" not in pathlib.Path(p).parts]
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *entries])
    return env


def _run_child(source: str):
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], cwd=ROOT,
                         env=_child_env(), capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def test_importing_the_backend_pulls_in_neither_ttnn_nor_jax():
    """`bindcraft2.pin_card` can only work before ttnn is imported, so importing the module must
    not import it. jax is BindCraft 2's dependency, not tt-bio's, and stays out too."""
    printed = _run_child("""
        import sys
        from tt_bio import bindcraft2
        print("ttnn" in sys.modules, "jax" in sys.modules)
    """)
    assert printed.split() == ["False", "False"]


def test_pinning_a_card_after_ttnn_is_imported_raises():
    printed = _run_child("""
        import sys, types
        sys.modules["ttnn"] = types.ModuleType("ttnn")
        from tt_bio import bindcraft2
        try:
            bindcraft2.pin_card(3)
        except RuntimeError as exc:
            print("raised", "TT_VISIBLE_DEVICES=3" in str(exc))
    """)
    assert printed.split() == ["raised", "True"]


def test_a_missing_checkpoint_is_named(tmp_path):
    pool = bindcraft2.TrunkPool(tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        pool.require(["model_1_ptm"])
    assert str(tmp_path / "params_model_1_ptm.npz") in str(caught.value)


def test_a_model_the_pool_was_never_given_is_refused(tmp_path):
    present = tmp_path / "params_model_1_ptm.npz"
    present.touch()
    pool = bindcraft2.TrunkPool({"model_1_ptm": present})
    assert pool.names == ("model_1_ptm",)
    with pytest.raises(KeyError):
        pool.use("model_3_multimer_v3")
    with pytest.raises(KeyError):
        pool.require(["model_3_multimer_v3"])


def test_the_mask_cache_separates_two_binder_lengths_in_one_bucket():
    """Two trajectories of different length land on the same padded size, and the masks that
    separate them differ only in their values. A shape-keyed cache folds the second one's padding
    as real residues."""
    key = bindcraft2.EvoformerOnDevice._key
    short = torch.zeros(1, 224)
    short[:, :200] = 1.0
    longer = torch.zeros(1, 224)
    longer[:, :211] = 1.0
    assert short.shape == longer.shape
    assert key(short) != key(longer)
    assert key(short) == key(short.clone())


# ----------------------------------------------------------------- with BindCraft 2 installed


def _bindcraft_root():
    bindcraft = pytest.importorskip("bindcraft", reason="BindCraft 2 is not on sys.path")
    return pathlib.Path(bindcraft.__file__).resolve().parents[1]


def _af2_params():
    from tt_bio import weights
    params = weights.resolve("af2-params")
    if params is None or not (pathlib.Path(params) / "params_model_1_ptm.npz").exists():
        pytest.skip("no AlphaFold 2 parameters; run `tt-bio weights --download af2ig`")
    return pathlib.Path(params)


def test_the_predictor_conforms_to_bindcrafts_protocol():
    _bindcraft_root()
    from bindcraft.af2 import AlphaFoldDesignModel
    from bindcraft.prediction import DifferentiableProteinPredictor

    cls = bindcraft2.design_model_class()
    assert issubclass(cls, AlphaFoldDesignModel)
    # `DifferentiableProteinPredictor` is runtime-checkable, so this is a method-presence check
    # and it is the one `campaign.py` relies on.
    assert issubclass(cls, DifferentiableProteinPredictor)


def test_an_unknown_trunk_is_refused():
    _bindcraft_root()
    cls = bindcraft2.design_model_class()
    with pytest.raises(ValueError):
        cls(trunk="cuda")
    with pytest.raises(ValueError):
        cls(trunk="device", pool=None)


GRADIENT_STEP = """
    import pathlib, sys
    import jax
    from tt_bio import bindcraft2, weights
    from bindcraft.preflight import cleaned_campaign_settings
    from bindcraft.settings import (build_design_settings, parse_setting_overrides,
                                    read_settings)
    from bindcraft.trajectory import initialize_design_trajectory

    assert not [p for p in sys.path if "perf" in pathlib.Path(p).parts], sys.path
    params = pathlib.Path(weights.resolve("af2-params"))
    settings = cleaned_campaign_settings(read_settings(
        {settings_file!r}, parse_setting_overrides(["binder_lengths=[60]", "campaign_seed=0"])))
    design_settings = build_design_settings(settings)
    protein_states, _, losses = initialize_design_trajectory(
        design_settings, jax.random.PRNGKey(design_settings.seed))

    with bindcraft2.predictor(trunk="jax") as build:
        model = build(presets="model_1_ptm", data_dir=str(params), max_cache_size=1,
                      num_recycle=1, length_bucket_size=32)
        predictions, gradients, loss = model.sequence_gradients(protein_states, losses)

    chain, gradient = sorted(gradients.items())[0]
    print("tokens", sum(len(p) for p in next(iter(predictions.values())).protein_complex.values()))
    print("gradient", chain, tuple(gradient.shape), bool((gradient != 0).any()),
          bool(jax.numpy.isfinite(gradient).all()))
    print("loss", float(loss))
"""


def test_the_control_arm_takes_one_gradient_step_from_tt_bio_alone():
    """The packaging test: a fresh process that imports only from `tt_bio`, builds the predictor
    and differentiates AlphaFold 2 through a sequence, with no `perf/` on `sys.path`."""
    root = _bindcraft_root()
    _af2_params()
    printed = _run_child(GRADIENT_STEP.format(
        settings_file=str(root / "examples" / "pdl1.json")))
    lines = dict(line.split(" ", 1) for line in printed.strip().splitlines())
    assert lines["gradient"].endswith("True True"), printed
    assert float(lines["loss"]) == float(lines["loss"]), printed
