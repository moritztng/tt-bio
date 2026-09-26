"""`tt_bio.bindcraft2`: BindCraft 2's design loop against tt-bio's AlphaFold 2 trunk.

The device leg lives in `test_bindcraft2_hw.py`. Everything here runs without a card, and the
gradient-step test runs BindCraft 2's own trunk (`trunk="jax"`), which is the control arm a
device result is read against.
"""
import contextlib
import os
import pathlib
import subprocess
import sys
import textwrap
import types

import numpy as np
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


def test_a_missing_checkpoint_is_recorded_rather_than_refused(tmp_path):
    """The pool holds what resolved and names what did not; the predictor routes the rest to
    BindCraft 2's own trunk. Refusing here is fatal to a campaign that has already done all of
    its design work, because validation is held out of design on purpose."""
    present = tmp_path / "params_model_1_ptm.npz"
    present.touch()
    pool = bindcraft2.TrunkPool(tmp_path)
    pool.require(["model_1_ptm", "model_3_multimer_v3"])
    assert pool.names == ("model_1_ptm",)
    assert pool.holds("model_1_ptm") and not pool.holds("model_3_multimer_v3")
    assert str(tmp_path / "params_model_3_multimer_v3.npz") in pool.absent["model_3_multimer_v3"]


def test_a_model_the_pool_was_never_given_is_refused(tmp_path):
    """`require` classifies, but `use` still refuses: its caller has already decided the card
    runs this fold, and folding it on another checkpoint's weights is invisible downstream."""
    present = tmp_path / "params_model_1_ptm.npz"
    present.touch()
    pool = bindcraft2.TrunkPool({"model_1_ptm": present})
    assert pool.names == ("model_1_ptm",)
    with pytest.raises(KeyError):
        pool.use("model_3_multimer_v3")
    pool.require(["model_3_multimer_v3"])
    assert not pool.holds("model_3_multimer_v3")


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


def test_bindcraft_keys_its_compile_cache_on_a_family_that_spans_checkpoints():
    """The premise the routing rests on, read off BindCraft 2 rather than assumed.

    Both compile caches key on `alphafold_model_family` (`af2.py:271` and `:331`), and that
    function collapses all five multimer checkpoints to one family and `model_1_ptm` with
    `model_2_ptm` to another, because their `CONFIG_DIFFS` entries are identical. So the
    device/host choice, which is read when haiku traces and baked into the cached program,
    cannot differ between two checkpoints of one family.
    """
    _bindcraft_root()
    from bindcraft.af2 import MONOMER_POOL, MULTIMER_POOL, alphafold_model_family

    assert len({alphafold_model_family(m) for m in MULTIMER_POOL}) == 1, MULTIMER_POOL
    assert len({alphafold_model_family(m) for m in MONOMER_POOL}) == 1, MONOMER_POOL
    assert alphafold_model_family(MULTIMER_POOL[0]) != alphafold_model_family(MONOMER_POOL[0])


def test_a_partly_resident_family_goes_to_the_host_whole(tmp_path):
    """Three of five multimer checkpoints on card is not a routable split: the other two would
    reuse the first one's compiled program. All five go to the host instead."""
    _bindcraft_root()
    from bindcraft.af2 import MONOMER_POOL, MULTIMER_POOL, alphafold_model_family

    cls = bindcraft2.design_model_class()
    names = (*MULTIMER_POOL, *MONOMER_POOL)

    def routes(resident):
        pool = bindcraft2.TrunkPool(tmp_path)
        for name in resident:
            (tmp_path / f"params_{name}.npz").touch()
        pool.require(names)
        stub = types.SimpleNamespace(
            presets=names, models=names, pool=pool,
            model_families={n: alphafold_model_family(n) for n in names})
        return cls._route_by_family(stub)

    partial = routes(MULTIMER_POOL[:3])
    assert set(partial.values()) == {"jax"}, partial

    whole = routes(MULTIMER_POOL)
    assert all(whole[m] == "device" for m in MULTIMER_POOL), whole
    assert all(whole[m] == "jax" for m in MONOMER_POOL), whole


def _campaign_factory_trunks(monkeypatch, **kwargs):
    """Which trunk each predictor a campaign builds gets, without opening a card."""
    _bindcraft_root()
    built = []

    def design(*args, **kw):
        built.append("device")
        return object()

    design.trunk, design.pool, design.evoformer = "device", object(), object()
    design.extra_msa = None
    design.exact = True

    @contextlib.contextmanager
    def fake_predictor(**_):
        yield design

    def fake_factory(*, trunk, pool, evoformer=None, extra_msa=None, exact=True):
        def build(*args, **kw):
            built.append(trunk)
            return object()
        return build

    monkeypatch.setattr(bindcraft2, "predictor", fake_predictor)
    monkeypatch.setattr(bindcraft2, "_factory", fake_factory)
    with bindcraft2.campaign_predictor(**kwargs) as build:
        from bindcraft import campaign
        assert campaign.AlphaFoldDesignModel is build
        build()  # the design model, campaign.py:262
        build()  # the validation ensemble, campaign.py:265
        build()  # and the one desperate_prediction_pools rebuilds mid-campaign
    return built


def test_the_validation_ensemble_folds_on_the_host_trunk_by_default(monkeypatch):
    """The stage that decides whether a design is accepted runs BindCraft 2's own trunk, so
    device numerics stay out of the instrument that grades the device."""
    assert _campaign_factory_trunks(monkeypatch) == ["device", "jax", "jax"]


def test_validation_can_be_put_on_card_explicitly(monkeypatch):
    assert _campaign_factory_trunks(monkeypatch, validation="device") == ["device"] * 3


def test_the_extra_msa_stack_stays_in_jax_unless_it_is_asked_for():
    """The second swap is off by default, so an Evoformer-only comparison keeps its program.

    `bcx-seeds` grades matched pairs on the Evoformer swap alone. A second default moving
    underneath that set would void it.
    """
    _bindcraft_root()
    params = _af2_params()
    with bindcraft2.predictor(trunk="device", checkpoints=str(params)) as build:
        assert build.extra_msa is None


def test_asking_for_the_extra_msa_swap_builds_one_on_the_evoformers_pool():
    """`predictor(extra_msa=True)` reaches the constructor and hands the caller its counters.

    The lever went in inert -- `ExtraMsaOnDevice` was defined and constructed nowhere, so the
    merge that landed it moved no shipped path. This is the guard against that recurring: a
    wired-and-inert lever is this fleet's most repeated failure. It checks construction only;
    that the card actually runs the stack is a counter read on a real round.
    """
    _bindcraft_root()
    from bindcraft.af.alphafold.model import layer_stack

    params = _af2_params()
    before = layer_stack.layer_stack
    with bindcraft2.predictor(trunk="device", checkpoints=str(params),
                              extra_msa=True) as build:
        extra = build.extra_msa
        assert isinstance(extra, bindcraft2.ExtraMsaOnDevice)
        # One pool across both swaps, or the two stacks fold on different trunks.
        assert extra.pool is build.pool
        assert extra.calls == {"primal": 0, "taped": 0, "backward": 0}
        assert extra.swapped == []
        assert layer_stack.layer_stack is not before
    assert layer_stack.layer_stack is before


def test_the_campaign_path_can_ask_for_the_extra_msa_swap_too():
    """`campaign_predictor` forwards the argument and re-exposes the handle.

    It takes `**kwargs` rather than naming `extra_msa`, so nothing in its signature says the swap
    reaches a campaign. A campaign is the entry point a real run uses, so this is what says it.
    """
    _bindcraft_root()
    params = _af2_params()
    with bindcraft2.campaign_predictor(checkpoints=str(params), extra_msa=True) as build:
        assert isinstance(build.extra_msa, bindcraft2.ExtraMsaOnDevice)
        assert build.extra_msa.pool is build.pool
    with bindcraft2.campaign_predictor(checkpoints=str(params)) as build:
        assert build.extra_msa is None


def test_the_gradient_runs_softmax_and_layer_norm_exact_by_default():
    """`exact` defaults on, and the check is the armed op tuple rather than the module global.

    `tape()` and `backward()` each read `exact_training_ops()` for their own extent, so that
    tuple is what a tape opened inside this scope would actually run. Reading
    `autograd._EXACT_TRAINING` instead would test the variable, not the scope.
    """
    _bindcraft_root()
    from tt_bio import autograd

    params = _af2_params()
    with bindcraft2.predictor(trunk="device", checkpoints=str(params)) as build:
        assert autograd.exact_training_ops() == autograd.EXACT_TRAINING_OPS
        assert build.exact is True


def test_the_exact_instrument_can_be_turned_off_through_the_predictor():
    """`predictor(exact=False)` is the only route a BindCraft 2 caller has to the off switch.

    Before this parameter the seam opened `trunk.taped.tape()` with no way out of it, so every
    BindCraft 2 round paid a host float64 round trip per softmax and per layer norm -- 2,880
    counted host entries a round at n=192, and 24.87x on the gradient call itself
    (`perf/bcx_exact/ROUND_AB.json`). The lever has to be inert-proof: an armed tuple that does
    not empty is a parameter that reaches nothing.
    """
    _bindcraft_root()
    from tt_bio import autograd

    params = _af2_params()
    armed = autograd.exact_training_ops()
    with bindcraft2.predictor(trunk="device", checkpoints=str(params), exact=False) as build:
        assert autograd.exact_training_ops() == ()
        assert build.exact is False
    # The scope is the predictor's, so it is gone with it and no later tape inherits it.
    assert autograd.exact_training_ops() == armed


def test_the_campaign_path_can_turn_the_exact_instrument_off_too():
    """`campaign_predictor` takes `**kwargs`, so nothing in its signature says `exact` arrives.

    A campaign is the entry point a real design run uses -- `campaign.run_campaign` -- and this
    is what says the parameter reaches it rather than being swallowed.
    """
    _bindcraft_root()
    from tt_bio import autograd

    params = _af2_params()
    with bindcraft2.campaign_predictor(checkpoints=str(params), exact=False) as build:
        assert autograd.exact_training_ops() == ()
        assert build.exact is False
    with bindcraft2.campaign_predictor(checkpoints=str(params)) as build:
        assert autograd.exact_training_ops() == autograd.EXACT_TRAINING_OPS
        assert build.exact is True


def test_the_extra_msa_swap_knows_both_names_alphafold_gives_its_closure():
    """The splice picks the extra-MSA `layer_stack` call out of four by the closure's name, and
    AlphaFold 2 uses two different names for it.

    `modules.py` (monomer) calls it `extra_msa_stack_fn`; `modules_multimer.py` calls it
    `extra_evoformer_fn`. Matching only the first is not a crash: `choose` falls through to
    BindCraft 2's own stack, the campaign runs the host implementation and every counter the
    caller would check reads zero, so `extra_msa=True` measures the arm it was meant to replace.
    That is what happened on the five-model `multimer_v3` pool this campaign's public
    configuration uses. Read off BindCraft 2's own source so an upstream rename fails here.
    """
    root = _bindcraft_root()
    model = root / "bindcraft" / "af" / "alphafold" / "model"
    defined = set()
    for source in (model / "modules.py", model / "modules_multimer.py"):
        for line in source.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("def extra_") and stripped.endswith("(x):"):
                defined.add(stripped[len("def "):stripped.index("(")])
    assert defined, f"no extra-MSA stack closure found under {model}"
    assert defined <= set(bindcraft2.EXTRA_MSA_FN_NAMES), (
        f"AlphaFold 2 defines {sorted(defined)}; the splice matches "
        f"{sorted(bindcraft2.EXTRA_MSA_FN_NAMES)}")


def test_the_masks_walk_reads_both_shapes_alphafold_builds_them_in():
    """`find_extra_msa_masks` against the two closures, without needing AlphaFold 2 to build one.

    The multimer path closes over one `extra_masks` dict; the monomer path has no dict and the
    walk has to assemble the pair from `batch["extra_msa_mask"]` and `mask_2d`. Both shapes are
    reproduced here with plain closures, which is all `_free_variable` ever sees.
    """
    msa_mask, pair_mask = np.zeros((1, 8), np.float32), np.ones((8, 8), np.float32)

    extra_masks = {"msa": msa_mask, "pair": pair_mask}

    def extra_evoformer_fn(x):        # modules_multimer.py:375
        return extra_masks, x

    got = bindcraft2.find_extra_msa_masks(extra_evoformer_fn)
    assert got["msa"] is msa_mask and got["pair"] is pair_mask

    batch, mask_2d = {"extra_msa_mask": msa_mask}, pair_mask

    def extra_msa_stack_fn(x):        # modules.py:1517
        return batch, mask_2d, x

    got = bindcraft2.find_extra_msa_masks(extra_msa_stack_fn)
    assert got["msa"] is msa_mask and got["pair"] is pair_mask

    # a closure that carries neither is a refusal, not a guess
    def evoformer_fn(x):
        return x
    assert bindcraft2.find_extra_msa_masks(evoformer_fn) is None


def test_the_extra_msa_swap_pads_bindcraft_2s_real_shapes_to_a_tile():
    """`_pad` and `_check_mask` on the shapes a real PD-L1 round hands the swap.

    Both are pure torch and numpy, so they are testable without a card, and both are the first
    thing the device path touches. n=275 is what `perf/bcx_extrawire/runs/grade_host` captured off
    a real `sequence_gradients`, and 275 buckets to 288.
    """
    extra = bindcraft2.ExtraMsaOnDevice(pool=None)

    pair = torch.arange(275 * 275 * 4, dtype=torch.float32).reshape(275, 275, 4)
    pair_mask = torch.ones(275, 275)
    padded, padded_mask, n = extra._pad(pair, pair_mask)
    assert n == 275
    assert padded.shape == (288, 288, 4), padded.shape
    assert padded_mask.shape == (288, 288), padded_mask.shape
    # the real region survives and the tile padding is zero, both of which the card relies on
    assert torch.equal(padded[:275, :275], pair)
    assert torch.equal(padded_mask[:275, :275], pair_mask)
    assert padded[275:].abs().sum() == 0 and padded[:, 275:].abs().sum() == 0
    assert padded_mask[275:].abs().sum() == 0 and padded_mask[:, 275:].abs().sum() == 0

    # a length already on a tile boundary is handed back untouched
    on_tile = torch.zeros(288, 288, 4)
    same, same_mask, n2 = extra._pad(on_tile, torch.zeros(288, 288))
    assert n2 == 288 and same is on_tile

    # BindCraft 2's real extra-MSA mask is all zero, which is the case this swap is correct for
    extra._check_mask(np.zeros((1, 275), dtype=np.float32))
    assert extra.mask_seen == {"calls": 1, "abs_max": 0.0}
    # and anything else is refused rather than folded against the wrong constant
    nonzero = np.zeros((1, 275), dtype=np.float32)
    nonzero[0, 7] = 1.0
    with pytest.raises(ValueError, match="nonzero"):
        extra._check_mask(nonzero)


def test_an_unknown_validation_trunk_is_refused():
    with pytest.raises(ValueError):
        with bindcraft2.campaign_predictor(validation="cuda"):
            pass


def test_the_campaign_rebinding_is_undone(monkeypatch):
    _bindcraft_root()
    from bindcraft import campaign
    before = campaign.AlphaFoldDesignModel
    _campaign_factory_trunks(monkeypatch)
    assert campaign.AlphaFoldDesignModel is before


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


def test_the_checkpoint_family_is_read_off_the_file_not_the_name(tmp_path):
    """`_Trunk` loads a checkpoint with the family `_is_multimer` reports, and the two families
    need a different weight remap. Loading one as the other does not return a wrong model, it
    raises a `KeyError` from inside the remap -- which is how the shipped `examples/pdl1.json`,
    whose five design checkpoints are all `multimer_v3`, became unrunnable from `tt_bio` while
    every test here still passed.

    The tell is the relative encoding, and it is read off the array names so a renamed or
    re-exported file cannot be loaded as the wrong family.
    """
    multimer = tmp_path / "renamed_to_something_else.npz"
    np.savez(multimer, **{
        "alphafold/alphafold_iteration/evoformer/~_relative_encoding/position_activations//weights":
            np.zeros((1, 1), dtype=np.float32)})
    monomer = tmp_path / "params_model_1_multimer_v3.npz"   # the misleading name is the point
    np.savez(monomer, **{
        "alphafold/alphafold_iteration/evoformer/pair_activiations//weights":
            np.zeros((1, 1), dtype=np.float32)})

    assert bindcraft2._is_multimer(multimer) is True
    assert bindcraft2._is_multimer(monomer) is False


def test_the_extra_msa_segment_survives_being_recomputed():
    """A checkpointed segment is run twice, so it may not capture anything it consumes.

    `_residual` consumes its `update` operand: it hands it to `ttnn.deallocate`
    (`tt_bio/af2.py::_residual`). A constant hoisted out of the loop and captured by the
    checkpoint closure is therefore a freed buffer by the time `autograd._recompute` re-enters
    the segment, and the first real gradient round through `predictor(extra_msa=True)` died
    exactly there, with TT_THROW "Buffer is not allocated". It got that far because the
    forward-only arm (`recompute=False`) uses each constant once and never sees it, so the whole
    card-free suite and the on-card forward passed while every backward was broken.

    Two stubs stand in for the card: a buffer that can be consumed once, and a `checkpoint` that
    runs the segment a second time the way a backward does.
    """
    class Buf:
        def __init__(self):
            self.alive = True

    class Block:
        @staticmethod
        def _residual(x, update):
            if not update.alive:
                raise RuntimeError("Buffer is not allocated")
            update.alive = False
            return x

        def __call__(self, z, *pair_masks):
            return z

    # The stub bites: consuming one buffer twice is what the shipped code did.
    dead = Buf()
    Block._residual(None, dead)
    with pytest.raises(RuntimeError, match="not allocated"):
        Block._residual(None, dead)

    built = []

    class Model:
        device_extra_msa = [Block(), Block()]
        opm_constant = [torch.zeros(3), torch.zeros(3)]

        @staticmethod
        def _up(_t):
            built.append(Buf())
            return built[-1]

    def checkpoint(fn, z):
        out = fn(z)     # the forward
        fn(z)           # the recompute the backward performs on the same closure
        return out

    trunk = object.__new__(bindcraft2._Trunk)
    trunk.model = Model()
    trunk.ag = types.SimpleNamespace(checkpoint=checkpoint)

    assert trunk.extra_msa("z", (), recompute=True) == "z"
    # One constant per block PER EXECUTION, two blocks run twice. Hoisting it out of the loop
    # body -- the shape that shipped -- builds 2 and reuses them, which is the failure above.
    assert len(built) == 4, built

    built.clear()
    assert trunk.extra_msa("z", (), recompute=False) == "z"
    assert len(built) == 2, built

    # The control: the loop body as it shipped, hoisting the constant out and letting the closure
    # capture it. Nothing above is production code, so without this the stubs could simply be
    # blind and every assertion would still pass.
    built.clear()
    model, z = trunk.model, "z"
    with pytest.raises(RuntimeError, match="not allocated"):
        for index, block in enumerate(model.device_extra_msa):
            const = model._up(model.opm_constant[index].reshape(1, 1, -1))
            z = checkpoint(lambda t, blk=block, c=const: blk(blk._residual(t, c)), z)
    assert len(built) == 1, built
