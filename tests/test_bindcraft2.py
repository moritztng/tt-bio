"""`tt_bio.bindcraft2`: BindCraft 2's design loop against tt-bio's AlphaFold 2 trunk.

The device leg lives in `test_bindcraft2_hw.py`. Everything here runs without a card, and the
gradient-step test runs BindCraft 2's own trunk (`trunk="jax"`), which is the control arm a
device result is read against.
"""
import contextlib
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import threading
import time
import traceback
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
    """The child names its own pin rather than inheriting one: `pin_card` returns early when the
    environment already names the card asked for, so a suite run under `TT_VISIBLE_DEVICES=3`
    (which is how this fleet runs anything that touches a card) used to fail this test and pass
    it on every other card."""
    printed = _run_child("""
        import os, sys, types
        os.environ["TT_VISIBLE_DEVICES"] = "0"
        sys.modules["ttnn"] = types.ModuleType("ttnn")
        from tt_bio import bindcraft2
        try:
            bindcraft2.pin_card(3)
        except RuntimeError as exc:
            print("raised", "TT_VISIBLE_DEVICES=3" in str(exc))
    """)
    assert printed.split() == ["raised", "True"]


def test_pinning_the_card_the_environment_already_names_is_a_no_op():
    """The other half of the same branch, and the reason the test above has to set its own pin:
    a process started with the card already in its environment is pinned, ttnn or no ttnn."""
    printed = _run_child("""
        import os, sys, types
        os.environ["TT_VISIBLE_DEVICES"] = "3"
        sys.modules["ttnn"] = types.ModuleType("ttnn")
        from tt_bio import bindcraft2
        bindcraft2.pin_card(3)
        print("no-op", os.environ["TT_VISIBLE_DEVICES"])
    """)
    assert printed.split() == ["no-op", "3"]


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
    design.template = None
    design.exact = True
    design.fast = None

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


def test_the_extra_msa_stack_runs_on_card_by_default():
    """Both AF2 swaps default ON, because leaving them in JAX is what costs the round.

    On a Wormhole Galaxy chip a 288-token round is 29.423 s with them in JAX against 16.267 s
    on card, 1.8087x, host 17.188 -> 2.417 s (`perf/bwx_perf/results/`); Blackhole agrees to
    3 %. Before this the shipped default was the one arm nobody had measured: every headline
    was taken through a harness that hardcoded them on.
    """
    _bindcraft_root()
    params = _af2_params()
    with bindcraft2.predictor(trunk="device", checkpoints=str(params)) as build:
        assert build.extra_msa is not None


def test_the_extra_msa_stack_can_be_kept_in_jax():
    """`extra_msa=False` is the opt-out an Evoformer-only comparison needs.

    `bcx-seeds` grades matched pairs on the Evoformer swap alone, so that set is re-run on
    this flag rather than on the default.
    """
    _bindcraft_root()
    params = _af2_params()
    with bindcraft2.predictor(trunk="device", checkpoints=str(params),
                              extra_msa=False) as build:
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
        assert isinstance(build.extra_msa, bindcraft2.ExtraMsaOnDevice)
    with bindcraft2.campaign_predictor(checkpoints=str(params), extra_msa=False) as build:
        assert build.extra_msa is None


def test_the_gradient_leaves_softmax_and_layer_norm_on_the_device_by_default():
    """`exact` defaults off, and the check is the armed op tuple rather than the module global.

    `tape()` and `backward()` each read `exact_training_ops()` for their own extent, so that
    tuple is what a tape opened inside this scope would actually run. Reading
    `autograd._EXACT_TRAINING` instead would test the variable, not the scope.

    The default is off because the host float64 instrument costs 24.87x on the gradient call
    (479.59 s against 19.285 s at n=192, `perf/bcx_exact/ROUND_AB.json`) and moves the worst
    gradient tensor 1.1 %, from 0.087998 to 0.088985 against a float64 reference, where
    bfloat16 alone already carries 0.075483 of it. This is the module-level default, not the
    engine's: `tt_bio.autograd.EXACT_TRAINING_OPS` is still what a bare tape arms, which is
    what OpenFold 3 training runs on and what `tests/test_exact_training_default.py` pins.
    """
    _bindcraft_root()
    from tt_bio import autograd

    params = _af2_params()
    armed = autograd.exact_training_ops()
    with bindcraft2.predictor(trunk="device", checkpoints=str(params)) as build:
        assert autograd.exact_training_ops() == ()
        assert build.exact is False
    assert autograd.exact_training_ops() == armed


def test_the_exact_instrument_can_be_asked_for_off_explicitly():
    """`predictor(exact=False)` spelled out, which is what every `perf/bcx_*` harness passes.

    The default test above covers the same branch, but not the keyword: renaming `exact` would
    leave that one green and break every caller. Before this parameter existed the seam opened
    `trunk.taped.tape()` with no way out of it, so every BindCraft 2 round paid a host float64
    round trip per softmax and per layer norm -- 2,880 counted host entries a round at n=192,
    and 24.87x on the gradient call itself (`perf/bcx_exact/ROUND_AB.json`). The lever has to
    be inert-proof: an armed tuple that does not empty is a parameter that reaches nothing.
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



def _fast_round_now():
    import importlib
    out = {}
    for module, owner, attr, _env, _value in bindcraft2._FAST_ROUND:
        target = importlib.import_module(f"tt_bio.{module}")
        out[attr] = getattr(getattr(target, owner) if owner else target, attr)
    return out


def test_fast_round_keeps_blackhole_only_levers_off_on_wormhole(monkeypatch):
    """The levers graded on Blackhole alone stay where they were on a Wormhole chip."""
    from tt_bio import tenstorrent
    for _module, _owner, _attr, env, _value in bindcraft2._FAST_ROUND:
        if env:
            monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: True)
    before = _fast_round_now()
    with bindcraft2.fast_round() as armed:
        for _m, _o, attr, _e, value in bindcraft2._FAST_ROUND:
            if attr == "TAPED_KERNELS_DEFAULT":
                assert armed[attr] == bindcraft2._FAST_ROUND_WORMHOLE_KERNELS
            elif attr in bindcraft2._BLACKHOLE_ONLY:
                assert armed[attr] == before[attr]
            else:
                assert armed[attr] == value
    assert _fast_round_now() == before
    assert bindcraft2._BLACKHOLE_ONLY <= {a for _m, _o, a, _e, _v in bindcraft2._FAST_ROUND}


def test_exact_false_arms_the_measured_round_and_puts_it_back(monkeypatch):
    """The round `docs/bindcraft2.md` quotes needs no env var: `exact=False` arms its levers.

    They are process state other models' tapes read too, so the check that matters as much as
    the arming is that the predictor's exit restores every one of them.
    """
    _bindcraft_root()
    for _module, _owner, _attr, env, _value in bindcraft2._FAST_ROUND:
        if env:
            monkeypatch.delenv(env, raising=False)
    from tt_bio import tenstorrent
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: False)
    params = _af2_params()
    before = _fast_round_now()
    want = {attr: value for _m, _o, attr, _e, value in bindcraft2._FAST_ROUND}
    assert before != want
    with bindcraft2.campaign_predictor(checkpoints=str(params), exact=False) as build:
        assert _fast_round_now() == want
        assert build.fast == want
    assert _fast_round_now() == before
    with bindcraft2.predictor(trunk="device", checkpoints=str(params)) as build:
        assert _fast_round_now() == want
        assert build.fast == want
    assert _fast_round_now() == before
    with bindcraft2.predictor(trunk="device", checkpoints=str(params), exact=True) as build:
        assert _fast_round_now() == before
        assert build.fast is None
    with bindcraft2.predictor(trunk="device", checkpoints=str(params), exact=False,
                              fast=False) as build:
        assert _fast_round_now() == before


def test_a_lever_env_var_still_takes_one_lever_out(monkeypatch):
    """An A/B has to be able to drop one lever from the armed round without editing code."""
    _bindcraft_root()
    from tt_bio import mm_layout, rne_add

    monkeypatch.setenv("TT_BIO_MM_LAYOUT", "0")
    monkeypatch.delenv("TT_BIO_WIDEN_ADD", raising=False)
    params = _af2_params()
    with bindcraft2.predictor(trunk="device", checkpoints=str(params), exact=False):
        assert mm_layout.MM_LAYOUT is False
        assert rne_add.WIDEN_ADD is True

def test_the_campaign_path_carries_the_exact_argument_both_ways():
    """`campaign_predictor` takes `**kwargs`, so nothing in its signature says `exact` arrives.

    A campaign is the entry point a real design run uses -- `campaign.run_campaign` -- and this
    is what says the parameter reaches it rather than being swallowed. Both directions, because
    `exact=True` is now the off-default side and nothing else card-free walks it.
    """
    _bindcraft_root()
    from tt_bio import autograd

    params = _af2_params()
    with bindcraft2.campaign_predictor(checkpoints=str(params), exact=True) as build:
        assert autograd.exact_training_ops() == autograd.EXACT_TRAINING_OPS
        assert build.exact is True
    with bindcraft2.campaign_predictor(checkpoints=str(params)) as build:
        assert autograd.exact_training_ops() == ()
        assert build.exact is False


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


def test_the_evoformer_path_pads_the_token_axis_before_anything_reaches_the_card():
    """The shape a BindCraft 2 round executes is `_pad32(n)`, never `n`.

    The swap's `_pad` is already pinned above; this is the Evoformer's, which is the path every
    round takes whether or not the extra-MSA stack is on card. It is worth its own test because
    the campaign's block harnesses were built at the PRE-PAD host shape and nothing caught it:
    `bcx-p10-devmap` and `bcx-p10-trimul` timed blocks at n=275 while the card ran 288, which
    inflated a per-family attribution table and produced a root cause the fold does not have
    (three fast paths decline on `% 32` at 275 and are open at 288).
    """
    pad = bindcraft2.EvoformerOnDevice._pad

    m = torch.arange(2 * 275 * 8, dtype=torch.float32).reshape(2, 275, 8)
    z = torch.arange(275 * 275 * 4, dtype=torch.float32).reshape(275, 275, 4)
    mask, pair_mask = torch.ones(2, 275), torch.ones(275, 275)
    mp, zp, maskp, pmp, n = pad(m, z, mask, pair_mask)

    assert n == 275, "the caller slices the result back with the LOGICAL length"
    assert mp.shape == (2, 288, 8) and zp.shape == (288, 288, 4)
    assert maskp.shape == (2, 288) and pmp.shape == (288, 288)
    # the real region survives and the padding is zero, which is what makes the mask exact
    assert torch.equal(zp[:275, :275], z) and torch.equal(mp[:, :275], m)
    assert zp[275:].abs().sum() == 0 and zp[:, 275:].abs().sum() == 0
    assert maskp[:, 275:].abs().sum() == 0
    assert pmp[275:].abs().sum() == 0 and pmp[:, 275:].abs().sum() == 0

    # already on a tile boundary: handed back untouched, same objects
    on_tile = (torch.zeros(2, 288, 8), torch.zeros(288, 288, 4),
               torch.zeros(2, 288), torch.zeros(288, 288))
    out = pad(*on_tile)
    assert out[4] == 288 and all(a is b for a, b in zip(out[:4], on_tile))


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

    def checkpoint(fn, z, offload=False):
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


# ------------------------------------------------ several trajectories on one card


def _campaign_calls(monkeypatch, rounds_before_finishing=3, tokens=None, **kwargs):
    """Run `bindcraft2.run_campaign` against a campaign that does nothing but count rounds.

    Each fake campaign call plays `rounds_before_finishing` gradient rounds through
    `duotraj.round_entered`, which is the hook the real design model calls, and records the
    thread and slot it ran on.
    """
    _bindcraft_root()
    from bindcraft import campaign
    from tt_bio import duotraj

    calls = []

    def fake_run_campaign(settings, project_folder, **kw):
        calls.append({"settings": settings, "project": project_folder, "kwargs": kw,
                      "slot": duotraj.slot(), "thread": threading.current_thread().name,
                      "gate": duotraj.GATE is not None, "order": len(calls)})
        for _ in range(rounds_before_finishing):
            duotraj.round_entered()
            time.sleep(0.01)
        return len(calls)

    monkeypatch.setattr(campaign, "run_campaign", fake_run_campaign)
    if tokens is not None:
        monkeypatch.setattr(bindcraft2, "design_tokens", lambda settings: tokens)
    returned = bindcraft2.run_campaign({"campaign_name": "t"}, "/tmp/project", **kwargs)
    return calls, returned


def test_one_trajectory_per_card_is_bindcrafts_own_call(monkeypatch):
    """The escape hatch changes nothing: no gate, no thread, the caller's own arguments."""
    from tt_bio import duotraj

    calls, returned = _campaign_calls(monkeypatch, trajectories_per_card=1,
                                      af2_weights="/w", mpnn_weights="/m")
    assert len(calls) == 1
    assert calls[0]["kwargs"] == {"af2_weights": "/w", "mpnn_weights": "/m"}
    assert calls[0]["gate"] is False and calls[0]["slot"] == ""
    assert calls[0]["thread"] == threading.current_thread().name
    assert returned == 1
    assert duotraj.GATE is None


def _box(monkeypatch, *, free_gb, rss_gb=0.0, card_gb=0.0, tokens=288, part_gb=31.875):
    """Pretend the box has this much free host memory, this process holds that much, the card has
    that much DRAM free (0 = no card open, which is what `auto` sees at entry), the part in the
    box is this big, and the design runs at this token axis.

    `part_gb` is pinned rather than read, so these expectations are the same on every box the
    suite runs on. 31.875 is Blackhole, which is where all of them were measured."""
    from tt_bio import duotraj

    monkeypatch.setattr(duotraj, "free_host_bytes", lambda: int(free_gb * 2**30))
    monkeypatch.setattr(duotraj, "host_rss_bytes", lambda: int(rss_gb * 2**30))
    monkeypatch.setattr(duotraj, "free_device_bytes", lambda: int(card_gb * 2**30))
    monkeypatch.setattr(duotraj, "card_total_bytes", lambda: int(part_gb * 2**30))
    monkeypatch.setattr(bindcraft2, "design_tokens", lambda settings: tokens)


def _roomy_box(monkeypatch):
    """Pin the box big enough to hold `duotraj.AUTO_CAP` trajectories, for the tests that are
    about the GATE and not about memory.

    Six of them asked for three trajectories and read the real hardware, so they passed on a
    Blackhole box and failed on a Wormhole Galaxy chip with
    `MemoryError: interleaving 3 trajectories of 0 tokens needs 16.01 GB on the card and it has
    11.40 GB for them` -- three times the fit's intercept is already more than a 12 GB card,
    whatever the design. The refusal is correct and is what `auto` errs low for; it is just not
    what a test of the header, the footer gate, the slot names or the summary lock is measuring.
    On a host with no BindCraft 2 the same six SKIP, which is why the suite looked green on
    every box anyone ran it on (`state/b2p-wh.md`).
    """
    _box(monkeypatch, free_gb=200.0)


def test_the_default_takes_as_many_trajectories_as_the_box_holds(monkeypatch, capsys):
    """The shipped default is `auto`, and on a roomy box that is the cap, not one."""
    from tt_bio import duotraj

    _box(monkeypatch, free_gb=200.0)
    calls, returned = _campaign_calls(monkeypatch, af2_weights="/w")
    assert len(calls) == duotraj.AUTO_CAP == 3
    assert all(call["gate"] for call in calls)
    assert all(call["kwargs"] == {"af2_weights": "/w"} for call in calls)
    # It says which it chose and why, on one line the user sees.
    line = capsys.readouterr().out
    assert "3 design trajectories on this card" in line and "288 tokens" in line


def test_a_box_that_holds_two_gets_two_not_the_cap(monkeypatch):
    """17 GB free is a two-trajectory box: two of them peaked at 14.2-15.3 GB and three at
    19.3-19.5, which is the reading that kept a third off pc."""
    _box(monkeypatch, free_gb=17.0)
    calls, _ = _campaign_calls(monkeypatch)
    assert len(calls) == 2


def test_a_small_box_gets_bindcrafts_own_loop(monkeypatch):
    """Auto floors at 1, and 1 is upstream's call: no gate, no thread."""
    from tt_bio import duotraj

    _box(monkeypatch, free_gb=9.0)
    calls, _ = _campaign_calls(monkeypatch)
    assert len(calls) == 1
    assert calls[0]["gate"] is False and calls[0]["slot"] == ""
    assert calls[0]["thread"] == threading.current_thread().name
    assert duotraj.GATE is None


def test_a_box_whose_free_memory_cannot_be_read_runs_one(monkeypatch, capsys):
    """An unreadable box is not a proven-roomy one. `free_host_bytes()` returns 0 when
    /proc/meminfo is missing, and choosing the cap there is how a default OOM-kills a campaign
    that worked yesterday."""
    _box(monkeypatch, free_gb=0.0)
    calls, _ = _campaign_calls(monkeypatch)
    assert len(calls) == 1
    assert calls[0]["gate"] is False
    assert "could not be read" in capsys.readouterr().out


def test_auto_does_not_put_more_trajectories_on_the_card_than_it_holds(monkeypatch):
    """The box is roomy and the card is not: the card is read too, when one is open.

    12 GB free holds two 288-token trajectories at 4.19 GB each with the 1 GB reserve left over;
    4 GB holds one. The count is what ALL of them need, not what the ones after the first need:
    `auto` runs before any trajectory has started, so none of that memory is spoken for yet, and
    charging the first one nothing is how the old estimate approved a second that had nowhere to
    go."""
    _box(monkeypatch, free_gb=200.0, card_gb=12.0)
    assert len(_campaign_calls(monkeypatch)[0]) == 2
    _box(monkeypatch, free_gb=200.0, card_gb=4.0)
    assert len(_campaign_calls(monkeypatch)[0]) == 1


def test_a_twelve_gib_wormhole_chip_is_not_priced_as_a_blackhole_one(monkeypatch):
    """The card `auto` prices before one is open is the part THIS host has.

    A Wormhole Galaxy chip holds 12 GiB where a Blackhole p150a or p300 chip holds 31.875, and
    `auto` runs before ttnn is imported, so nothing in the allocator can say which it is. Priced
    as Blackhole, a 288-token design got 3 trajectories: 12.6 GB asked of a card with 11.1.
    Measured on dev Galaxy .107 chip 30, one 288-token trajectory peaks at 4.119 GB and 2 of them
    interleaved ran a real campaign to its stop condition, so 2 is the answer whether or not a
    chip is open, and 3 is refused either way."""
    _box(monkeypatch, free_gb=462.0, part_gb=12.0)
    assert len(_campaign_calls(monkeypatch, tokens=288)[0]) == 2
    # Explicit counts are still the caller's: 2 is honoured on that chip, 3 is refused on it.
    _box(monkeypatch, free_gb=462.0, part_gb=12.0)
    assert len(_campaign_calls(monkeypatch, tokens=288, trajectories_per_card=2)[0]) == 2
    with pytest.raises(MemoryError, match="on the card"):
        _campaign_calls(monkeypatch, tokens=288, trajectories_per_card=3)
    # With the chip open and reporting its own free bytes, the count is the measured 2.
    _box(monkeypatch, free_gb=462.0, part_gb=12.0, card_gb=11.9)
    assert len(_campaign_calls(monkeypatch, tokens=288)[0]) == 2


def test_an_unknown_part_is_priced_as_the_smallest_one(monkeypatch, tmp_path):
    """A host that will not say which part it has gets the tighter answer, the way
    `tenstorrent.l1_resident_budget_bytes()` falls back to Wormhole's L1. Over-pricing the card
    starts trajectories that do not fit; under-pricing it starts one that does."""
    from tt_bio import duotraj

    monkeypatch.setattr(duotraj, "TT_SYSFS_CLASS", str(tmp_path / "nothing-here"))
    assert duotraj.card_total_bytes() == duotraj.CARD_BYTES == int(12 * 2**30)

    monkeypatch.setattr(duotraj, "TT_SYSFS_CLASS", str(tmp_path))
    node = tmp_path / "tenstorrent!0" / "device"
    node.mkdir(parents=True)
    (node / "device").write_text("0xdead\n")          # a part this table has never seen
    assert duotraj.card_total_bytes() == int(12 * 2**30)
    (node / "device").write_text("0xb140\n")
    assert duotraj.card_total_bytes() == int(31.875 * 2**30)
    (node / "device").write_text("0x401e\n")
    assert duotraj.card_total_bytes() == int(12 * 2**30)


def test_a_large_design_gets_fewer_trajectories_than_a_small_one(monkeypatch, capsys):
    """The regression this row exists for. The footprint grows with the square of the token axis,
    so a count chosen at 288 tokens is not a count that fits at 704: measured on qb2 card 0, one
    trajectory of a 704-token design peaks at 19.32 GB of the card's 31.875, and the three the
    size-blind default chose died in round 1 with the card 99.4 % allocated
    (`state/bgx-traj.md`). The same roomy box, the same card, five sizes."""
    _box(monkeypatch, free_gb=200.0, tokens=288)
    assert len(_campaign_calls(monkeypatch, tokens=288)[0]) == 3
    assert len(_campaign_calls(monkeypatch, tokens=448)[0]) == 3
    assert len(_campaign_calls(monkeypatch, tokens=512)[0]) == 2
    assert len(_campaign_calls(monkeypatch, tokens=576)[0]) == 1
    assert len(_campaign_calls(monkeypatch, tokens=704)[0]) == 1
    assert "704 tokens" in capsys.readouterr().out


def test_a_design_that_only_fits_once_runs_bindcrafts_own_loop(monkeypatch):
    """Where one fits, auto IS one: no gate, no thread, upstream's own call."""
    from tt_bio import duotraj

    _box(monkeypatch, free_gb=200.0, tokens=704)
    calls, _ = _campaign_calls(monkeypatch, tokens=704)
    assert len(calls) == 1
    assert calls[0]["gate"] is False and calls[0]["slot"] == ""
    assert calls[0]["thread"] == threading.current_thread().name
    assert duotraj.GATE is None


def test_a_design_whose_token_axis_cannot_be_read_runs_one(monkeypatch, capsys):
    """An unknown size is not a small one. `design_tokens` answers 0 when BindCraft 2 cannot
    size the design, and pricing that as the 288-token reference is how the guard was wrong."""
    _box(monkeypatch, free_gb=200.0, tokens=0)
    calls, _ = _campaign_calls(monkeypatch, tokens=0)
    assert len(calls) == 1
    assert "token axis could not be read" in capsys.readouterr().out


def test_the_estimate_is_above_every_footprint_measured_on_the_card(monkeypatch):
    """The guard errs LOW on the count, which means erring HIGH on the footprint. Measured peak
    minus the shared round-boundary floor, one shipped trajectory, qb2 card 0
    (`perf/bgx_traj/out/`), against what `trajectory_bytes` charges for it; and the padded
    route's resident peaks on qb1's p150a with the pad-up on, floor included
    (`state/b2p-ceiling.md`)."""
    from tt_bio import duotraj

    fused = {288: 2.881, 384: 5.084, 448: 6.896, 512: 8.985,
             576: 11.506, 640: 14.184, 704: 17.139}
    for tokens, gb in fused.items():
        charged = duotraj.trajectory_bytes(tokens) / 2**30
        assert charged > gb, (tokens, charged, gb)
    padded = {544: 13.00e9, 608: 16.01e9, 736: 23.07e9, 832: 29.18e9, 864: 31.39e9}
    for tokens, resident in padded.items():
        charged = duotraj.trajectory_bytes(tokens) + duotraj.trajectory_floor_bytes(tokens)
        assert charged > resident, (tokens, charged / 1e9, resident / 1e9)


def test_the_composed_surcharge_is_priced_only_where_the_composed_path_can_run(monkeypatch):
    """With the pad-up off the fused arm declines at 544 and the composed path held 20.02 GB over
    the floor there, 25.75 GB resident on qb1's p150a; the charge covers both. With it on, that
    path does not run and charging it opened one trajectory where two fit at 512 to 608."""
    from tt_bio import duotraj

    monkeypatch.setattr(duotraj, "_PAD_UP_ON", False)
    assert duotraj.trajectory_bytes(544) / 2**30 > 20.018
    assert (duotraj.trajectory_bytes(544) + duotraj.trajectory_floor_bytes(544)) / 2**30 > 25.75
    monkeypatch.setattr(duotraj, "_PAD_UP_ON", True)
    assert duotraj.trajectory_bytes(544) / 2**30 < 20.018


def test_the_pad_up_default_read_before_ttnn_matches_the_one_the_kernel_uses():
    """`duotraj` cannot import `tenstorrent` (it imports ttnn), so it reads the variable itself.
    The two defaults must be the same number or `auto` prices a path the fold does not take."""
    import re
    from pathlib import Path
    from tt_bio import duotraj

    def default(mod):
        src = Path(mod).read_text()
        return int(re.search(r'env_int\("TT_BIO_TRIATT_HIFI_PAD_UP", (\d+)\)', src).group(1))

    here = Path(duotraj.__file__).parent
    assert default(here / "duotraj.py") == default(here / "tenstorrent.py")


@pytest.mark.parametrize("tokens,count", [(448, 3), (512, 2), (544, 2), (576, 1), (832, 1)])
def test_auto_opens_two_trajectories_where_the_served_path_fits_two(monkeypatch, tokens, count):
    """At 544 one trajectory holds 13.00 GB of a p150a on the padded route, so two fit in the
    29.2 GB `auto` compares against; pricing the composed path's 25.8 GB there opened one. 576 is
    the first axis where two do not fit: 16.01 GB each measured, charged 16.7."""
    from tt_bio import duotraj

    monkeypatch.setattr(duotraj, "_PAD_UP_ON", True)
    monkeypatch.setattr(duotraj, "free_host_bytes", lambda: int(200 * 2**30))
    monkeypatch.setattr(duotraj, "host_rss_bytes", lambda: 0)
    monkeypatch.setattr(duotraj, "free_device_bytes", lambda: 0)
    monkeypatch.setattr(duotraj, "card_total_bytes", lambda: int(31.875 * 2**30))
    assert duotraj.auto_trajectories(tokens)[0] == count


def _galaxy_chip(monkeypatch, free_gb):
    from tt_bio import duotraj

    monkeypatch.setattr(duotraj, "free_host_bytes", lambda: int(free_gb * 2**30))
    monkeypatch.setattr(duotraj, "host_rss_bytes", lambda pid="self": 0)
    monkeypatch.setattr(duotraj, "free_device_bytes", lambda: 0)
    monkeypatch.setattr(duotraj, "card_total_bytes", lambda: int(12 * 2**30))
    return duotraj


def test_campaigns_started_together_on_a_galaxy_do_not_plan_more_host_than_the_box_has(monkeypatch):
    """32 campaigns from one shell loop all read MemAvailable before any has grown. Measured on a
    Wormhole Galaxy (.107, 2026-10-01): 278.6 GB free, and each alone would take 2 at 288 tokens,
    32 x 14.5 GB = 464 GB of plan. With claims a second trajectory opens only where it fits; past
    that every campaign runs one, which is the floor `auto` cannot go under."""
    duotraj = _galaxy_chip(monkeypatch, 278.6)
    alive = {}
    monkeypatch.setattr(duotraj, "_start_time", lambda pid: alive.get(pid, ""))
    counts = []
    for pid in range(1000, 1032):
        alive[pid] = "1"
        monkeypatch.setattr(duotraj.os, "getpid", lambda pid=pid: pid)
        counts.append(duotraj.auto_trajectories(288)[0])
    assert counts[0] == 2 and counts[-1] == 1, counts
    assert counts == sorted(counts, reverse=True), counts
    # Every second trajectory was opened with room for it: the campaigns that took two, and the
    # one-trajectory floor every later campaign still needs, fit in what was free.
    two = counts.count(2)
    planned = (two * (duotraj.AUTO_BASE_HOST_BYTES + duotraj.AUTO_EXTRA_HOST_BYTES)
               + duotraj.AUTO_HOST_RESERVE_BYTES) / 2**30
    assert planned <= 278.6, (planned, counts)
    assert two < 32


def test_a_sibling_counts_only_for_what_it_has_not_grown_into(monkeypatch):
    """A sibling holding its whole plan is already out of MemAvailable; charging it again would
    open fewer trajectories than fit."""
    duotraj = _galaxy_chip(monkeypatch, 20.0)
    monkeypatch.setattr(duotraj, "_start_time", lambda pid: "1")
    claims = pathlib.Path(duotraj.HOST_CLAIMS_DIR)
    claims.mkdir()
    (claims / "4242-1").write_text(str(int(14.5 * 2**30)))
    monkeypatch.setattr(duotraj, "host_rss_bytes", lambda pid="self": 0)
    n, why = duotraj.auto_trajectories(288)
    assert n == 1 and "other campaigns on this box planned" in why, why
    monkeypatch.setattr(duotraj, "host_rss_bytes",
                        lambda pid="self": int(14.5 * 2**30) if pid == 4242 else 0)
    assert duotraj.auto_trajectories(288)[0] == 2


def test_a_claim_left_by_a_dead_campaign_is_dropped(monkeypatch):
    duotraj = _galaxy_chip(monkeypatch, 20.0)
    claims = pathlib.Path(duotraj.HOST_CLAIMS_DIR)
    claims.mkdir()
    dead = claims / "4242-1"
    dead.write_text(str(int(500 * 2**30)))
    monkeypatch.setattr(duotraj, "_start_time", lambda pid: "" if pid == 4242 else "7")
    assert duotraj.auto_trajectories(288)[0] == 2
    assert not dead.exists()
    assert [f.name for f in claims.glob("*-*")] == [f"{os.getpid()}-7"]


def test_an_unwritable_claims_dir_leaves_auto_as_it_was(monkeypatch):
    duotraj = _galaxy_chip(monkeypatch, 278.6)
    monkeypatch.setattr(duotraj, "HOST_CLAIMS_DIR", "/proc/no-such-dir/claims")
    assert duotraj.auto_trajectories(288)[0] == 2


def test_the_line_calls_its_own_estimate_an_upper_bound_at_the_largest_axis_that_fits(monkeypatch):
    """At 832 tokens one trajectory is charged more than half of what a p150a has left, so `auto`
    runs one. Phrased as "one trajectory holds about N GB of the card" that reads as a refusal at
    the largest axis that works."""
    from tt_bio import duotraj

    monkeypatch.setattr(duotraj, "free_host_bytes", lambda: int(200 * 2**30))
    monkeypatch.setattr(duotraj, "host_rss_bytes", lambda: 0)
    monkeypatch.setattr(duotraj, "free_device_bytes", lambda: 0)  # what `auto` sees at entry
    monkeypatch.setattr(duotraj, "card_total_bytes", lambda: int(31.875 * 2**30))

    count, why = duotraj.auto_trajectories(832)
    assert count == 1
    assert "priced at up to" in why and "does not fit" in why, why
    assert "holds about" not in why, why


def test_an_explicit_count_is_honoured_even_when_it_will_not_fit(monkeypatch):
    """Auto lowers itself; a number the caller wrote is not quietly lowered. It still raises."""
    _bindcraft_root()
    from bindcraft import campaign

    started = []
    monkeypatch.setattr(campaign, "run_campaign", lambda *a, **kw: started.append(1))
    _box(monkeypatch, free_gb=2.0)
    with pytest.raises(MemoryError, match="HOST"):
        bindcraft2.run_campaign({}, "/tmp/project", trajectories_per_card=3)
    assert not started


def test_an_explicit_count_too_big_for_the_card_is_refused_on_the_card(monkeypatch):
    """The host is roomy and the card is not: an explicit 3 at 704 tokens is the run that died,
    and it is refused before a thread exists rather than in round 1."""
    _bindcraft_root()
    from bindcraft import campaign

    started = []
    monkeypatch.setattr(campaign, "run_campaign", lambda *a, **kw: started.append(1))
    _box(monkeypatch, free_gb=200.0, tokens=704)
    with pytest.raises(MemoryError, match="on the card"):
        bindcraft2.run_campaign({}, "/tmp/project", trajectories_per_card=3)
    assert not started


def test_a_string_that_is_not_auto_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="auto"):
        _campaign_calls(monkeypatch, trajectories_per_card="all")


@pytest.mark.parametrize("trajectories", [0, -1])
def test_fewer_than_one_trajectory_is_refused(monkeypatch, trajectories):
    with pytest.raises(ValueError):
        _campaign_calls(monkeypatch, trajectories_per_card=trajectories)


def test_three_trajectories_run_one_campaign_each_under_the_gate(monkeypatch):
    _roomy_box(monkeypatch)
    from tt_bio import duotraj

    calls, returned = _campaign_calls(monkeypatch, trajectories_per_card=3)
    assert len(calls) == 3
    assert sorted(call["slot"] for call in calls) == ["t1", "t2", "t3"]
    assert {call["thread"] for call in calls} == {"duotraj:t1", "duotraj:t2", "duotraj:t3"}
    assert all(call["gate"] for call in calls)
    assert returned == 3
    # The gate is installed for the campaign and taken down with it.
    assert duotraj.GATE is None


def test_a_trajectory_waits_for_the_one_before_it_to_clear_its_compile_round(monkeypatch):
    """No two trajectories trace at the same time: i starts on i-1's SECOND round."""
    _bindcraft_root()
    from bindcraft import campaign
    from tt_bio import duotraj

    seen = []

    def fake_run_campaign(settings, project_folder, **kw):
        slot = duotraj.slot()
        for round_number in range(1, 4):
            seen.append((slot, round_number))
            duotraj.round_entered()
            time.sleep(0.05)

    monkeypatch.setattr(campaign, "run_campaign", fake_run_campaign)
    bindcraft2.run_campaign({}, "/tmp/project", trajectories_per_card=2)

    assert seen[0] == ("t1", 1)
    # t2's first round cannot appear before t1's second, which is the event that releases it.
    assert seen.index(("t2", 1)) > seen.index(("t1", 2))


def test_a_trajectory_that_stops_early_does_not_strand_the_next_one(monkeypatch):
    """A campaign that never reaches a second round still releases its follower."""
    _roomy_box(monkeypatch)
    _bindcraft_root()
    from bindcraft import campaign

    ran = []
    monkeypatch.setattr(campaign, "run_campaign",
                        lambda settings, project_folder, **kw: ran.append(1))
    bindcraft2.run_campaign({}, "/tmp/project", trajectories_per_card=3, stagger_timeout=30.0)
    assert len(ran) == 3


def test_a_failing_trajectory_comes_back_out(monkeypatch):
    _bindcraft_root()
    from bindcraft import campaign
    from tt_bio import duotraj

    def fake_run_campaign(settings, project_folder, **kw):
        if duotraj.slot() == "t2":
            raise RuntimeError("t2 fell over")

    monkeypatch.setattr(campaign, "run_campaign", fake_run_campaign)
    with pytest.raises(RuntimeError, match="t2 fell over"):
        bindcraft2.run_campaign({}, "/tmp/project", trajectories_per_card=2,
                                stagger_timeout=30.0)


def test_a_box_that_cannot_hold_them_is_refused_before_any_campaign_starts(monkeypatch):
    """The host is the limit that binds, and it is read before a thread exists."""
    _bindcraft_root()
    from bindcraft import campaign
    from tt_bio import duotraj

    started = []
    monkeypatch.setattr(campaign, "run_campaign",
                        lambda *a, **kw: started.append(1))
    monkeypatch.setattr(duotraj, "free_host_bytes", lambda: 2 * 2**30)
    monkeypatch.setattr(bindcraft2, "design_tokens", lambda settings: 288)
    with pytest.raises(MemoryError, match="HOST"):
        bindcraft2.run_campaign({}, "/tmp/project", trajectories_per_card=3)
    assert not started


def test_the_campaign_header_is_printed_once_not_once_per_trajectory(monkeypatch):
    _roomy_box(monkeypatch)
    _bindcraft_root()
    from bindcraft import campaign

    real = campaign.print_campaign_header
    headers = []
    monkeypatch.setattr(campaign, "print_campaign_header", lambda *a, **kw: headers.append(1))
    monkeypatch.setattr(campaign, "run_campaign",
                        lambda settings, project_folder, **kw:
                        campaign.print_campaign_header(settings, project_folder, None))
    bindcraft2.run_campaign({}, "/tmp/project", trajectories_per_card=3, stagger_timeout=30.0)
    assert headers == [1]
    assert campaign.print_campaign_header is not real or True


def test_the_campaign_is_announced_over_once_by_the_last_trajectory_out(monkeypatch):
    """`run_campaign` prints `campaign done: ...` when `design_worker_index()` is None, which is
    per PROCESS. N threads share one environment, so all N announced the end and the earlier ones
    did it while another trajectory was still printing stage lines."""
    _roomy_box(monkeypatch)
    _bindcraft_root()
    from bindcraft import campaign

    real = campaign.design_worker_index
    spoke, still_running = [], []

    def fake_run_campaign(settings, project_folder, **kw):
        still_running.append(1)
        time.sleep(0.05)
        if campaign.design_worker_index() is None:
            spoke.append(len(still_running))
        still_running.pop()

    monkeypatch.setattr(campaign, "run_campaign", fake_run_campaign)
    bindcraft2.run_campaign({}, "/tmp/project", trajectories_per_card=3, stagger_timeout=30.0)

    # Once, and by the arm that found itself alone: nothing was still running behind it.
    assert spoke == [1]
    assert campaign.design_worker_index is real


def test_a_real_worker_process_keeps_bindcrafts_own_footer_gate(monkeypatch):
    """With BINDCRAFT_WORKER_ID set the process is one of several on the project and upstream
    means nobody to announce the campaign; the thread gate must not talk over that."""
    _roomy_box(monkeypatch)
    _bindcraft_root()
    from bindcraft import campaign

    monkeypatch.setenv("BINDCRAFT_WORKER_ID", "1")
    indices = []

    monkeypatch.setattr(campaign, "run_campaign",
                        lambda settings, project_folder, **kw:
                        indices.append(campaign.design_worker_index()))
    bindcraft2.run_campaign({}, "/tmp/project", trajectories_per_card=3, stagger_timeout=30.0)
    assert indices == [1, 1, 1]


def test_the_closing_summary_writer_is_serialised(monkeypatch):
    """N trajectories share a stop condition, so they reach the unlocked summary rewrite at
    once; overlapping writes to its one partial file would produce a summary that is neither."""
    _roomy_box(monkeypatch)
    _bindcraft_root()
    from bindcraft import campaign

    real = campaign.write_campaign_summary
    inside, overlapped = [], []

    def fake_run_campaign(settings, project_folder, **kw):
        campaign.write_campaign_summary(project_folder)

    def slow_summary(project_folder, *a, **kw):
        overlapped.append(len(inside))
        inside.append(1)
        time.sleep(0.05)
        inside.pop()

    monkeypatch.setattr(campaign, "run_campaign", fake_run_campaign)
    monkeypatch.setattr(campaign, "write_campaign_summary", slow_summary)
    bindcraft2.run_campaign({}, "/tmp/project", trajectories_per_card=3, stagger_timeout=30.0)

    assert overlapped == [0, 0, 0]
    assert campaign.write_campaign_summary is slow_summary
    monkeypatch.setattr(campaign, "write_campaign_summary", real)


def test_each_interleaved_trajectory_gets_its_own_design_model(monkeypatch):
    """`campaign_predictor` calls the first build of a campaign the design model. With N
    trajectories that is the first build PER TRAJECTORY, or the second one designs on the host
    control arm while its caller believes it is on the card."""
    _bindcraft_root()
    from bindcraft import campaign
    from tt_bio import duotraj

    built = []

    def design(*args, **kw):
        built.append((duotraj.slot(), "device"))
        return object()

    design.trunk, design.pool, design.evoformer = "device", object(), object()
    design.extra_msa = design.template = None
    design.exact = True
    design.fast = None

    @contextlib.contextmanager
    def fake_predictor(**_):
        yield design

    def fake_factory(*, trunk, pool, **kw):
        def build(*args, **kwargs):
            built.append((duotraj.slot(), trunk))
            return object()
        return build

    monkeypatch.setattr(bindcraft2, "predictor", fake_predictor)
    monkeypatch.setattr(bindcraft2, "_factory", fake_factory)

    def fake_run_campaign(settings, project_folder, **kw):
        campaign.AlphaFoldDesignModel()     # the design model, campaign.py:262
        campaign.AlphaFoldDesignModel()     # the validation ensemble, campaign.py:265

    monkeypatch.setattr(campaign, "run_campaign", fake_run_campaign)
    with bindcraft2.campaign_predictor():
        bindcraft2.run_campaign({}, "/tmp/project", trajectories_per_card=2,
                                stagger_timeout=30.0)

    assert sorted(built) == [("t1", "device"), ("t1", "jax"),
                             ("t2", "device"), ("t2", "jax")]


# --------------------------------------------------------------------- the refusal a user reads

#: tt-metal's own words, copied verbatim off an hTNFa + 100-residue-binder fold on qb1 card 1,
#: 2026-09-29, which runs on a 608-token axis: the refused buffer is exactly [608,4,608,608]
#: fp32 (608**3 * 4 * 4 = 3,596,091,392 B). JAX wraps this in a `JaxRuntimeError` behind ten
#: Python frames and forty lines of C++ hex, and the size that caused it appears nowhere.
REFUSAL_608 = (
    "TT_FATAL @ /project/tt_metal/impl/allocator/bank_manager.cpp:439: false\n"
    "info:\n"
    "Out of Memory: Not enough space to allocate 3596091392 B DRAM buffer across 8 banks, "
    "where each bank needs to store 449511424 B, but bank size is 4278190016 B "
    "(allocated: 3762941504 B, free: 515248512 B, largest free block: 326674368 B)\n"
    "backtrace:\n"
    " --- /home/ttuser/bcx_e2e_venv/lib/python3.12/site-packages/jaxlib/libjax_common.so"
    "(+0x3aad4e1) [0x70a33d99d4e1]\n")


def test_a_refusal_above_the_measured_ceiling_names_the_size_and_the_way_down(monkeypatch):
    """The four things the traceback does not say: how big, how much, why, and what to do."""
    # The allocator line is verbatim; the complex length is a fixture, since the message logic
    # is what is under test and the seam reports the real one at runtime. 608 refuses only with
    # the pad-up off, and that is the ceiling the message must then quote.
    from tt_bio import tenstorrent
    monkeypatch.setattr(tenstorrent, "_TRIATT_HIFI_PAD_UP_TILES", 0)
    got = bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_608),
                                         phase="backward", n=586, padded=608)
    msg = str(got)
    assert isinstance(got, MemoryError)
    assert "608 tokens" in msg and "586 residues" in msg     # the axis and how it was composed
    assert "3.596 GB" in msg and "326.7 MB" in msg           # asked for, and the block it got
    assert "fragmentation, not a full card" in msg           # 4.122 GB free covers 3.596 GB
    assert "576 tokens" in msg                               # one bucket down, and it serves
    assert "10 residues off the binder" in msg               # 586 - 576, the concrete action
    assert "backward" in msg


def test_the_p150a_ceiling_quoted_follows_the_pad_up(monkeypatch):
    """864 with the pad-up on (the default), 576 with it off: a refusal at 608 with the pad-up off
    must not tell the user 608 fits, and one at 896 with it on must quote 864, not 576."""
    from tt_bio import tenstorrent
    monkeypatch.setattr(tenstorrent, "_TRIATT_HIFI_PAD_UP_TILES", 2)
    assert bindcraft2._measured_board(bindcraft2.P150A_DRAM_BYTES)[1] == 864
    monkeypatch.setattr(tenstorrent, "_TRIATT_HIFI_PAD_UP_TILES", 0)
    assert bindcraft2._measured_board(bindcraft2.P150A_DRAM_BYTES)[1] == 576
    assert bindcraft2._measured_board(bindcraft2.WH_GALAXY_DRAM_BYTES)[1] == 512


def test_the_refusal_does_not_teach_the_false_token_arithmetic(monkeypatch):
    """The axis is the padded complex, NOT target + binder.

    hHSA at 736 tokens carries a 706-residue complex where the target counts 578 residues and
    the binder 100 -- the sum is 678, a whole two buckets low. A refusal that explains the axis
    as that sum sends the user to re-size against a number the seam does not use, which is the
    mistake this ladder made in its own harness before `axis_census.sh` measured the seam.
    736 is over the card's ceiling with the pad-up off, which is the branch whose wording this pins.
    """
    from tt_bio import tenstorrent
    monkeypatch.setattr(tenstorrent, "_TRIATT_HIFI_PAD_UP_TILES", 0)
    better = bindcraft2._size_aware_refusal(
        RuntimeError(REFUSAL_608), phase="backward", n=706, padded=736)
    assert better is not None
    text = str(better)
    assert "706 residues" in text
    assert "LARGER than target residues + binder length" in text
    assert "size the job off the 706" in text
    # the superseded clause, verbatim, must be gone
    assert "token axis is target residues + binder" not in text


def test_a_refusal_on_a_genuinely_full_card_is_not_called_fragmentation():
    """Same words from the allocator, opposite remedy: retrying smaller is all that is left."""
    full = REFUSAL_608.replace("free: 515248512 B", "free: 51524851 B") \
                      .replace("largest free block: 326674368 B", "largest free block: 51524851 B")
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(full),
                                             phase="forward", n=586, padded=608))
    assert "The card is full" in msg
    assert "fragmentation" not in msg


def test_a_card_held_to_its_last_percent_is_full_even_when_the_sliver_would_cover_the_request():
    """Wormhole's boundary, verbatim off the 544-token rung on dev .107 card 30
    (`perf/bwx_bringup/out/ceiling6/rung_b398/round.logtxt`): 177.5 MB free covers a 151.5 MB
    request, but the fold holds 98.6 % of the chip. Calling that "not a full card" sent a
    Wormhole user at 544-608 looking for a fragmentation fix, while a neighbouring rung in the
    same state, a few MB the other side of its request, was told the card was full."""
    boundary = ("Out of Memory: Not enough space to allocate 151519232 B DRAM buffer across 12 "
                "banks, where each bank needs to store 12627968 B, but bank size is 1073741792 B "
                "(allocated: 1058951296 B, free: 14790496 B, largest free block: 3954656 B)")
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(boundary),
                                             phase="backward", n=531, padded=544))
    assert "The card is full: 12.707 GB of 12.885 GB is held by this fold" in msg
    assert "177.5 MB left is in pieces of at most 4.0 MB a bank" in msg
    assert "fragmentation" not in msg
    assert "run a smaller complex" in msg and "512 tokens" in msg


@pytest.mark.parametrize("dram,banks,pad_up", [
    (bindcraft2.P150A_DRAM_BYTES, 8, 2),
    (bindcraft2.P150A_DRAM_BYTES, 8, 0),
    (bindcraft2.WH_GALAXY_DRAM_BYTES, 12, 2),
])
def test_no_refusal_on_a_measured_board_names_a_size_that_board_refuses(monkeypatch, dram, banks,
                                                                        pad_up):
    """Every padded axis from 224 to 1536 and every complex length that pads to it: the size the
    message sends the user to is at or under this board's measured ceiling, reachable by losing
    a positive number of residues, and never below the smallest bucket. Once was a 608-token
    Wormhole fold sent to 576, which refuses too."""
    from tt_bio import tenstorrent
    monkeypatch.setattr(tenstorrent, "_TRIATT_HIFI_PAD_UP_TILES", pad_up)
    cap = bindcraft2._measured_board(dram)[1]
    bank = dram // banks
    line = (f"Out of Memory: Not enough space to allocate {banks * 1024} B DRAM buffer across "
            f"{banks} banks, where each bank needs to store 1024 B, but bank size is {bank} B "
            f"(allocated: {bank - 512} B, free: 512 B, largest free block: 512 B)")
    for padded in range(224, 1537, 32):
        for n in (padded - 31, padded):
            msg = str(bindcraft2._size_aware_refusal(RuntimeError(line), phase="backward",
                                                     n=n, padded=padded))
            hit = re.search(r"(\d+) residues? (?:off the binder takes this fold to|to reach) "
                            r"(\d+) tokens", msg)
            assert hit, msg
            drop, landing = int(hit.group(1)), int(hit.group(2))
            assert drop == n - landing and drop > 0
            assert landing >= bindcraft2.TOKEN_BUCKET
            if padded > cap:
                assert landing <= cap and "run a smaller complex" in msg
            else:
                assert landing == padded - 32 and "trajectories_per_card=1" in msg


def test_a_refusal_at_a_size_that_fits_blames_the_card_not_the_size():
    """320 tokens is measured to fit alone, so a refusal there is company on the chip, and the
    knob that removes it is BindCraft 2's own one-at-a-time loop, not a shorter binder."""
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_608),
                                             phase="backward", n=300, padded=320))
    assert "trajectories_per_card=1" in msg
    assert "288 tokens" in msg


def test_the_way_down_lands_on_a_size_that_fits_not_one_bucket_down(monkeypatch):
    """The one actionable sentence must not name another size that refuses.

    A 736-token fold on a p150a was told "2 residues off the binder takes this fold to 704
    tokens". 704 is four buckets above the 576 this board is measured to complete, so the user
    pays another trace and compile to be refused again -- and the binder in that fold is 100
    residues long, so the advice was also arithmetically fine and physically impossible to
    follow beyond a point the message never named. Found on Wormhole at 608 (where the next
    bucket down, 576, refuses too) and it was latent on Blackhole all along, so the fix is on
    the shared path and not per board (`state/b2p-wh.md`). 576 is the ceiling with the pad-up
    off, so that is the branch this pins; with it on, 736 is inside the range.
    """
    from tt_bio import tenstorrent
    monkeypatch.setattr(tenstorrent, "_TRIATT_HIFI_PAD_UP_TILES", 0)
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_608),
                                             phase="backward", n=706, padded=736))
    assert "to reach 576 tokens" in msg              # the ceiling, not 704
    assert "lose 130 residues" in msg                # 706 - 576, the whole way down
    assert "704, refuses on this board too" in msg   # why the obvious step is not the answer
    assert "the binder on its own may not be long enough" in msg
    assert "takes this fold to 704" not in msg


def test_one_bucket_down_is_still_named_when_one_bucket_down_serves():
    """586 residues at 608 tokens on a p150a is one bucket over a 576 that completes, so the
    cheap step IS the answer and the message must stay the short version."""
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_608),
                                             phase="backward", n=586, padded=608))
    assert "10 residues off the binder takes this fold to 576 tokens" in msg
    assert "refuses on this board too" not in msg


def test_the_wormhole_way_down_skips_the_buckets_that_also_refuse():
    """On a Wormhole Galaxy chip 544, 576 and 608 all refuse, measured. A 608-token fold told
    to try 576 is told to try a size this row watched refuse ten minutes earlier."""
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_WORMHOLE),
                                             phase="backward", n=584, padded=608))
    assert "to reach 512 tokens" in msg              # this board's ceiling
    assert "lose 72 residues" in msg                 # 584 - 512
    assert "576, refuses on this board too" in msg
    assert "takes this fold to 576" not in msg


def test_a_board_nobody_laddered_keeps_the_one_bucket_step():
    """`cap` on an unmeasured card is a p150a's number, so clamping the way down to it would
    promise a size nobody has run there. One bucket down is the only honest step."""
    odd = REFUSAL_WORMHOLE.replace("across 12 banks", "across 4 banks")
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(odd),
                                             phase="backward", n=706, padded=736))
    assert "2 residues off the binder takes this fold to 704 tokens" in msg
    assert "refuses on this board too" not in msg
    assert "its own ceiling is lower" in msg


#: A Wormhole Galaxy chip's own words: 12 banks of 1,073,741,792 B, about 12.885 GB. Copied
#: off the 544-token rung of the ceiling ladder on dev .107 card 30, 2026-09-29.
REFUSAL_WORMHOLE = (
    "Out of Memory: Not enough space to allocate 1638400000 B DRAM buffer across "
    "12 banks, where each bank needs to store 136533344 B, but bank size is "
    "1073741792 B (allocated: 1028249600 B, free: 45492192 B, "
    "largest free block: 21491680 B)")


def test_a_wormhole_refusal_under_its_own_ceiling_blames_the_card_not_the_size():
    """320 tokens completes a gradient round on a Wormhole Galaxy chip held alone, so a refusal
    there means something else holds the card, most often an interleave count the chip cannot
    carry. Telling that user to shrink the complex sends them the wrong way."""
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_WORMHOLE),
                                             phase="backward", n=300, padded=320))
    assert "fits on a Wormhole Galaxy chip" in msg and "512" in msg
    assert "trajectories_per_card=1" in msg
    assert "run a smaller complex" not in msg
    assert "34.226 GB" not in msg          # the p150a's DRAM is not this card's business


def test_a_wormhole_refusal_quotes_the_ceiling_measured_on_a_wormhole():
    """The board in hand decides which measured ceiling the message may quote.

    A Wormhole Galaxy chip completes a gradient round at 512 tokens and refuses at 544; a p150a
    goes to 576. Quoting 576 here tells a user whose 544-token fold just died that it should
    have fitted, and the only remedy that reading suggests -- go find the co-tenant -- is for a
    co-tenant that does not exist. Measured in `state/bwx-bringup.md`.
    """
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_WORMHOLE),
                                             phase="backward", n=531, padded=544))
    assert "Wormhole Galaxy chip" in msg and "512 tokens" in msg
    assert "12.885 GB" in msg
    assert "run a smaller complex" in msg
    assert "19 residues off the binder" in msg     # 531 -> 512, the concrete way down
    assert "576" not in msg                        # the p150a's ceiling, not this card's


def test_a_board_nobody_laddered_still_gets_the_conservative_reference():
    """An unmeasured card must not be silently promoted to a measured one.

    A 5 % match is what identifies a board, so a card at half a p150a matches neither row and
    falls back to the p150a comparison, which says only that its own ceiling is lower -- a
    direction, not a number, because nobody has run the ladder on it.
    """
    odd = REFUSAL_WORMHOLE.replace("across 12 banks", "across 4 banks")
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(odd),
                                             phase="backward", n=300, padded=320))
    assert "its own ceiling is lower" in msg
    assert "Wormhole Galaxy chip" not in msg


def test_an_unrelated_failure_is_re_raised_unchanged():
    """A wrapper that repaints the shape of an unrelated bug is worse than no wrapper."""
    boom = ValueError("holds 24 Evoformer blocks and this splice was built for 48")
    with pytest.raises(ValueError) as caught:
        with bindcraft2._refusal_names_the_size("forward", 586, 608):
            raise boom
    assert caught.value is boom
    assert bindcraft2._size_aware_refusal(boom, phase="forward", n=586, padded=608) is None


def test_the_allocators_own_refusal_is_kept_as_the_cause():
    """The added message is for the user; the original line is what a bug report needs."""
    original = RuntimeError(REFUSAL_608)
    with pytest.raises(MemoryError) as caught:
        with bindcraft2._refusal_names_the_size("backward", 586, 608):
            raise original
    assert caught.value.__cause__ is original
    assert "largest free block: 326674368 B" in str(caught.value.__cause__)


# ------------------------------- a refusal raised inside a pure_callback, as the user gets it


class _JaxRuntimeErrorLookalike(RuntimeError):
    """What `jax.pure_callback` hands the caller when the callback raised.

    Not `jaxlib`'s own class, deliberately: these tests must run on a host with no jaxlib, and
    what is under test is that the refusal is recovered from an exception that is NOT a
    `MemoryError` and only carries the refusal's text. Matching on the class would be matching
    on the wrong thing -- JAX has moved this exception between modules more than once.
    """


def _jax_wrapped(exc: BaseException) -> _JaxRuntimeErrorLookalike:
    """`exc`, stringified into a callback error the way JAX does it.

    Measured shape: on a Wormhole Galaxy chip at 544 tokens the refusal reaches the caller as
    `INTERNAL: CpuCallback error calling callback: Traceback (most recent call last): ...`
    with six frames of `jax/_src/callback.py` and `contextlib` in front of the refusal's own
    text (`state/b2p-wh.md`). The frames are built here by raising `exc` for real, so the
    chained `__cause__` the allocator's line hangs off is in the string exactly as it would be.
    """
    try:
        raise exc
    except BaseException:                                                 # noqa: BLE001
        return _JaxRuntimeErrorLookalike(
            "INTERNAL: CpuCallback error calling callback: " + traceback.format_exc())


def test_a_refusal_stringified_by_jax_still_reaches_the_caller_as_a_memory_error():
    """On a Wormhole Galaxy chip EVERY refusal above 512 tokens is a stringified one.

    The card saturates in the Evoformer backward, which runs inside a `jax.pure_callback`, so
    the size-aware refusal is turned into a `JaxRuntimeError` carrying its text and six frames
    of JAX internals in front of it. Two things break at once: the user reads a traceback
    before the useful message, and `except MemoryError` -- the handler the refusal is written
    for -- stops catching it. A p150a passes its own 576 ceiling in a forward outside the
    callback and arrives clean, so without this the same mistake reads differently on the two
    boards (`state/b2p-wh.md`).
    """
    with pytest.raises(MemoryError) as raised:
        with bindcraft2._refusal_names_the_size("backward", 531, 544):
            raise RuntimeError(REFUSAL_WORMHOLE)
    wrapped = _jax_wrapped(raised.value)
    assert not isinstance(wrapped, MemoryError)        # the defect this test exists for

    with pytest.raises(MemoryError) as caught:
        with bindcraft2.refusals_unwrapped():
            raise wrapped
    msg = str(caught.value)
    assert "Evoformer backward at 544 tokens" in msg
    assert "Wormhole Galaxy chip" in msg and "512 tokens" in msg
    assert "19 residues off the binder" in msg
    # The refusal is handed back, not reconstructed, so the allocator's own line is still the
    # cause of the cause and a bug report loses nothing.
    assert "largest free block: 21491680 B" in str(caught.value.__cause__)
    assert "CpuCallback" not in msg


def test_the_unwrapped_refusal_is_the_object_the_seam_raised():
    """Handed back, not parsed back out of the traceback it was printed into.

    A parse would have to guess where the message ends, and the message ends with the
    allocator's own refusal -- so a parse either truncates it or swallows the frames it was
    meant to remove.
    """
    with pytest.raises(MemoryError) as raised:
        with bindcraft2._refusal_names_the_size("backward", 531, 544):
            raise RuntimeError(REFUSAL_WORMHOLE)
    assert bindcraft2.unwrap_device_refusal(_jax_wrapped(raised.value)) is raised.value


def test_an_unrelated_callback_failure_is_not_repainted_as_a_refusal():
    """A callback that died of something else must arrive as itself.

    The recovery matches on the text of a refusal this process actually raised, so a wrapper
    around an unrelated bug -- or around a refusal from a different fold -- is left alone.
    """
    with pytest.raises(MemoryError):
        with bindcraft2._refusal_names_the_size("backward", 531, 544):
            raise RuntimeError(REFUSAL_WORMHOLE)          # arms the recovery
    other = _jax_wrapped(ValueError("no live tape for token 7"))
    assert bindcraft2.unwrap_device_refusal(other) is other
    with pytest.raises(_JaxRuntimeErrorLookalike) as caught:
        with bindcraft2.refusals_unwrapped():
            raise other
    assert caught.value is other


def test_a_refusal_that_was_never_wrapped_passes_through_untouched():
    """The common case on a p150a: the ceiling is passed in a forward outside the callback."""
    clean = MemoryError("BindCraft 2 ran out of device memory")
    assert bindcraft2.unwrap_device_refusal(clean) is clean
    with pytest.raises(MemoryError) as caught:
        with bindcraft2.refusals_unwrapped():
            raise clean
    assert caught.value is clean
    assert caught.value.__cause__ is None      # not re-chained onto itself


# ------------------------------------------------- the degradation that used to be silent

def _splice_without_a_card():
    """An `EvoformerOnDevice` with nothing built: the note reads counters and prints, no more."""
    splice = object.__new__(bindcraft2.EvoformerOnDevice)
    splice._fused_checked = set()
    return splice


def test_a_fused_arm_that_declined_every_call_says_so(monkeypatch, capsys):
    """512 tokens is 1.8x the memory and 1.4x the round of 544 and nothing said why."""
    monkeypatch.setattr(bindcraft2, "_fused_hifi_counts", lambda: (0, 972))
    splice = _splice_without_a_card()
    splice._note_if_the_fused_arm_declined(512, (0, 0))
    err = capsys.readouterr().err
    assert "declined all 972 calls at 512 tokens" in err
    assert "do not fit L1" in err
    assert "[512,4,512,512]" in err              # what the composed path holds instead
    assert "2.147 GB" in err                     # 16 * 512**3, computed not quoted
    assert "one 32-token bucket UP as well as one down" in err
    assert 512 in splice._fused_checked          # said once, not once a round


def test_the_note_quotes_no_measured_axis(monkeypatch, capsys):
    """It used to name 512 and 544 as the bad and good sizes. Those came from arithmetic over a
    target file, not from the seam, and when the arithmetic proved a bucket off the message
    advised moving to the very axis it was declining at. The computed tensor size cannot rot."""
    monkeypatch.setattr(bindcraft2, "_fused_hifi_counts", lambda: (0, 96))
    splice = _splice_without_a_card()
    splice._note_if_the_fused_arm_declined(544, (0, 0))
    err = capsys.readouterr().err
    assert "declined all 96 calls at 544 tokens" in err
    assert "2.576 GB" in err                     # 16 * 544**3
    for stale in ("25.75", "14.23", "69.2", "49.4"):
        assert stale not in err


def test_a_fused_arm_that_served_is_not_reported(monkeypatch, capsys):
    """544 serves all 972. A note there would train the user to ignore the note."""
    monkeypatch.setattr(bindcraft2, "_fused_hifi_counts", lambda: (972, 0))
    splice = _splice_without_a_card()
    splice._note_if_the_fused_arm_declined(544, (0, 0))
    assert capsys.readouterr().err == ""


def test_the_note_reads_a_delta_not_a_running_total(monkeypatch, capsys):
    """The counters are process-wide and a campaign varies the binder length, so a trajectory
    that served at an earlier axis must not silence the note at this one."""
    monkeypatch.setattr(bindcraft2, "_fused_hifi_counts", lambda: (972, 972))
    splice = _splice_without_a_card()
    splice._note_if_the_fused_arm_declined(512, (972, 0))     # 972 served BEFORE this axis
    assert "declined all 972 calls at 512 tokens" in capsys.readouterr().err


def test_a_run_that_never_reached_the_arm_is_silent(monkeypatch, capsys):
    """`(0, 0)` is the host trunk and every non-device path. Silence is the only honest note."""
    monkeypatch.setattr(bindcraft2, "_fused_hifi_counts", lambda: (0, 0))
    splice = _splice_without_a_card()
    splice._note_if_the_fused_arm_declined(288, (0, 0))
    assert capsys.readouterr().err == ""


# --- the memory modes -------------------------------------------------------------------
#
# `memory='auto'` is a DEFAULT, so the first thing these pin is the thing a user would never
# forgive: that nothing which runs today starts running differently. The rest pin the one
# actionable sentence a refusal gives and the direction the retry ladder may move in.

def test_auto_leaves_every_size_that_runs_today_in_the_fast_mode():
    """The modes buy room by spending time, so a fold that already fits must not pay.

    Both boards, every bucket fast was measured to hold: `auto` has to answer `fast`. A single
    bucket answering `lean` here would make every campaign on that board slower for nothing, and
    it is the kind of regression a footprint ladder never catches because the fold still runs.

    "Measured to hold" is fast's own top, not the board's: on a p150a fast refuses 768 and 832
    while the board serves both in lean, so the buckets above 736 are the ones that MUST pay.
    """
    for card, top in ((bindcraft2.WH_GALAXY_DRAM_BYTES, bindcraft2.MEASURED_MAX_TOKENS_WH_GALAXY),
                      (bindcraft2.P150A_DRAM_BYTES, 736)):
        assert bindcraft2.max_tokens("fast", card) == top, card
        for padded in range(bindcraft2.TOKEN_BUCKET, top + 1, bindcraft2.TOKEN_BUCKET):
            assert bindcraft2.memory_mode("auto", padded, card) == "fast", (padded, card)


def test_auto_takes_the_cheapest_mode_that_fits_and_never_skips_one():
    """Above the fast ceiling `auto` steps one mode at a time, not straight to the slowest."""
    card = bindcraft2.WH_GALAXY_DRAM_BYTES
    caps = {m: bindcraft2.max_tokens(m, card) for m in bindcraft2.MEMORY_MODES}
    assert caps["fast"] < caps["lean"] < caps["offload"], caps
    assert bindcraft2.memory_mode("auto", caps["fast"] + bindcraft2.TOKEN_BUCKET, card) == "lean"
    assert bindcraft2.memory_mode("auto", caps["lean"] + bindcraft2.TOKEN_BUCKET, card) == "offload"
    # Too big for every mode: the leanest one still runs, so the user gets the allocator's own
    # numbers at the size they asked for rather than a guess from a table.
    assert bindcraft2.memory_mode("auto", caps["offload"] * 2, card) == "offload"


def test_an_unknown_memory_mode_is_refused_by_name():
    with pytest.raises(ValueError) as caught:
        bindcraft2.memory_mode("cheap", 512, bindcraft2.WH_GALAXY_DRAM_BYTES)
    assert "cheap" in str(caught.value) and "auto" in str(caught.value)


def test_a_refusal_in_a_lean_mode_does_not_send_the_user_below_what_lean_already_clears():
    """The measured board rows are FAST-mode ladders, and a lean fold is already past them.

    A Wormhole chip's row says 512. A `lean` fold that refuses is running a size `fast` could
    never reach, so capping its way down at 512 would tell that user to drop below a size they
    have already completed -- 544 runs in `lean`, measured, at a 7.527 GB peak.
    """
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_WORMHOLE),
                                             phase="backward", n=700, padded=736, mode="lean"))
    assert "512 tokens" not in msg
    assert str(bindcraft2.max_tokens("lean", bindcraft2.WH_GALAXY_DRAM_BYTES)) in msg


def test_a_mode_ceiling_claims_exactly_the_evidence_it_has_and_no_more():
    """The number a refusal sends the user to must not claim more evidence than it has.

    Three kinds of ceiling, and the message has to say which it is holding, because the user
    resizes their job around that one figure:

    * a board's FAST-mode row is a ladder of folds that were run, so it is measured;
    * `offload` on a Wormhole chip is now measured too, on ADJACENT buckets -- 896 completes,
      928 refuses -- so the ceiling is 896 and the message may say so;
    * `lean` is not. It completes at 544 and refuses at 768, and nobody ran the buckets in
      between, so the law's estimate inside that bracket is the best available number and must
      be offered as an estimate.

    The case this pins is the second one changing. Before the `wh7` ladder the `offload` ceiling
    was the law's 960, and a researcher who trimmed to 960 would have refused again -- sent to a
    second failure by the message meant to rescue the first.
    """
    card = bindcraft2.WH_GALAXY_DRAM_BYTES
    fast = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_WORMHOLE),
                                              phase="backward", n=520, padded=544, mode="fast"))
    assert "largest axis measured" in fast, "a measured board row is still quoted as measured"

    # An estimate inside a wide bracket stays an estimate, and never exceeds the known refusal.
    lean_top, lean_measured = bindcraft2.mode_ceiling("lean", card)
    assert not lean_measured and lean_top <= 768 - bindcraft2.TOKEN_BUCKET
    lean = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_WORMHOLE), phase="backward",
                                              n=716, padded=736, mode="lean"))
    assert f"{lean_top} tokens" in lean and "estimated" in lean
    assert "measured to complete a gradient round on one" not in lean, lean

    # The adjacent bracket IS a measurement, and the message says so rather than hedging.
    off_top, off_measured = bindcraft2.mode_ceiling("offload", card)
    assert (off_top, off_measured) == (896, True)
    off = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_WORMHOLE), phase="backward",
                                             n=972, padded=992, mode="offload"))
    assert "largest axis measured to complete a gradient round" in off, off
    # 896 is the size to aim at, and it is the ONLY size offered as one that holds. 960 may still
    # appear -- the way-down sentence names it to say it refuses too, which is the message
    # working -- so what matters is that no sentence offers 960 as a size that fits.
    assert "tops out at 896 tokens" in off and "in the 'offload' mode is 896 tokens" in off, off
    for wrong in ("tops out near 960", "tops out at 960", "should hold 960", "is 960 tokens",
                  "reach 960"):
        assert wrong not in off, (wrong, off)


def test_auto_runs_lean_where_fast_was_measured_to_refuse_on_a_p150a():
    """On a p150a the law puts fast's top at 864, and fast refused 768 and 832 there, twice each,
    while lean served both. Without a measured row `auto` chose fast at both and the default
    refused a size v0.11.0 runs (832) and one it crashes on (768). With it, every axis from 768
    up runs lean, and the axes fast was measured to hold still run fast.
    """
    card = bindcraft2.P150A_DRAM_BYTES
    assert bindcraft2.mode_ceiling("fast", card) == (736, False)
    for padded in (512, 544, 736):
        assert bindcraft2.memory_mode("auto", padded, card) == "fast", padded
    for padded in (768, 800, 832, 864, 896):
        assert bindcraft2.memory_mode("auto", padded, card) == "lean", padded
    # The 896 lean served is a floor under lean, not its ceiling, so lean is not capped there.
    assert bindcraft2.max_tokens("lean", card) >= 896
    # No offload row: every offload rung on the measuring box was killed by host RAM.
    assert "offload" not in bindcraft2._MEASURED_MODE_LADDERS["p150a"]


def test_a_measured_refusal_caps_the_memory_law_everywhere_it_is_quoted():
    """`auto`, the roomier-mode offer and the ceiling sentence must not disagree.

    The law puts `offload` near 960 on a Wormhole Galaxy chip and 928 was measured to refuse, so
    a 928-token fold must not be told that `offload` will hold it. One ceiling, three readers.
    """
    card = bindcraft2.WH_GALAXY_DRAM_BYTES
    assert bindcraft2.max_tokens("offload", card) == 896
    assert bindcraft2.round_device_bytes("offload", 960) <= card * bindcraft2._MODE_USABLE, (
        "if the law no longer reaches 960 this test has stopped covering the disagreement")

    # The roomier-mode offer is the sentence a refusing fold acts on first.
    at_928 = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_WORMHOLE), phase="backward",
                                                n=908, padded=928, mode="lean"))
    assert "should hold 928 tokens" not in at_928, at_928
    # And a size offload DOES hold is still offered it.
    at_768 = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_WORMHOLE), phase="backward",
                                                n=748, padded=768, mode="lean"))
    assert "'offload' mode should hold 768 tokens" in at_768, at_768

    # `auto` decides on the same number: 896 is the last axis it sends to offload.
    assert bindcraft2.memory_mode("auto", 896, card) == "offload"
    assert bindcraft2.memory_mode("auto", 544, card) == "lean"
    assert bindcraft2.memory_mode("auto", 512, card) == "fast"


def test_a_refusal_names_the_roomier_mode_and_how_to_reach_it():
    """A refusal that does not mention the escape hatch is a user who crops their target."""
    msg = str(bindcraft2._size_aware_refusal(RuntimeError(REFUSAL_WORMHOLE),
                                             phase="backward", n=520, padded=544, mode="fast"))
    assert "'lean'" in msg
    assert "memory='auto'" in msg and "memory='lean'" in msg
    assert "docs/bindcraft2.md" in msg


def test_the_k_block_ladder_only_ever_narrows_what_the_plan_already_chose():
    """The L1 retry must start at today's width, so a call that fits is never re-planned.

    `bmm_k_blocks` is the ladder `_bmm_narrower_k` walks, and it skips its own first entry. If
    that first entry were not exactly what `bmm_program_config` picks, the retry would either
    re-try the width that just refused or quietly skip a width that fits.
    """
    from tt_bio import autograd as ag

    class _Shape:
        def __init__(self, shape):
            self.shape = shape

    for k, expected_first in ((768, 8), (128, 4), (256, 8), (32, 1), (96, 3)):
        widths = ag.bmm_k_blocks(_Shape([2, 8, 768, k]), _Shape([2, 8, k, 32]))
        kt = -(-k // 32)
        assert widths[0] == expected_first, (k, widths)
        assert widths == sorted(widths, reverse=True), (k, widths)
        assert all(kt % w == 0 for w in widths), (k, widths)


L1_REFUSAL = (
    "RuntimeError: TT_THROW @ /project/tt_metal/impl/program/program.cpp:1043: tt::exception\n"
    "info:\nStatically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=8)] grow to "
    "1856800 B which is beyond max L1 size of 1499136 B\nbacktrace:\n --- 0x1aa6601\n")


def test_an_l1_refusal_is_not_reported_as_the_card_running_out_of_memory():
    """L1 is 1.5 MB inside each Tensix and the card can be gigabytes free when it refuses.

    Read as an OOM it costs a user their target: they crop, or they reach for a slower memory
    mode, and neither touches a per-core buffer. Measured at 768 tokens in the offload mode with
    7.2 GB of DRAM free.
    """
    better = bindcraft2._l1_refusal_names_the_size(RuntimeError(L1_REFUSAL),
                                                   phase="Evoformer backward", n=738, padded=768)
    assert better is not None
    msg = str(better)
    assert "768 tokens" in msg and "738 residues" in msg
    assert "1.9 MB" in msg and "1.5 MB" in msg
    assert "NOT the card running out of memory" in msg
    assert "800" in msg                       # the next bucket, which is measured to run
    assert bindcraft2._l1_refusal_names_the_size(RuntimeError("something else"),
                                                 phase="x", n=1, padded=32) is None


def test_an_allocator_refusal_is_still_read_as_a_size_and_not_as_an_l1_clash():
    """The two wrappers must not catch each other's refusal: the remedies are opposites."""
    assert bindcraft2._l1_refusal_names_the_size(RuntimeError(REFUSAL_WORMHOLE),
                                                 phase="backward", n=520, padded=544) is None
