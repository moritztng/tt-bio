"""The float64 softmax and layer norm are off by default and scoped to the tape when on.

of3t-stackexact measured the model-frame trunk gradient at 0.9822570327981535x the bar with
both exact against 1.4511706984958472x on the device ops, and of3t-stackship made that the
default. of3t-p10exact then cleared the same bar on the device alone (0.99736x) with the fp32
softmax backward, and of3t-p10default turned the instrument off. These pin the default and the
gate: under `exact_training(True)`, `install()` opens it until its `uninstall()`, `tape()` and
`backward()` open it for their own extent, put back exactly what they replaced, and leave an
outer owner's install alone. No environment variable, and no route from inference.
"""
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "tt_bio"


def _mods():
    ag = pytest.importorskip("tt_bio.autograd")
    tt = pytest.importorskip("tt_bio.taped_ttnn")
    import ttnn
    return ag, tt, ttnn


def _state(ag, tt, ttnn):
    return {"softmax": ttnn.softmax is ag._exact_softmax_raw,
            "softmax_verb": tt._VERBS["softmax"] is ag._v_exact_softmax,
            "layer_norm": ttnn.layer_norm is ag._exact_layer_norm_raw,
            "layer_norm_verb": tt._VERBS["layer_norm"] is ag._v_exact_layer_norm,
            "layer_norm_hook": ag._TAPED["layer_norm"] is ag._v_exact_layer_norm}


def _all(v):
    return {k: v for k in ("softmax", "softmax_verb", "layer_norm", "layer_norm_verb",
                           "layer_norm_hook")}


def test_the_default_is_the_device_path(monkeypatch):
    """A tape and a backward opened with no switch leave softmax and layer norm on the device,
    and the fp32 softmax backward that makes that path accurate is on."""
    ag, tt, ttnn = _mods()
    seen = []
    monkeypatch.setattr(ag, "_backward", lambda roots, seeds: seen.append(_state(ag, tt, ttnn)))
    assert ag.exact_training_ops() == ()
    with ag.tape():
        assert _state(ag, tt, ttnn) == _all(False)
    ag.backward([], [])
    assert seen == [_all(False)]
    assert ag.SOFTMAX_BW_FP32 is True


def test_tape_runs_softmax_and_layer_norm_exact_and_puts_them_back():
    ag, tt, ttnn = _mods()
    was = (ttnn.softmax, ttnn.layer_norm, tt._VERBS["layer_norm"], ag._TAPED["layer_norm"])
    assert _state(ag, tt, ttnn) == _all(False)
    with ag.exact_training(True), ag.tape():
        assert _state(ag, tt, ttnn) == _all(True)
        with ag.tape():                       # nested: the inner block changes nothing
            assert _state(ag, tt, ttnn) == _all(True)
        assert _state(ag, tt, ttnn) == _all(True)
    assert (ttnn.softmax, ttnn.layer_norm, tt._VERBS["layer_norm"],
            ag._TAPED["layer_norm"]) == was


def test_backward_runs_exact_for_its_own_extent(monkeypatch):
    """The backward runs after the tape block has closed and recomputes checkpointed blocks
    and `triangle_attention`'s scores, so it opens the scope itself."""
    ag, tt, ttnn = _mods()
    seen = []
    monkeypatch.setattr(ag, "_backward", lambda roots, seeds: seen.append(_state(ag, tt, ttnn)))
    with ag.exact_training(True):
        ag.backward([], [])
    assert seen == [_all(True)]
    assert _state(ag, tt, ttnn) == _all(False)


def test_exact_training_false_inside_true_is_the_off_switch(monkeypatch):
    ag, tt, ttnn = _mods()
    seen = []
    monkeypatch.setattr(ag, "_backward", lambda roots, seeds: seen.append(_state(ag, tt, ttnn)))
    with ag.exact_training(True):
        with ag.exact_training(False):        # innermost block wins
            assert ag.exact_training_ops() == ()
            with ag.tape():
                assert _state(ag, tt, ttnn) == _all(False)
            ag.backward([], [])
        assert ag.exact_training_ops() == ("softmax", "layer_norm")
    assert seen == [_all(False)]
    assert ag.exact_training_ops() == ()


def test_an_outer_install_is_left_to_its_owner():
    """A caller that armed the exact softmax for a longer scope keeps it after a tape closes,
    and a harness that replaced the taped layer norm gets its own verb back, not the shipped
    one -- the tape restores what it found, not what it thinks was there."""
    ag, tt, ttnn = _mods()
    mine = lambda shipped, args, kwargs: None
    saved = (tt._VERBS["layer_norm"], ag._TAPED["layer_norm"])
    tt._VERBS["layer_norm"] = ag._TAPED["layer_norm"] = mine
    try:
        with ag.exact_softmax(), ag.exact_training(True):
            with ag.tape():
                assert _state(ag, tt, ttnn) == _all(True)
            assert ttnn.softmax is ag._exact_softmax_raw
            assert tt._VERBS["layer_norm"] is mine and ag._TAPED["layer_norm"] is mine
        assert ttnn.softmax is not ag._exact_softmax_raw
    finally:
        tt._VERBS["layer_norm"], ag._TAPED["layer_norm"] = saved


def test_install_is_the_gate_and_pairs_by_nesting():
    """`install()` is what `walked_weights` brackets the discovery forward with and the recipe
    brackets the fit with, and the step depends on what discovery ran: with discovery on the
    device ops the trunk gradient came back 1958 of 2736 tensors off the clearing arm. An inner
    install/uninstall pair must not disarm what the outer one armed."""
    ag, tt, ttnn = _mods()
    import tt_bio.ops as ops
    with ag.exact_training(True):
        prev = ag.install()
        try:
            assert _state(ag, tt, ttnn) == _all(True)
            ag.install()
            ag.uninstall()
            assert _state(ag, tt, ttnn) == _all(True), "an inner pair disarmed the outer install"
            with ag.tape():
                pass
            assert _state(ag, tt, ttnn) == _all(True), "a tape closing disarmed the install"
        finally:
            ag.uninstall()
            ops.set_grad_hook(prev)
    assert _state(ag, tt, ttnn) == _all(False)
    ag.uninstall()                            # a bare uninstall is still harmless
    assert _state(ag, tt, ttnn) == _all(False)


def test_tape_does_not_leave_an_install_behind():
    """`tape()` re-arms the hooks on entry. If it went through `install()` it would push an
    exact scope nobody pops, and every later fold in the process would run exact."""
    ag, tt, ttnn = _mods()
    depth = len(ag._INSTALL_EXACT)
    with ag.exact_training(True), ag.tape():
        pass
    assert len(ag._INSTALL_EXACT) == depth
    assert _state(ag, tt, ttnn) == _all(False)


def test_no_environment_variable_and_no_inference_route():
    """Moritz's hard constraint: the float64 ops are training-only. A global env default read
    on shared call sites is how `tt-bio-shared-diffusion-global-env-default-regression`
    happened; this lever has no variable, and nothing outside the tape names it."""
    src = (SRC / "autograd.py").read_text(errors="replace")
    body = src[src.index("EXACT_SOFTMAX_STATS ="):src.index("def mul(")]
    assert "environ" not in body and "env_flag" not in body
    names = ("_exact_layer_norm_raw", "_v_exact_layer_norm", "_training_exact",
             "EXACT_LAYER_NORM_STATS", "_install_exact", "_INSTALL_EXACT", "_install_hooks")
    callers = {}
    for path in sorted(SRC.rglob("*.py")):
        if "_vendor" in path.parts or path.name in ("autograd.py", "taped_ttnn.py"):
            continue
        hits = sorted(n for n in names if n in path.read_text(errors="replace"))
        if hits:
            callers[str(path.relative_to(SRC))] = hits
    assert not callers, f"the exact training ops are named outside the tape: {callers}"
