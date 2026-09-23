"""Softmax and layer norm run exact on every training tape, gated on the tape itself.

of3t-stackexact measured the model-frame trunk gradient at 0.9822570327981535x the bar with
both exact against 1.4511706984958472x on the device ops, and of3t-stackship made that the
default. These pin the gate: `tape()` and `backward()` open the scope for their own extent,
put back exactly what they replaced, leave an outer owner's install alone, and
`exact_training(False)` turns it off. No environment variable, and no route from inference.
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


def test_tape_runs_softmax_and_layer_norm_exact_and_puts_them_back():
    ag, tt, ttnn = _mods()
    was = (ttnn.softmax, ttnn.layer_norm, tt._VERBS["layer_norm"], ag._TAPED["layer_norm"])
    assert _state(ag, tt, ttnn) == _all(False)
    with ag.tape():
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
    ag.backward([], [])
    assert seen == [_all(True)]
    assert _state(ag, tt, ttnn) == _all(False)


def test_exact_training_false_is_the_off_switch(monkeypatch):
    ag, tt, ttnn = _mods()
    seen = []
    monkeypatch.setattr(ag, "_backward", lambda roots, seeds: seen.append(_state(ag, tt, ttnn)))
    with ag.exact_training(False):
        assert ag.exact_training_ops() == ()
        with ag.tape():
            assert _state(ag, tt, ttnn) == _all(False)
        ag.backward([], [])
        with ag.exact_training(True):         # innermost block wins
            assert ag.exact_training_ops() == ag.EXACT_TRAINING_OPS
    assert seen == [_all(False)]
    assert ag.exact_training_ops() == ("softmax", "layer_norm")


def test_an_outer_install_is_left_to_its_owner():
    """A caller that armed the exact softmax for a longer scope keeps it after a tape closes,
    and a harness that replaced the taped layer norm gets its own verb back, not the shipped
    one -- the tape restores what it found, not what it thinks was there."""
    ag, tt, ttnn = _mods()
    mine = lambda shipped, args, kwargs: None
    saved = (tt._VERBS["layer_norm"], ag._TAPED["layer_norm"])
    tt._VERBS["layer_norm"] = ag._TAPED["layer_norm"] = mine
    try:
        with ag.exact_softmax():
            with ag.tape():
                assert _state(ag, tt, ttnn) == _all(True)
            assert ttnn.softmax is ag._exact_softmax_raw
            assert tt._VERBS["layer_norm"] is mine and ag._TAPED["layer_norm"] is mine
        assert ttnn.softmax is not ag._exact_softmax_raw
    finally:
        tt._VERBS["layer_norm"], ag._TAPED["layer_norm"] = saved


def test_install_alone_arms_nothing():
    """`install()` routes `tt_bio.ops` through the hook; it is what `walked_weights` and the
    recipe bracket discovery and the fit with. Only a tape or a backward runs exact."""
    ag, tt, ttnn = _mods()
    import tt_bio.ops as ops
    prev = ag.install()
    try:
        assert _state(ag, tt, ttnn) == _all(False)
    finally:
        ag.uninstall()
        ops.set_grad_hook(prev)


def test_no_environment_variable_and_no_inference_route():
    """Moritz's hard constraint: the float64 ops are training-only. A global env default read
    on shared call sites is how `tt-bio-shared-diffusion-global-env-default-regression`
    happened; this lever has no variable, and nothing outside the tape names it."""
    src = (SRC / "autograd.py").read_text(errors="replace")
    body = src[src.index("EXACT_SOFTMAX_STATS ="):src.index("def mul(")]
    assert "environ" not in body and "env_flag" not in body
    names = ("_exact_layer_norm_raw", "_v_exact_layer_norm", "_training_exact",
             "EXACT_LAYER_NORM_STATS", "_install_exact")
    callers = {}
    for path in sorted(SRC.rglob("*.py")):
        if "_vendor" in path.parts or path.name in ("autograd.py", "taped_ttnn.py"):
            continue
        hits = sorted(n for n in names if n in path.read_text(errors="replace"))
        if hits:
            callers[str(path.relative_to(SRC))] = hits
    assert not callers, f"the exact training ops are named outside the tape: {callers}"
