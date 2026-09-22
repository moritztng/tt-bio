"""The host float64 softmax is a per-site path, and it ships off at every site.

Tenstorrent's fp32 is a few mantissa bits short of IEEE fp32. On [1,16,384,384] fp32 against a
float64 softmax of the same values the op's own default reads 2.029e-02, `precise_config()`
1.646e-03 and the 5-op `_accurate_softmax` chain 5.156e-04, where real fp32 reads ~1e-7. Upstream
trains in IEEE fp32 on GPU, so a host round trip is not overshooting them, it is the only route to
what they already do. It costs a round trip per softmax and it is a training-path lever.

Four properties to hold. Nothing ships it on. Every site that owns a replaceable `ttnn.softmax`
has one, so the path cannot silently miss a site the precise lever already reaches. With the site
off, `site_softmax` is `ttnn.softmax` and nothing else, which is what makes a fold with the path
present byte-identical to one without it.

And the site being on is not enough. `TT_BIO_HOST_F64_SOFTMAX_AB` is an environment variable and
these call sites are shared with every model's inference, so the selector alone left a
training-only lever one `export` away from a user's fold, which is the second time a global flag
has been the mechanism of a shared-path regression. The implementation lives in `tt_bio.autograd`
and `site_softmax` reaches it through the hook `autograd.install` fills, so with no tape open
there is no function to reach. Pinned in both directions below: with a tape the host path runs,
without one it does not, and the entry point itself refuses rather than folding slower in silence.
"""
import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "tt_bio"

# Every construction site that resolves a softmax token, with the file that owns it. These are
# the sites whose `ttnn.softmax` this path can replace -- the same set `softmax_ckc` reaches,
# because the two levers act on the same call.
SITES = {
    "openfold3.diffusion_transformer": "tt_bio/openfold3_diffusion_transformer.py",
    "openfold3.atom_transformer": "tt_bio/openfold3_atom_transformer.py",
    "protenix.atom_transformer": "tt_bio/protenix.py",
}

ENV = "TT_BIO_HOST_F64_SOFTMAX_AB"


def _selector():
    from tt_bio.tenstorrent import host_f64_softmax_site
    return host_f64_softmax_site


def _live():
    sel = _selector()
    return {s for s in SITES if sel(s)}


def test_env_unset_ships_every_site_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert _live() == set()


def test_empty_env_ships_every_site_off(monkeypatch):
    monkeypatch.setenv(ENV, "")
    assert _live() == set()


@pytest.mark.parametrize("token", sorted(SITES))
def test_a_named_site_turns_on_only_itself(monkeypatch, token):
    monkeypatch.setenv(ENV, token)
    assert _live() == {token}


def test_all_and_minus_all_move_every_site_without_a_token_of_its_own(monkeypatch):
    monkeypatch.setenv(ENV, "all")
    assert _live() == set(SITES)
    monkeypatch.setenv(ENV, "-all")
    assert _live() == set()
    monkeypatch.setenv(ENV, "-all,openfold3.atom_transformer")
    assert _live() == {"openfold3.atom_transformer"}


def test_no_site_ships_the_path_on():
    """A `default=True` here would put a host round trip into a user's fold. There is none, and
    a flip would have to move this line as well as the call site."""
    call = re.compile(r"host_f64_softmax_site\(\s*[^)]*default\s*=\s*True")
    offenders = [str(p.relative_to(ROOT)) for p in sorted(SRC.rglob("*.py"))
                 if call.search(p.read_text())]
    assert offenders == [], "the host float64 softmax ships on at: %s" % ", ".join(offenders)


#: Sites that reach the host float64 softmax through `_fp32_softmax_attention` rather than
#: through `site_softmax`, and therefore have no per-site precise-config lever: that route takes
#: its compute kernel config from the process-wide `TT_BIO_SOFTMAX_CKC`, not from a token. Every
#: other float64 site is also a `softmax_ckc` site, and this list is the only exception the
#: anti-drift check allows.
FLOAT64_ONLY = {"af2.tri_att", "af2.msa"}


def _lever_sites():
    """The tokens each lever reaches, read off the tree.

    A token arrives two ways: written at the call (`softmax_ckc("x")`) or handed to a module that
    resolves it (`softmax_site="x"`, which `AttentionPairBias` gives to both levers and
    `TriangleAttention` to the float64 one). Reading only the first form is what let the triangle
    attention sites stay invisible to this check while D225 was open.
    """
    ckc, f64 = set(), set()
    for path in sorted(SRC.rglob("*.py")):
        t = path.read_text()
        passed = set(re.findall(r"softmax_site=\s*\"([^\"]+)\"", t))
        ckc |= set(re.findall(r"softmax_ckc\(\s*\"([^\"]+)\"", t))
        f64 |= set(re.findall(r"host_f64_softmax_site\(\s*\"([^\"]+)\"", t)) | passed
        # `AttentionPairBias` resolves BOTH levers from the token it is handed, so a token passed
        # to it counts for both. `TriangleAttention` resolves only the float64 one.
        ckc |= {tok for tok in passed if tok not in FLOAT64_ONLY}
    return ckc, f64


def test_every_precise_softmax_site_also_has_a_float64_one():
    """The anti-drift check. Where both levers exist they decide the same call, so a site that
    gained one and not the other is a site this path misses. The float64 lever now reaches
    strictly more sites than the precise one -- `_fp32_softmax_attention` has no site-level
    compute kernel config -- and every one of those is named in `FLOAT64_ONLY`, so drift in
    either direction still fails here."""
    ckc, f64 = _lever_sites()
    assert not (ckc - f64), "sites with a precise lever but no float64 one: %s" % sorted(ckc - f64)
    assert f64 - ckc == FLOAT64_ONLY, (
        "float64-only sites changed: %s, expected %s. A new one is fine, but it has to be named "
        "here with the route it is on." % (sorted(f64 - ckc), sorted(FLOAT64_ONLY)))
    assert set(SITES) <= ckc, "the site list in this file is stale: %s" % sorted(set(SITES) - ckc)


def test_the_fp32_softmax_route_gates_on_the_tape(monkeypatch):
    """The route D225 was missing, at the gate rather than at the call.

    `_fp32_softmax_attention` cannot call `site_softmax` -- its device implementations consume
    their input -- so it asks `host_softmax_or_none`. Both have to answer the same way, and the
    census has to separate a site that declined from one that asked and found no tape."""
    import tt_bio.tenstorrent as T

    before = dict(T.HOST_F64_SOFTMAX_STATS)
    assert T.host_softmax_or_none(False) is None
    assert T.HOST_F64_SOFTMAX_STATS["declined"] == before["declined"] + 1

    assert T.host_softmax_or_none(True) is None, "no tape open, so there is nothing to serve"
    assert T.HOST_F64_SOFTMAX_STATS["refused"] == before["refused"] + 1

    _fake_tape(monkeypatch, lambda x, dim: "HOST")
    assert T.host_softmax_or_none(True)("SC", -1) == "HOST"
    assert T.HOST_F64_SOFTMAX_STATS["refused"] == before["refused"] + 1


def test_a_selected_site_that_never_reaches_a_call_is_reported(monkeypatch):
    """D225's reporting gap. served/declined/refused all need the call to ARRIVE, so a selector
    wired to a branch that never consults it reads the same zeros as a model nobody ran."""
    import tt_bio.tenstorrent as T

    monkeypatch.setitem(T.HOST_F64_SOFTMAX_STATS, "selected", 0)
    monkeypatch.setitem(T.HOST_F64_SOFTMAX_STATS, "served", 0)
    monkeypatch.setitem(T.HOST_F64_SOFTMAX_STATS, "refused", 0)
    monkeypatch.setitem(T.HOST_F64_SOFTMAX_STATS, "declined", 0)
    assert "not selected" in T.host_f64_softmax_reach()

    monkeypatch.setenv(ENV, "openfold3.atom_transformer")
    assert T.host_f64_softmax_site("openfold3.atom_transformer") is True
    assert T.HOST_F64_SOFTMAX_STATS["selected"] == 1
    assert "NEVER REACHED" in T.host_f64_softmax_reach()

    T.host_softmax_or_none(True)          # the call arrives, and finds no tape
    assert "NEVER REACHED" not in T.host_f64_softmax_reach()
    assert "reached" in T.host_f64_softmax_reach()


def test_no_wired_site_still_calls_ttnn_softmax_directly():
    """Every call that takes `self._softmax_ckc` must go through `site_softmax`, or the site has
    a float64 selector that decides nothing."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "ttnn.softmax(" in line and "_softmax_ckc" in line:
                offenders.append("%s:%d" % (path.relative_to(ROOT), i))
    assert offenders == [], "ttnn.softmax still called directly at: %s" % ", ".join(offenders)


def test_site_softmax_off_is_ttnn_softmax_and_nothing_else(monkeypatch):
    """The byte-identity argument in one assertion: off, the wrapper forwards every argument it
    was given to `ttnn.softmax` and returns its result unchanged."""
    import tt_bio.tenstorrent as T

    seen = {}

    def fake(x, **kw):
        seen["x"], seen["kw"] = x, kw
        return "OUT"

    monkeypatch.setattr(T.ttnn, "softmax", fake)
    out = T.site_softmax("SC", dim=-1, numeric_stable=True, compute_kernel_config="CKC",
                         host_f64=False)
    assert out == "OUT"
    assert seen == {"x": "SC", "kw": {"dim": -1, "numeric_stable": True,
                                      "compute_kernel_config": "CKC"}}


def _fake_tape(monkeypatch, fn):
    """Fill the hook slot `autograd.install` fills, without importing the tape."""
    import tt_bio.ops as ops
    monkeypatch.setattr(ops, "host_softmax_hook", lambda: fn)


def test_site_softmax_on_under_a_tape_does_not_reach_ttnn_softmax(monkeypatch):
    import tt_bio.tenstorrent as T

    def fake(*a, **k):
        raise AssertionError("ttnn.softmax ran with the host float64 path selected")

    monkeypatch.setattr(T.ttnn, "softmax", fake)
    _fake_tape(monkeypatch, lambda x, dim=-1: ("HOST", x, dim))
    assert T.site_softmax("SC", dim=-1, compute_kernel_config="CKC",
                          host_f64=True) == ("HOST", "SC", -1)


def test_site_softmax_on_without_a_tape_is_still_ttnn_softmax(monkeypatch):
    """The safety property. A person can set `TT_BIO_HOST_F64_SOFTMAX_AB` and these sites are
    every model's inference, so the selector firing with no tape open must leave the call
    exactly what it was: `ttnn.softmax`, the caller's own arguments, nothing dropped."""
    import tt_bio.tenstorrent as T

    seen = {}

    def fake(x, **kw):
        seen["x"], seen["kw"] = x, kw
        return "OUT"

    monkeypatch.setattr(T.ttnn, "softmax", fake)
    before = dict(T.HOST_F64_SOFTMAX_STATS)
    out = T.site_softmax("SC", dim=-1, numeric_stable=True, compute_kernel_config="CKC",
                         host_f64=True)
    assert out == "OUT"
    assert seen == {"x": "SC", "kw": {"dim": -1, "numeric_stable": True,
                                      "compute_kernel_config": "CKC"}}
    assert T.HOST_F64_SOFTMAX_STATS["refused"] == before["refused"] + 1
    assert T.HOST_F64_SOFTMAX_STATS["served"] == before["served"]


def test_the_hook_slot_is_empty_until_a_tape_installs_it():
    """`site_softmax` gates on `ops.host_softmax_hook()`, and an inference process never fills
    it. Reading it must not import the tape either, or the gate would drag in what it exists to
    keep out."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, tt_bio.ops as ops; "
         "print(ops.host_softmax_hook(), 'tt_bio.autograd' in sys.modules)"],
        capture_output=True, text=True, cwd=str(ROOT))
    if out.returncode != 0:
        pytest.skip("importing tt_bio.ops needs ttnn")
    assert out.stdout.strip() == "None False", out.stdout.strip()


def test_the_host_path_refuses_an_inference_tensor():
    """The entry point states the rule once more for a caller that found it some other way. It
    costs a host round trip per softmax and returns different numbers, so reaching it without a
    tape is a mistake in either direction and it says so."""
    ag = pytest.importorskip("tt_bio.autograd")
    assert not ag.installed(), "a tape was left installed by an earlier test"
    with pytest.raises(RuntimeError, match="needs an open tape"):
        ag.host_f64_softmax(object(), -1)


def test_install_fills_the_slot_and_uninstall_empties_it():
    """The other direction, so the gate cannot be satisfied by a hook nobody ever installs."""
    ag = pytest.importorskip("tt_bio.autograd")
    import tt_bio.ops as ops

    assert ops.host_softmax_hook() is None
    prev = ag.install()
    try:
        assert ops.host_softmax_hook() is ag.host_f64_softmax
    finally:
        ag.uninstall()
        ops.set_grad_hook(prev)
    assert ops.host_softmax_hook() is None


def test_the_census_counts_all_three_answers(monkeypatch):
    """A lever that fires and is inert is the failure mode this campaign keeps meeting, so the
    path publishes every answer rather than leaving any of them invisible. `refused` is the one
    that matters most: it is how "the flag was set and the tape was not open" reads as a number
    instead of as a fold that quietly did the ordinary thing."""
    import tt_bio.tenstorrent as T

    monkeypatch.setattr(T.ttnn, "softmax", lambda x, **kw: "OUT")
    before = dict(T.HOST_F64_SOFTMAX_STATS)
    T.site_softmax("SC", host_f64=False)
    T.site_softmax("SC", host_f64=True)
    _fake_tape(monkeypatch, lambda x, dim=-1: "HOST")
    assert T.site_softmax("SC", host_f64=True) == "HOST"
    assert T.HOST_F64_SOFTMAX_STATS["declined"] == before["declined"] + 1
    assert T.HOST_F64_SOFTMAX_STATS["refused"] == before["refused"] + 1


def test_a_backward_recompute_still_takes_the_host_path():
    """The gate must not be tighter than the forward it protects.

    `tape()` restores the grad hook on the way out, and a checkpointed segment recomputes its
    forward from inside `backward`, which runs after the block has closed. `recompute_scope`
    puts the hook back for exactly that reason. If the gate said no there, the recomputed
    forward would run a different softmax from the taped one and the gradient would be taken
    on values the forward never produced.
    """
    ag = pytest.importorskip("tt_bio.autograd")
    import tt_bio.ops as ops
    from tt_bio.taped_ttnn import recompute_scope

    prev = ag.install()
    try:
        ops.set_grad_hook(prev)                  # what `tape()` does on the way out
        assert ops.host_softmax_hook() is None, "the slot alone must not open the path"
        with recompute_scope():
            assert ops.host_softmax_hook() is ag.host_f64_softmax
        assert ops.host_softmax_hook() is None
    finally:
        ag.uninstall()
        ops.set_grad_hook(prev)


def test_the_backward_reads_the_float64_forward_not_the_device_copy():
    """The whole point of the path. Taking the Jacobian on the rounded output that went back to
    the card would put a device-precision softmax straight back into the gradient."""
    src = (ROOT / "tt_bio/autograd.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "host_f64_softmax")
    bw = next(n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n.name == "bw")
    names = {n.id for n in ast.walk(bw) if isinstance(n, ast.Name)}
    assert "y64" in names, "the backward does not read the float64 forward output"


# --- the exact softmax installed from the package -----------------------------------------
#
# The site selector above is one of two ways to reach this arithmetic, and it is the one an
# inference fold shares a call site with. The other is `autograd.install(exact_softmax=True)`,
# which replaces the taped `softmax` verb and `ttnn.softmax` itself for the life of the
# install. It reads 0.4175214198121818 against the float64 reference at crop 384 where the
# device softmax reads 0.702981502944001, and the reason it is safe to have at all is
# structural rather than a matter of what the default is: an inference fold never calls the
# training entry points. These pin that structure, and the reach the number depends on.


def _shim():
    tt = pytest.importorskip("tt_bio.taped_ttnn")
    return tt, tt.taped_ttnn()


def test_the_exact_softmax_ships_off(monkeypatch):
    """`install()` on its own must not touch a softmax. The lever costs a host round trip per
    call and returns different numbers; it is asked for or it is not there."""
    ag = pytest.importorskip("tt_bio.autograd")
    import ttnn

    before = ttnn.softmax
    prev = ag.install()
    try:
        assert not ag.exact_softmax_installed()
        assert ttnn.softmax is before
    finally:
        ag.uninstall()
        import tt_bio.ops as ops
        ops.set_grad_hook(prev)


def test_install_and_uninstall_put_back_the_objects_they_replaced():
    """Identity, not equality. A teardown that rebuilds what it thinks the original was leaves
    a second definition of the shipped softmax in the process."""
    ag = pytest.importorskip("tt_bio.autograd")
    tt, _ = _shim()
    import ttnn

    was = {"raw": ttnn.softmax, "raw_ip": ttnn.softmax_in_place,
           "verb": tt._VERBS["softmax"], "verb_ip": tt._VERBS["softmax_in_place"]}
    ag.install(exact_softmax=True)
    try:
        assert ag.exact_softmax_installed()
        assert ttnn.softmax is ag._exact_softmax_raw
        assert ttnn.softmax_in_place is ag._exact_softmax_raw
        assert tt._VERBS["softmax"] is ag._v_exact_softmax
        assert tt._VERBS["softmax_in_place"] is ag._v_exact_softmax
    finally:
        ag.uninstall()
    assert not ag.exact_softmax_installed()
    assert ttnn.softmax is was["raw"] and ttnn.softmax_in_place is was["raw_ip"]
    assert tt._VERBS["softmax"] is was["verb"]
    assert tt._VERBS["softmax_in_place"] is was["verb_ip"]


def test_the_context_manager_is_the_same_switch():
    """`exact_softmax()` exists for the SCOPE, not for different behaviour: the backward
    recomputes after the tape block has closed, so a lever scoped to the tape would leave every
    recomputed softmax on the card."""
    ag = pytest.importorskip("tt_bio.autograd")
    import ttnn

    before = ttnn.softmax
    with ag.exact_softmax():
        assert ttnn.softmax is ag._exact_softmax_raw
        with ag.exact_softmax():            # idempotent, and the inner block does not undo it
            assert ttnn.softmax is ag._exact_softmax_raw
        assert ttnn.softmax is ag._exact_softmax_raw
    assert ttnn.softmax is before


def test_the_two_reaches_are_two_different_entry_points():
    """The verb and the raw op are separate claims and the counters keep them separate.

    The verb makes the forward AND the Jacobian exact; it cannot reach
    `autograd.triangle_attention._scores`, which calls `ttnn.softmax` directly from a module
    `taped_ttnn._NEVER_SHIM` excludes. With the verb alone the pair track runs on the card, and
    `raw` reading 0 is how that says so.
    """
    ag = pytest.importorskip("tt_bio.autograd")
    tt, _ = _shim()

    assert ag._v_exact_softmax is not ag._exact_softmax_raw
    assert set(ag.EXACT_SOFTMAX_STATS) == {"verb", "raw", "raw_elements"}
    # `_scores` calls the raw op, so the raw half is the only one that can reach it.
    src = (SRC / "autograd.py").read_text(errors="replace")
    body = src[src.index("def triangle_attention("):]
    body = body[:body.index("\ndef ", 1)]
    assert "ttnn.softmax(" in body, (
        "triangle_attention no longer calls ttnn.softmax directly; the module-wide half of the "
        "install was justified by that call and needs re-deriving")
    assert "tt_bio.autograd" in tt._NEVER_SHIM


def test_forget_shim_bindings_makes_a_verb_swap_visible():
    """`_Ttnn.__getattr__` caches a closure that captured the `_VERBS` entry AND the shipped
    callable, and `_SHIM` is a module singleton that outlives a `tape()`. So a lever installed
    after anything has gone through the shim once reads as perfectly inert -- a failure mode a
    patch applied at process start never meets, and the reason the package install needs this
    and `dev_cot.py` did not."""
    ag = pytest.importorskip("tt_bio.autograd")
    tt, shim = _shim()

    getattr(shim, "softmax")
    assert "softmax" in shim.__dict__, "the shim is expected to cache its bindings"

    on_tape = ag.Tensor.__new__(ag.Tensor)      # `_on_tape` asks isinstance and nothing else
    saved = tt._VERBS["softmax"]
    try:
        tt._VERBS["softmax"] = lambda shipped, args, kwargs: "SWAPPED"
        assert "softmax" in shim.__dict__, "the swap alone is invisible, which is the point"
        tt.forget_shim_bindings("softmax")
        assert "softmax" not in shim.__dict__
        assert shim.softmax(on_tape) == "SWAPPED"
    finally:
        tt._VERBS["softmax"] = saved
        tt.forget_shim_bindings("softmax")


def test_the_exact_softmax_install_drops_the_cached_bindings():
    """The install above, wired: the same invalidation, done for it rather than by its caller."""
    ag = pytest.importorskip("tt_bio.autograd")
    tt, shim = _shim()

    for n in ("softmax", "softmax_in_place"):
        getattr(shim, n)
        assert n in shim.__dict__
    with ag.exact_softmax():
        for n in ("softmax", "softmax_in_place"):
            assert n not in shim.__dict__, (
                f"`{n}` kept its pre-install binding; every call site that had already used it "
                f"would take the device softmax and the arm would read as inert")


def test_no_inference_fold_has_a_route_to_the_exact_softmax():
    """Moritz's hard constraint, argued structurally rather than by a default.

    The site selector is an environment variable read on call sites shared with every model's
    inference (`tt-bio-shared-diffusion-global-env-default-regression` is what that shape did
    last time). This install has no environment variable and no inference caller: the only
    things that turn it on are `autograd.install` and `autograd.exact_softmax`, and
    `tests/test_training_opt_in.py::test_no_inference_module_imports_training` is what keeps
    the inference modules from importing that module at all.
    """
    callers = {}
    for path in sorted(SRC.rglob("*.py")):
        if "_vendor" in path.parts or path.name in ("autograd.py", "taped_ttnn.py"):
            continue
        text = path.read_text(errors="replace")
        hits = sorted({m for m in ("exact_softmax", "_install_exact_softmax",
                                   "_exact_softmax_raw", "_v_exact_softmax")
                       if m in text})
        if hits:
            callers[str(path.relative_to(SRC))] = hits
    assert not callers, (
        f"the exact softmax is named outside the training stack: {callers}. It must stay "
        f"reachable only from the training entry points, or it is a global flag again.")
    # And no environment variable of its own, which is the mechanism being avoided.
    src = (SRC / "autograd.py").read_text(errors="replace")
    body = src[src.index("EXACT_SOFTMAX_STATS ="):src.index("def mul(")]
    assert "environ" not in body and "env_flag" not in body, (
        "the exact softmax grew an environment variable; that is the shape the site selector "
        "already has and the reason this install exists")


def test_an_unrelated_uninstall_does_not_remove_the_exact_softmax():
    """`uninstall()` is not this lever's private teardown, and two callers already use
    `install()`/`uninstall()` as a scoped pair around something far narrower than a step:
    `train/lora.py:608-615` brackets the DISCOVERY forward and `train/recipes.py:211` the fit.

    Tearing down on ANY `uninstall()` made the lever collateral damage of whichever pair closed
    first. Measured, not hypothesised: the packaged arm came back 0.702981502944001 -- the
    device-softmax control to sixteen digits -- with all 1,742 of its exact softmaxes spent in
    the discovery forward and none in the step that was scored.
    """
    ag = pytest.importorskip("tt_bio.autograd")
    import tt_bio.ops as ops
    import ttnn

    with ag.exact_softmax():
        assert ttnn.softmax is ag._exact_softmax_raw
        # Exactly what `walked_weights` does around the discovery forward.
        prev = ops.grad_hook()
        ag.install()
        try:
            assert ttnn.softmax is ag._exact_softmax_raw
        finally:
            ag.uninstall()
        assert ttnn.softmax is ag._exact_softmax_raw, (
            "an unrelated install/uninstall pair removed the exact softmax")
        ops.set_grad_hook(prev)
    assert ttnn.softmax is not ag._exact_softmax_raw


def test_install_exact_softmax_is_still_undone_by_its_own_uninstall():
    """The other direction, so the ownership fix does not simply leak the lever."""
    ag = pytest.importorskip("tt_bio.autograd")
    import tt_bio.ops as ops
    import ttnn

    was = ttnn.softmax
    prev = ag.install(exact_softmax=True)
    try:
        assert ttnn.softmax is ag._exact_softmax_raw
    finally:
        ag.uninstall()
        ops.set_grad_hook(prev)
    assert ttnn.softmax is was
    assert not ag.exact_softmax_installed()
