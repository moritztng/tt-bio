"""The accurate-softmax lever is per-site, and only measured sites ship it on.

`ttnn.softmax` normalises against a denominator its own numerators do not match (rows sum to
0.977 on [1,16,512,512] fp32), and `_accurate_softmax` replaces it with a 5-op chain that costs
up to 4.22x on the op. Whether a given model should pay that is a per-model, per-site verdict.
Triangle attention is its own site (`rf3.tri_att`) because it was not always one: widening
`accurate_softmax` onto `TriangleAttention` activated RF3's years-older opt-in for
AttentionPairBias at the biggest softmax site in the stack, costing 1.376x on the published
512 aa cell. A parameter that reaches two sites under one name cannot express that verdict.
Protenix-v2 and OpenDDE have theirs: both cleared 20/256/512/768 aa with the chain on, worst
reading +2.0% against a +-15% band. Nothing has scored the ESMFold2 or OpenFold3 sites, so those
still ship off.

Two properties to hold. The split above is what ships, and no unmeasured site joins it by
accident. And every site stays individually overridable after it ships, in both directions,
because `protenix.*` and `opendde.*` name the SAME two code sites and a switch they share is the
shape that cost OpenDDE 60x once already.
"""
import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "tt_bio"

# Every construction site that owns a ttnn.softmax the lever can replace, with the model that
# owns it. protenix.* and opendde.* name the SAME two code sites: OpenDDE instantiates
# tt_bio.protenix.Protenix, so the scope is what keeps the two models' verdicts separable.
SITES = {
    "esmfold2.token_transformer": "tt_bio/esmfold2.py",
    "openfold3.trunk": "tt_bio/openfold3_trunk.py",
    "openfold3.confidence": "tt_bio/openfold3_confidence.py",
    "openfold3.template": "tt_bio/openfold3_template.py",
    "openfold3.msa": "tt_bio/openfold3_msa_embedder.py",
    "protenix.trunk": "tt_bio/protenix.py",
    "protenix.confidence": "tt_bio/protenix.py",
    "opendde.trunk": "tt_bio/protenix.py",
    "opendde.confidence": "tt_bio/protenix.py",
    "opendde.refiner": "tt_bio/opendde.py",
    "rf3.tri_att": "tt_bio/rf3/remap.py",
}

# The sites whose verdict landed, i.e. the ones passing `default=True`. protenix.py's two sites
# are scoped, so flipping them flips four tokens; opendde.py's refiner is the fifth.
ON_BY_DEFAULT = {
    "protenix.trunk", "protenix.confidence",
    "opendde.trunk", "opendde.confidence", "opendde.refiner",
}
OFF_BY_DEFAULT = set(SITES) - ON_BY_DEFAULT


def _selector():
    from tt_bio.tenstorrent import accurate_softmax_site
    return accurate_softmax_site


def _live():
    """The answer each site gets as it is constructed, under the current environment.

    The default is an argument at the construction site, not a table inside the selector, so a
    test has to supply it. `test_every_site_ships_the_default_this_file_declares` is what keeps
    ON_BY_DEFAULT honest against the source.
    """
    sel = _selector()
    return {s for s in SITES if sel(s, default=s in ON_BY_DEFAULT)}


def test_env_unset_ships_the_measured_sites_on_and_nothing_else(monkeypatch):
    monkeypatch.delenv("TT_BIO_ACCURATE_SOFTMAX_AB", raising=False)
    assert _live() == ON_BY_DEFAULT


def test_empty_env_ships_the_measured_sites_on_and_nothing_else(monkeypatch):
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "")
    assert _live() == ON_BY_DEFAULT


def test_a_named_site_turns_on_only_itself(monkeypatch):
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "openfold3.trunk")
    assert _live() == ON_BY_DEFAULT | {"openfold3.trunk"}


@pytest.mark.parametrize("token", sorted(ON_BY_DEFAULT))
def test_a_shipped_site_can_still_be_forced_off_alone(monkeypatch, token):
    """The reason the selector outlives the flip: a shipped site stays A/B-able."""
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "-" + token)
    assert _live() == ON_BY_DEFAULT - {token}


def test_all_and_minus_all_move_every_site_without_a_token_of_its_own(monkeypatch):
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "all")
    assert _live() == set(SITES)
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "-all")
    assert _live() == set()
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "-all,openfold3.msa")
    assert _live() == {"openfold3.msa"}


def test_protenix_and_opendde_do_not_drag_each_other(monkeypatch):
    """The shared-construction-site trap. protenix.py builds BOTH models' Pairformers, so an
    override on one must not move the other, in either direction."""
    sel = _selector()
    on = lambda t: sel(t, default=t in ON_BY_DEFAULT)

    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "-protenix.trunk,-protenix.confidence")
    assert not on("protenix.trunk") and not on("protenix.confidence")
    assert on("opendde.trunk") and on("opendde.confidence") and on("opendde.refiner")

    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB",
                       "-opendde.trunk,-opendde.confidence,-opendde.refiner")
    assert on("protenix.trunk") and on("protenix.confidence")
    assert not on("opendde.trunk") and not on("opendde.confidence") and not on("opendde.refiner")


@pytest.mark.parametrize("token", sorted(SITES))
def test_site_is_reachable_and_wired_in_the_named_file(monkeypatch, token):
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", token)
    assert _selector()(token), "%s is not reachable through the selector" % token
    src = (ROOT / SITES[token]).read_text()
    site = token.split(".", 1)[1]
    tail = ", default=True)" if token in ON_BY_DEFAULT else ")"
    literal = 'accurate_softmax_site("%s"%s' % (token, tail)
    scoped = 'accurate_softmax_site(f"{softmax_scope}.%s"%s' % (site, tail)
    assert literal in src or scoped in src, "%s: no site wired with default=%s in %s" % (
        token, token in ON_BY_DEFAULT, SITES[token])


# The token expression each shipped-on site passes, as it is written in the source. Scoped ones
# are one expression covering two models, which is exactly why they are listed by expression.
SHIPPED_ON_EXPRESSIONS = {
    'f"{softmax_scope}.trunk"',
    'f"{softmax_scope}.confidence"',
    '"opendde.refiner"',
}


def test_every_site_ships_the_default_this_file_declares():
    """No unmeasured site joins the flip. The set of `default=True` call sites in tt_bio has to
    be exactly the ones ON_BY_DEFAULT names, or this file is describing code that moved."""
    found = set()
    call = re.compile(r"accurate_softmax_site\(\s*(f?\"[^\"]*\")\s*,\s*default=True\s*\)")
    for path in sorted(SRC.rglob("*.py")):
        found |= set(call.findall(path.read_text()))
    assert found == SHIPPED_ON_EXPRESSIONS, \
        "sites shipping the lever on changed: %s" % sorted(found)


# RF3 ships the lever on with a literal True, and that is the one place a literal is allowed. Its
# own port scored it: the row deficit is the whole of AttentionPairBias's 13.43x on RF3's
# pairformer, and PAIRFORMER_FLAGS reaches only RF3's own stack. Everywhere else the answer goes
# through accurate_softmax_site(), so a site keeps a name and an override even once it ships on.
LEVER_ON_ALLOWED = {"tt_bio/rf3/remap.py"}


def test_no_shared_construction_site_hardcodes_the_lever_on():
    """A flip moves a `default=`, it does not bypass the selector with a literal True. An
    unnamed site cannot be forced back off, and that is how a shared default becomes a 60x."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel in LEVER_ON_ALLOWED:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"accurate_softmax\s*=\s*True", line):
                offenders.append("%s:%d" % (rel, i))
    assert offenders == [], "accurate_softmax hardcoded ON at: %s" % ", ".join(offenders)


def test_the_rf3_allowance_stays_inside_rf3s_own_flag_dict():
    """The allowlist is not a blanket: exactly one True, and it is in `PAIRFORMER_FLAGS`."""
    src = (ROOT / "tt_bio/rf3/remap.py").read_text()
    hits = re.findall(r"accurate_softmax\s*=\s*True", src)
    assert len(hits) == 1, "RF3 hardcodes the lever %d times, expected 1" % len(hits)
    tree = ast.parse(src)
    flags = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(getattr(t, "id", None) == "PAIRFORMER_FLAGS" for t in n.targets)]
    assert len(flags) == 1, "PAIRFORMER_FLAGS is not a single module-level assignment"
    kwargs = {k.arg: k.value for k in flags[0].value.keywords}
    on = kwargs.get("accurate_softmax")
    assert isinstance(on, ast.Constant) and on.value is True, \
        "RF3's True is not the one inside PAIRFORMER_FLAGS"


def test_shared_primitives_default_the_keyword_off():
    """Every function/class that accepts the keyword must default it False, so the decision
    stays at the construction site and no model inherits another model's verdict."""
    bad = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            names = [a.arg for a in args.args + args.kwonlyargs]
            if "accurate_softmax" not in names:
                continue
            pos_defaults = dict(zip([a.arg for a in args.args][-len(args.defaults):] if args.defaults else [],
                                    args.defaults))
            kw_defaults = {a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults)}
            default = pos_defaults.get("accurate_softmax", kw_defaults.get("accurate_softmax"))
            if not (isinstance(default, ast.Constant) and default.value is False):
                bad.append("%s:%d %s" % (path.relative_to(ROOT), node.lineno, node.name))
    assert bad == [], "accurate_softmax not defaulted False at: %s" % ", ".join(bad)


# --- triangle attention is a site of its own -------------------------------------------------
# `PairformerLayer`/`Pairformer` take `tri_att_accurate_softmax`, and it inherits
# `accurate_softmax` when left as `None`. The inheritance is what makes the blast radius zero by
# construction: every caller that does not pass it keeps the identical value at every site, so no
# other model needs re-measuring.
import inspect


def _param(cls, name):
    return inspect.signature(cls.__init__).parameters[name]


@pytest.mark.parametrize("cls_name", ["PairformerLayer", "Pairformer"])
def test_tri_att_site_inherits_when_unset(cls_name):
    import tt_bio.tenstorrent as T
    assert _param(getattr(T, cls_name), "tri_att_accurate_softmax").default is None, \
        "%s must inherit accurate_softmax, not default to a bool" % cls_name


def test_pairformer_layer_routes_the_two_triangle_attentions_through_the_new_name():
    """The parameter has to actually reach both TriangleAttention children and nothing else.
    Read off the AST rather than built layers, which would need weights and a card."""
    tree = ast.parse((ROOT / "tt_bio/tenstorrent.py").read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "PairformerLayer")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    passed = {}
    for node in ast.walk(init):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        for kw in node.keywords:
            if kw.arg == "accurate_softmax":
                passed.setdefault(node.func.id, []).append(ast.unparse(kw.value))
    assert passed.get("TriangleAttention") == ["tri_acc", "tri_acc"], \
        "both TriangleAttentions must take the triangle-attention site: %s" % passed
    assert passed.get("AttentionPairBias") == ["accurate_softmax"], \
        "AttentionPairBias must keep the layer-level flag: %s" % passed


def test_rf3_ships_the_chain_at_attention_pair_bias_and_not_at_triangle_attention(monkeypatch):
    monkeypatch.delenv("TT_BIO_ACCURATE_SOFTMAX_AB", raising=False)
    import importlib
    import tt_bio.rf3.remap as remap
    flags = importlib.reload(remap).PAIRFORMER_FLAGS
    assert flags["accurate_softmax"] is True
    assert flags["tri_att_accurate_softmax"] is False, \
        "the triangle-attention chain is the 1.376x; it does not ship on"


def test_the_rf3_triangle_attention_site_stays_ab_able(monkeypatch):
    """Recovering the regressed route without a checkout is how the fix was measured, and how
    anyone re-opens the accuracy question later."""
    import importlib
    import tt_bio.rf3.remap as remap
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "rf3.tri_att")
    assert importlib.reload(remap).PAIRFORMER_FLAGS["tri_att_accurate_softmax"] is True
    monkeypatch.delenv("TT_BIO_ACCURATE_SOFTMAX_AB")
    assert importlib.reload(remap).PAIRFORMER_FLAGS["tri_att_accurate_softmax"] is False
