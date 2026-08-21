"""The accurate-softmax lever must be off at every construction site by default.

`ttnn.softmax` normalises against a denominator its own numerators do not match (rows sum to
0.977 on [1,16,512,512] fp32), and `_accurate_softmax` replaces it with a 5-op chain that costs
up to 4.22x on the op. Whether a given model should pay that is a per-model, per-site verdict.
Until one lands, every site stays on today's path, and this test is what makes "no-op by
construction" a property rather than a claim.
"""
import ast
import os
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
}


def _selector():
    from tt_bio.tenstorrent import accurate_softmax_site
    return accurate_softmax_site


def test_env_unset_means_every_site_is_off(monkeypatch):
    monkeypatch.delenv("TT_BIO_ACCURATE_SOFTMAX_AB", raising=False)
    sel = _selector()
    assert [s for s in SITES if sel(s)] == []


def test_empty_env_means_every_site_is_off(monkeypatch):
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "")
    sel = _selector()
    assert [s for s in SITES if sel(s)] == []


def test_a_named_site_turns_on_only_itself(monkeypatch):
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "openfold3.trunk")
    sel = _selector()
    assert [s for s in SITES if sel(s)] == ["openfold3.trunk"]


def test_protenix_and_opendde_do_not_drag_each_other(monkeypatch):
    """The shared-construction-site trap: a Protenix-v2 flip must not reach OpenDDE."""
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "protenix.trunk,protenix.confidence")
    sel = _selector()
    assert not sel("opendde.trunk") and not sel("opendde.confidence")
    assert not sel("opendde.refiner")


@pytest.mark.parametrize("token", sorted(SITES))
def test_site_is_reachable_and_wired_in_the_named_file(monkeypatch, token):
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", token)
    assert _selector()(token), "%s is not reachable through the selector" % token
    src = (ROOT / SITES[token]).read_text()
    model, site = token.split(".", 1)
    literal = 'accurate_softmax_site("%s")' % token
    scoped = 'accurate_softmax_site(f"{softmax_scope}.%s")' % site
    assert literal in src or scoped in src, "%s: no wiring found in %s" % (token, SITES[token])


def test_no_construction_site_hardcodes_the_lever_on():
    """A default flip is release-gated. `accurate_softmax=True` in tt_bio/ is that flip."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"accurate_softmax\s*=\s*True", line):
                offenders.append("%s:%d" % (path.relative_to(ROOT), i))
    assert offenders == [], "accurate_softmax defaulted ON at: %s" % ", ".join(offenders)


def test_shared_primitives_default_the_keyword_off():
    """Every function/class that accepts the keyword must default it False."""
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
