"""The fused SDPA's ragged-tile-tail mask is per site, and RF3 is the one site it ships on.

ttnn's fused SDPA reduces over the TILE-PADDED physical key length while a mask sized to the
logical length leaves the ragged tail at a bias of 0. exp(0) = 1 beats real scores that mostly sit
below 0, so up to 31 padded key columns take a real share of every row's softmax mass. Measured in
fp64 on captured RF3 operands the fused path is 71-76x the materialised path's error at every
length that is not a multiple of 32, and flat at ~1e-2 either way when it is.

So the mask is a correctness fix, not a tuning knob -- and it is still per site, because scoring it
on the three models that reach a ragged fused call did not give one answer:

    rf3 7ROA L117          X 1.6415 A (outside floor)  ->  X 0.1780 A          9.2x BETTER
    protenix-v2 hsa L585   committed GAP               ->  X 0.674 / 0.695     GAP -> PASS
    protenix-v2 ubq L76    committed GAP-evidenced     ->  X 1.828 / 1.923     GAP -> PASS
    opendde-prot-prod      X 2.215 / 2.102 (PASS)      ->  X 7.289 / 3.683     PASS -> GAP
    opendde-trpcage-nomsa  X 0.361 / 0.374 (PASS)      ->  X 0.428 / 0.475     19% WORSE

OpenDDE also gets less self-consistent with the mask on (device floor 2.102 -> 3.683), the opposite
of what it does to RF3, so that is not diffusion chaos being misread. Until it is root-caused, the
site that measured a win is the only site that ships it. Protenix-v2 improving at both targets is
a follow-up worth taking, not something to fold in here.

`TT_BIO_SDPA_RAGGED_PAD` still forces the mask on at EVERY site; that global is how the table above
was measured and it stays as the cross-model A/B switch.
"""
import ast
import inspect
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "tt_bio"

# Construction sites wired to the selector, with the file that wires them.
SITES = {"rf3.tri_att": "tt_bio/rf3/remap.py"}

# RF3 ships ON. Unlike the HiFi4 sibling this lever has a fold-level number behind it at this site,
# and shipping RF3 OFF is not neutral: the fast arm without the mask reads 1.6415 A at 7ROA against
# a 0.361 A floor, which is worse than the materialised route it replaced.
ON_BY_DEFAULT = {"rf3.tri_att"}


def _live():
    from tt_bio.tenstorrent import sdpa_ragged_pad_site
    return {s for s in SITES if sdpa_ragged_pad_site(s, default=s in ON_BY_DEFAULT)}


def test_env_unset_ships_rf3_on_and_nothing_else(monkeypatch):
    monkeypatch.delenv("TT_BIO_SDPA_RAGGED_PAD_AB", raising=False)
    assert _live() == ON_BY_DEFAULT


def test_empty_env_ships_rf3_on_and_nothing_else(monkeypatch):
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "")
    assert _live() == ON_BY_DEFAULT


@pytest.mark.parametrize("token", sorted(SITES))
def test_a_named_site_moves_only_itself(monkeypatch, token):
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "-" + token)
    assert _live() == ON_BY_DEFAULT - {token}
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", token)
    assert _live() == ON_BY_DEFAULT | {token}


def test_all_and_minus_all_move_every_site_without_a_token_of_its_own(monkeypatch):
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "all")
    assert _live() == set(SITES)
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "-all")
    assert _live() == set()
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "-all,rf3.tri_att")
    assert _live() == {"rf3.tri_att"}


def test_the_three_selectors_do_not_move_each_other(monkeypatch):
    """All three share `_site_flag`, so the one thing that refactor can break is two of them
    reading each other's environment variable."""
    from tt_bio.tenstorrent import (accurate_softmax_site, sdpa_ragged_pad_site,
                                    triatt_sdpa_hifi_site)
    for var in ("TT_BIO_SDPA_RAGGED_PAD_AB", "TT_BIO_TRIATT_SDPA_HIFI_AB",
                "TT_BIO_ACCURATE_SOFTMAX_AB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "all")
    assert sdpa_ragged_pad_site("rf3.tri_att")
    assert not triatt_sdpa_hifi_site("boltz2.trunk")
    assert not accurate_softmax_site("openfold3.trunk")
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "-all")
    monkeypatch.setenv("TT_BIO_TRIATT_SDPA_HIFI_AB", "all")
    assert triatt_sdpa_hifi_site("boltz2.trunk")
    assert not sdpa_ragged_pad_site("rf3.tri_att", default=True)


@pytest.mark.parametrize("token", sorted(SITES))
def test_site_is_wired_in_the_named_file(token):
    src = (ROOT / SITES[token]).read_text()
    assert 'sdpa_ragged_pad_site("%s"' % token in src, \
        "%s is not wired in %s" % (token, SITES[token])


def test_opendde_and_protenix_do_not_get_a_token():
    """The whole point of scoping. A token for either would ship the regression OpenDDE measured,
    or ship Protenix an improvement nobody signed off on."""
    offenders = []
    call = re.compile(r"sdpa_ragged_pad_site\(\s*f?\"([^\"]*)\"")
    for path in sorted(SRC.rglob("*.py")):
        offenders += [t for t in call.findall(path.read_text()) if t not in SITES]
    assert offenders == [], "unlisted sites wired to the pad selector: %s" % sorted(offenders)


def test_no_site_bypasses_the_selector_with_a_literal():
    """A flip moves a `default=`, it does not hardcode the keyword: an unnamed site cannot be
    forced back off, and being unable to turn this one OFF is what makes the OpenDDE reading
    unreproducible."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]  # prose that NAMES the keyword is not a bypass
            if re.search(r"(tri_att_)?sdpa_ragged_pad\s*=\s*True", code):
                offenders.append("%s:%d" % (path.relative_to(ROOT), i))
    assert offenders == [], "the mask is hardcoded ON at: %s" % ", ".join(offenders)


@pytest.mark.parametrize("cls_name,param", [
    ("TriangleAttention", "sdpa_ragged_pad"),
    ("PairformerLayer", "tri_att_sdpa_ragged_pad"),
    ("Pairformer", "tri_att_sdpa_ragged_pad"),
])
def test_every_layer_on_the_path_defaults_the_keyword_off(cls_name, param):
    """Every model that did not opt in must be byte-for-byte unaffected, which holds only if the
    keyword defaults off the whole way down."""
    import tt_bio.tenstorrent as T
    assert inspect.signature(getattr(T, cls_name).__init__).parameters[param].default is False


def test_pairformer_layer_routes_the_flag_to_both_triangle_attentions_and_nothing_else():
    """Read off the AST: building the layer would need weights and a card. Triangle attention has
    a start and an ending variant and a ragged length is ragged in both."""
    tree = ast.parse((ROOT / "tt_bio/tenstorrent.py").read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "PairformerLayer")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    passed = {}
    for node in ast.walk(init):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for kw in node.keywords:
                if kw.arg == "sdpa_ragged_pad":
                    passed.setdefault(node.func.id, []).append(ast.unparse(kw.value))
    assert passed == {"TriangleAttention": ["tri_att_sdpa_ragged_pad"] * 2}, \
        "the flag must reach both TriangleAttentions and nothing else: %s" % passed


def test_the_global_still_forces_every_site_on():
    """`TT_BIO_SDPA_RAGGED_PAD` is how the cross-model table in this module's docstring was
    measured. If the per-site flag replaced it rather than joining it, that measurement stops
    being reproducible."""
    src = (ROOT / "tt_bio/tenstorrent.py").read_text()
    assert "if not (ragged and (_SDPA_RAGGED_PAD or pad)):" in src


def test_rf3_ships_the_mask_on():
    """The regression guard for the half-applied flip. RF3 on the fused arm WITHOUT the mask is
    worse than the materialised route it replaced, so `fp32_softmax=False` and this flag are one
    change and must not drift apart."""
    from tt_bio.rf3.remap import PAIRFORMER_FLAGS
    assert PAIRFORMER_FLAGS["fp32_softmax"] is False
    assert PAIRFORMER_FLAGS["tri_att_sdpa_ragged_pad"] is True
