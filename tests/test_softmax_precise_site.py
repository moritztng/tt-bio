"""The per-site softmax compute-kernel-config lever.

`softmax_ckc` is called at import-time-constructed model modules, so a site that forgot to
import it fails at model LOAD, not at the softmax -- which is how protenix-v2 came back
"name 'softmax_ckc' is not defined" from a fold that had loaded fine for OpenFold3. The
import check below is the cheap version of that fold.
"""
import importlib
import os

import pytest

from tt_bio import tenstorrent as T

#: Every module that reads `softmax_ckc` at construction time. A new site belongs here.
SITE_MODULES = [
    "tt_bio.tenstorrent",
    "tt_bio.protenix",
    "tt_bio.openfold3_diffusion_transformer",
    "tt_bio.openfold3_atom_transformer",
]

#: The tokens the shipped sites name. Pinned so a rename has to be deliberate: the tokens are
#: the user-facing surface of TT_BIO_SOFTMAX_PRECISE_AB.
SITE_TOKENS = [
    "openfold3.diffusion_transformer",
    "openfold3.atom_transformer",
    "protenix.atom_transformer",
    "pairformer",
    "miniformer",
    "diffusion_transformer.token",
    "diffusion_transformer.atom",
    # protenix-v2's token DiT. It used to construct AttentionPairBias without softmax_site, so
    # its lever sat under the shared "default" token and could not be switched on its own --
    # and of3t-softmax's model/site table attributes protenix to diffusion_transformer.token,
    # which is a DIFFERENT construction. of3t-fwdkcfg's runtime census is what separated them.
    "protenix.token_dit",
]


@pytest.mark.parametrize("mod", SITE_MODULES)
def test_every_site_module_can_resolve_softmax_ckc(mod):
    m = importlib.import_module(mod)
    src = open(m.__file__).read()
    if "softmax_ckc(" not in src:
        pytest.skip(f"{mod} does not call softmax_ckc")
    assert hasattr(m, "softmax_ckc"), (
        f"{mod} calls softmax_ckc but does not import it -- this fails at model load, not at "
        f"the softmax, so no unit test of the softmax itself would catch it")


def test_the_shipped_default_is_off_at_every_site(monkeypatch):
    monkeypatch.delenv("TT_BIO_SOFTMAX_PRECISE_AB", raising=False)
    for tok in SITE_TOKENS:
        assert T.softmax_ckc(tok) is None, (
            f"{tok} ships the precise config on. Measured NO-GO on 2026-09-20: it moves "
            f"OpenFold3's 1UBQ structure 0.3237 A against a 0.6250 A seed floor, so turning "
            f"it on by default is a release-gate decision, not a code change")


def test_the_ab_grammar_reaches_every_site(monkeypatch):
    monkeypatch.setenv("TT_BIO_SOFTMAX_PRECISE_AB", "all")
    for tok in SITE_TOKENS:
        assert T.softmax_ckc(tok) is not None, f"{tok} is unreachable by TT_BIO_SOFTMAX_PRECISE_AB"
    # and a per-site minus wins over the blanket on, or the A/B cannot isolate one site
    monkeypatch.setenv("TT_BIO_SOFTMAX_PRECISE_AB", "all,-pairformer")
    assert T.softmax_ckc("pairformer") is None
    assert T.softmax_ckc("openfold3.diffusion_transformer") is not None


def test_the_config_is_the_precise_one():
    """A config that is not HiFi4 + fp32 dest acc buys none of the measured 12.3x."""
    import ttnn
    c = T._SOFTMAX_PRECISE_CKC
    assert c.math_fidelity == ttnn.MathFidelity.HiFi4
    assert c.math_approx_mode is False
    assert c.fp32_dest_acc_en is True


def test_the_docstring_no_longer_claims_a_config_changes_nothing():
    """The sentence this row was dispatched to delete. It said a compute kernel config
    changes nothing; it changes 2.029e-02 into 1.646e-03."""
    doc = T._accurate_softmax.__doc__
    assert "compute kernel config change" not in doc.replace("\n", " ").replace("  ", " ")
    assert "2.029e-02" in doc and "1.646e-03" in doc
