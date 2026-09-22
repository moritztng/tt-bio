"""OpenFold3's sample-ranking score, and the single-chain case that made it a defect.

Two properties, and the second is the one the change exists for:

  1. With an interface, the score is AF3 SI 5.9.3 unchanged, to the bit.
  2. Without one, ipTM is zero by construction and its 0.8 weight moves to pLDDT instead of
     evaporating. The collapsed rule ranked five samples of one target on a pTM spread of
     0.011 and selected worse than random over nine seeds; see perf/of3t_confhead/.
"""
import pytest

from tt_bio.openfold3_fold import sample_ranking_score


def test_with_interface_is_af3_unchanged():
    """A complex keeps upstream's formula exactly, so nothing an interface reaches moves."""
    kw = dict(iptm=0.7, ptm=0.6, plddt=0.9, disorder=0.2, has_clash=0.0)
    assert sample_ranking_score(**kw) == pytest.approx(
        0.8 * 0.7 + 0.2 * 0.6 + 0.5 * 0.2 - 100.0 * 0.0)
    clashing = dict(kw, has_clash=1.0)
    assert sample_ranking_score(**clashing) < -99.0, "the clash veto must survive"


def test_without_interface_ranks_on_plddt():
    """Single chain: ipTM is 0 by construction, so pLDDT carries its weight."""
    kw = dict(iptm=0.0, ptm=0.6, plddt=0.9, disorder=0.0, has_clash=0.0)
    assert sample_ranking_score(**kw) == pytest.approx(0.8 * 0.9 + 0.2 * 0.6)


def test_without_interface_the_old_rule_would_have_tied_these_and_the_new_one_does_not():
    """The failure this change is for, at the measured scale.

    Two samples of one monomer: pTM differs by 0.011, which is the whole spread the old rule
    had to work with, and pLDDT differs by 0.006 the other way. The old rule serves the sample
    with the higher pTM; the new one serves the one the head is more confident in per residue.
    """
    higher_ptm = dict(iptm=0.0, ptm=0.6831, plddt=0.7957, disorder=0.0, has_clash=0.0)
    higher_plddt = dict(iptm=0.0, ptm=0.6720, plddt=0.8023, disorder=0.0, has_clash=0.0)
    old = lambda s: 0.8 * s["iptm"] + 0.2 * s["ptm"] + 0.5 * s["disorder"]
    assert old(higher_ptm) > old(higher_plddt)
    assert sample_ranking_score(**higher_plddt) > sample_ranking_score(**higher_ptm)


def test_disorder_is_still_read():
    """The RASA term measured 0.0 on ubiquitin; that is a property of that target, not of the
    rule, and a target that does have disorder must still see it."""
    a = sample_ranking_score(iptm=0.0, ptm=0.6, plddt=0.9, disorder=0.0, has_clash=0.0)
    b = sample_ranking_score(iptm=0.0, ptm=0.6, plddt=0.9, disorder=0.4, has_clash=0.0)
    assert b - a == pytest.approx(0.5 * 0.4)
