"""One sample-ranking rule for the AF3-lineage models, and the two properties it has to have.

  1. **With an interface, every site keeps the expression it already had, to the bit.** Each
     of the three was AF3 SI 5.9.3 over the terms it computes, so a change to the
     no-interface case must be a no-op for every complex. The old expressions are transcribed
     verbatim below and compared over a randomized sweep, not argued from the source.
  2. **Without one, ipTM's 0.8 goes to pLDDT.** ipTM is identically zero on a single chain, so
     the three sites improvised three different collapsed rules -- 0.2*pTM, 1.2*pTM and pTM --
     all of which order the samples by pTM alone. Measured against true Ca-RMSD over nine
     seeds, pTM is the worst-ordering output the head produces; see perf/of3t_confhead/ and
     perf/of3t_rankunify/.
"""
import random

import pytest

from tt_bio.ranking import ranking_score

# The three sites as they stood before unification, transcribed from
# openfold3_fold.sample_ranking_score, rf3.confidence.ranking_score and
# worker._protenix_emit._score at wk/of3t 7aed7253b.
def _old_of3(iptm, ptm, plddt, disorder, clash):
    if iptm > 0.0:
        return 0.8 * iptm + 0.2 * ptm + 0.5 * disorder - 100.0 * clash
    return 0.8 * plddt + 0.2 * ptm + 0.5 * disorder - 100.0 * clash


def _old_rf3(iptm, ptm, plddt, disorder, clash):
    if iptm is None:
        iptm = ptm if ptm is not None else 0.0
    return 0.8 * iptm + 0.2 * (ptm or 0.0) - 100 * int(clash)


def _old_protenix(iptm, ptm, plddt, disorder, clash):
    if iptm > 0.0:
        return 0.8 * iptm + 0.2 * ptm
    return ptm if ptm > 0.0 else plddt


def _cases(n=400, iptm_zero=False):
    rng = random.Random(20260921)
    for _ in range(n):
        yield dict(iptm=0.0 if iptm_zero else rng.uniform(0.05, 1.0),
                   ptm=rng.uniform(0.0, 1.0), plddt=rng.uniform(0.0, 1.0),
                   disorder=rng.choice([0.0, rng.uniform(0.0, 1.0)]),
                   has_clash=float(rng.random() < 0.15))


def test_interface_branch_is_af3_unchanged_to_the_bit():
    """Deliverable: a multi-chain target scores exactly AF3's formula."""
    for c in _cases():
        assert ranking_score(**c) == (0.8 * c["iptm"] + 0.2 * c["ptm"]
                                      + 0.5 * c["disorder"] - 100.0 * c["has_clash"])


@pytest.mark.parametrize("old,terms", [
    (_old_of3, ("disorder", "has_clash")),
    (_old_rf3, ("has_clash",)),
    (_old_protenix, ()),
])
def test_interface_branch_is_byte_identical_at_every_site(old, terms):
    """Each site passes 0.0 for the terms it does not compute, so with an interface it gets
    back exactly its own old expression. Equality, not approx."""
    for c in _cases():
        passed = {k: (c[k] if k in terms else 0.0) for k in ("disorder", "has_clash")}
        assert ranking_score(iptm=c["iptm"], ptm=c["ptm"], plddt=c["plddt"], **passed) == \
            old(c["iptm"], c["ptm"], c["plddt"], passed["disorder"], passed["has_clash"])


def test_the_three_sites_disagreed_without_an_interface_and_now_do_not():
    """The defect, stated as a test. On one monomer the three sites give three different
    numbers; rf3 and protenix both order by pTM; OpenFold3, repaired by of3t-confhead, does
    not. Unifying means rf3 and protenix adopt OpenFold3's rule, and OpenFold3 does not move.
    """
    a = dict(iptm=0.0, ptm=0.6831, plddt=0.7957, disorder=0.0, has_clash=0.0)
    b = dict(iptm=0.0, ptm=0.6720, plddt=0.8023, disorder=0.0, has_clash=0.0)
    olds = [_old_of3, _old_rf3, _old_protenix]
    assert len({round(f(0.0, a["ptm"], a["plddt"], 0.0, 0.0), 6) for f in olds}) == 3
    for f in (_old_rf3, _old_protenix):
        assert f(0.0, a["ptm"], a["plddt"], 0.0, 0.0) > f(0.0, b["ptm"], b["plddt"], 0.0, 0.0)
    assert ranking_score(**b) > ranking_score(**a)


def test_openfold3_does_not_move():
    """of3t-confhead already put OpenFold3 on this rule and folded it. Unification must not
    change OpenFold3 again, on either branch, or its evidence stops applying."""
    for zero in (False, True):
        for c in _cases(iptm_zero=zero):
            assert ranking_score(**c) == _old_of3(c["iptm"], c["ptm"], c["plddt"],
                                                  c["disorder"], c["has_clash"])


def test_without_interface_iptm_weight_goes_to_plddt():
    assert ranking_score(iptm=0.0, ptm=0.6, plddt=0.9) == pytest.approx(0.8 * 0.9 + 0.2 * 0.6)
    assert ranking_score(iptm=None, ptm=0.6, plddt=0.9) == ranking_score(
        iptm=0.0, ptm=0.6, plddt=0.9), "rf3 passes None where of3 passes 0.0"


def test_clash_veto_and_disorder_survive():
    assert ranking_score(iptm=0.7, ptm=0.6, plddt=0.9, disorder=0.2, has_clash=1.0) < -99.0
    a = ranking_score(iptm=0.0, ptm=0.6, plddt=0.9, disorder=0.0)
    b = ranking_score(iptm=0.0, ptm=0.6, plddt=0.9, disorder=0.4)
    assert b - a == pytest.approx(0.5 * 0.4)


def test_every_site_routes_through_the_shared_rule():
    """A site left un-wired is the failure mode this row exists to prevent, so the sites are
    called here rather than the shared function being tested alone."""
    from tt_bio.rf3.confidence import ranking_score as rf3_rank
    assert rf3_rank(0.7, 0.6, 0.9, False) == ranking_score(iptm=0.7, ptm=0.6, plddt=0.9)
    assert rf3_rank(None, 0.6, 0.9, False) == ranking_score(iptm=None, ptm=0.6, plddt=0.9)
    assert rf3_rank(None, 0.6, 0.9, True) < -99.0


def test_rf3_orders_on_the_full_precision_score():
    """rf3 publishes its ranking score rounded to 4 decimals in summary_confidences.json, and
    it used to ORDER on that rounded number, so two samples whose scores differ below 1e-4
    were ordered by sample index instead. These are the real scalars of rf3/multimer seed 1
    samples 4 and 0: both round to 0.7400 and the wrong one was served at rank 3. Two of the
    five samples of that single fold collided this way, so it is the common case on a target
    whose samples are close, not a corner. The published rounding stays; the ordering must not
    use it.
    """
    from tt_bio.rf3.confidence import ranking_score as rf3_rank
    a = rf3_rank(0.7338814735412598, 0.7646858096122742, 0.783822, False)
    b = rf3_rank(0.7342625260353088, 0.7630373835563660, 0.784785, False)
    assert round(a, 4) == round(b, 4) == 0.74, "the published rounding collapses this pair"
    assert a > b, "and the full-precision score does not"
