"""Known-answer controls for the two-size family arithmetic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import two_size_family as m


def test_recovers_a_synthetic_fixed_term():
    # Build a world with a known F and known work terms, then recover F from the family.
    F, w298 = 4.0, 7000.0
    r = 2.0
    t298 = F + w298 / m.CLOCK
    t512 = F + r * w298 / m.CLOCK
    saved = (m.T512, m.T298)
    m.T512, m.T298 = t512, t298
    try:
        e = m.family(r)
        assert abs(e["fixed_s"] - F) < 1e-9, e
        assert abs(e["W298_Mcycles"] - w298) < 1e-6, e
        assert abs(e["W512_Mcycles"] - r * w298) < 1e-6, e
    finally:
        m.T512, m.T298 = saved


def test_family_is_monotone_in_the_work_ratio():
    fs = [m.family(r)["fixed_s"] for r in (1.6, 1.8, 2.0, 2.5, 3.0)]
    assert fs == sorted(fs), "a larger work ratio must imply a larger fixed term"


def test_refuses_a_ratio_that_makes_512_cheaper_than_298():
    for bad in (1.0, 0.5, -1.0):
        try:
            m.family(bad)
        except ValueError:
            continue
        raise AssertionError("ratio %r must be refused" % bad)


def test_fixed_term_never_exceeds_the_smaller_fold():
    for r in (1.6, 2.0, 3.0, 5.07, 50.0):
        assert m.family(r)["fixed_s"] < m.T298


def test_goal_flag_agrees_with_the_percentage():
    for r in (1.6, 2.5, 5.07):
        e = m.family(r)
        d = m.demand(e)
        assert d["fixed_cut_alone_reaches_goal"] == (d["cut_with_fixed_at_1.0s_pct"] <= 0.0)
