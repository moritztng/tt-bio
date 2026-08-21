"""The cascade band's two silent-failure modes.

`cascade_band.py` decides which designs a device-first filter may resolve without paying for a
host reference, so a bug in it is a wrong accept/reject rather than a wrong number. Two things
can break without any output looking odd: the referral rule can start passing designs whose
reference verdict it cannot actually predict, and the holdout inflation can stop firing and
quietly hand back an un-widened band.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "af2_port"))

import cascade_band as cb  # noqa: E402


def _row(rid, design, dev, ref):
    keys = ("plddt", "i_ptm", "i_pae")
    return {"id": rid, "design": design,
            "device": dict(zip(keys, dev)), "reference": dict(zip(keys, ref)),
            "delta": {k: d - r for k, d, r in zip(keys, dev, ref)}}


def test_referral_rule_only_resolves_what_it_can_predict():
    """Every design the cascade resolves must be one where the band's own promise -- |delta| <=
    W -- forces the reference verdict. Checked by brute force: for each resolved design, move
    every criterion by the full band in the direction that would flip it, and require the
    verdict to survive."""
    band = {"plddt": 0.06, "i_ptm": 0.10, "i_pae": 0.10}
    rows = [
        _row("clear-reject", "a", (0.95, 0.20, 0.70), (0.95, 0.20, 0.70)),
        _row("clear-accept", "a", (0.95, 0.90, 0.10), (0.95, 0.90, 0.10)),
        _row("on-the-line", "a", (0.95, 0.52, 0.33), (0.95, 0.45, 0.39)),
    ]
    out = cb.cascade(rows, band)
    for rid in set(r["id"] for r in rows) - set(out["referred_ids"]):
        row = next(r for r in rows if r["id"] == rid)
        for c, (sense, bar) in cb.BARS.items():
            worst = row["device"][c] + (-band[c] if sense == ">" else band[c])
            assert (worst > bar) == (row["device"][c] > bar) if sense == ">" else \
                   (worst < bar) == (row["device"][c] < bar), \
                   f"{rid} resolved but {c} can cross its bar inside the band"
    assert out["referred_ids"] == ["on-the-line"]
    assert out["decisions_reproduced"] == out["n"]


def test_conjunction_rule_resolves_a_confident_reject_the_symmetric_rule_refers():
    """af2_easy is a conjunction, so one criterion failing by more than its band settles it even
    when another sits inside its band. If that stops holding the cascade gets needlessly
    expensive, which is a regression no accuracy check would catch."""
    band = {"plddt": 0.06, "i_ptm": 0.10, "i_pae": 0.10}
    rows = [_row("far-reject-near-ptm", "a", (0.95, 0.55, 0.90), (0.95, 0.55, 0.90))]
    assert cb.cascade(rows, band, asymmetric=True)["device_resolved"] == 1
    assert cb.cascade(rows, band, asymmetric=False)["device_resolved"] == 0


def _two_scale_population(quiet_step=0.001, loud_step=0.02):
    """13 designs on a quiet backbone and 13 on one 20x noisier, every criterion moving, so the
    holdout has a real tail to miss."""
    rows = []
    for tag, step in (("quiet", quiet_step), ("loud", loud_step)):
        for i in range(13):
            rows.append(_row(f"{tag}{i}", tag,
                             (0.9, 0.6, 0.2),
                             (0.9 - 0.3 * step * i, 0.6 - step * i, 0.2 + 0.8 * step * i)))
    return rows


def test_inflation_is_floored_at_one_and_fires_when_the_holdout_misses():
    """A model that covers out of sample must not be shrunk, and one that misses must be widened
    by at least the shortfall."""
    picked = cb.choose(_two_scale_population())
    for c, p in picked.items():
        assert p["unbounded"] or p["inflation"] >= 1.0
        # Either the model covered out of sample, or the inflation did the work.
        assert p["unbounded"] or p["inflation"] > 1.0 or p["holdout_shortfall_halfnorm"] <= 1.0
    ptm = picked["i_ptm"]
    # The two backbones differ 20x in scale, so a fit on the quiet one cannot see the loud one's
    # tail and the inflation has to do the work.
    assert ptm["holdout_shortfall_halfnorm"] > 1.0
    assert ptm["inflation"] > 1.0
    assert ptm["width"] > ptm["base_width"]


def test_a_criterion_one_backbone_never_moves_is_reported_unbounded_not_narrow():
    """A backbone that shows zero device error on a criterion must not produce a zero band while
    another backbone shows a real one: no finite inflation of nothing covers it."""
    rows = [_row(f"q{i}", "quiet", (0.9, 0.6, 0.2), (0.9, 0.6, 0.2)) for i in range(13)]
    rows += [_row(f"l{i}", "loud", (0.9, 0.6, 0.2), (0.9, 0.6 - 0.02 * (i + 1), 0.2))
             for i in range(13)]
    p = cb.choose(rows)["i_ptm"]
    assert p["unbounded"] is True
    assert p["width"] == cb.FULL_RANGE
    assert p["inflation"] is None
    # And an unbounded criterion refers the whole population rather than resolving any of it.
    assert cb.cascade(rows, {c: cb.choose(rows)[c]["width"] for c in cb.BARS})["device_resolved"] == 0


def test_observed_max_content_falls_once_clustering_is_counted():
    """The whole argument for widening: the observed maximum is a weaker tolerance limit than a
    naive count of designs suggests."""
    p = cb.choose(_two_scale_population())["i_ptm"]
    assert p["n_eff"] < p["n"]
    assert p["observed_max_content_at_neff"] < p["observed_max_content_at_n"]
