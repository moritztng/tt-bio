"""How wide does the device-first cascade's referral band have to be to be a bound?

Pass 12 built a device-first cascade: score every design on device, refer the ones sitting near
an `af2_easy` bar to the host reference, decide the rest on device. It set the band to the worst
per-design delta it had seen, 0.0710 of i_pTM, and said in terms that anyone shipping the cascade
has to widen it, because an observed maximum over 26 designs is not a bound on the 27th.

This derives the width instead of observing it, and the derivation has to survive one specific
worry. The 26 designs are 13 sequences on each of two generated backbones, and the two backbones
do not behave the same: the device delta's sd on i_pTM is 0.0149 on `binder` and 0.0315 on
`binder_1`, so the delta's SCALE is a property of the backbone, not a constant of the port. A
third backbone can therefore be worse than either, and any method that treats the 26 rows as 26
independent draws is answering an easier question than the one the cascade asks.

Four things follow, in order.

1.  **How weak the observed maximum is.** For n independent draws, P(the next draw is below the
    observed max) = n/(n+1), and the max is a one-sided nonparametric tolerance limit whose
    content p at confidence 1-alpha solves alpha = p^n. At n = 26 that is p = 0.891: with 95%
    confidence, at least 89% of future designs fall inside the pass-12 band, so up to one in nine
    falls outside. Reaching p = 0.99 nonparametrically needs n >= 299 designs, and a host
    reference costs 298 s of CPU a design here, so 299 designs is 24.7 h of reference alone. The
    nonparametric route is not affordable at the content the cascade needs, which is why the
    width below is parametric.

2.  **How much less than 26 the 26 rows are worth.** One-way random effects on |delta|, backbone
    as the grouping factor, gives an intraclass correlation around 0.24 on all three criteria and
    so an effective sample size n_eff = n / (1 + (m-1) * ICC) of about 6.5. Every tolerance
    factor below is evaluated at n_eff, not at 26. That single substitution is most of the
    widening: it takes the chi-square upper bound on the scale from 1.31x to 1.84x. Read the
    other way, the observed maximum is only a 64%-content bound once the clustering is counted,
    not an 89% one.

3.  **Which model to put the tail through, and how much to inflate it**, both decided by holding
    a backbone out rather than by looking at a fit statistic. Fit the scale on one backbone, and
    require the resulting width to cover the OTHER backbone's worst delta -- the exact
    extrapolation the cascade performs when it meets a new design. Two things come out of it.
    First the model: the half-normal recipe covers in both directions on the two interface
    criteria while the naive iid version of the same recipe does not, under-bounding i_pTM by
    1.41x, so the clustering correction is load-bearing rather than decorative. On pLDDT the
    half-normal is refuted, and so is the distribution-free unimodal bound
    (Vysochanskij-Petunin, P(|X-mu| >= lambda*sigma) <= 4/(9*lambda^2)) -- pLDDT's delta has
    skew -2.9 and excess kurtosis 9.3, and one design carries a delta 19x the quiet backbone's
    rms, so a single backbone does not see the tail at all. pLDDT falls back to the
    assumption-light model. Second the inflation: whatever the model, the width is multiplied by
    the worst out-of-sample shortfall the holdout measured, floored at 1. i_pTM and i_pAE cover
    already and are not inflated; pLDDT is inflated 1.51x. That makes the recipe one recipe --
    fit, hold a backbone out, pay for the shortfall -- instead of a per-criterion judgement call.

4.  **What the width costs**, in designs the cascade still resolves on device. Reported under the
    referral rule that exploits `af2_easy` being a conjunction: a device REJECT is safe as soon
    as ONE criterion fails by more than its band, because the reference then fails that criterion
    too and rejects for the same reason, while a device ACCEPT needs all three margins outside
    their bands. That asymmetry is worth more than the widening costs.

    PYTHONPATH=. python3 scripts/af2_port/cascade_band.py \\
        --flip-rate scripts/af2_port/parity_artifacts/designpop_bg119/flip_rate.json \\
        --out scripts/af2_port/parity_artifacts/designpop_bg119/cascade_band.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import chi2, norm

# af2_easy's three confidence criteria and their sense. The fourth, bound-unbound RMSD < 3.5, is a
# comparison between two predictions rather than a scalar with a device delta, so it is scored by
# bound_unbound_rmsd.py and reported by filter_flip_rate.py, not banded here.
BARS = {"plddt": (">", 0.8), "i_ptm": (">", 0.5), "i_pae": ("<", 0.35)}

CONTENT = 0.99      # the band has to cover 99% of designs...
CONFIDENCE = 0.95   # ...and we have to be 95% sure it does
HOST_SECONDS = 298.0    # median host reference per design, pass 12
DEVICE_SECONDS = 6.0    # warm device filter per design, pass 12

# A criterion whose width cannot be bounded from this population refers every design, which is
# what a band as wide as the metric's own range does.
FULL_RANGE = 1.0
UNBOUNDED = float("inf")


def _finite(x: float) -> float | None:
    """None rather than a JSON-invalid infinity."""
    return None if x == UNBOUNDED else round(x, 3)


def margin(value: float, criterion: str) -> float:
    """Signed distance to the bar, positive when the criterion PASSES."""
    sense, bar = BARS[criterion]
    return (value - bar) if sense == ">" else (bar - value)


def icc_neff(groups: list[list[float]]) -> tuple[float, float]:
    """Intraclass correlation of |delta| across backbones, and the effective sample size.

    One-way random effects with equal group sizes. A positive ICC means two designs on the same
    backbone tell us less than two designs on different backbones, which is the whole reason the
    tolerance factors below are not evaluated at n = 26.
    """
    k = len(groups)
    n = sum(len(g) for g in groups)
    m = n / k
    grand = sum(sum(g) for g in groups) / n
    means = [sum(g) / len(g) for g in groups]
    msb = sum(len(g) * (mu - grand) ** 2 for g, mu in zip(groups, means)) / (k - 1)
    msw = sum(sum((x - mu) ** 2 for x in g) for g, mu in zip(groups, means)) / (n - k)
    var_b = max(0.0, (msb - msw) / m)
    icc = var_b / (var_b + msw) if var_b + msw > 0 else 0.0
    return icc, n / (1 + (m - 1) * icc)


def scale_upper(values: list[float], neff: float) -> float:
    """Upper (1-CONFIDENCE) confidence bound on the rms of the delta, at `neff` degrees of
    freedom. The rms rather than the sd because the delta's mean is small against its spread
    (i_pTM: 0.0038 against 0.0242) and the cascade needs a bound on |delta| about zero, not
    about the sample mean."""
    rms = math.sqrt(sum(x * x for x in values) / len(values))
    return rms * math.sqrt(neff / chi2.ppf(1 - CONFIDENCE, neff))


def w_halfnorm(values: list[float], neff: float) -> float:
    """Half-normal width: the CONTENT quantile of a normal with the upper-bounded scale."""
    return norm.ppf(CONTENT) * scale_upper(values, neff)


def w_unimodal(values: list[float], neff: float) -> float:
    """Vysochanskij-Petunin width. Assumes only that the delta is unimodal, so it survives the
    skew and the excess kurtosis that break the half-normal on pLDDT."""
    lam = math.sqrt(4 / (9 * (1 - CONTENT)))
    mean = sum(values) / len(values)
    return abs(mean) + lam * scale_upper([x - mean for x in values], neff)


def holdout(rows: list[dict], criterion: str, icc: float) -> list[dict]:
    """Fit the scale on one backbone, require the width to cover the other backbone's worst
    delta. Reported for the clustered recipe and for the naive iid one, because the naive one
    failing is the evidence that the clustering correction is load-bearing rather than decorative.
    """
    backbones = sorted({r["design"] for r in rows})
    out = []
    for fit in backbones:
        v = [r["delta"][criterion] for r in rows if r["design"] == fit]
        held = max(abs(r["delta"][criterion]) for r in rows if r["design"] != fit)
        m = len(v)
        # One backbone on its own: the between-backbone variance is entirely unobserved, so the
        # same design effect applies with the pooled ICC.
        neff_cluster = m / (1 + (m - 1) * icc)
        for label, neff in (("clustered", neff_cluster), ("iid", float(m))):
            for model, fn in (("halfnorm", w_halfnorm), ("unimodal", w_unimodal)):
                w = fn(v, neff)
                out.append({"fit_on": fit, "recipe": label, "model": model,
                            "width": round(w, 6), "heldout_max": round(held, 6),
                            "covers": bool(w >= held)})
    return out


def choose(rows: list[dict]) -> dict:
    """Per criterion: pick the model the holdout does not refute, then inflate by the shortfall
    the holdout measured. Both steps are out-of-sample, so neither is a judgement call."""
    picked = {}
    for c in BARS:
        d = [r["delta"][c] for r in rows]
        icc, neff = icc_neff([[abs(r["delta"][c]) for r in rows if r["design"] == b]
                              for b in sorted({r["design"] for r in rows})])
        hold = holdout(rows, c, icc)

        def shortfall(model: str) -> float:
            """Worst heldout_max / width over the fit backbones. <= 1 means the model covered.

            A zero width means that backbone saw no device error at all on this criterion. If the
            held-out backbone did see one, no finite inflation of nothing reaches it, and the
            criterion is reported unbounded rather than given a made-up width.
            """
            ratios = []
            for h in hold:
                if h["recipe"] != "clustered" or h["model"] != model:
                    continue
                if h["width"] > 0:
                    ratios.append(h["heldout_max"] / h["width"])
                else:
                    ratios.append(1.0 if h["heldout_max"] == 0 else UNBOUNDED)
            return max(ratios)

        hn_ok = shortfall("halfnorm") <= 1.0
        iid_ok = all(h["covers"] for h in hold
                     if h["recipe"] == "iid" and h["model"] == "halfnorm")
        model = "halfnorm" if hn_ok else "unimodal"
        inflation = max(1.0, shortfall(model))
        base = (w_halfnorm if hn_ok else w_unimodal)(d, neff)
        # An unbounded criterion refers every design rather than resolving one it cannot predict.
        unbounded = inflation >= UNBOUNDED
        width = FULL_RANGE if unbounded else base * inflation
        observed = max(abs(x) for x in d)
        picked[c] = {
            "bar": BARS[c][1], "sense": BARS[c][0], "n": len(d),
            "icc": round(icc, 4), "n_eff": round(neff, 2),
            "observed_max_abs_delta": round(observed, 6),
            "base_width": round(base, 6),
            # What the observed max is actually worth as a tolerance limit, before and after the
            # clustering correction: alpha = p^n solved for p.
            "observed_max_content_at_n": round(CONFIDENCE and (1 - CONFIDENCE) ** (1 / len(d)), 4),
            "observed_max_content_at_neff": round((1 - CONFIDENCE) ** (1 / neff), 4),
            "model": model,
            "model_reason": ("half-normal survives leave-one-backbone-out"
                             if hn_ok else
                             "half-normal refuted by leave-one-backbone-out, "
                             "distribution-free unimodal bound instead"),
            "iid_recipe_survives_holdout": iid_ok,
            "holdout_shortfall_halfnorm": _finite(shortfall("halfnorm")),
            "holdout_shortfall_unimodal": _finite(shortfall("unimodal")),
            "inflation": (None if unbounded else round(inflation, 3)),
            "unbounded": bool(unbounded),
            "width": round(width, 6),
            "width_over_observed_max": (round(width / observed, 3) if observed > 0 else None),
            "halfnorm_width": round(w_halfnorm(d, neff), 6),
            "unimodal_width": round(w_unimodal(d, neff), 6),
            "holdout": hold,
        }
    return picked


def cascade(rows: list[dict], width: dict[str, float], asymmetric: bool = True) -> dict:
    """Run the device-first cascade on the population and check it against the reference.

    The referral test reads the DEVICE value, which is all a cascade has when it decides. Its
    soundness is one line: if |device_margin| > W and |delta| <= W then the reference sits on the
    same side of the bar, so the criterion's verdict transfers.
    """
    resolved, referred, wrong = [], [], []
    for r in rows:
        mg = {c: margin(r["device"][c], c) for c in BARS}
        device_accept = all(mg[c] > 0 for c in BARS)
        reference_accept = all(margin(r["reference"][c], c) > 0 for c in BARS)
        if asymmetric and not device_accept:
            # One criterion failing by more than its band settles a conjunction.
            safe = any(mg[c] < 0 and abs(mg[c]) > width[c] for c in BARS)
        else:
            safe = all(abs(mg[c]) > width[c] for c in BARS)
        if not safe:
            referred.append(r["id"])
            continue
        resolved.append(r["id"])
        if device_accept != reference_accept:
            wrong.append(r["id"])
    n = len(rows)
    refer_frac = len(referred) / n
    seconds = DEVICE_SECONDS + refer_frac * HOST_SECONDS
    return {
        "rule": "asymmetric-conjunction" if asymmetric else "symmetric",
        "n": n,
        "device_resolved": len(resolved),
        "device_resolved_fraction": round(len(resolved) / n, 4),
        "referred": len(referred),
        "referred_ids": referred,
        "decisions_reproduced": n - len(wrong),
        "mis_resolved_ids": wrong,
        "seconds_per_design": round(seconds, 1),
        "speedup_vs_host_only": round(HOST_SECONDS / seconds, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip-rate", required=True,
                    help="filter_flip_rate.py report carrying the paired per-design rows")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = json.loads(Path(args.flip_rate).read_text())["rows"]
    picked = choose(rows)
    width = {c: picked[c]["width"] for c in BARS}
    observed = {c: picked[c]["observed_max_abs_delta"] for c in BARS}

    report = {
        "source": args.flip_rate,
        "content": CONTENT,
        "confidence": CONFIDENCE,
        "criteria": picked,
        "band": width,
        "nonparametric_n_for_content": math.ceil(math.log(1 - CONFIDENCE) / math.log(CONTENT)),
        "cascade_derived_band": cascade(rows, width),
        "cascade_derived_band_symmetric": cascade(rows, width, asymmetric=False),
        "cascade_observed_max_band": cascade(rows, observed),
    }
    for c, p in picked.items():
        print("%-6s bar %s%.2f | ICC %.3f n_eff %.2f | observed max %.6f "
              "(content %.3f at n, %.3f at n_eff)" %
              (c, p["sense"], p["bar"], p["icc"], p["n_eff"], p["observed_max_abs_delta"],
               p["observed_max_content_at_n"], p["observed_max_content_at_neff"]))
        print("       -> %s x%s inflation -> width %.6f = %sx the observed max (%s)" %
              (p["model"], "inf" if p["unbounded"] else "%.2f" % p["inflation"], p["width"],
               p["width_over_observed_max"], p["model_reason"]))
        for h in p["holdout"]:
            if h["model"] == p["model"] or h["model"] == "halfnorm":
                print("          holdout fit=%-9s %-9s %-9s %.6f vs held-out %.6f  %s" %
                      (h["fit_on"], h["recipe"], h["model"], h["width"], h["heldout_max"],
                       "COVERS" if h["covers"] else "MISSES"))
    print("\n%d designs need a nonparametric bound at %.0f%% content / %.0f%% confidence"
          % (report["nonparametric_n_for_content"], CONTENT * 100, CONFIDENCE * 100))
    for key in ("cascade_observed_max_band", "cascade_derived_band_symmetric",
                "cascade_derived_band"):
        c = report[key]
        print("%-34s %s: %d/%d on device (%.0f%%), %d referred, %d/%d decisions reproduced, "
              "%.1f s/design (%.2fx vs host-only)" %
              (key, c["rule"], c["device_resolved"], c["n"],
               100 * c["device_resolved_fraction"], c["referred"],
               c["decisions_reproduced"], c["n"], c["seconds_per_design"],
               c["speedup_vs_host_only"]))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print("-> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
