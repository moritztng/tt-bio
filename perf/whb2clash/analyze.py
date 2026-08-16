#!/usr/bin/env python3
"""Apply the Part B decision rule to the scored arms, verbatim as pre-registered.

The rule lives in `~/.coworker/state/wh-boltz2-640aa-clash-rootcause_PREREG.md` and is frozen.
This script does not choose anything: it reads the arms, runs the four gates in the order the
pre-registration states them, and prints ACCEPT / REJECT / AMBIGUOUS per arm.

  1. HARD KILL      backbone integrity is categorical -- one instance kills the arm.
  2. HARNESS INVALID  at band 768 K3 is provably inert (768 % 256 == 0), so `k3` must be
                    byte-identical to `base` and `ship` to `cc`. This is the gate's own A/A and
                    it is checked before any verdict. A digest mismatch means nothing is
                    adjudicated until the harness is fixed.
  3. ACCEPT         no hard kill, AND two-sided exact sign test p > 0.05, AND bootstrap 95 % CI
                    on mean(d_t) contains 0, AND CI upper bound <= sigma_bar.
  4. REJECT         sign test p <= 0.05 with a majority of d_t > 0, OR CI lower bound > 0, OR a
                    hard kill.
  Anything else is AMBIGUOUS, and AMBIGUOUS resolves against the lever.

sigma_bar is the pooled within-target standard deviation of clash_frac across the 5 diffusion
samples of the `null` arm. It is computed before any corner is compared.

Usage: analyze.py PARTB_DIR [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BAND640 = ["P22303", "P27694", "P03951", "P20794", "P17405", "O14744"]
BAND768 = ["P54802", "P42224", "P18074", "P47712", "Q05823", "O15111"]
TARGETS = BAND640 + BAND768
CORNERS = ["ship", "k3", "cc"]
IN_BAND_FAIL = 0.90
N_BOOT = 10000
BOOT_SEED = 0


def load(root: Path):
    arms = {}
    missing = []
    for arm in ["base"] + CORNERS + ["null"]:
        for t in TARGETS:
            p = root / f"{t}_{arm}" / "score.json"
            if not p.exists():
                missing.append(f"{t}_{arm}")
                continue
            d = json.loads(p.read_text())
            if d.get("error") or not d.get("structures"):
                missing.append(f"{t}_{arm} ({d.get('error', 'no structures')})")
                continue
            arms[(arm, t)] = d
    return arms, missing


def probe_ok(d, arm):
    """The corner an arm claims must match the corner its probe recorded."""
    want = {"base": (0, 608), "k3": (1, 608), "cc": (0, 1088), "ship": (1, 1088),
            "null": (0, 608)}[arm]
    return (int(bool(d.get("SDPA_DIV_K"))) == want[0]
            and int(d.get("SEQ_LEN_MORE_CHUNKING", -1)) == want[1])


def sign_test(d):
    """Two-sided exact sign test. Ties are dropped, as a sign test must."""
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail), pos, neg


def bootstrap_ci(d, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap on mean(d), with a fixed seed so the number is reproducible."""
    import random
    rng = random.Random(seed)
    n = len(d)
    means = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    arms, missing = load(a.root)
    rep = {"missing": missing, "targets": len(TARGETS)}
    print(f"loaded {len(arms)} arm-target cells; {len(missing)} missing")
    if missing:
        for m in missing[:12]:
            print(f"  missing {m}")

    # Provenance first. A cell whose probe disagrees with its corner is never scored.
    bad = [f"{arm}/{t}" for (arm, t), d in sorted(arms.items()) if not probe_ok(d, arm)]
    rep["probe_mismatch"] = bad
    if bad:
        print(f"PROBE MISMATCH on {len(bad)} cells -- these are not scoreable: {bad[:8]}")

    def s0(arm, t):
        return arms[(arm, t)]["structures"][0]

    # ---- gate 2 first in practice: the harness A/A at band 768 -------------------
    aa = {"k3_vs_base": [], "ship_vs_cc": []}
    for t in BAND768:
        for lhs, rhs, key in (("k3", "base", "k3_vs_base"), ("ship", "cc", "ship_vs_cc")):
            if (lhs, t) in arms and (rhs, t) in arms:
                same = s0(lhs, t)["digest"] == s0(rhs, t)["digest"]
                aa[key].append({"target": t, "identical": same,
                                "lhs": s0(lhs, t)["digest"], "rhs": s0(rhs, t)["digest"]})
    rep["band768_AA"] = aa
    aa_fail = [r for v in aa.values() for r in v if not r["identical"]]
    print(f"\nband-768 A/A (K3 provably inert): "
          f"{sum(1 for v in aa.values() for r in v if r['identical'])} identical, "
          f"{len(aa_fail)} differing")
    for r in aa_fail:
        print(f"  DIFFERS {r['target']}: {r['lhs']} vs {r['rhs']}")

    # ---- null calibration, computed before any corner is compared ---------------
    within = []
    for t in TARGETS:
        if ("null", t) not in arms:
            continue
        fr = [s["clash_frac"] for s in arms[("null", t)]["structures"] if s["clash_frac"] is not None]
        if len(fr) < 2:
            continue
        m = sum(fr) / len(fr)
        var = sum((x - m) ** 2 for x in fr) / (len(fr) - 1)
        within.append({"target": t, "n": len(fr), "mean": m, "sd": math.sqrt(var),
                       "counts": [s["n_clash"] for s in arms[("null", t)]["structures"]]})
    sigma_bar = (math.sqrt(sum(w["sd"] ** 2 for w in within) / len(within))
                 if within else None)
    rep["null"] = {"per_target": within, "sigma_bar": sigma_bar}
    print(f"\nsigma_bar (pooled within-target SD of clash_frac over the null arm's 5 samples): "
          f"{sigma_bar if sigma_bar is None else f'{sigma_bar:.6f}'}"
          f"  from {len(within)} targets")
    for w in within:
        print(f"  {w['target']:8s} counts={w['counts']} sd={w['sd']:.6f}")

    # ---- per-arm verdicts -------------------------------------------------------
    rep["arms"] = {}
    for arm in CORNERS:
        avail = [t for t in TARGETS if (arm, t) in arms and ("base", t) in arms]
        d = [s0(arm, t)["clash_frac"] - s0("base", t)["clash_frac"] for t in avail]
        kills = []
        for t in avail:
            A, B = s0(arm, t), s0("base", t)
            if A["backbone_gaps"] > 0 and B["backbone_gaps"] == 0:
                kills.append(f"{t}: {A['backbone_gaps']} backbone gap(s), base has 0")
            if (A["in_band_frac"] is not None and B["in_band_frac"] is not None
                    and A["in_band_frac"] < IN_BAND_FAIL <= B["in_band_frac"]):
                kills.append(f"{t}: in_band {A['in_band_frac']} below {IN_BAND_FAIL}, "
                             f"base {B['in_band_frac']}")
        out = {"n": len(avail), "targets": avail, "d": d, "hard_kills": kills}
        if len(avail) < 2:
            out["verdict"] = "INSUFFICIENT"
        else:
            p, pos, neg = sign_test(d)
            lo, hi = bootstrap_ci(d)
            mean = sum(d) / len(d)
            out.update({"sign_p": p, "n_worse": pos, "n_better": neg, "mean_d": mean,
                        "ci_lo": lo, "ci_hi": hi, "sigma_bar": sigma_bar})
            accept = (not kills and p > 0.05 and lo <= 0 <= hi
                      and sigma_bar is not None and hi <= sigma_bar)
            reject = bool(kills) or (p <= 0.05 and pos > neg) or lo > 0
            out["verdict"] = ("REJECT" if reject else "ACCEPT" if accept else "AMBIGUOUS")
        rep["arms"][arm] = out

        print(f"\n--- arm {arm} (n={out['n']}) ---")
        for t, dd in zip(avail, d):
            print(f"  {t:8s} base={s0('base', t)['n_clash']:3d} {arm}={s0(arm, t)['n_clash']:3d} "
                  f"d={dd:+.6f}")
        if out["verdict"] != "INSUFFICIENT":
            print(f"  worse on {out['n_worse']}, better on {out['n_better']}, "
                  f"sign p={out['sign_p']:.4f}")
            print(f"  mean d={out['mean_d']:+.6f}  95% CI [{out['ci_lo']:+.6f}, "
                  f"{out['ci_hi']:+.6f}]  sigma_bar="
                  f"{'n/a' if sigma_bar is None else f'{sigma_bar:.6f}'}")
        for k in kills:
            print(f"  HARD KILL {k}")
        print(f"  => {out['verdict']}")

    incomplete = bool(missing) or bool(bad) or bool(aa_fail) or sigma_bar is None
    print(f"\nGATE COMPLETE: {not incomplete}"
          + ("" if not incomplete else "  (verdicts above are provisional)"))
    if aa_fail:
        print("HARNESS INVALID: band-768 A/A differs -- nothing is adjudicated until it is fixed.")
    rep["gate_complete"] = not incomplete
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
