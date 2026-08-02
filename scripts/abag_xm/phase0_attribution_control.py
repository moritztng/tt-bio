"""PHASE 0 attribution control verdict (deep-N saturation fullpanel, galaxy window 1).

Pre-registered in state/abag-xm-deepn-saturation-fullpanel.md: the galaxy p2 panel showed
protenix-v2 and esmfold2 systematically DockQ-LOW vs the qb1 tier_a distribution (~0.01-0.02),
confounded between (i) mps-1-vs-5 chunk-width numerics, (ii) WH-vs-BH architecture, (iii) code
drift. The control re-folds 7 GT targets on the galaxy at N=16 with NEW disjoint seeds
(protenix 30000 mps=5 -- the BH-matching config; esmfold2 50000 at the exact p2 config,
single-sequence 10/100) and answers:

  protenix-v2: control consistent with tier_a  -> mps explains the offset (class: chunk
      numerics) -> run the px campaign leg at mps=5. Control still galaxy-low -> class: arch
      -> STOP the px galaxy leg and report the first clean WH-vs-BH numerics signal.
  esmfold2: control consistent with tier_a -> the p2 offset was code drift (fixed at this
      HEAD) -> proceed. Control still galaxy-low AND control ~= p2 galaxy values -> arch ->
      STOP the esm leg. (esmfold2 never took an mps value in either PHASE 0 arm, so mps is
      not a candidate cause -- pass-16 correction.)

Verdict bars are the phase0 v2 ones: per stat, exceedance (frac targets |delta| > q95_within)
<= 0.15 AND ratio (med|delta_cross| / med|delta_within|) <= 2 -> CONSISTENT. n=7 targets gives
exceedance granularity 1/7 ~= 0.143.

Runs on qb1 (needs pandas/numpy/scipy + the harvested p25 control tree + scored TSVs):
  python3 scripts/abag_xm/phase0_attribution_control.py \
      --p25 /home/ttuser/abag_xm/deepn/phase0 --p0 /home/ttuser/abag_xm/deepn/phase0 \
      --tier_a /home/ttuser/abag_xm/tier_a --out .../attribution_verdict.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "p0check", pathlib.Path(__file__).with_name("phase0_cross_hardware_check.py"))
p0check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(p0check)

MODELS = {"protenix_v2": ("protenix", "confidence_score"),
          "esmfold2": ("esmfold2", "plddt")}
GAL_SUFFIX = {"protenix_v2": "_protenix", "esmfold2": "_esmfold2"}
CTRL_TARGETS = ["21tw", "9d3j", "9ma0", "9obn", "9q6y", "9udq", "9wpm"]


def load_control(p25: pathlib.Path, md: str, conf_key: str) -> dict:
    """p25 control arm: conf from results.json all_runs, DockQ from the scored TSV, by rank."""
    dockq: dict[str, dict[int, float]] = {}
    tsv = p25 / f"p25ctrl_{md}_dockq.tsv"
    for line in tsv.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3 or parts[2].startswith("ERR"):
            continue
        dockq.setdefault(parts[0], {})[int(parts[1])] = float(parts[2])
    out = {}
    for t in CTRL_TARGETS:
        rjs = sorted((p25 / f"p25ctrl_{md}" / t).glob("*_results_*/results.json"))
        if not rjs or t not in dockq:
            continue
        runs = json.loads(rjs[0].read_text())[0].get("all_runs", [])
        conf = {int(r["rank"]): float(r[conf_key]) for r in runs
                if r.get(conf_key) is not None and np.isfinite(float(r[conf_key]))}
        ranks = sorted(set(conf) & set(dockq[t]))
        if len(ranks) < 8:
            continue
        out[t] = {"conf": np.array([conf[r] for r in ranks]),
                  "dockq": np.array([dockq[t][r] for r in ranks])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p25", required=True, type=pathlib.Path)
    ap.add_argument("--p0", default="/home/ttuser/abag_xm/deepn/phase0", type=pathlib.Path)
    ap.add_argument("--tier_a", default="/home/ttuser/abag_xm/tier_a", type=pathlib.Path)
    ap.add_argument("--out", default=None, type=pathlib.Path)
    args = ap.parse_args()

    verdict = {}
    for md, (dirname, conf_key) in MODELS.items():
        ctrl = load_control(args.p25, dirname, conf_key)
        ta = p0check.load_tiera(args.p0, args.tier_a, md)
        gal = p0check.load_galaxy(args.p0, GAL_SUFFIX[md])
        gal = {t: gal[t] for t in CTRL_TARGETS if t in gal}

        rows, pooled = p0check.compare(ctrl, ta, seed=20260803)
        summ = p0check.summarize(md, rows, pooled)

        # side-by-side: control vs p2-galaxy on the same targets (same-hardware seed check)
        side = {}
        for t in sorted(set(ctrl) & set(gal)):
            cs = p0check.stats_of(ctrl[t]["conf"], ctrl[t]["dockq"])
            gs = p0check.stats_of(gal[t]["conf"], gal[t]["dockq"])
            side[t] = {k: {"ctrl": cs.get(k), "p2gal": gs.get(k),
                           "delta": (cs[k] - gs[k]) if k in cs and k in gs else None}
                       for k in ("dq_mean", "oracle", "user", "conf_mean")}

        key_stats = {k: v for k, v in summ["stats"].items()
                     if k in ("dq_mean", "oracle", "user", "dq_q90", "conf_mean", "spearman")}
        consistent = all(v["exceed_q95_within"] <= 0.15 and v["ratio_med"] <= 2.0
                         for v in key_stats.values() if not np.isnan(v.get("ratio_med", np.nan)))
        bias = {k: v.get("bias_frac_above") for k, v in key_stats.items()}
        verdict[md] = {"n_control_targets": len(ctrl), "vs_tier_a": key_stats,
                       "consistent_with_tier_a": consistent, "bias_frac_above": bias,
                       "control_vs_p2galaxy": side,
                       "targets": sorted(ctrl)}

    text = json.dumps(verdict, indent=1, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
