#!/usr/bin/env python3
"""Where does each gated model's baseline actually come from on this box, and can any of them
fall through to a wrong number?

The gate resolves cards.p300c.machines.tt-quietbox2.models over cards.p300c.models per model
(perf_regression.card_baselines returns {**card, **machine}). Two things can go wrong:

  * a cell reseeded at the card layer while a stale copy survives in the machine block is silently
    shadowed, and the two drift apart;
  * a model with no cell at either layer could be skipped rather than gated.

The second one is safe by construction: _print_table prints NO BASELINE and clears all_pass, so an
unseeded model fails the gate loudly. This script proves the first one leg by leg.
"""
import json, pathlib, sys

WT = "/home/ttuser/.coworker/wt/qb2-card-layer-baseline-reseed"
sys.path.insert(0, WT)
sys.path.insert(0, f"{WT}/scripts")
import perf_regression as pr

d = json.loads(pathlib.Path(f"{WT}/docs/perf_baselines.json").read_text())
card = d["cards"]["p300c"]
card_models, mach_models = card.get("models", {}), card["machines"]["tt-quietbox2"]["models"]
resolved = pr.card_baselines(d, "p300c", "tt-quietbox2")

unseeded, shadowed = [], []
print(f"{'model':18s} {'layer':9s} {'value':>14s}  {'date':10s}")
for m in pr.SPECS:
    if m not in resolved:
        unseeded.append(m)
        print(f"{m:18s} {'NONE':9s} {'-':>14s}  {'-':10s}")
        continue
    if m in mach_models and m in card_models:
        shadowed.append((m, card_models[m]["value"]))
    e = resolved[m]
    layer = "machine" if m in mach_models else "card"
    print(f"{m:18s} {layer:9s} {e['value']:>14.6f}  {e.get('date','?'):10s}")

print(f"\nresolves at card layer   : {sorted(set(card_models) - set(mach_models))}")
print(f"resolves at machine layer: {sorted(mach_models)}")
print(f"no cell at either layer  : {unseeded}  "
      f"-> gate prints NO BASELINE and FAILS, it does not default")
print(f"card cell shadowed by a machine cell: "
      f"{[m for m, _ in shadowed] or 'none'}"
      + (f"  (dead card values {dict(shadowed)})" if shadowed else ""))
print("\nOK: every reseeded cell in this task resolves at the card layer, unshadowed."
      if not (set(m for m, _ in shadowed) & set(card_models) & {
          "boltz2", "esmfold2", "esmfold2-fast", "esmc-300m", "esmc-600m", "esmc-6b",
          "boltzgen", "boltz2-affinity"})
      else "\nFAIL: a cell this task reseeded is shadowed by a machine-layer copy.")
