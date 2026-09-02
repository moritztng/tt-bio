#!/usr/bin/env python3
"""Where does each of the 19 SPECS models get its baseline from on this box, and is any cell dead?

The gate resolves cards.p300c.machines.tt-quietbox2.models over cards.p300c.models per model
(perf_regression.card_baselines returns {**card, **machine}). Two things this task acts on:

  * a model with NO cell at either layer: _print_table prints NO BASELINE and clears all_pass, so
    the gate fails loudly on it. Loud, but it fails, so a full-SPECS gate cannot pass on qb2 while
    any model is uncovered.
  * a card-layer cell shadowed by a machine-layer copy: dead weight the gate never reads, free to
    drift away from the value that is actually used.

Prints both lists so the before/after of the dead-cell deletion can be diffed.
"""
import json, pathlib, sys

WT = "/home/ttuser/.coworker/wt/p300c-baseline-coverage-gap"
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

print(f"\nSPECS models              : {len(pr.SPECS)}")
print(f"no cell at either layer   : {unseeded or 'none'}")
print(f"dead (shadowed) card cells: {dict(shadowed) or 'none'}")
print("\nresolution fingerprint (what the gate actually reads):")
for m in sorted(resolved):
    if m in pr.SPECS:
        print(f"  {m}={resolved[m]['value']:.6f}")
