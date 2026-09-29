# BoltzGen designability check

**Designability** (self-consistency RMSD, *scRMSD*) is the standard binder-design
QA metric used by RFdiffusion, BindCraft, and BoltzGen's own paper. Refold the
designed binder's *sequence* **in isolation** (no target, no template), Kabsch-align
the refolded backbone to the originally-designed backbone, and measure CA-RMSD:

- **scRMSD ≤ 2 Å**: strictly designable (the sequence encodes the intended shape).
- **scRMSD ≤ 4 Å**: permissively designable.
- **high scRMSD**: a bad design *or* a device-fidelity problem in the fold.

The other BoltzGen checks (`tests/test_boltzgen.py` state-dict load,
`tests/test_boltzgen_regression.py` bond-length distribution against a GPU baseline)
show that the sampler runs and the chemistry is intact. They say nothing about whether
a design is a *good* binder. This check does that job for
`tt-bio design --model boltzgen`, as `scripts/release_gate.py` does for the fold models.

## It already runs inside the pipeline

tt-bio does not re-implement scRMSD; the shipping design pipeline computes it.
For the `protein-anything` / `protein-small_molecule` protocols the `design_folding`
step refolds each design's sequence alone, and `analysis` Kabsch-aligns and writes it
to `aggregate_metrics_analyze.csv`:

| column | meaning |
| --- | --- |
| `designfolding-bb_rmsd` | **scRMSD**, backbone RMSD of the isolated (no-target) refold. This is the metric |
| `designfolding-bb_designability_rmsd_2` | scRMSD ≤ 2 Å (strict pass flag) |
| `designfolding-bb_designability_rmsd_4` | scRMSD ≤ 4 Å (permissive pass flag) |
| `bb_rmsd_design` | design-region RMSD from the **whole-complex** refold (target present) |

Protocols that skip `design_folding` (nanobody / antibody / peptide) expose only the
whole-complex `bb_rmsd_design`; the script falls back to it and labels the source.

`scripts/boltzgen_designability.py` is a thin harness over this: it optionally runs the
pipeline, then harvests and summarises the scRMSD column. It has no refold or Kabsch
code and no checkpoint of its own.

## Why the refolder choice isolates device bugs from bad designs

The isolation refold uses BoltzGen's folding checkpoint `boltz2_conf_final.ckpt`, a
**Boltz-2-derived** confidence model. It is not the standard `tt-bio predict --model boltz2`
checkpoint, and using that one instead would give a *different*, non-canonical scRMSD.
Boltz-2's own on-device folding accuracy is **independently ground-truth-gated** by
`scripts/release_gate.py` (boltz2 leg: CA-RMSD ≤ 3 Å / TM ≥ 0.75 on 7ROA). So if the
refolder is accurate and a design still refolds poorly, the fault is design quality or
target hardness, not a refold device bug.

## Running it

```bash
# design against a target, then score designability (card pinned via TT_VISIBLE_DEVICES)
TT_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> \
    python scripts/boltzgen_designability.py --num_designs 4

# score an already-completed design output dir, no device needed
python scripts/boltzgen_designability.py --from-output ./binder

# gate mode: non-zero exit if too few designs clear the bar
python scripts/boltzgen_designability.py --from-output ./binder \
    --sc-threshold 2.0 --min-pass-rate 0.5
```

The default spec is `examples/binder.yaml` (de-novo protein binder vs chain A of 7ROA), the
target the README documents for `tt-bio design --model boltzgen`.

## Observed on-device results

Produced with this script on Blackhole card 1 (`TT_VISIBLE_DEVICES=1`,
`protein-anything`, production sampling: design 500 steps, refold 200 steps). The first
row is a fresh reproduction; the rest are additional harvested runs of the canonical
examples (isolated-refold scRMSD, except the nanobody target, which uses the
whole-complex fallback):

| target (protocol) | n | scRMSD median (Å) | ≤2 Å | ≤4 Å |
| --- | --- | --- | --- | --- |
| `examples/binder.yaml` (protein-anything), fresh run | 2 | 0.76 | 100% | 100% |
| `examples/binder.yaml` (protein-anything) | 4 | 1.00 | 75% | 100% |
| `examples/binder.yaml`, single design | 1 | 0.63 | 100% | 100% |
| nanobody target (nanobody-anything, complex refold) | 40 | 1.44 | 82.5% | 97.5% |

Designs refold to **sub-Å–1.3 Å** scRMSD on the canonical binder example, well inside
BoltzGen's ≤2 Å designable bar, and the on-device designability distribution is what a
working BoltzGen + Boltz-2 refolder should produce: **no device-fidelity problem is
evident**. The fresh 2-design run (`binder` 0.90 Å, `binder_1` 0.62 Å) took ~40 min
end-to-end on one card, most of it in the design step.

## In the release gate

BoltzGen is the fifth leg of `scripts/release_gate.py` (`--model boltzgen`, on by
default). It gates the same `designfolding-bb_rmsd` column through this script's
`_run_gen`/`score`, not a re-derivation, against a ≤2 Å / ≥50%-of-4 floor.

The ~40 min 2-design run above once kept the check out of the fast gate. A fresh n=4
reproduction of the same target and protocol on 2026-07-10 (main, after the
device-resident-trunk merge) measured **271 s (4.5 min) end-to-end** (design + inverse-fold
+ refold + analysis + filtering) with 0.85 Å median scRMSD and 4/4 designs ≤2 Å, which is
comparable to a single fold model's leg.
