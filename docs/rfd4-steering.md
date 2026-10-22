# Steering an RFD4 design run

`tt-bio design --model rfd4` runs the shipped recipe and writes one CIF per design; the
[README](../README.md#design) has the input grammar and the thinking budget. This page is the
rest of the surface: the trajectory, the per-design conditions, guidance, and the things upstream
offers that tt-bio refuses rather than approximates.

## Watching the trajectory

```
tt-bio design spec.yaml --model rfd4 --save_trajectory_every 10
```

writes `<out_dir>/<design>_trajectory.cif` beside the design: one model per saved step, CA only,
loadable in PyMOL or Mol*. The final step is always written, whatever the cadence.

It is CA-only because the all-atom form would run the decoder on every frame, which multiplies
the card time of a run by the number of saved steps, and because upstream's other format (XTC)
needs mdtraj, which tt-bio does not ship.

## A different condition per design

A batch of designs normally shares one input: same contig, same hotspots, only the noise differs.
A sampling transform runs once per design instead, so a batch can cover a target's surface rather
than repeat one patch of it. That is how upstream's VHH recipe works.

```python
from tt_bio import rfd4
from tt_bio.rfd4.replicates import resample_hotspots

features, atoms = rfd4.featurize("input.cif", seed=0)
engine = rfd4.get_engine()
out = engine.design(features, num_designs=8, nsteps=60, seed=0,
                    transforms=[resample_hotspots(3)])   # 3 of the marked hotspots, per design
```

Each design draws from the hotspots the input marks, seeded by its own index: design 2 is the
same whether you asked for 3 designs or 30, so a bigger batch extends a run instead of re-rolling
it. A transform may resample a condition, not resize the example.

## Metric guidance

A differentiable metric can push the flow each step. The metric is evaluated on the clean
prediction, and its gradient, signed by a per-modality weight schedule, is added to the state.
A negative weight minimizes the metric.

```python
from tt_bio.rfd4 import guidance, sampler

terms = {"contract_ca.sq_norm": {sampler.BB_CA: lambda t: guidance.window(t, w=-0.1, t_start=0.0, t_end=0.1)}}
out = engine.design(features, num_designs=1, nsteps=60, seed=0,
                    guidance=guidance.gradient_guidance(terms))
```

`contract_ca.sq_norm` (the summed squared CA norm) is the one differentiable metric upstream
ships, and the shipped config is the window above. Writing your own means adding an entry to
`guidance.METRICS` whose value is differentiable in the state.

## What is refused, and why

**Classifier-free guidance.** `guidance_w` other than 1.0 is refused. The blend needs a second,
unconditional pass, and upstream builds that pass by dropping `cath_code` / `x_motif` from the
batch. The released checkpoint conditions through the condition palette instead, so there is no
unconditional model to blend with; upstream's own code asserts on this and its BioIR backend
rejects the weight outright. Use metric guidance for a steered run.

**Dependency-bound metrics in `tt-bio evaluate --model rfd4`.** The tmol interface energies, the
QM IPA interaction energy, the PyRosetta charge-complementarity score and Foldseek novelty or
clustering are refused by name, each naming the dependency it would need. They are opt-in
upstream too (not in `all_inline_metrics.yaml`), and a novelty number against a database tt-bio
does not ship would not be reproducible for you anyway. Scoring structures you already have, with
a `task_csv` for per-structure tags, works: see [`docs/rfd4-evaluate.md`](rfd4-evaluate.md).

## Scoring one design against several partners

The beam-search oracle can answer more than affinity. A pre-oracle transform rewrites a design
before the fold, and `OffTargetSubstitution` replants the designed chain against each off-target
you list, which is a specificity assay:

```python
from tt_bio.rfd4.pre_oracle import OffTargetSubstitution, collect, expand

jobs, routing = expand(requests, [OffTargetSubstitution({"paralog": paralog_cif})])
metrics = collect(oracle.fold_batch(jobs), routing, len(requests))
```

The untransformed design is folded first and keeps its own keys; each variant's scores arrive
under `off_target.<name>.<key>`, so a variant that fails to fold costs the variant and not the
design you are ranking.
