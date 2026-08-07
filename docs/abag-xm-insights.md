# AbAg-XM deep-N: what 383,000 labelled samples say about sampling and selection

Analysis of the AbAg-XM deep-N asset (four architecturally independent structure predictors,
161 antibody-antigen targets, N up to 256 samples per target, every sample DockQ-labelled).
Source data is frozen and unmodified; this document is the technical backing for the site.

Reproduce every number:

```
python3 scripts/abag_xm_insights/build_insights.py -o site/data/insights.json
python3 scripts/abag_xm_insights/summarise.py site/data/insights.json
```

Inputs: `~/abag_xm/deepn/dataset/{model}_samples.parquet` and
`~/abag_xm/deepn/galaxy/fleet_results.jsonl`. Nothing is fabricated, hand-entered or carried
over from a previous document.

---

## Method

**Pools.** Rungs nest exactly (verified: rung-128 chunk 0 is bit-identical to rung-256 chunk 0),
so the 256-sample rung is a strict superset of every lower rung. All curves are computed by
subsampling inside that one pool rather than by comparing separately folded arms, which removes
the differing-target-set problem that forces DATASHEET section 4 to caveat its rows. Targets
whose top rung landed short (boltz2 9ua5, opendde 9rye/9gvn/9xqn, protenix 9d73) are dropped so
that every N compares the same depth.

**Exact curves, no Monte Carlo.** For a pool of n samples ordered worst-first by any ranking key,
the probability that sorted position j is the top-ranked member of a uniformly drawn k-subset is
`C(j, k-1) / C(n, k)`. Applying that one weight matrix to different value columns gives the
oracle best-of-k curve, the confidence selector's expected pick, threshold-crossing probability,
maximum epitope overlap, and any candidate selector — all exact, at every k from 1 to 256. The
engine is verified against brute-force enumeration over all k-subsets. It reproduces the
campaign's own B=200 Monte-Carlo curves (DATASHEET section 7) to ~1e-3, so it supersedes them
without contradicting them.

**Ties** in a ranking key are resolved against the selector: a selector that cannot separate two
samples is credited with the worse one. All confidence flavours are continuous float64, so this
is a discipline rather than a knob.

**Bootstrap.** B=20000 resamples over targets, one shared resample draw (seed 20260802, the same
convention as DATASHEET section 6), so CIs from different models and metrics are comparable and
their differences are genuinely paired. Implemented as a count-matrix matmul, which keeps the
full 256-column curve bootstrap in tens of MB.

**Cost.** Per-model, per-target card-seconds come from `galaxy/fleet_results.jsonl` (seconds of
one Wormhole Galaxy chip per completed 64-sample chunk), the source DATASHEET section 8 names as
the cost authority. Median cost per sample over the four-model common target set: esmfold2
14.1 s, boltz2 17.8 s, opendde-abag 31.0 s, protenix-v2 31.0 s.

> **Do not cost from the `wall_s` column in the packaged sample parquets.** It is byte-identical
> across all four models for the same (target, chunk) — 631 of 631 shared chunks — and takes only
> ~90 distinct minute-quantised values. It is a packaging artifact, not a per-model fold time.
> This is a defect in the packaged asset, found here; the fleet log is unaffected.

---

## Q1. Sampling scales. Selection does not.

`oracle(k)` = expected best DockQ among k samples. `user(k)` = expected DockQ of the sample the
model's own selector returns from those k. At k=1 the two coincide, and both equal the per-target
mean DockQ — the value of drawing one sample at random.

| model | targets | random | oracle@16 | oracle@256 | user@16 | user@256 | gap@256 |
|---|---|---|---|---|---|---|---|
| boltz2 | 160 | 0.2266 | 0.3388 | 0.4285 | 0.2464 | 0.2559 | +0.1726 [+0.1427, +0.2035] |
| opendde-abag | 153 | 0.5039 | 0.5663 | 0.6140 | 0.5097 | 0.5018 | +0.1122 [+0.0929, +0.1333] |
| protenix-v2 | 159 | 0.2963 | 0.4111 | 0.5328 | 0.3181 | 0.3207 | +0.2121 [+0.1827, +0.2429] |
| esmfold2 | 160 | 0.2497 | 0.3311 | 0.4055 | 0.2745 | 0.2754 | +0.1301 [+0.1118, +0.1496] |

Going from 16 to 256 samples is 16x the compute. What it buys:

| model | oracle gain 16→256 | user gain 16→256 |
|---|---|---|
| boltz2 | +0.0897 [+0.0717, +0.1093] | +0.0094 [-0.0043, +0.0236] |
| opendde-abag | +0.0476 [+0.0374, +0.0595] | **-0.0079 [-0.0147, -0.0013]** |
| protenix-v2 | +0.1216 [+0.1011, +0.1433] | +0.0026 [-0.0066, +0.0118] |
| esmfold2 | +0.0744 [+0.0618, +0.0885] | +0.0009 [-0.0030, +0.0052] |

Every oracle gain is large and its CI excludes zero. Three of the four user gains are
indistinguishable from zero. The fourth, opendde-abag, is significantly **negative**: on the
antibody-specialised model, sampling 16x deeper makes the delivered answer measurably worse,
because deeper pools surface more high-confidence wrong poses than high-confidence right ones.

**Selection efficiency** `SE(k) = (user(k) - random) / (oracle(k) - random)` is the share of the
ceiling that sampling unlocks which actually reaches the user. At k=256: boltz2 0.145
[0.057, 0.236], protenix-v2 0.103 [0.028, 0.180], esmfold2 0.165 [0.085, 0.250], opendde-abag
-0.019 [-0.121, +0.088] (indistinguishable from zero, i.e. no better than picking at random).
SE decays with k for every model.

**Effective N** — invert the oracle curve at the delivered accuracy: 256 samples plus the
model's own confidence delivers what a perfect selector would have got from

| model | N_eff (mean DockQ) | N_eff (≥0.23) | N_eff (≥0.49) | N_eff (≥0.80) |
|---|---|---|---|---|
| boltz2 | 1.98 [1.42, 2.85] | 2.16 [1.36, 3.45] | 2.42 [1.30, 4.84] | 1.91 [1.00, 6.87] |
| opendde-abag | 1.00 [1.00, 1.57] | 1.44 [1.00, 3.46] | 2.72 [1.34, 14.61] | 1.00 [1.00, 1.58] |
| protenix-v2 | 1.76 [1.22, 2.48] | 1.89 [1.19, 3.15] | 3.01 [1.40, 5.96] | 1.00 [1.00, 2.30] |
| esmfold2 | 2.09 [1.60, 2.85] | 2.47 [1.42, 4.83] | 3.02 [1.56, 8.80] | 1.81 [1.00, 4.23] |

**256 samples in, 1 to 3 samples out.** The upper CI bound never exceeds 15 on any model,
threshold or metric.

---

## Q2. Confidence knows which target is easy. It does not know which sample is right.

Spearman correlation between a confidence score and DockQ, computed two ways: *within* a target
over its 256 samples, then summarised across targets; and *across* targets between target-mean
confidence and target-mean DockQ.

| model | flavour | within-target median | within-target mean | across-target |
|---|---|---|---|---|
| boltz2 | confidence_score | +0.029 | +0.051 [+0.015, +0.088] | 0.670 |
| boltz2 | iptm | +0.049 | +0.043 [+0.007, +0.079] | 0.702 |
| boltz2 | ptm | +0.039 | +0.043 [+0.008, +0.079] | 0.656 |
| boltz2 | complex_plddt | +0.024 | +0.032 [-0.016, +0.080] | 0.537 |
| opendde-abag | confidence_score | +0.069 | +0.040 [-0.015, +0.096] | 0.774 |
| opendde-abag | iptm | **-0.021** | +0.023 [-0.032, +0.078] | 0.754 |
| opendde-abag | complex_plddt | +0.025 | +0.013 [-0.040, +0.065] | 0.595 |
| protenix-v2 | confidence_score | +0.185 | +0.157 [+0.097, +0.218] | 0.723 |
| protenix-v2 | iptm | +0.182 | +0.156 [+0.096, +0.218] | 0.737 |
| protenix-v2 | complex_plddt | -0.008 | -0.011 [-0.062, +0.040] | 0.217 |
| esmfold2 | plddt (its selector) | +0.086 | +0.116 [+0.069, +0.165] | 0.681 |
| esmfold2 | ptm | +0.153 | +0.179 [+0.131, +0.228] | 0.788 |

Across-target correlation is 0.54 to 0.79 everywhere. Within-target correlation is 0.03 to 0.19.
The scores are informative about problem difficulty and near-uninformative about which of their
own samples to hand over.

`iptm` is the score the field uses for interfaces. Its within-target median is +0.049 on boltz2
and **-0.021** on opendde-abag. It does not rank samples within a target.

Protenix-v2 is the strongest within-target ranker of the four (+0.185), which corroborates the
earlier N=12 to N=23 pilot finding that protenix-v2's confidence is the best Ab-Ag trust signal
available — now at ~700x the sample scale. It is still an order of magnitude short of usable.

**Is the failure concentrated on hard targets?** No. Median within-target rho by target-difficulty
quartile (selector flavour), hardest to easiest: boltz2 +0.045 / -0.026 / +0.028 / +0.043;
opendde-abag +0.185 / -0.068 / +0.114 / -0.041. Both fail uniformly. Protenix-v2 (+0.183 / +0.179
/ +0.506 / +0.112) and esmfold2 (+0.019 / +0.049 / +0.179 / +0.295) do rank better on easier
targets, but never well enough on the hard ones, which are the ones that matter.

**Control.** The "user" pick is genuinely the model's own shipped selector: the `selector` column
equals `confidence_score` for 100% of targets on all three co-folders, and for esmfold2 holds
plddt (which the parquet stores nowhere else). Spearman(selector, file `rank`) = -0.87 to -0.99,
i.e. `rank` is just the selector ordering — correctly excluded as an independent signal.

---

## Q3. Failure is a site-discovery failure, not a pose-refinement failure.

The per-sample predicted-vs-native epitope overlap (`epitope_jaccard`, EJ) is bimodal. The trough
between the modes, searched over the interior of the range and taken as the median of the four
per-model troughs, gives EJ* = 0.558 (per-model troughs 0.458 to 0.625). Every (model, target) is
then one of three states.

| model | solved | right site, wrong pose | never finds site | share of failures that never find the site |
|---|---|---|---|---|
| boltz2 | 93 | 21 | 46 | 69% |
| opendde-abag | 117 | 13 | 23 | 64% |
| protenix-v2 | 106 | 7 | 25 | 78% |
| esmfold2 | 76 | 18 | 44 | 71% |

Median max-EJ over the whole pool, unsolved vs solved targets: boltz2 0.318 vs 0.826,
opendde-abag 0.403 vs 0.867, protenix-v2 0.400 vs 0.862, esmfold2 0.264 vs 0.866. The separation
is clean on all four. Two thirds to four fifths of all failures are targets where no sample ever
lands on the right epitope.

**Does depth buy site discovery?** Less than it buys pose quality. Over k = 1 to 256 (boltz2):
P(at least one sample finds the site) 0.409 → 0.544, a 33% relative gain, while P(at least one
acceptable pose) goes 0.301 → 0.581, a 93% relative gain. Protenix-v2 over k = 1 to 128: +30% vs
+58%. Esmfold2 over k = 1 to 64: +16% vs +34%. Opendde-abag is the exception, +16% vs +14% — the
antibody-specialised model gains on both at the same rate.

**A sub-claim that did NOT hold.** The relative shape of the *mean* max-EJ curve and the mean
max-DockQ curve is nearly identical for every model (boltz2, normalised gain at k=2/8/32/128:
EJ 0.15/0.42/0.66/0.88 vs DockQ 0.15/0.44/0.66/0.88). Site discovery does not visibly plateau
while pose accuracy climbs. The site-vs-pose asymmetry above is real but shows up in the
threshold-crossing rates, not in a plateau of the mean. Reported as measured.

**Coverage caveat.** EJ labels are complete for boltz2 (161/161 at N=256) and opendde-abag
(153/156). For protenix-v2 and esmfold2 the epitope scorer ran on a subset of the 64-sample
chunks, so those two are analysed at their deepest chunk-aligned depth covering ≥100 targets
(128 and 64) and are never quoted at N=256. Missingness is chunk-aligned rather than per-sample,
and DockQ means are close between labelled and unlabelled samples (protenix-v2 0.294 vs 0.300),
so it is a scorer-coverage gap and not informative missingness. Partial coverage can only
under-count site discovery, so "never finds the site" is a conservative count for those two.

**Do the models fail on the same targets?** Partly. Pairwise Jaccard of failure sets ranges 0.21
(opendde-abag vs esmfold2) to 0.56 (boltz2 vs esmfold2) — the two generic co-folders fail most
alike; the antibody-specialised model fails most differently. On the 117 targets where all four
carry EJ labels, only **10 are failed by all four models**, while the best single model fails 26.

---

## Q4. At matched compute, spend it on different models, not on more samples.

Per target and per budget in card-hours, a strategy assigns each model
`n_m = min(256, share / cost_m(target))` samples. The union's oracle DockQ and threshold
probabilities are computed exactly from the product of the per-model best-of-k CDFs. The
pre-declared comparison is single-model-deep against an even four-way split; no subset search
feeds it. 151 targets carry pools and fleet cost records for all four models.

Oracle mean DockQ:

| card-h / target | 0.04 | 0.08 | 0.15 | 0.5 | 1.0 | 2.5 |
|---|---|---|---|---|---|---|
| boltz2 alone | 0.307 | 0.332 | 0.353 | 0.391 | 0.414 | 0.426 |
| opendde-abag alone | 0.537 | 0.554 | 0.566 | 0.588 | 0.599 | 0.610 |
| protenix-v2 alone | 0.354 | 0.385 | 0.413 | 0.467 | 0.499 | 0.529 |
| esmfold2 alone | 0.310 | 0.329 | 0.346 | 0.379 | 0.395 | 0.401 |
| **even four-way** | **0.517** | **0.598** | **0.624** | **0.660** | **0.677** | **0.698** |

Fraction of targets reaching DockQ ≥ 0.23:

| card-h / target | 0.04 | 0.08 | 0.15 | 0.5 | 1.0 | 2.5 |
|---|---|---|---|---|---|---|
| boltz2 alone | 0.411 | 0.449 | 0.479 | 0.528 | 0.555 | 0.570 |
| opendde-abag alone | 0.699 | 0.713 | 0.723 | 0.742 | 0.750 | 0.760 |
| protenix-v2 alone | 0.500 | 0.543 | 0.580 | 0.656 | 0.707 | 0.758 |
| esmfold2 alone | 0.392 | 0.415 | 0.439 | 0.500 | 0.533 | 0.543 |
| **even four-way** | **0.689** | **0.783** | **0.809** | **0.851** | **0.874** | **0.898** |

From 0.08 card-h/target upward the even four-way split beats every single-model strategy at every
budget, on both metrics. The headline comparison:

**Four models at 0.08 card-h/target — 11.9 samples in total, about 3 per model — reach DockQ ≥
0.23 on 0.783 [0.719, 0.843] of targets. The best single model at 2.5 card-h/target, 31x the
compute and 233.8 samples, reaches 0.760 [0.692, 0.826].** Oracle mean DockQ over the same
comparison: 0.598 [0.549, 0.647] vs 0.610 [0.559, 0.662] — the twelve mixed samples match 234
deep ones on mean quality and beat them on solve rate.

Model diversity is worth roughly 30x its cost in sampling depth.

**The specialisation claim, verified.** opendde-abag alone at 0.08 card-h/target (9.8 samples)
scores oracle 0.554, well above boltz2 at 2.5 card-h/target (255.3 samples, 0.426) at 31x less
compute. Architecture and training domain dominate sample count.

**And the catch, reported in full.** Nobody can currently harvest the union's ceiling. Taking the
globally highest-confidence sample across the four pools (exact computation; confidence is not
calibrated across models, which is the point) delivers 0.450 at 0.08 card-h/target, then
**declines to 0.406 as the budget grows to 2.5** — worse than simply using opendde-abag alone
(0.500 to 0.509, flat). The union has by far the highest ceiling and the worst naive delivery.
This is consistent with the earlier pre-declared cross-model consensus-confidence pilot, which
was also a null result.

---

## Q6. Sampling alone does not get there.

Fitting the measured k = 1..256 curves with a saturating family `y = a - b·N^(-alpha)` (asymptote
bounded to 1.0, which a DockQ or a fraction cannot exceed) and a log-linear family
`y = c + d·log2(N)`:

The saturating fit is **unidentifiable** over the measured range for boltz2, protenix-v2 and
esmfold2 — it walks its asymptote to the 1.0 bound, i.e. it degenerates into the log-linear form.
Only opendde-abag admits a genuine finite ceiling (a = 0.792, alpha = 0.086, rmse 0.0002). Within
N ≤ 256 the curves carry no evidence of saturation. This is the quantitative form of the
campaign's own conclusion that N* = 256 was a decision cap and not a measured knee.

The only defensible extrapolation is therefore the log-linear one, which assumes no ceiling and
is consequently **optimistic** — a lower bound on the N required. Under it, reaching 80% of
targets:

| model | measured @256 (≥0.23) | N for 80% @≥0.23 | card-h/target | N for 80% @≥0.49 | card-h/target |
|---|---|---|---|---|---|
| protenix-v2 | 0.761 | 5.6e2 | 4.5 | 1.3e4 | 106 |
| opendde-abag | 0.765 | 2.3e3 | 20 | 3.8e6 | 3.2e4 |
| boltz2 | 0.581 | 2.9e4 | 142 | 9.1e6 | 4.4e4 |
| esmfold2 | 0.544 | 1.4e5 | 546 | 6.0e12 | 2.3e10 |

Merely acceptable poses on 80% of targets is within reach for the two strongest models
(hundreds to thousands of samples). Medium-quality poses on 80% of targets needs 10^4 samples for
the best model and 10^6 to 10^12 for the rest — 3.2e4 card-hours per target on opendde-abag.
Every entry beyond N = 256 is an extrapolation past the measured range, by factors of 2 to 10^10,
and the 10^12 entry should be read as "not by sampling", not as a number.

---

## Q7. You cannot get interface accuracy and loop accuracy from the same sample.

Deep sampling does improve CDR-H3: best-of-k H3 RMSD falls from 1.37 Å at k=1 to 0.77 Å at k=256
on boltz2, 1.02 → 0.70 Å on opendde-abag. But within a target's pool, DockQ and H3 accuracy are
essentially uncorrelated — median Spearman(DockQ, -H3 RMSD) is +0.061 (boltz2), +0.067
(opendde-abag), +0.066 (protenix-v2), -0.007 (esmfold2), and fewer than 2.5% of targets exceed
rho = 0.5 on any model.

Taking the DockQ-best sample instead of the H3-best sample costs, in mean H3 RMSD: boltz2 +0.45
[+0.38, +0.53] Å, opendde-abag +0.30 [+0.20, +0.41] Å, protenix-v2 +0.38 [+0.27, +0.50] Å,
esmfold2 +0.46 [+0.35, +0.57] Å. If you need both the interface and the loop, you need two
different samples from the pool, and no available signal tells you which.

---

## Limitations

- **161 of 164 targets.** 9ly2 / 9ly3 / 9lz2 are 3-way Ab:Ag hetero-hexamers that the DockQ
  scorer's interface model does not support. After dropping short top rungs, per-model target
  counts are 153 to 160; the four-model common set is 151.
- **Hardware exclusions.** opendde-abag drops 9i3p / 9ivj / 9j4c / 9q7y and protenix-v2 /
  esmfold2 drop 9j4c on the Wormhole Galaxy, all measured DRAM capacity boundaries. None carries
  to Blackhole.
- **Epitope and CDR labels are chunk-partial** for protenix-v2 and esmfold2 (Q3, Q7 run at
  reduced depth there and say so).
- **N* = 256 is a decision cap, not a measured knee.** Three of four models were still gaining
  above the seed-noise floor at the cap; protenix-v2 had its largest marginal gain there.
- **`wall_s` in the packaged parquets is not a per-model cost** (see Method). Costs here come
  from the fleet log.
- **Four specific models at specific settings.** esmfold2 is single-sequence throughout;
  opendde-abag is antibody-specialised and its advantage here should not be read as a general
  co-folding result. A different sampler, temperature or MSA depth is a different experiment.
- **Single hardware for the deep rungs.** All N=256 pools were folded on the Wormhole Galaxy.
  Cross-hardware consistency was gated at N=16 and N=64 (DATASHEET section 9); the residual
  Wormhole/Blackhole difference is chaotic amplification of reduction-order numerics, reproduced
  on-Galaxy by an mps 1→5 control.
- **Q6 extrapolates far past the measured range**, under a fit family that assumes no ceiling.
- **Not a published-ranker benchmark.** This measures each model's own shipped confidence, not
  ipSAE / pDockQ2 / AntiConf / DeepRank-Ab / ABAG-Rank, which need PAE matrices this asset does
  not carry. Whether a learned ranker closes the gap is untested here.

## What is new here

Confidence-vs-oracle gaps in co-folding are known, and per-target ranking failure has been
reported before at small scale. What this adds is scale and decomposition on one panel:
383,000 DockQ-labelled samples, four architecturally independent generators, N to 256, the same
161 targets throughout — enough to separate within-target from across-target ranking cleanly, to
convert the gap into an effective-N number, to attribute two thirds of all failures to epitope
discovery rather than pose refinement, and to price model diversity against sampling depth at
measured compute. The claims about *which* mechanism fails, and *what a fixed budget should buy*,
are the parts that need this asset.
