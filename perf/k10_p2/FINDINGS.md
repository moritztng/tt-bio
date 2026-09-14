# The standing lDDT deficit is the device path, not the port

`k10-p1-accuracy-anchor` found a CA-lDDT deficit against upstream `boltz==2.2.1` fp32 that
predates the whole K10 campaign, and `perf/roof_shared` showed it survives shared draws:
-0.00586 at 298 aa and -0.01487 / -0.01462 on the two 512 aa pseudo-domains, with the sampler's
noise removed as an explanation. That single number covers three layers at once: tt-bio's own
torch code, the bf16 dtype boundary, and ttnn. This splits them.

## Method

tt-bio carries a torch CPU path for Boltz-2, the one `scripts/full_parity_gate.py` uses as its
`reference_fp32` and `reference_bf16` arms. Running the same fixture through it at seed 0 with
`TT_BIO_SHARED_DRAW_SEED=0` puts four arms on one noise realisation:

| arm | what it is |
|---|---|
| `gpurefshared-s0` | upstream boltz 2.2.1, fp32, RTX 6000 Ada, committed by the anchor |
| `ttcpufp32-s0` | tt-bio's torch code, CPU, fp32, `use_kernels=False` |
| `ttcpubf16-s0` | the same under `TT_BIO_REF_BF16=1` (torch bf16 autocast) |
| `ttshared-s0` | tt-bio on a Wormhole chip, bf16, committed by `perf/roof_shared` |

The shared noise is verified, not assumed. The CPU arms' first three sampler draws are
`84c748a64d2a8963`, `0221d7a064b23ef4`, `d4598badfc09f9ba` at shapes (1,2400,3), (1,4), (1,1,3),
identical to the device arm's, and both consume `n_randn = 898` draws over the fold.

Scored with `perf/b2z2_fusebias/score.py` unmodified. Before any new number was taken the pipeline
reproduced the committed controls: `ttshared` 0.43277 A, `gpuref` against `gpurefshared` 0.92565 A,
`gpurefshared-s0` CA-lDDT against 1HCL 0.98351.

## The result, 298 aa, seed 0, shared draws

| arm | all-atom vs upstream | CA vs upstream | CA-lDDT vs upstream | CA-lDDT vs 1HCL | CA RMSD vs 1HCL | plDDT (CIF) |
|---|---|---|---|---|---|---|
| `gpurefshared-s0` | - | - | - | **0.98351** | 0.60389 A | 0.910272 |
| `ttcpufp32-s0` | **0.03420 A** | 0.01280 A | **1.00000** | **0.98340** | 0.60452 A | 0.910437 |
| `ttshared-s0` | 0.43277 A | 0.31302 A | 0.99288 | **0.97765** | 0.77810 A | 0.909273 |
| `ttcpubf16-s0` | 1.21511 A | 1.15733 A | 0.78905 | **0.76777** | 1.36863 A | 0.861969 |

The plDDT column is `plddt_cif`, mean CA B-factor read from each CIF. The first cut of this table
used the plDDT each fold script reported, which is not one quantity across the two stacks; see
"The plDDT column was two instruments" below.

**Our torch code costs -0.00011 CA-lDDT. The deficit is -0.00586. The port's own code is 1.9 % of
it.** Two fp32 implementations of the same checkpoint on the same noise land 0.0342 A apart on a
fixture whose seed floor is 0.80 to 0.93 A, with CA-lDDT against each other of 1.00000.

There is no third direction to look in: the device arm is the same distance from our fp32 torch
code as from upstream, 0.43282 A against 0.43277 A, a ratio of 1.0001.

## The result, 512 aa, seed 0, shared draws

Same protocol at the size where the deficit is 2.5x larger. Folded on pc (12 cores, no watchdog)
after qb2 hard-reset five times in 2.5 hours and killed the first two attempts; 2810.9 s for the
fp32 arm on 6 threads and 2120.0 s for the bf16 arm on 5. The move is controlled: the 298 aa
`ttcpufp32` fold re-run on pc lands 1e-5 A from the qb2 one, with identical lDDT, RMSD and plDDT
to five decimals on a different CPU and a different thread count.

Shared noise verified again at this size. All three of `ttcpufp32`, `ttcpubf16` and the committed
device arm draw `5936883ad8f94f53`, `96ffa2884fbdcc3d`, `a3469063c51c337b` at (1,4128,3), (1,4),
(1,1,3) and consume `n_randn = 1112`.

Distances against `gpurefshared-s0`, worst per-pseudo-domain, and CA-lDDT against 1HCL per
pseudo-domain:

| arm | d1 all-atom | d2 all-atom | CA-lDDT vs upstream d1/d2 | CA-lDDT vs 1HCL d1 | d2 | plDDT (CIF) |
|---|---|---|---|---|---|---|
| `gpurefshared-s0` | - | - | - | **0.94304** | **0.94169** | 0.863003 |
| `ttcpufp32-s0` | **0.02368 A** | **0.02533 A** | **1.00000 / 1.00000** | **0.94294** | **0.94206** | 0.863725 |
| `ttshared-s0` | 0.68922 A | 0.53884 A | 0.98328 / 0.98483 | **0.92817** | **0.92707** | 0.859296 |
| `ttcpubf16-s0` | 1.74382 A | 2.06855 A | 0.71654 / 0.68612 | **0.65868** | **0.63947** | 0.777739 |
| seed floor | 2.04178 A | 1.19920 A | - | - | - | - |

**Our torch code costs -0.00010 CA-lDDT on copy 1 and +0.00037 on copy 2, where the deficit is
-0.01487 and -0.01462. That is 0.7 % and a sign flip.** The split holds at both sizes and gets
cleaner at the larger one, which is the opposite of what a size-dependent divergence in the port
would look like. Our fp32 code is fractionally ahead of upstream on copy 2.

No third direction here either. Re-scoring with `ttcpufp32` as the baseline
(`out/score_512_refcpufp32.json`) puts the device arm 0.68626 A and 0.53240 A away, ratios 0.996
and 0.988 against its distance to upstream. The two fp32 implementations are 0.024 A apart, so
the device is the same distance from either.

## bf16 is the class, but not the magnitude

`ttcpubf16` is the release gate's own bf16 reference arm, the same code under `torch.autocast`.
It misses the fp32 reference by 1.21511 A and loses **0.21574** of CA-lDDT against 1HCL, its
plDDT falling from 0.910437 to 0.861969. The device loses 0.00586. **The device is 37x more
accurate than a torch bf16 recomputation of the same model.**

At 512 aa the same arm loses **0.28436** and **0.30222** of CA-lDDT where the device loses
0.01487 and 0.01462, so the device is 19 to 21x more accurate than the torch bf16 recomputation.
The torch bf16 arm also leaves the sampler's own distribution at that size, 2.06855 A on domain 2
against a 1.19920 A seed floor, while the device arm stays inside it on both domains (0.68922
against 2.04178, 0.53884 against 1.19920).

So the deficit is a precision effect, at the good end of the bf16 class rather than the bad one.
ttnn's dtype boundary (bf16 activations, fp32 accumulate, fp32 destination) is much closer to fp32
than `torch.autocast` is, and what is left after 200 closed-loop sampling steps is 0.43 A of
coordinates and 0.006 of CA-lDDT. Nothing here is a correctness bug: there is no transform that is
wrong rather than imprecise, and the two arms that should agree do agree.

A side reading, worth recording because the release gate depends on it: on this fixture the
envelope denominator `d(reference_bf16, reference_fp32)` is 1.21511 A while the numerator
`d(device_bf16, reference_fp32)` is 0.43277 A, so the bound sits 2.81x above the device. An
envelope that wide passes the device comfortably and cannot resolve a 0.006 CA-lDDT offset. That is
the envelope working as designed, not a defect, but it is the reason this deficit was never
surfaced by the gate.

## What the static diff says, and it agrees

Checked before the folds ran, against the `boltz==2.2.1` sdist, class by class and function by
function (69 of 120 name-matched definitions byte-identical once comments and docstrings are
stripped):

* `tt_bio/data/featurizer.py` against `boltz/data/feature/featurizerv2.py`: the import block is
  the only difference, 2352 lines against 2354.
* `tt_bio/data/tokenize.py` against `boltz/data/tokenize/boltz2.py`: the import block plus the
  inlined `Tokenizer` ABC.
* `AttentionPairBias` (attentionv2), `AdaLN`, `DiffusionTransformer`, `DiffusionTransformerLayer`,
  `AtomTransformer`, `ConditionedTransitionBlock` and `weighted_rigid_align` are identical modulo
  formatting and import aliases. Upstream's bare `LayerNorm` in `transformersv2.py` is
  `torch.nn.LayerNorm`, `sqrt` is `math.sqrt`, and `einsum(a, b, "spec")` became
  `torch.einsum("spec", a, b)` with the subscripts unchanged.
* Diffusion process args equal `Boltz2DiffusionParams` term for term, `--step_scale` defaults to
  1.5 on both, and both leave `contact_guidance_update` True with potentials off.
* The MSA dropout values that differ (0.15 / 0.25 here against 0.0 / 0.0 upstream) are inert:
  `get_dropout_mask` multiplies the rate by `training`.
* What remains on the coordinate path is the port's own guarded levers, `_host_levers()`,
  `_block_pairwise()`, the `z_to_p` fused-bias argument, and an `add_additional_atom_features`
  option that defaults False.

So featurization, tokenization and sampler configuration were never candidates, and the fold
confirms it quantitatively.

## The plDDT column was two instruments

Caught while reading the 512 aa table: our fp32 arm reported plDDT 0.813571 against upstream's
0.863003 on structures 0.025 A apart. That is not a model difference. `perf/k10_anchor/gpu_ref_fold.py`
reports mean CA B-factor from the CIF, every TT and CPU fold script reports tt-bio's own metrics
plDDT, and the campaign's plDDT column has been comparing the two.

`score.py` now emits `plddt_cif` beside the reported `plddt`, mean CA B-factor from the CIF it
already loads, so the comparable column exists in every report the instrument produces. On it,
512 aa: upstream 86.3003, our fp32 86.3725 (+0.07), the device 85.9296 (-0.37), torch bf16
77.7739 (-8.53) -- the same ordering as CA-lDDT, and our fp32 arm is fractionally ahead of
upstream rather than 0.05 behind it. The refactor was verified by re-scoring the 512 aa set and
diffing: every pre-existing block byte-identical, `plddt_cif` the only new key.

One thing this turned up and did not answer. tt-bio's reported plDDT is not the mean of the
per-atom plDDT it writes into its own B-factor column: 0.811365 against 0.859296 (CA) and
0.860440 (all atoms) at 512 aa, 0.918296 against 0.910437 at 298 aa, so the sign flips with size.
Upstream's `complex_plddt` equals its own CA B-factor mean to six decimals on all 13 committed
reference runs. Both of ours come from one confidence dict in `tt_bio/worker.py`, `metrics` from
`c["plddt"]` and the B-factors from `c["plddt_atom"]`, so localising it is cheap. It touches no
lDDT or RMSD number in this campaign, those come from coordinates, but the reported confidence is
user-facing on JapanFold and a 0.05 discrepancy at 512 aa is worth one task.

## Predictions, scored

`PREDICTED.md`, written before the first fold.

1. `ttcpufp32` within 0.05 to 0.25 A of upstream: **low**, 0.03420 A, better than the band.
2. Its CA-lDDT within 0.003 of 0.98351: **right**, -0.00011.
3. `d(ttshared, ttcpufp32) >= 0.8x d(ttshared, gpurefshared)`: **right**, 1.0001x.
4. `ttcpubf16` reads 0.15 to 0.5 A with CA-lDDT within 0.005: **badly wrong**, 1.21511 A and
   -0.21574. The reasoning was that autocast is mixed precision like the anchor's GPU
   `bf16-mixed` control, which cost 0.01993 A. CPU autocast is not that. The direction of the
   miss matters: it means ttnn is the more accurate bf16 of the two, not the less.
5. The alternative, that our torch code drifted from upstream: **refuted**.

The 512 aa half, predicted before those two folds were scored:

6. `ttcpufp32` 0.05 to 0.20 A from upstream, CA-lDDT >= 0.998: **low again**, 0.02368 / 0.02533 A
   and 1.00000. Both sizes came in below the predicted band, so the bias was in the reasoning,
   not the fixture: two fp32 implementations of one checkpoint on one noise realisation agree
   much more tightly than op ordering suggested.
7. The port-code share stays under 20 % of the deficit: **right**, 0.7 % on copy 1 and a sign
   flip on copy 2. This is the prediction that would have broken the conclusion.
8. `d(ttshared, ttcpufp32)` within 15 % of `d(ttshared, gpurefshared)`: **right**, 0.4 % and
   1.2 %.
9. `ttcpubf16` 1.5 to 3.5 A with CA-lDDT down more than 0.15: **right**, 1.74382 / 2.06855 A and
   -0.28436 / -0.30222.

## Reproducing

```
fold_cpu_ref.py --out out/cpu_folds.json --cifdir cif --sizes 298 --arms fp32,bf16
../roof_shared/assemble.py \
    --tt-runs ../roof_shared/out/folds.json,out/cpu_folds.json \
    --tt-cif  ../roof_shared/cif,cif --sizes 298 \
    --out-cifdir scorecif --out-runs out/score_runs.json
../b2z2_fusebias/score.py scorecif --runs out/score_runs.json --split 298 \
    --out out/score_298.json --arms gpurefshared,ttcpufp32,ttcpubf16,ttshared
```

512 aa, folded on pc, arms run as separate processes so the second could be niced behind the
first:

```
fold_cpu_ref.py --out out/cpu_folds_512_fp32.json --cifdir cif --sizes 512 --arms fp32
fold_cpu_ref.py --out out/cpu_folds_512_bf16.json --cifdir cif --sizes 512 --arms bf16
../roof_shared/assemble.py \
    --tt-runs ../roof_shared/out/folds.json,out/cpu_folds_512_fp32.json,out/cpu_folds_512_bf16.json \
    --tt-cif  ../roof_shared/cif,cif,cif --sizes 512 \
    --out-cifdir scorecif_512 --out-runs out/score_runs_512.json
../b2z2_fusebias/score.py scorecif_512 --runs out/score_runs_512.json --split 298 \
    --out out/score_512.json --arms gpurefshared,ttcpufp32,ttshared,ttcpubf16
```

`--arms` takes arm PREFIXES, not full tags. Full tags leave the `lever` block empty and populate
only `native`, which costs a scoring run to notice.

Folds on qb2's host CPU under `benchlock.sh` at 298 aa, 388 s (fp32) and 298 s (bf16), model load
13.6 s. 512 aa on pc, 2810.9 s (fp32, 6 threads) and 2120.0 s (bf16, 5 threads, niced). No device
opened by any of them.

## Owed

Nothing for the measurement. Both sizes are scored, the split is the same at both, and the
mechanism is named with a control.

Two things stay open and neither is this task's question:

* n = 1 shared draw on the reference side, inherited from the anchor. A second shared-draw
  reference run would put an error bar on the third decimal. The sign and the 0.7 to 1.9 % split
  do not depend on it.
* the port's own plDDT aggregation, above.

The follow-up this measurement argues for, out of scope here because it is a lever with a perf
cost: the deficit is one-sided (negative on all three pseudo-domains at both sizes) and grows
with size, -0.00586 at 298 aa against -0.0147 at 512 aa. That is the signature of accumulated
bf16 rounding rather than a fixed offset, and it is the same one-sidedness that predicted a
payoff from widening on AF2-IG. A targeted fp32 widening experiment on the trunk would say how
much of the 0.0147 is buyable and what it costs, but it is a release-gated accuracy/perf trade
and needs its own task.
