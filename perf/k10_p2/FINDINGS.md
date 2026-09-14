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

| arm | all-atom vs upstream | CA vs upstream | CA-lDDT vs upstream | CA-lDDT vs 1HCL | CA RMSD vs 1HCL | plDDT |
|---|---|---|---|---|---|---|
| `gpurefshared-s0` | - | - | - | **0.98351** | 0.60389 A | 0.910272 |
| `ttcpufp32-s0` | **0.03420 A** | 0.01280 A | **1.00000** | **0.98340** | 0.60452 A | 0.918296 |
| `ttshared-s0` | 0.43277 A | 0.31302 A | 0.99288 | **0.97765** | 0.77810 A | 0.915212 |
| `ttcpubf16-s0` | 1.21511 A | 1.15733 A | 0.78905 | **0.76777** | 1.36863 A | 0.854852 |

**Our torch code costs -0.00011 CA-lDDT. The deficit is -0.00586. The port's own code is 1.9 % of
it.** Two fp32 implementations of the same checkpoint on the same noise land 0.0342 A apart on a
fixture whose seed floor is 0.80 to 0.93 A, with CA-lDDT against each other of 1.00000.

There is no third direction to look in: the device arm is the same distance from our fp32 torch
code as from upstream, 0.43282 A against 0.43277 A, a ratio of 1.0001.

## bf16 is the class, but not the magnitude

`ttcpubf16` is the release gate's own bf16 reference arm, the same code under `torch.autocast`.
It misses the fp32 reference by 1.21511 A and loses **0.21574** of CA-lDDT against 1HCL, its
plDDT falling from 0.918296 to 0.854852. The device loses 0.00586. **The device is 37x more
accurate than a torch bf16 recomputation of the same model.**

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

Folds on qb2's host CPU under `benchlock.sh`, 388 s (fp32) and 298 s (bf16) at 298 aa, model load
13.6 s, no device opened.

## Owed

Both CPU arms at 512 aa, where the deficit is 2.5x larger. Queued behind another worker's
benchlock at the time of writing; roughly 20 min per arm.
