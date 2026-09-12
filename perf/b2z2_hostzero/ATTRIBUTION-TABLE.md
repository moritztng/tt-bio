# The Boltz-2 512 aa fold's host residual on current main, by where the second goes

One fold, whglx card 6 (Wormhole galaxy chip, 8x9 grid, 32 torch threads), commit `6587e773`
(`origin/main` plus the bias-stack fusion, which is off by default and was off here). Instrument:
`perf/b2x_host_residual/host_residual.py --phases plain,attrib`, `time.perf_counter` wall brackets
over all of `predict_one`, **no `synchronize_device` anywhere and no `thread_time` anywhere** --
on this stack the calling thread spins while it waits for dispatch-queue room, so CPU time reads
as host work when it is device wait, which is the artifact that produced wave 1's refuted
"dispatch-bound" bulletin. Raw JSON: `base_whglx_c6.json`.

Fold 40.6672 s, plain fold on the same card 41.0571 s, CIF `2e3b25a5520c8ff8` on both.
**Unattributed 0.0114 s = 0.03 %**, so the tree is a measured decomposition, not an inference.

WH, not BH. The mechanism and the ranking transfer; the seconds do not. A Wormhole galaxy chip
runs the device part of this fold in 37.09 s against Blackhole's ~16 s, so the host residual's
*share* here (8.8 %) is about half what it is on the cell (18.9 %) even though the host seconds
are comparable. Rank levers off this table; take ratios on qb2.

## Device vs host

| | s | share |
|---|---|---|
| device brackets (`trunk` 26.610, `denoise_device` 9.459, `pairformer_conf` 1.025) | 37.094 | 91.2 % |
| **host residual** | **3.573** | **8.8 %** |

This is the zero-device residual plus glue. It does **not** separate the host seconds exposed
*inside* the three device brackets (1.136 s on the cell, the dispatch axis, closed at 1.050x) --
that needs a trace-replay device floor per bracket, which is `b2x-op-cost-curve`'s instrument.

## The residual, row by row

| row | s | % of residual | what it is | fix |
|---|---|---|---|---|
| `diffusion_conditioning` | **1.067** | 29.9 % | | **ported, see below** |
| ├ `pairwise_conditioner` | 0.539 | 15.1 % | cat(z, relpos), LayerNorm, Linear, 2 Transitions | device |
| ├ bias stacks (24+3+3), exclusive | 0.329 | 9.2 % | `token_trans_proj_z` is 24 LayerNorm+Linear over one tensor | device + fusion |
| └ `atom_encoder` | 0.198 | 5.5 % | `z_to_p_trans` is the only thing it reads `z` for | device (the projection) |
| confidence head, host part | 0.746 | 20.9 % | 8-9 full passes over `[1, 512, 512, 128]` in torch | device port, not built |
| `predict_step` glue, exclusive | 0.465 | 13.0 % | `z_init` assembly, distogram, the `dict_out` plumbing | marginal |
| `prepare` | 0.353 | 9.9 % | parse + MSA resolve + tokenize + featurise | host by nature |
| sampler loop, exclusive | 0.225 | 6.3 % | the serial EDM arithmetic between 200 denoiser calls | host by nature |
| `weighted_rigid_align`, 200 calls | 0.188 | 5.3 % | 0.94 ms/call | host by nature |
| `write_result` | 0.149 | 4.2 % | CIF write + metrics | host by nature |
| `rel_pos`, confidence | 0.114 | 3.2 % | `RelativePositionEncoder`, second call site | device |
| `rel_pos`, trunk | 0.099 | 2.8 % | same module, first call site | device |
| `input_embedder` | 0.081 | 2.3 % | | device |
| `compute_random_augmentation`, 200 calls | 0.037 | 1.0 % | | host by nature |
| denoiser wrapper, 200 calls | 0.040 | 1.1 % | | glue |
| **unattributed** | **0.011** | 0.3 % | fold wall minus every top-level region | measured |

## What this ranks

**1.067 s of 3.573 s is one module with no device implementation**, and it sits between the trunk
and the sampler with no device work to hide behind. That is this row's target and it is now
`TT_BIO_DEVICE_CONDITIONING`.

The second-biggest row, the confidence head's 0.746 s, is the same shape -- passes over the same
`[1, 512, 512, 128]` pair tensor -- and shares `PairwiseConditioning` with the conditioning, so
one device module serves both. It is not built here.

Below those two, 1.152 s is featurisation, MSA parse, CIF write and the sampler's serial EDM
arithmetic, which is host by nature, and 0.505 s is glue. There is no third lever in this table.
