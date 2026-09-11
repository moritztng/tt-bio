# The Boltz-2 512 aa fold's host path, stage by stage

`b2x-op-cost-curve` bracketed the fold by `tt_bio.tenstorrent` module class and found trunk
12.882 s, sampler 7.151 s, confidence 0.465 s and a **3.212 s remainder with zero device work and
79 ttnn calls in the whole fold**. That remainder was the gap left after the rows, not a row. This
is the rows.

## Provenance, and what is still owed

Every number below is `time.perf_counter` wall clock on **pc** (AMD Ryzen 5 8600G, 6 cores /
12 threads, 16 MB L3, torch 2.8.0+cpu at its default 6 threads), running the shipped code on the
shipped fixture (`perf/size512/fixtures/cdk2x2_512.yaml` + its 35-row a3m, 512 tokens,
**4128 atoms**) with the real checkpoint. No device is involved in any of it, which is the point:
these stages issue no ttnn calls at all.

`thread_time()` appears nowhere. On this stack the calling thread spins while it waits for
dispatch-queue room, so CPU time reads as host work when it is device wait -- the artifact that
made a published verdict wrong on 2026-09-11. Wall clock is the honest instrument here.

**Owed, and not in this file:** the same table taken inside a real fold on qb2 card 2, which is
what converts these seconds into a percentage of that box's 3.212 s. qb2 hard-hung at 19:55 CEST
on 2026-09-11 with all four cards in use and needs a physical power trip (`state/qb2-offline`,
asks 7572 / 7670), so the on-card leg of this pass did not run. pc's cores are not qb2's; treat
the shares as measured and the absolute seconds as pc's.

## The table

| stage | pc ms | share | what it is | device impl? |
|---|---|---|---|---|
| `diffusion_conditioning` | **1126.7** | 60.7 % | `DiffusionConditioning.forward` | **none** |
| ├ `pairwise_conditioner` | 493.3 | 26.6 % | cat(z,relpos) 256 MB, LayerNorm+Linear, 2 residual Transitions | none |
| ├ bias stacks (24+3+3) | 453.8 | 24.5 % | `token_trans_proj_z` / `atom_enc_proj_z` / `atom_dec_proj_z` | none |
| └ `atom_encoder` | 179.6 | 9.7 % | `AtomEncoder.forward` | none |
| `prepare` | 289.3 | 15.6 % | parse + MSA resolve + tokenize + featurize | n/a, host by nature |
| sampler loop, 200 steps | 156.0 | 8.4 % | the host arithmetic between the 200 denoiser calls | n/a, see below |
| `input_embedder` | 111.5 | 6.0 % | `InputEmbedder.forward` | none |
| `write_result` | 115.0 | 6.2 % | CIF write + metrics | n/a, host by nature |
| `rel_pos` | 56.2 | 3.0 % | `RelativePositionEncoder.forward`, O(n^2 * 139) | none |
| `to_batch` | 0.07 | 0.0 % | unsqueeze + move | n/a |
| **total named** | **1854.8** | 100 % | | |

The pc legs time each stage on its own, so this table has no `unattributed` row to report --
that row is a property of a fold, and it comes from the on-card instrument. On the one on-card
leg that did run (128 aa, card 2) the bracket tree covered `predict_one` completely:
**unattributed 0.000 s of 4.974**, so the method produces a closed table; the 512 aa fold's own
row is owed.

The sampler-loop row, per step at 4128 atoms (`sampler_host_cost.py`):

| stage | ms/step | s/fold |
|---|---|---|
| `weighted_rigid_align` | 0.417 | 0.083 |
| `compute_random_augmentation` | 0.122 | 0.024 |
| rotate / rotate denoised | 0.101 | 0.020 |
| `randn` for eps | 0.052 | 0.010 |
| centre / centre denoised | 0.048 | 0.010 |
| the EDM step arithmetic | 0.027 | 0.005 |
| **per step** | **0.778** | **0.156** |

## Two predictions this falsifies

**The residual is not the sampler loop.** `d0183cc0` put "13.7 ms/step of host in the sampler" on
record and this task's own prediction built on it, expecting 2.5-2.9 s of the 3.212 there. The
loop's own host arithmetic is **0.778 ms/step, 0.156 s per fold** -- 18x smaller, and 8 % of the
host path rather than 85 %. Whatever the 13.7 ms/step figure measured, it was not this.

**GC is not in it.** The hypothesis was that ~10 fresh tensors per step against a resident
Boltz-2 plus every cached ttnn tensor would make generation-2 passes expensive. Measured with
`gc.callbacks` over a whole fold on card 2 (the 128 aa preflight, the one on-card leg that did
run): **1.65 ms total**, 10 gen-0 collections and 1 gen-1, no gen-2 at all. Dead.

## What the table says instead

**60 % of the fold's host path is one module with no device implementation.**
`tt_bio.tenstorrent` has no `DiffusionConditioning` and no `PairwiseConditioning`;
`Boltz2.forward` calls the torch module between the trunk and the sampler, and it is 120 GFLOP of
dense fp32 matmul at 512 tokens, measured at 243 GFLOP/s on 6 cores -- near that CPU's peak, so
there is no host-side arithmetic win hiding in it. The reachable set is a port, not a
micro-optimisation.

The three genuinely unavoidable stages -- featurisation, the CIF write and the sampler's serial
EDM arithmetic -- are **0.560 s, 30.2 % of the host path**. The brief's plausible outcome (an
unavoidable MSA parse plus a CIF write, `RECOVERED: 0.0 s`) is refuted: most of this block is
model code that has simply never been moved to the device.

## Levers, measured with real checkpoint weights

Bit-exact means `torch.equal` on the stage output with the production checkpoint, at 512 tokens,
not a tolerance. Random-weight bit-exactness is not enough -- the blocked `PairwiseConditioning`
passed with random weights and failed with real ones.

| lever | shipped ms | patched ms | ratio | bit-exact | default |
|---|---|---|---|---|---|
| `rel_pos` gather instead of a 139 MB one-hot | 56.2 | 55.2 | 1.02x | **yes** | on |
| bias stacks, one pass in L3-sized row blocks | 527.5 | 239.5 | **2.20x** | **yes** | on |
| ├ preallocated output, no blocking | 527.5 | 468.0 | 1.13x | yes | (part of it) |
| `PairwiseConditioning` in row blocks | 472.4 | 253.2 | 1.87x | **no** | **off** |
| **the two defaults, host stages end to end** | **1294.4** | **1017.8** | **1.27x** | **yes** | |

277 ms off the host path, bit-exact. Against a 23.710 s fold that is 1.2 % if qb2's host scales
like pc's, and qb2's residual is 3.212 s against pc's 1.855 s of named stages, so the honest
range is 1.2-2.1 % pending the on-card A/B.

### Why the pairwise lever is not bit-exact, exactly

Split into its steps against the real weights: `LayerNorm`, `fc1` and `fc2` are all bit-exact
under blocking at every block size tried (256 to 65536 rows). The divergence is in the
transition's gate, `silu(fc1(x)) * fc2(x)` -- SiLU's vectorised path and its scalar tail disagree
by about 1 ULP, and blocking moves that boundary. 11351 of 33.5M elements at rows=2048, max
3.8e-6; 48 elements at rows=819. A ULP entering a 200-step diffusion trajectory can move the CIF,
so it stays off pending a control.

### Block size

The budget is a working-set size read from `/sys/devices/system/cpu/cpu0/cache/index3/size`, not
a row count and not a constant, because the optimum is last-level cache and hosts differ. On pc's
16 MB L3 the curve peaks exactly there and is broad:

| block footprint | 2 MB | 4 MB | 8 MB | **16 MB** | 32 MB | 64 MB | 128 MB |
|---|---|---|---|---|---|---|---|
| ratio | 1.50x | 1.51x | 1.74x | **2.20x** | 1.90x | 1.31x | 1.07x |

## Files

All under `~/.coworker/scratch/b2x-host-residual/` on pc until qb2 comes back and they can be
committed to the worktree:

* `hostpath_cost.py` -> `hostpath_512_pc.json` -- the stage table.
* `sampler_host_cost.py` -- the per-step sampler-loop breakdown.
* `bitexact.py` -- the first-pass bit-exactness probe (one-hot vs gather, layer-norm collapse).
* `hostlever_bench.py` -- the lever sweep at fold shapes, random weights.
* `pairwise_exact.py` -- where blocking stops being bit-exact, real weights.
* `bias_sweep.py` -- cat-elimination vs blocking, and the block-size curve.
* `stage_parity.py` -> `stage_parity_{shipped,patched}_pc.json` -- per-stage sha256, two
  processes, one overlay.
* `apply_host_levers.py` -- the patch, anchor-checked and idempotent.

On card 2, committed on `wk/b2x-host-residual` at `1a79949b`:

* `perf/b2x_host_residual/host_residual.py` -- the on-card instrument: wall brackets over all of
  `predict_one`, a 1 ms stack sampler, `gc.callbacks` timing. 100 % attribution on the 128 aa
  preflight (unattributed 0.000 s of 4.974).
* `perf/b2x_host_residual/preflight_128.json` -- that preflight, including the GC result.
