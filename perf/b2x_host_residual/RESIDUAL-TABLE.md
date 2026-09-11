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

The pc table below is the stage-by-stage screen: it times each stage on its own, so it ranks the
stages and prices the levers, but its seconds are pc's cores. The fold-level table further down is
the one taken inside a real fold on qb2 card 2, and it is where the attribution percentage comes
from. The two agree on the ranking and disagree on the absolute seconds, as they should: qb2's
host cores are faster than pc's on every row.

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

## The on-card table (card 2, qb2, the one that turns shares into a percentage)

Taken inside a real 512 aa fold on Blackhole card 2, `time.perf_counter` wall brackets over all of
`predict_one`, no `synchronize_device` anywhere, no stack sampler running. Levers on (the shipped
default after this branch). Fold 23.1107 s, benchlocked, A/A floor from the two plain folds of the
same process 23.0814 / 23.0440 s = **0.16 %**.

`b2x-op-cost-curve` bracketed by `tt_bio.tenstorrent` module class. Those three brackets are
exactly three rows of this tree, and they reproduce: `TrunkModule` 12.9058 s (campaign 12.882),
`DiffusionModule` 7.1539 s (campaign 7.151), the confidence pairformer 0.4653 s (campaign 0.465).
The residual is the rest of the fold: **23.1107 - 12.9058 - 7.1539 - 0.4653 = 2.5857 s**.

| row | s | share of residual | instrument |
|---|---|---|---|
| `diffusion_conditioning` total | **0.9198** | **35.6 %** | brackets |
| ├ `pairwise_conditioner` | 0.5063 | 19.6 % | brackets |
| ├ `atom_encoder` | 0.1870 | 7.2 % | brackets |
| └ the 24+3+3 bias stacks (its own exclusive time) | 0.2266 | 8.8 % | brackets |
| confidence head, host part | 0.5879 | 22.7 % | brackets |
| `predict_step` glue (its own exclusive time) | 0.3153 | 12.2 % | brackets |
| sampler loop host, 200 steps | 0.2444 | 9.5 % | brackets |
| ├ `weighted_rigid_align` | 0.0963 | 3.7 % | brackets, 200 calls |
| ├ loop body outside the called regions | 0.1021 | 3.9 % | brackets |
| ├ `compute_random_augmentation` | 0.0248 | 1.0 % | brackets, 200 calls |
| └ denoiser wrapper | 0.0212 | 0.8 % | brackets, 200 calls |
| `prepare` (parse + MSA resolve + tokenize + featurize) | 0.1884 | 7.3 % | brackets |
| `input_embedder` | 0.1040 | 4.0 % | brackets |
| `rel_pos`, trunk | 0.0807 | 3.1 % | brackets |
| `rel_pos`, confidence | 0.0840 | 3.2 % | brackets |
| `write_result` (CIF write + metrics) | 0.0580 | 2.2 % | brackets |
| `to_batch` | 0.0001 | 0.0 % | brackets |
| **`unattributed`** | **0.0032** | **0.12 %** | fold wall minus the named rows |

**RESIDUAL-ATTRIBUTED: 99.88 %.** The tree wraps every call `predict_one` makes, so the
unattributed row is a measured remainder and not an inference.

### The bias-stack row moves with box noise, and that is the finding under the lever

Two attribution folds ran on card 2 with identical code. The unlocked one put the bias stacks at
0.6790 s and `diffusion_conditioning` at 1.3636 s; the benchlocked one puts them at 0.2266 and
0.9198. Every other row of the tree agrees to within a few ms, including `prepare`,
`write_result` and all 200 sampler steps. A row that is 3x slower when the box is busy and stable
when it is quiet is bandwidth bound, not arithmetic bound, which is what the blocking lever
assumes and what the pc block-size curve already showed. It also means this row is the wrong
place to read a small effect from a single unlocked fold; the A/B below is interleaved and
benchlocked for that reason.

## The A/B: what came off the fold

Card 2, benchlocked, twelve ABAB pairs in one process with one device open, arms flipped through
`TT_BIO_HOST_LEVERS` between folds.

| arm | median | range over 12 folds |
|---|---|---|
| A, levers off (incumbent) | 23.7082 s | 23.491 - 23.992 |
| B, levers on | 23.2864 s | 23.113 - 23.533 |

Median ratio **1.0181x**. The paired statistic is the one to read, because it cancels the box's
slow drift: **paired mean +0.3951 s**, sd 0.1066, se 0.0308, 95 % CI **[+0.335, +0.455]**,
t = 12.84, **12 of 12 pairs positive**. **A/A floor 0.0407 s = 0.17 %**, the median absolute
difference between consecutive repeats of the identical incumbent arm inside the same run, which
agrees with the attribution run's two quiet plain folds (23.0814 / 23.0440 = 0.16 %). The effect
is about ten times the floor.

`RECOVERED: 0.395 s.` Against a residual of 2.981 s with the levers off, that is 13 % of the
zero-device block and 1.018x end to end.

A five-pair replicate ran earlier on a loaded box (a sibling campaign worker at 205 % CPU on
card 3, loadavg reaching 6.7). Same effect — ratio 1.0189x, paired mean +0.4032 s — and
unresolvable: its own A/A floor was **1.16 %** and one of five pairs came out negative. Both
runs are committed. The floor has to be measured in the run that carries the claim.

Every one of the 38 folds run on this branch — 17 A/B pairs across both runs, 2 plain, 2
attribution — wrote the identical CIF, sha256
`4f3995a69be5d6106f2b6a2aca57b29f8bb84810bf751e39a0258956b09d15e5`, the campaign reference.

Bit-exactness is not pinned to 512 tokens: repeated at **1024 tokens / 8256 atoms** on qb2,
host-only, all 12 recorded stage hashes are identical with the levers on and off
(`stage_parity_1024_{,leversoff_}qb2.json`).

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

277 ms off pc's host path, bit-exact. On card 2 the same two levers took 0.395 s off the fold
end to end, 1.018x -- see the A/B section above, which is the number that counts.

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
