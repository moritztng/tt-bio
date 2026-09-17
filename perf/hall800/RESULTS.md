# Protenix-v2 at 800 tokens on one Blackhole p300c chip

Fixture `hall800.yaml`: 797 residues over 2 protein chains (HSA 585 aa + 9ncy light chain 212 aa),
800 padded tokens. Served config: `--fast --diffusion_samples 1 --max_parallel_samples 1 --seed 42
--msa_dir perf/hall800/msa --msa_cache_only`, engine defaults for recycling (10) and sampling (200).
qb2 cards 0 and 3, 11x10 grid, tt-bio `eab845ad1`.

**Every number below was taken at 1350 MHz, forced and sampled during the fold.** The card idles at
800 MHz, so the clock is not incidental here.

## It folds, 3 times out of 5

| # | card | outcome |
|---|---|---|
| 1 | 0 | ok, 380.8 s (cold) |
| 2 | 0 | ok, 222.2 s (warm) |
| 3 | 0 | host-spin wedge at `trunk 8/10` |
| 4 | 3 | ok (census fold) |
| 5 | 3 | host-spin wedge at `trunk 3/10` |

No OOM, no allocator refusal, no size ceiling: 800 tokens is far below the 1088 this model reaches.
plDDT 0.89784, pTM 0.79315, ipTM 0.644198, identical across both completed folds at the same seed.
What fails is finishing reliably. See "The wedge" below.

## Fold time, when it finishes

| | seconds | cycles at 1350 MHz |
|---|---|---|
| cold fold (first of the process) | 380.8 | 5.14e11 |
| warm fold | **222.2** | **3.00e11** |

Warm stage split, from the run's own timestamps:

| stage | s | share |
|---|---|---|
| prep, MSA featurize | 9 | 4 % |
| trunk, 10 recycles | 188 (17-19 each) | 85 % |
| diffusion, 200 steps | 26 | 12 % |
| confidence + saving | 4 | 2 % |

The fold is trunk-recycle-bound. Only one warm rep exists: the run that would have given a spread
wedged on its first fold, so 222.2 s carries no sigma yet.

`docs/size_ladder_baseline.d/protenix-v2.json` reads 112.5 s at 768 on the same card type. That rung
is single-sequence, 6 sampling steps, 1 sample. It is not a capacity number and 222.2 s is not a
regression against it.

## Which levers fire at 800 tokens, measured

`scripts/lever_census.py` on the served config, artifact `runs/census_h800/census.json`.

**`TRANSPOSE_L1_RESIDENT` is dark and cannot be extended.** 160 served of 1208, frac 0.132. The pair
tensor is `N^2 * c_z(256) * 2` bytes: 134.22 MB at 512, 327.68 MB at 800, against 168.57 MB of total
grid L1 (110 x 1 532 416 B). The shipped headroom 1.25 last fits at 512, headroom 1.0 at 544. This is
arithmetic, not a tuning miss.

**The fused triangle-attention path is dark on `dtype`, and `--fast` is the cause.**
`TRIATT_PERSISTENT_MASK` serves 0 of 2416, rejecting on `dtype` x2400, `fill_preconditions` x8,
`pm_over_l1` x7. `triatt_sdpa.sdpa` requires bfloat16 on every operand (`triatt_sdpa.py:338`) and
`--fast` puts the Protenix trunk in bf8 (`protenix.py:1881`) — which is 85 % of this fold. Two more
levers go down on the same precondition: `TRIMUL_IN_PROJ_DUAL_NOC` 16 of 2256 (`dtype` x2240,
`mm_dualnoc.py:86`) and `TRIATT_HEAD_MAJOR_QKV` 8 of 1208 (`dtype_or_memory` x1200).

So **`--fast` on vs off at 800 aa is an open question with the chip count riding on it**: bf8 makes
the trunk arithmetic cheaper while switching the fused paths off, the two point opposite ways, and
only a measurement settles it. `run.sh <name> <reps> --cards <n> --nofast` is the arm.

**Correction to an earlier reading in this file.** It previously said the q-chunk ladder at 800 being
`(800, 256)` with 256 not dividing 800 was "the real tuning gap", and that
`TT_BIO_TRIATT_NARROW_Q_FALLBACK=1` would recover it. The census refutes that: the divisibility route
accounts for 8 refused calls of 2416. That claim was read off the source instead of measured, and the
runtime L1-refusal caches it depended on (`_SDPA_Q_CHUNK_OVER_L1`, `_PM_OVER_L1`) are only populated
by real allocator refusals during a fold, so it could never have been settled statically. The
divisibility effect is real and negligible; the lever is refuted at this size.

Healthy at 800: `ADALN_S_HOIST` 1.000 (13209 calls), `REBLOCK_PERMUTE` 1.000, `REBLOCK_PERMUTE_BACK`
1.000, `TRANSITION_H_CHUNK` 0.995, `TRIMUL_TAIL_F1` 0.868, `QKV_MM_CONFIG` 0.704.

## The recycle dial, and why it is the caller's choice

The fold is trunk-bound, so the chip count is almost linear in `n_cycle`. Derived from the warm
fold's own per-recycle timestamps (9 steady recycles, mean 18.22 s, min 17, max 19; the tenth is
24 s because it also runs the trunk-output head):

| n_cycle | fold s | chips for 20k/day |
|---|---|---|
| 10 (upstream default, what we serve) | 222.2 | 51.4 |
| 8 | 185.8 | 43.0 |
| 6 | 149.3 | 34.6 |
| 5 | 131.1 | 30.3 |
| 3 | 94.6 | 21.9 |
| 1 | 58.2 | 13.5 |

This is not a lever we get to pull. 10 is Protenix-v2's own `N_cycle` from the checkpoint
(`main.py:2671`), and serving fewer recycles to make the number look better is doing less of the
model's own work. The table is here because the largest single term in the capacity answer is a
config value the caller owns.

Stage-sum arithmetic checks to 227 s against the reported 222.2 s; the 5 s gap is the log's
1-second timestamp resolution across ~14 stage boundaries, so per-recycle figures carry ~±1 s.

## Host cores are a co-requirement, not a footnote

`dp-throughput-host-core-ceiling` measured Boltz-2 512 aa DP on a Wormhole Galaxy at widths 1-32:
per-fold host cost is flat at ~1.85-1.9 cores before the shipped thread-cap/OMP-parking lever and
~1.35-1.49 after it, and it does not fall as width grows. Applied to 51.4 chips that is **69-98 host
cores**, and a host short of that stretches the folds themselves rather than merely scaling
sub-linearly (41.43 s to 103.01 s at width 32 on a 32-core box). qb2's 16 cores ceil at 8.4-11.9
concurrent folds against only 4 chips, so a 1-vs-2-chip test here is not host-limited and should
come out linear.

Labelled as transferred, not measured here: that is Boltz-2 at 512 aa on Wormhole, and Protenix-v2
runs 10 recycles against Boltz-2's 3, so this fold may cost more host cores per fold rather than
fewer. Measuring cores/fold belongs to the scaling arm.

## The wedge

Two of five folds stopped emitting mid-trunk and spun at 106-113 % CPU in state R, 81 threads, still
holding `/dev/tenstorrent/<n>` on two fds with ARC answering. `SigIgn` is `0x1001000` (SIGPIPE,
SIGXFSZ) and SIGTERM is in `SigCgt`, so the handler is installed and never runs: the spin is in
native code. Killing the launcher does not clear it; recovery is a reset, and on a p300 that resets
the board pair, so one wedge costs two chips.

Same signature as the single prior sighting at 768 aa, with two additions: it is **not chip-specific**
(two chips, two boards) and **both wedges are in the trunk**, never in the 200-step diffusion loop.
Rate is 2 of 5 here, ~3 of 25 pooled with the 768 aa sighting. A service at this size needs this
fixed, not a watchdog around it.
