# Protenix-v2 at 800 tokens on one Blackhole p300c chip

Fixture `hall800.yaml`: 797 residues over 2 protein chains (HSA 585 aa + 9ncy light chain 212 aa),
800 padded tokens. Served config: `--fast --diffusion_samples 1 --max_parallel_samples 1 --seed 42
--msa_dir perf/hall800/msa --msa_cache_only`, engine defaults for recycling (10) and sampling (200).
qb2 card 0, 11x10 grid, tt-bio `eab845ad1`.

**Every number below was taken at 1350 MHz, forced and sampled during the fold: 63 453 samples,
mean 1350.0 MHz, all at target.** The card idles at 800 MHz, so the clock is not incidental here.

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

The fold is trunk-recycle-bound. plDDT 0.89784, pTM 0.79315, ipTM 0.644198, identical across both
folds at the same seed.

`docs/size_ladder_baseline.d/protenix-v2.json` reads 112.5 s at 768 on the same card type. That rung
is single-sequence, 6 sampling steps, 1 sample. It is not a capacity number and 222.2 s is not a
regression against it.

## Two 512-tuned gates are dark at 800 tokens, for different reasons

**`TRANSPOSE_L1_RESIDENT` cannot be extended here.** The pair tensor is `N^2 * c_z(256) * 2` bytes:
134.22 MB at 512, 327.68 MB at 800, against 168.57 MB of total grid L1 (110 x 1 532 416 B). The
shipped headroom 1.25 last fits at 512 and headroom 1.0 last fits at 544. The committed census
reads frac 1.000 at 512 and 0.132 from 640 up, which is this arithmetic and not a tuning miss.

**The fused-SDPA q-chunk is dark by divisibility and has a lever.** At 800 the default ladder is
`(800, 256)`; 256 does not divide 800, so the fold lands on a padding q_chunk, which sets
`use_padded_mask` and declines the fused persistent-mask path outright (`tenstorrent.py:1518`).
768 and 1024 both divide and serve. `TT_BIO_TRIATT_NARROW_Q_FALLBACK=1` (merged, default off) makes
the 800 ladder `(800, 160, 32, 256)`. Worth measuring; not yet measured.

For anyone choosing target sizes: 768 tokens is a better length for this model than 800.

## The recycle dial, and why it is the customer's call

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

The third identical fold stopped emitting at `trunk 8/10` and spun at 118 % CPU in state R with the
chip still answering ARC. SIGTERM did not clear it; recovery is a reset. Same signature as the
single prior sighting at 768 aa, now seen at 800 on the third fold after two clean ones. A service
at this size needs a per-fold watchdog.
