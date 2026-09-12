# b2z2-msa-axis-shard — pre-registered prediction

Written and committed BEFORE the first device run of this row. Git holds the timestamp.

## The question

`MSALayer` is 3.8337 s of a 41.4333 s Wormhole fold (9.25 %, `b2z2-msa-layer-census`, 16 calls,
238.5071 ms/call at 1024 padded rows for 35 real ones). The trunk shards on the token axis and the
sampler shards on the atom axis. Does the MSA track divide on its **row** axis?

## Step 2 of the brief: predict from the byte ratio, and score it

The atom axis worked because an atom layer's weights (~0.55 MB) are small against its activations
(4.6 MB), so a row split divides the large term; the token DiT failed inverted.

`MSALayer` at Boltz-2 config (`c_m` 64, `c_z` 128, PWA 8 heads x 32, TriAtt 4 heads x 32):

| | params | bf16 |
|---|---|---|
| `msa_transition` | 49,152 | 0.098 MB |
| `pair_weighted_averaging` | 50,560 | 0.101 MB |
| `outer_product_mean` | 135,296 | 0.271 MB |
| `pairformer_layer` | ~525,312 | 1.051 MB |
| **total weights** | **~760,320** | **1.52 MB** |

Against activations of **67.1 MB** per full-depth tensor (`m` = [1024, 512, 64] bf16, `z` =
[512, 512, 128] bf16) and a 536.9 MB pre-`proj_o` intermediate inside OPM.

**WEIGHT-ACTIVATION-RATIO: 2.27 % (1.52 MB / 67.1 MB).** Four times more favourable than the atom
track's 12 % (0.55 / 4.6). **On the atom axis's own heuristic this is a clear GO.**

**And I am pre-registering that the heuristic is the WRONG discriminator on this block, and that
it will mis-predict.** The atom track's constant was small because everything in an atom layer
carries an atom index. `MSALayer` does not: **one of its four sub-units has no MSA row index at
all.** `pairformer_layer(z)` is pure pair track. On an MSA-row split every chip computes it whole.
That is a replicated term the byte ratio cannot see, and it is 33 % of the block.

## PREDICTED

Decomposition of the census's own sub-unit table (`split_layer_wh_c1.json`), by whether the
sub-unit carries an MSA row index:

| sub-unit | kernel ms | share | row axis? |
|---|---|---|---|
| `pairformer_layer` | 78.381 | 33.3 % | **NO — pure `z`, fully replicated** |
| `pair_weighted_averaging` | 73.987 | 31.4 % | mostly yes; `token_weight` x 8 + the `z` layer_norm are `z`-only |
| `outer_product_mean` | 61.401 | 26.1 % | `project_ab` yes; the post-contraction `z` stage is `z`-only |
| `msa_transition` | 21.518 | 9.1 % | yes, every op is per row |

* **P1 — CONSTANT-FRACTION 43 %, band 36-50 %**, at the shipped 1024-row bucket.
  32.9 % `pairformer_layer` + ~2 % PWA's `z` side + ~8 % OPM's post-contraction `z` stage.
  **The falsifier bar is 45 %. I am predicting 43 %, inside it by two points.** This row is a coin
  flip on its own pre-registered bar and resolving it is the deliverable either way.
* **P2 — free-split track ratios 1.40x at 2 chips (band 1.33-1.47) and 1.75x at 4 (band
  1.60-1.86)**; with the collectives paid, **1.30x at 2 and 1.55x at 4**.
* **P3 — the collective is an OPM all-reduce, and it is `proj_o`-side.** `z_rows` contracts over
  the row axis, so each chip holds a partial. `proj_o` and the `1/n_msa` scale are both linear in
  that partial, so the reduce can be moved AFTER them: **67.1 MB** ([512, 512, 128] bf16) instead
  of the 536.9 MB pre-`proj_o` intermediate. Predicted 0.9-1.6 ms per call at the measured 20 GB/s
  class of link, so 2-4 % of the call.
* **P4 — the OPM reduce is NOT bit-exact and that is the row's real risk.** Splitting a contraction
  over S into per-chip partial sums reassociates a bf16 accumulation; `tenstorrent.py`'s own
  `project_depth_parts` says so in its docstring about the same contraction. Predicted: PWA and
  `msa_transition` shard bit-exactly (every op is per row), OPM does not. **If that holds, the
  brief's requirement 3 — `torch.equal`, max abs 0.0, at every width — is only reachable by
  all-gathering the c=32 projections `a`/`b` (33.6 MB each) and leaving the contraction whole.**
* **P5 — replicated floor, bytes vs time.** Byte side ~39 % (`pairformer_layer` moves 4548 of the
  11,646 MB the census counted). Time side HIGHER than the byte side, not lower as on the atom
  axis, because here the replicated term is compute the mesh cannot divide rather than a halo that
  grows with it: predicted **43-50 % time-side**.

## FALSIFIER

**Constant fraction above ~45 % kills the row**: the MSA track is then the token DiT's machine and
does not shard. That is a complete one-pass result and it retires the last open stage of the fold.
