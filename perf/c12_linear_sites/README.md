# c12-linear-fusion-census — the 27 `linear` keys, their call sites, and what fusing them is worth

No device. Every number is read off instruments already committed to this repo, joined on capture +
op index: the 26 module graph captures in `perf/roof_budget/captures`, `roof_true/true_floor.py`
(capture walk, launch arm, census key string), `roof_residual/split_units.py` (which `unit::<Class>`
frame owns each op), `roof_budget/exec_flops.py` (operand address + shape), and
`c10-fold-census`'s priced budget, vendored here as `census_budget_sweep2.json` (origin
`wk/c10-fold-census` @ 9199f75f6, blob 4f9034809, sha256 4c21cafccce4edf5c8fecd27d859c608094806).

Clock: 1350 MHz, pinned and sampled during every interval by `c10-fold-census`. Seconds are
clock-dependent on this fixture, so every row also carries cycles (Mc = s x 1350).

    python3 perf/c12_linear_sites/sites.py      # call sites + sibling sets  -> sites_512.{json,txt}
    python3 perf/c12_linear_sites/price.py      # rate pricing per set       -> priced_512.{json,txt}
    python3 perf/c12_linear_sites/test_sites.py # the join + both guards, with negative controls

`sites.py` exits non-zero unless all 27 keys and all 108,608 calls are reproduced exactly, which is
the only thing that makes the owner column believable.

## Call sites

`sites_512.txt` holds the generated table. Sites are `tt_bio/tenstorrent.py`:

| key | calls | fold s | site |
|---|---|---|---|
| `1x16x512x512 K=128`  | 17,920 | 1.1671 | `Transition.swiglu` fc1 :8211, fc2 :8222 (pair track, 32 row chunks) |
| `1x16x512x128 K=512`  |  8,960 | 0.7920 | `Transition.swiglu` fc3 :8233 |
| `1x512x768 K=768`     | 38,600 | 0.5658 | `AdaLN.s_terms` :9395 + :9403 (19,200); `AttentionPairBias` g :8159, o :8169 (9,600); `DiffusionTransformerLayer` :9610 (4,800); `ConditionedTransitionBlock` :9519 (4,800); `Diffusion` :10679 (200) |
| `1x512x512x128 K=128` |    560 | 0.5276 | `TriangleMultiplication` p_out :6789 via `_pair_proj_linear` :4348. One linear per trimul: the g projection is already fused into the in-projection `generic_op` |
| `1x512x1536 K=768`    | 15,200 | 0.3418 | `ConditionedTransitionBlock` :9492, :9498, :9506 (14,400); DiT `Transition` fc1/fc2 (800) |
| `1x512x3072 K=768`    |  4,800 | 0.1766 | `AttentionPairBias` qkv :7971 (already one fused projection) |
| `1024x512x32 K=64`    |    288 | 0.1752 | `PairWeightedAveraging.head_out` v :9867, g :9883 (256); `OuterProductMean` :10097, :10104 (32) |
| `512x512x128 K=1024`  |     16 | 0.1578 | `OuterProductMean` out :10316 |
| `1x512x768 K=1536`    |  5,200 | 0.1238 | `ConditionedTransitionBlock` b_to_a :9526 (4,800); DiT `Transition` fc3 (400) |
| `1x512x512x16 K=128`  |    264 | 0.1006 | `AttentionPairBias` pair-bias z projection :7789/:8006 |
| `1x140x128x256 K=128` |  1,200 | 0.0974 | `AttentionPairBias` kv :8128 (atom windows) |
| `1x140x32x256 K=128`  |  3,600 | 0.0891 | `ConditionedTransitionBlock` :9492, :9498, :9506 (atom) |
| `1x140x32x128 K=128`  |  4,800 | 0.0775 | `AttentionPairBias` q :8120, g :8159, o :8169 (3,600); `ConditionedTransitionBlock` :9519 (1,200) |
| `1024x512x64 K=32`    |    128 | 0.0773 | `PairWeightedAveraging.head_out` o :9896 |
| `1x16x512x256 K=64`   |  2,048 | 0.0641 | `Transition.swiglu` fc1/fc2 (MSA track) |
| `1x16x512x64 K=256`   |  1,024 | 0.0446 | `Transition.swiglu` fc3 (MSA track) |
| `1x140x32x128 K=256`  |  1,200 | 0.0377 | `ConditionedTransitionBlock` b_to_a :9526 (atom) |
| `1x512x1536 K=384`    |    792 | 0.0122 | trunk `Transition` fc1/fc2 (528); trunk `AttentionPairBias` qkv :7971 (264) |
| `1x4480x768 K=128`    |    200 | 0.0072 | `Diffusion` atom_to_token :10705 |
| `512x512x8 K=128`     |     16 | 0.0061 | `PairWeightedAveraging` z projection :9829 |
| `1x512x384 K=1536`    |    264 | 0.0053 | trunk `Transition` fc3 |
| `1x512x384 K=384`     |    528 | 0.0043 | trunk `AttentionPairBias` g :8159, o :8169 |
| `1x4480x128 K=3`      |    200 | 0.0030 | `Diffusion` r_to_q :10688 |
| `1x512x128 K=768`     |    200 | 0.0022 | `Diffusion` s_to_a :10758 |
| `1x4480x3 K=128`      |    200 | 0.0016 | `Diffusion` feat_to_pos :10822 |
| `1x768 K=256`         |    200 | 0.0012 | `Diffusion` conditioner_fourier_single :10736 |
| `1x256 K=256`         |    200 | 0.0011 | `Diffusion` conditioner_fourier_embed :10720 |

The owner column is the frame the capture recorded for all but the last six rows. ttnn drops a
`function_end` when a frame below it leaks, so for the six 200-call `Diffusion` keys the innermost
open frame reads as `DiffusionTransformerLayer`; those are pinned by shape and weight instead.
`Diffusion.a_to_q` :10788 issues no linear in any of the three captures, and since the captures
reproduce all 108,608 calls it does not fire at 512 aa.

## What a sibling set is here

>= 2 `ttnn.linear` calls that read the SAME activation buffer address at the same (batch, M, K)
with DISTINCT weight tensors. Two guards, both with a negative control in `test_sites.py`:

* **no reallocation between members.** A loop that frees and refills one buffer hands every
  iteration the same address. Without this guard the pair transition's 32 row chunks read as one
  64-member set and `PairWeightedAveraging`'s 8 per-head outputs as one 8-member set.
* **no weight tensor twice.** Keyed on the weight ADDRESS this was wrong: `head_out` slices 16
  per-head weights out of two parent tensors one at a time and ttnn hands every slice the same
  recycled address, which split all 16 apart and dropped a real 256-call, 0.1557 s set. Bytes still
  dedupe on address; only "is this a different weight" reads the tensor id.

No op-index window is used. A window wide enough to span an attention body is also wide enough to
merge two instances: at 40 it merged 24 DiT layers into one set.

Coverage: 13 frame-local sets plus 2 cross-frame sets cover 82,112 of 108,608 calls (75.6 %) and
2.5789 of the class's 4.6604 s (55.3 %).
