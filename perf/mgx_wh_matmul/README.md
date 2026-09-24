# Wormhole fp32 accumulation: single elements off by -2^k

On Wormhole, a matmul with `fp32_dest_acc_en` returns rare elements off by exactly a negative power of
two, one to three binades above the running sum. Seen at 1536 tokens in every pair model we
censused (boltz2, protenix-v2, openfold3, opendde, nesso1), at roughly 1e-7 to 3e-6 per output
element. Blackhole shows none on the same inputs. There is no ttnn-level setting that removes it
without giving up accuracy, so it is open upstream. `repro.py` reproduces both forms on one tile.

## What we measured

`stress.py`, randn triangle product at Kt 64, 268M output elements per arm, whglx:

| arm | misses | rel. L2 vs float64 |
|---|---|---|
| HiFi4, in0_block_w 8, packer L1 acc | 150 | 1.18e-3 |
| HiFi4, in0_block_w 1, packer L1 acc | 0 (but see real data below) | 4.87e-4 |
| HiFi3, in0_block_w 8 | 1 | 5.10e-4 |
| packer L1 acc off, in0_block_w 1 / 2 / 8 | 0 / 39 / 62 | 1.28e-2 / 6.40e-3 / 1.79e-3 |
| both operands non-negative, w 8 and w 1 | 0 and 0 | 3.5e-4 |
| in1 non-negative, in0 mixed sign | 158 | 1.49e-3 |
| sign split (4 non-negative products, P - N) | 0 | 3.18e-3 |

On real activations the small K block is not safe either. `fold_census.py --save` captured wrong
elements from a protenix-v2 fold that ran every matmul at in0_block_w 1. `replay.py` reproduces
them bit for bit on one tile. They fail at in0_block_w 1, 2 and 4 with packer L1 accumulation and
are right at 8. `locate.py` puts each one at a single packer add: a partial holding about 10
significand bits plus an opposite-signed addend 2^7 to 2^15 smaller comes back 2^(e+1) too low.

## Where it happens

Two different fp32 adders, in hardware:

- dest accumulation: `_llk_math_matmul_` in
  `tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h` (MVMUL, "D = B*A", ttnn 0.68.0 wheel), which
  `matmul_block` iterates over a K block (`bmm_large_block_zm_fused_bias_activation.cpp:298-301`).
- packer L1 accumulation across K blocks: `llk_pack_reconfig_l1_acc(1)` in the same kernel
  (lines 337-372), `_llk_pack_reconfig_l1_acc_` in `llk_pack_common.h:180`.

Both need negative products. Turning packer L1 accumulation off sends the partials through
`reload_from_cb_to_dst` (kernel line 84), which truncates them and costs 27x in accuracy.

## Tried and reverted

A K block of 1 on every Wormhole HiFi4 fp32-dest matmul (branch commits e77615d0a..26930e50d). It
cleared the dest form in boltz2, nesso1 and openfold3, but protenix-v2 still had 428 wrong elements
(457 without it), all in the packer form. Cost 1.02-1.12x at the fold. Reverted in bab0c3dd7.
