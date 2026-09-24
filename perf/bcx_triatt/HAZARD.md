# The batched reuse matmul corrupts output when one batch's M is split over blocks

qb1 card 1 (p150a, `subsystem_device 0x0040`), AICLK 1350, tt-bio-dev env ttnn, 2026-09-24.

`ttnn.MatmulMultiCoreReuseProgramConfig` on a batched `[B,H,M,K] x [B,H,K,N]` product:

- `per_core_N` must equal N. Anything else raises `TT_FATAL ... N == per_core_N`
  (`bmm_stress.json`, the `pn4` rows).
- With `per_core_M < M`, so one batch's output takes two or more M blocks, the kernel returned
  wrong output inside a real fwd+bwd pair unit: QK^T `[256,4,256,32] x [256,4,256,32]^T`,
  `per_core_M=4` of `Mt=8`, gave 3 non-finite elements in the first blocks of core 0, and the
  finite elements did not match a rerun on the same inputs. Inputs were checked finite, and a
  device sync after every matmul did not make it go away (3/252 and 7/252 calls, with and
  without `packer_l1_acc`). It never failed in isolation: 0/60 repeats per shape, bit-identical
  (`bmm_stress.json`). An explicit transpose instead of `transpose_b` still failed, so the
  transpose flag is not the cause. With the whole output matrix per block (`per_core_M = Mt`),
  0/1344 calls at chunk 128 and 0/336 at chunk 256 were bad, and two runs of the float64-graded
  e2e gave bit-identical logit gradients at both chunks.

So `autograd.bmm_program_config` takes the whole output matrix per block (at most 64 tiles), or
declines. It is a region the evidence covers, not a fix of the kernel.

Splitting the contraction is a different axis and is not the hazard. At n=384 the K=384
products run in two K blocks of 6 tiles (`in0_block_w` 6 of `Kt` 12): 0 bad calls in 1134
in-block checks (`n384/hazard_new.log`, `n384/hazard_probe.py`), two float64-graded e2e runs
gave bit-identical logit gradients (`e2e/e2e_n384_kblock_{a,b}.log`), and n=256, where
`Kt` 8 fits one block, is bit-exact against the single-K-block commit.

Instrument: `/tmp`-style wrappers around `ttnn.matmul` inside `block_ab.py`'s arms that checked
input and output finiteness per call; the reproducible parts are `bmm_stress.py` and the
`e2e/e2e_n256_verbs_on_c{128,256}_{a,b}.log` pairs.
