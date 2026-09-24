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

So `autograd.bmm_program_config` takes the whole output matrix and the whole contraction per
block, or declines. It is a region the evidence covers, not a fix of the kernel.

Instrument: `/tmp`-style wrappers around `ttnn.matmul` inside `block_ab.py`'s arms that checked
input and output finiteness per call; the reproducible parts are `bmm_stress.py` and the
`e2e/e2e_n256_verbs_on_c{128,256}_{a,b}.log` pairs.
