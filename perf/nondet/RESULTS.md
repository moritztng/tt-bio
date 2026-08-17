# protenix-v2 run-to-run determinism — qb1 card 3 leg

Fixture: CDK2 (1HCL) sequence tiled/truncated to N residues (targets/cdk2_N.yaml),
`--model protenix-v2 --single_sequence --seed 0`, fresh process per fold, one at a time
on TT_VISIBLE_DEVICES=3, ttnn 0.67.4, branch wk/protenix-v2-nondeterminism-rootcause
(one merge behind origin/main 198981b7; the delta is a boltz2 re-baseline, no protenix code).

Verdict = sha256 over ATOM/HETATM lines of the two CIFs (compare.py).

| size | rep a vs rep b | atom sha |
|---|---|---|
| 128 | IDENTICAL | be53212ba1e28570 |
| 160 | IDENTICAL | c7d24c5ae42893f1 |
| 192 | IDENTICAL | c1b909281a801608 |
| 224 | IDENTICAL | afc583ffa37511e0 |
| 256 | IDENTICAL | 0879651e337c7fcc |
| 384 | IDENTICAL | 02acf18a1e279727 |
| 512 | IDENTICAL | 43322b70734c8e70 |

qb1 card 3 shows NO run-to-run nondeterminism at any size 128-512, where pc card 0
(japanfold-wh-correctness-close §14.5, ttnn version TBD) showed NONDETERMINISTIC at 256/384
and the pc-card0-512aa memory shows 4/4 distinct CIFs at 512. The phenomenon is
host/card-conditioned, not size-conditioned in the code alone.

## n>=5 floor extension (2026-08-16, same setup)

| size | n folds (independent processes) | atom sha |
|---|---|---|
| 256 | 7 (a,b + c,d,e,f,g) | ALL 0879651e337c7fcc |
| 384 | 5 (a,b + c,d,e) | ALL 02acf18a1e279727 |

qb1 card 3 is bit-stable with n=7 / n=5 at 256/384 aa — a hash-equality bar on qb1 needs
no noise-floor allowance at these sizes. pc card 0 shows rmsd 1.05-2.79 A run-to-run at
160-256 aa (out/solo_160_*, solo_224_*, solo_256_pc_* on the pc worktree).

## Canonical pair-cond probe references (qb1, `paircond_repeat.py`, 2026-08-16)

Per-stage `out_sha16` for the nine pair-cond stages, in order relpe_linear, concat,
layer_norm, linear_z, transition_z1, add_z1, transition_z2, add_z2, full_paircond. Fixed
input/weight seeds make these comparable across hosts, cards and processes: any host that
reproduces these shas is bit-identical to qb1 on this chain.

| N | prec | card | K | shas |
|---|---|---|---|---|
| 256 | fp32 | 1, 2, 3 (all equal) | 8 | `4362d54b,b576fe5e,003e9ca9,f02ef7cb,e79944a5,1b9124b5,c2fd7917,fa695ce1,fa695ce1` |
| 160 | fp32 | 3 | 8 | `3f668b97c9aa9ee4,46a8f05e4bfa899f,5c107a6f4ad140ca,d5e3f6ef691d14bc,e6a5c0f2d9d8737c,80fd2643a4a5b75f,4e27fc896db2761b,ab1a70bd898ca817,ab1a70bd898ca817` |
| 160 | bf16 | 3 | 8 | `42d9ce218fcfa2f6,468edfaddfbc186c,36f18abc8d7121a4,cdc2b01c8dfc1a45,c745660cb07edbe5,55893e38e1f1510a,07fc151fd2fe70b7,d96f1ac1c8ff7633,d96f1ac1c8ff7633` |
| 384 | fp32 | 3 | 6 | `08de56b217e6168b,cf0a0c76b64ee29c,ea50d0ae73a21ad8,820ab3d9254546f3,45941a8de3bec19a,d96e115247bef4e2,b5b34844fb6e18f5,12e9985fde81f8ff,12e9985fde81f8ff` |

Every run above: all stages bit-identical across repeats, readback bit-exact, upload control
bit-exact. 160 aa is the smallest size at which pc card 0 is known unstable, so it is the
cheapest pc comparison point and now has a reference at both precisions.

`dram_stability.py 256 <iters> <bf16|fp32>` (zero compute: read-read, read-written,
copy-copy) is clean on qb1 card 3 at both precisions, 2 iters each. 256 MB is ~4x the 256 aa
fp32 pair_z tensor and is the validated working size; 1024 MB aborts inside ttnn on tensor
size, not on any fault.

## qb1 card 0 — the last untested fleet card (2026-08-17)

Card 0 is the qb1 chip with a wedge history, so it was the one card that could plausibly have
broken the "every qb1 card is clean" claim. It does not.

| probe | result |
|---|---|
| `dram_stability.py 256 4 fp32` | 0 mismatched elements over 4 x 256 MB (read-read, read-written, copy-copy all 0) |
| `paircond_repeat.py 256 8 fp32`, process r1 | 9/9 stages bit-identical, readback + upload bit-exact |
| `paircond_repeat.py 256 8 fp32`, process r2 | 9/9 stages bit-identical; every sha equals r1 |

Card 0 per-stage shas are the canonical values, byte for byte:
`4362d54b,b576fe5e,003e9ca9,f02ef7cb,e79944a5,1b9124b5,c2fd7917,fa695ce1,fa695ce1`.
Storage grid 13x10, CORE_GRID_MAIN (13,10), same as cards 1/2/3.

All four qb1 cards now produce the same nine shas, within a process and across processes. The
canonical pair-cond answer is a 4-card, 2-process result, not a 3-card one.

### Cross-card fold-level agreement (qb1 card 0 vs card 3)

| N | card 3 canonical | card 0 | rmsd | atoms moved >1e-6 A |
|---|---|---|---|---|
| 160 | `c7d24c5ae42893f1` (n=2) | `c7d24c5ae42893f1` | 0.0000 A | 0/1278 |
| 256 | `0879651e337c7fcc` (n=7) | `0879651e337c7fcc` | 0.0000 A | 0/2060 |

Real weights, real data, fresh process, seed 0. The canonical output is cross-card, so pc card
0 is producing a wrong answer rather than an equally valid alternative reduction order.
