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
