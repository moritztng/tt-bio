# P21: the in-matmul pair bias is AF2-scoped

Evidence that scoping `linear_o.bias` to AF2-IG's triangle-attention blocks leaves RF3 folding
exactly what main folds, and leaves AF2-IG's own tap gate where P19 measured it. All legs on qb2
(qb1's cards were held by the v0.6.6 release gate).

## RF3, card 1

`perf/rf3/fold_fix_ab.py --fix cdk2x2_<size> --seeds 0[,0] --sampling-steps 5 --dump-distogram`,
10 recycles, seed 0. The distogram is read before the sampler, so it carries no sampler noise.
`main` arms are main `0062bb4f`: `tenstorrent.py`, `protenix.py` and `protenix_data.py`, which is
every file this branch changes that RF3's import graph reaches.

| dir | size | arm | CIF sha256 | distogram sha256 | plDDT |
|---|---|---|---|---|---|
| `main/` | 298 | main, files checked out over the branch | `abd218c5158a5983` | `c4dcf66cfd3315cc` | 81.6548 |
| `main298iso/` | 298 | main, separate tree built by `git archive` | `abd218c5158a5983` | `c4dcf66cfd3315cc` | 81.6548 |
| `narrow/` | 298 | this branch, two folds at one seed for an A/A control | `abd218c5158a5983` | `c4dcf66cfd3315cc` | 81.6548 |
| `sens_o/` | 298 | this branch with `TT_BIO_PAIR_BIAS_IN_MATMUL=o`, P20's default | `0aeb6928e3df890d` | `c46d27883c86abe5` | 81.6199 |
| `main512/` | 512 | main, separate tree | `24c48f0cf8d5c2e7` | `9dbe842f420ffe1e` | 77.929 |
| `narrow512/` | 512 | this branch, two folds at one seed | `24c48f0cf8d5c2e7` | `9dbe842f420ffe1e` | 77.929 |

`sens_o` is the control that makes the agreements mean something: it moves both digests, and its
`pair_bias_stats` flips from 1088 `o_add` a fold to 1088 `o_in_matmul_exit_l1` and zero `o_add`, so
the fixture is demonstrably able to see this change. `main298iso` is the method control, the same
main answer through a different way of building the arm.

## AF2-IG tap gate, card 2

`scripts/af2_port/tap_gate.py --device --template-host`, `OMP_NUM_THREADS=8`. `tap_head.json`
against `scripts/af2_port/parity_artifacts/host_bisect/p19_committed.json`, same host:

| leg | taps_failed | scalars_failed | pcc_min | envelope_ratio_max |
|---|---|---|---|---|
| P19 committed, qb2 card 0 | 9 | 2 | 0.9969005520359849 | 9.115207053809002 |
| P21 HEAD, qb2 card 2 | 9 | 2 | 0.9969005520359849 | 9.115207053809002 |

Bit-identical, so both the narrowing and the merge of main are inert for AF2-IG.

`distogram.npy` and the CIFs are not committed; every digest above is recorded in the `fold.json`
beside it.
