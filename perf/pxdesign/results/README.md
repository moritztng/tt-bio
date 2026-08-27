# PXDesign perf-page cells, 2026-08-26

The four cells the `design` row publishes, and the A/B that justified the stack they were measured
on. `cells_2026_08_26.json` is the pooled summary; the per-run reports carry everything it was
derived from.

| directory | box | what it holds |
|---|---|---|
| `h200_2026_08_26/` | H200, Xeon 8488C | A0 on the pinned stack, +8.5 % against the published cell. The outlier box. |
| `h200b_2026_08_26/` | H200, Xeon 8480+ | A0 retry (+0.45 %), the pinned and modern `gen` sweeps, and `B1.json` |
| `a100_2026_08_26/` | A100 SXM4 40 GB, EPYC 7713 | the a100 cell |
| `b200_2026_08_26/` | B200 sm_100, Xeon 8592+ | the b200 cell |

Each `run_<label>_rep<N>.json` is one process: five rounds, round 0 cold and dropped, seeds
0,1,2,3,0. Read `warm_median_cell_s` for the leg, `rounds[].gen_cell_s` for the samples,
`digests` for the coordinate digest per seed, `gpu_exclusive` and `compute_apps_before/after` for
the exclusivity evidence, `stages` for the per-stage counts, and `ds_counter_selftest` /
`jax_counter_selftest` for the two positive controls behind H1.

**The written CIFs are not committed.** They are 9.4 MB for one A/B and they are not the evidence:
`digests` in `run_laczc512_gen_pinned_rep*.json` and `run_laczc512_gen_modern_rep*.json` already
proves the point those files would, because the digest is taken over the CIF's own coordinate text.
The two stacks agree seed for seed:

    seed 0  cf5fe1a04b549b09    seed 1  48ffc413f2acf702
    seed 2  532083b869f0fe12    seed 3  a5471e16fbdb3dbb

`B1.json` is the same result computed the other way, from the coordinates rather than the digest:
direct RMSD 0.0000 A and max atom deviation 0.0000 A on every cross-stack pair, against an
inter-seed scale of 46.3 A. Regenerate the CIFs by re-running the sweep; `stack_ab.py` recomputes
`B1.json` from a results directory that still has them on disk.
