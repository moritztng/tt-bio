# of3-1y57-template-tail-rootcause — evidence

Question: the faithful templates-ON 1y57 arm (`test_template_lowers_rmsd[1y57]`) showed a wider
upper tail than upstream's MPS envelope at N=8 (4 of 8 samples above their 15.5 Å observed max).
Is that (a) a model/input property, (b) small-N noise, or (c) a port defect?

## Device arm, N=24 (done 2026-08-18)

`runs/device_on_n24/`: faithful ON arm (one-row dummy MSA + `template_alignments/1y57.npz`),
seed 42, 24 diffusion samples, 200 steps, 3 recycles, qb2 card 0, worktree at `6fc864c9`
(== origin/main). Command (reconstructed from the log):

```
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:of3-1y57-template-tail-rootcause \
OF3_TEMPLATE_STRUCTURES=$PWD/examples/of3_upstream/template_structures \
PYTHONPATH=$PWD /home/ttuser/tt-bio/env/bin/python3 -m tt_bio.main predict \
  examples/of3_upstream/1y57_template_on_dummymsa.yaml --model openfold3 \
  --out_dir perf/of3-1y57-template-tail-rootcause/runs/device_on_n24 \
  --diffusion_samples 24 --sampling_steps 200 --recycling_steps 3 --seed 42
```

Scored with upstream's own `best_ca_rmsd` (`scripts/of3_upstream_score.py`, upstream venv on pc):
`dev_on_n24_scored.json`. n=24, mean 13.356 Å, sd 3.866, max 22.21. Sorted:
8.36, 8.38, 8.58, 8.61, 8.73, 8.73, 9.28, 9.42, 10.76, 12.55, 13.70, 15.16, 15.33, 15.45,
15.46, 15.56, 15.67, 15.75, 15.83, 15.84, 16.02, 17.04, 18.14, 22.21.

Seed/protocol verification: the port seeds each diffusion sample as seed+sample_index
(`tt_bio/openfold3_fold.py`), so samples 0-7 of this run replay the committed qb1 seed-42 run
(`docs/implementation-parity-data/openfold3-upstream-suite/tmpl_1y57_on_dummymsa.json`). 6 of 8
values match within 0.022 Å; two moved (19.08→18.14, 11.43→10.76), consistent with chaotic
trajectory amplification across hosts (qb1 p150a vs qb2 P300), not a protocol difference.

## CPU control, N=24 (running)

Upstream openfold3 0.4.4, fp32, checkpoint of3-p2-155k, on pc (the calibrated control host):
3 × 8 samples, seeds 42/43/44, same query, same alignment (the template cache npz the runner
consumes is array-identical to the committed `1y57.npz`: index, release_date, idx_map all equal).
One-row dummy MSA is upstream's own no-MSA behavior, matching the faithful arm.
