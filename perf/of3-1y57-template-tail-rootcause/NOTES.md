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

## CPU control, N=24 (done 2026-08-18/19)

Upstream openfold3 (0.4.6.dev12+g72fc3a953, the calibrated control env), fp32, checkpoint
of3-p2-155k, on pc: 3 × 8 samples, seeds 42/43/44, same query, same alignment (the template cache
npz the runner consumes is array-identical to the committed `1y57.npz`: index, release_date, idx_map
all equal). One-row dummy MSA is upstream's own no-MSA behavior, matching the faithful arm. Scored
with the same `scripts/of3_upstream_score.py`, `--layout upstream`: `cpu_on_s{42,43,44}_scored.json`.

Combined N=24: mean **12.524 Å**, sd 2.657, max **15.84**. Sorted:
8.21, 9.19, 9.27, 9.40, 9.51, 9.52, 9.57, 9.97, 10.53, 10.59, 13.41, 13.46, 13.55, 14.15, 14.16,
14.18, 14.61, 14.70, 14.94, 15.12, 15.43, 15.43, 15.83, 15.84.
0 of 24 above 16 Å; 2 of 24 above upstream's MPS observed max (15.5). This reproduces upstream's
MPS band (8.2-15.5, 12.07 ± 2.83) almost exactly.

## Verdict: (a) model/input property, not a port defect

- Upstream's own calibration note (`test_templates.py`, 1y57 case) records their CUDA-family runs
  spreading ON per-sample over **8.6-22.8 Å** (5 distinct draws replayed across two machines). Our
  entire device range (8.36-22.21) sits inside upstream's own envelope, at a matching tail rate
  (>=1/5 above 16 Å vs our 4/24).
- The tight band is specific to CPU/MPS-class numerics: our matched-N CPU control of upstream's own
  model stays <= 15.84. Upstream's eval default enables triton triangle kernels on CUDA; CPU/MPS run
  the reference path. The wide family is upstream-CUDA + this port's ttnn kernels.
- Port-specific suspects all controlled: same npz both arms (array-identical), same one-row dummy
  MSA, same checkpoint, same metric (upstream's own `best_ca_rmsd`), diffusion sampler fp32 on device
  (`OF3_DIFFUSION_FP32_DEVICE=1` default, no override in the run), noise drawn host-side on MT19937
  (same RNG backend as the CPU control), per-sample seeding replays committed values across hosts to
  0.022 Å, confidence ranking does not enter the per-sample distribution.
- Stats: means indistinguishable (13.356 vs 12.524, Welch t 0.869, p 0.39). Tail rate >= 16 Å:
  device 4/24 vs CPU 0/24, Fisher p 0.055 (borderline). Upstream's own internal split is the same
  order: CUDA 1/5 vs MPS 0/8, Fisher p 0.38. Pooled wide family (device 24 + upstream CUDA 5) vs
  tight family (CPU 24 + upstream MPS 8): Fisher p 0.020. KS D 0.375, p 0.051.
- Not (b): the tail persists at N=24 on device and appears in upstream's own n=5. Not (c): no
  port-specific suspect survives; our max is below upstream's own observed max.

Result JSON: `verdict.json` here; state `~/.coworker/state/of3-1y57-template-tail-rootcause.md`.
