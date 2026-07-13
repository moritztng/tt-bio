# OpenDDE MSA wiring (P7) — run artifacts + numbers

Branch: `wk/tt-bio-opendde-msa-wiring`. Real device runs on pc, card 0
(`TT_VISIBLE_DEVICES=0`), real `opendde_abag.pt` weights, PDB 9dsg, 10 recycles / 200
diffusion steps. Env: `~/tt-bio/env` (editable `tt_bio` maps to the shared checkout, so
runs use `PYTHONPATH=<worktree>` to pick up the worktree code). DockQ via `DockQ==2.1.3`
installed into a throwaway `--target` lib (not the project env, not a runtime dep).

## Code (committed 07674bc7)

- `tt_bio/worker.py::_predict_opendde_one` reuses the Protenix-v2 MSA stage
  (`_generate_esmfold2_a3m` + `_resolve_a3m_text` + `build_complex_features` block-diag MSA).
- `tt_bio/main.py::_resolve_msa_default` now covers `opendde` / `opendde-abag`.
- `metrics["msa"]` now reflects real per-chain MSA usage.

## Run artifacts (outside the worktree, not committed)

- 1-sample + MSA: `/home/moritz/.coworker/run-opendde-msa-9dsg/n1/`
  - `boltz_results_9dsg_abag/results.json` (msa=true, ipTM 0.712, pLDDT 0.892)
  - `boltz_results_9dsg_abag/structures/9dsg_abag.cif`
  - `dockq.json` (A-H 0.0113 / fnat 0, H-L 0.494, global 0.253)
  - `msa/*.a3m` (antigen 12651 rows, heavy 117, light 10950) — reused by the n5 run
- 5-sample + MSA: `/home/moritz/.coworker/run-opendde-msa-9dsg/n5/`
  - `boltz_results_9dsg_abag/results.json` (5 samples, rank 0 ipTM 0.7147; all_runs tight)
  - `structures/9dsg_abag.cif` (rank 0) + `.._model_{1..4}.cif`
- DockQ lib: `/home/moritz/.coworker/run-opendde-msa-9dsg/dockq_lib/`

## DockQ across all 5 samples (P7, MSA + best-of-5)

| sample (rank) | A-H DockQ | A-H fnat | H-L DockQ | global |
|---|---:|---:|---:|---:|
| 0 (conf. best) | 0.0113 | 0 | 0.4974 | 0.2544 |
| 1 | 0.0112 | 0 | 0.4843 | 0.2477 |
| 2 | 0.0112 | 0 | 0.4832 | 0.2472 |
| 3 | 0.0109 | 0 | 0.4808 | 0.2459 |
| 4 | 0.0113 | 0 | 0.4766 | 0.2439 |

## Verdict (honest)

MSA is wired and consumed (ipTM 0.549 -> 0.712, Fab H-L 0.377 -> 0.497), but Ab-Ag
DockQ stays 0.011 / fnat 0 across one and five samples — the antigen is never placed in
the paratope. The distribution is degenerate at fnat=0, so confidence- AND oracle-based
best-of-N both can't help. This is a genuine port/model issue distinct from the
missing-input gap; MSA + N_sample=5 (the paper's standard pipeline) does not close it.

## Reproduce

```
cd <worktree>
TT_VISIBLE_DEVICES=0 PYTHONPATH=<worktree> ~/tt-bio/env/bin/python -c \
  "from tt_bio.main import cli; cli()" predict examples/9dsg_abag.yaml \
  --model opendde-abag --use_msa_server --recycling_steps 10 \
  --sampling_steps 200 --diffusion_samples 5 --out_dir <runs>/n5
PYTHONPATH=<worktree>/../run-opendde-msa-9dsg/dockq_lib ~/tt-bio/env/bin/python \
  scripts/opendde_dockq.py <runs>/n5/boltz_results_9dsg_abag/structures/9dsg_abag.cif \
  examples/ground_truth_structures/9dsg.cif
```
