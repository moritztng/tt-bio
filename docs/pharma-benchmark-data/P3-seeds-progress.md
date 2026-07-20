# P3-seeds extension — progress / handoff note

Branch: `wk/tt-bio-pharma-benchmark-p3-seeds`. Task: extend OpenDDE (trpcage + 7ROA prod) and the 3 Boltz-2 affinity targets (FKBP12, DHFR, trypsin) from 3+3 to 5+5 seeds.

## DONE (committed on this branch)

### OpenDDE — COMPLETE (5+5, both legs PASS)
- trp-cage (reduced 4c/20s/1sample): ref seeds 3,4 generated on qb2 CPU (pinned OpenDDE a0d5134, fp32, torch kernels, no MSA). Device seeds 3,4 on qb1 card 0 (p150a). Fixture: `docs/pharma-benchmark-data/ref-fixtures/opendde/trpcage/nomsa_4cycle_20step_1sample_fp32_reduced/seed{3,4}`. RDX: X 0.509 +/- 0.156, R 0.374, D 0.517, X/floor 0.98 -> PASS (all 4 metrics).
- 7ROA production (10c/200s/1sample): ref seeds 3,4 on qb2 CPU (same pinned commit/settings). Device seeds 3,4 on qb1 card 0. Fixture: `docs/pharma-benchmark-data/ref-fixtures/opendde/prot/nomsa_10cycle_200step_1sample_fp32_prod/seed{3,4}`. RDX: X 4.671 +/- 3.322, R 1.499, D 6.042, X/floor 0.77 -> PASS (all 4 metrics). Device still more seed-stochastic than ref (D 6.04 vs R 1.50), same bf16 property; floor device-dominated.
- Both reproduce the 3+3 verdicts within noise. No vast.ai used (CPU was faster/cheaper this run). JSON: `opendde.json`, `opendde-prod-leg.json`. Doc rows + footnote updated.

### Affinity FKBP12 — COMPLETE (5+5, verdict unchanged)
- ref seeds 3,4 generated on qb1 CPU (official boltz 2.2.1, same pinned settings). Device seeds 3,4 on qb1 card 0. Fixture: `docs/pharma-benchmark-data/ref-fixtures/boltz2/affinity_fkg/nomsa_200step_5affsample_3recycle_bf16_mwcorr/seed{3,4}`. RDX: affinity_pred_value X 0.264 +/- 0.151 (R 0.047, D 0.196, X/floor 1.35, within floor+sigma YES); affinity_probability_binary X 0.018 (X/floor 1.07, YES); ligand-pose RMSD X 0.319 (X/floor 1.04, within floor+sigma YES); 1-pocket-lDDT X 0.120 (X/floor 4.68, NO -> GAP). Verdict unchanged from 3+3 (PASS/PASS/PASS/GAP). JSON: `boltz2-affinity-fkbp12-5x5.json`. Doc row + pass-5 note updated.

## PENDING (next relaunch)

### Affinity DHFR + trypsin — NOT done (still 3+3 in doc)
- ref seeds 3,4: GENERATED on qb1 CPU, sitting at `/tmp/affinity_ref/ref_dhfr_s{3,4}` and `/tmp/affinity_ref/ref_tryp_s{3,4}` (raw boltz output, NOT yet harvested into the fixture). Harvest into `docs/pharma-benchmark-data/ref-fixtures/boltz2/affinity_{dhfr,tryp}/.../seed{3,4}/` (copy `predictions/<id>/affinity_<id>.json` -> `seedN/affinity_<id>.json`, `predictions/<id>/<id>_model_0.cif` -> `seedN/structures/<id>.cif`), update fixture meta seeds -> [0,1,2,3,4].
- device seeds 0-4 for BOTH dhfr and tryp: RUNNING on qb1 card 0 via nohup (`/tmp/affinity_dev_rest.sh`, pid 177735-ish, log `/tmp/affinity_dev_rest.log`). Output at `/tmp/affinity_dev/dev_{dhfr,tryp}_s{0..4}/boltz_results_affinity_{dhfr,tryp}/`. The existing 3+3 device seeds 0-2 for dhfr/tryp were EPHEMERAL (lived in /tmp on the pass-6 host, now gone), so ALL 5 device seeds are regenerated fresh here. Check `/tmp/affinity_dev_rest.log` for `AFFINITY_DEV_REST_ALL_DONE` before harvesting.
- After both land: recompute with `scripts/boltz2_affinity_parity.py --ref-dirs <fixture>/seed{0,1,2,3,4} --dev-dirs /tmp/affinity_dev/dev_{dhfr,tryp}_s{0..4}/boltz_results_affinity_{dhfr,tryp} --target-id affinity_{dhfr,tryp}`; update the doc pass-6 table rows + the `boltz2-affinity-{dhfr,tryp}.json` files; confirm PASS/GAP unchanged.

### Release gate + state file — NOT done
- After dhfr/tryp land, re-run the FULL release gate (accuracy+perf+UX) per task item 3. Must still pass.
- Then write `~/.coworker/state/tt-bio-pharma-benchmark-p3-seeds.md` (item 4) and only THEN emit DONE:.

## Notes
- vast.ai: NOT used ($0 spent, no instance rented). CPU was faster for OpenDDE this run; boltz 2.2.1 CPU affinity ref ran on qb1/qb2 CPU. No teardown needed.
- All R/D/X numbers above are real measured values from `scripts/pharma_parity.py` / `scripts/boltz2_affinity_parity.py`, not fabricated.
- Worktree is on qb1 (`/home/ttuser/.coworker/wt/tt-bio-pharma-benchmark-p3-seeds`). The shared checkout `/home/ttuser/tt-bio-dev` is NOT touched.
