# BindCraft 2 on one H200 — the reference this campaign divides by

Measured 2026-09-25 on vast.ai instance 52585411, 1x NVIDIA H200, driver 615.71.09, jax 0.11.2,
BindCraft 2 pinned at `7a2dfdb8a285232a6f881899fe135c6dc48679f1`. hPDL1 chain A (115 residues),
binder length 146, `--core benchmark`. Full reading in `state/bcx-gpuref.md`.

    warm/                       the measurement: 1,698 warm steps, 14 trajectories, 4 accepted designs
      denominator_warm.json     the distributions and the per-trajectory phase split
      steps_warm.jsonl          every step and phase record, both processes, in order
      designs_warm.json         validate_designs.py's verdict on the four accepted designs
      !_Ranked.csv              the accepted designs with sequences and metrics
      clocks_warm.txt           SM clock sampled every 5 s during the run; 1980 MHz on all 253
      stamp_warm.txt            card, driver, settings
      rental.txt                offer, instance, teardown
      sha256.txt                hashes as the files stood on the box, before 2_Refolded was archived
      pdl1_warm/                BC2's own output tree; 2_Refolded is tarred to keep the repo small
    arm1-first-four/            the first four trajectories, superseded by warm/ which contains them
    aborted-shared-box-.../     a start on the mgx-reference box that was abandoned, kept for the ledger

Regenerate the figures with `python3 perf/bcx_gpuref/report.py results/warm/denominator_warm.json
--designs results/warm/designs_warm.json --ranked results/warm/'!_Ranked.csv'`.
