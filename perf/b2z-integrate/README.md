# b2z-integrate — the Blackhole confirmation instrument

The b2z swarm screens levers on the 32-chip Wormhole galaxy. The published Boltz-2 512 aa cell is a
p300c **Blackhole** processor. A mechanism found on Wormhole generalises; a number does not. Nothing
the swarm finds is a result until it is re-measured here: on qb2, under benchlock, paired against
its own incumbent in one process, n=5, with the session's own A/A floor printed next to it.

## Files

* `ab_arms.py` — the harness. Arms are named on the command line, so a lever that lands overnight is
  measured without editing code first. Protocol constants (3 recycles, 200 sampling steps, 1
  diffusion sample, seed 0, `diffusion_trace` off) are imported from
  `perf/b2x-flag-levers/ab_flag_levers.py`, not copied, so this cannot drift off the cell's
  protocol. The CIF layout it writes is the one `perf/b2x-flag-levers/score298.py` and
  `domain_split.py` already read, so parity scoring is those two scripts unchanged.
* `run_bh.sh` — the qb2 wrapper: benchlock, one card, `PYTHONPATH` at the worktree so `tt_bio`
  comes from the checkout rather than the venv's installed package.
* `test_summarize.py` — unit test for the arithmetic that decides GO/NO-GO. No device.

## Running it

    perf/b2z-integrate/run_bh.sh 1 arm1 \
      --arm 'sdpa=tt_bio.tenstorrent._B2_CUSTOM_SDPA=1' \
      --arm 'bfp8=tt_bio.tenstorrent._PAIR_PROJ_BFP8=1' \
      --arm 'all=tt_bio.tenstorrent._B2_CUSTOM_SDPA=1,tt_bio.tenstorrent._PAIR_PROJ_BFP8=1' \
      --combined all

`base` is implicit and runs immediately before every arm. `--combined` turns on the additivity
block: the sum of the separate savings, the measured combined saving, and the discount between
them. The discount is measured here, never projected.

## What the output protects against

| output | the mistake it catches |
|---|---|
| `paired_speedup` (median of per-pair ratios) | a ratio of two medians taken hours apart, where a drift in the box lands in one arm |
| `AA_floor.fold_AA_ratio` | reporting a 1.02x that the session cannot resolve |
| `NO-OBSERVABLE-EFFECT` flag | a lever that never applied, reported as "1.00x, no win". Same time **and** base's CIF digest means it did not take. |
| `INSIDE-AA-FLOOR` flag | same time, different answer: the lever applied and bought nothing |
| `shipped_defaults` in `env` | arm leakage. Every site any arm names is snapshotted at startup and restored before each fold. |
| startup env-pin check | an env var pinned outside the process making every arm the same arm |

CIFs stay on the box; their sha256 digests are in the JSON, which is the committed evidence.
