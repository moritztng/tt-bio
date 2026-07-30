# AbAg-XM structural spread (spec 2.9)

Per-fold structural convergence of the 50-sample ensemble (mean pairwise TM over all 1225 sample pairs; sd in parens) split by success. Failing = no sample at DockQ >= 0.23. A LOW mean pairwise TM means the samples disagree structurally (several basins); HIGH means they converged on one pose. The 3 unscorable targets (9ly2/9ly3/9lz2) are excluded.

| generator | n fail | n succ | median mean-TM fail (sd) | median mean-TM succ (sd) | MWU p | Cliff's d |
|---|---|---|---|---|---|---|
| opendde-abag | 41 | 120 | 0.828 (0.094) | 0.976 (0.022) | 6.07e-08 | -0.57 |
| protenix-v2 | 56 | 105 | 0.761 (0.111) | 0.881 (0.083) | 2.31e-04 | -0.35 |
| boltz2 | 81 | 80 | 0.796 (0.103) | 0.912 (0.070) | 3.32e-05 | -0.38 |

## Failure class per generator (split at the generator median mean-TM)

| generator | multi-basin wrong (seeds might help) | converged-wrong (seeds won't help) | dominant class |
|---|---|---|---|
| opendde-abag | 34 | 7 | multi-basin |
| protenix-v2 | 38 | 18 | multi-basin |
| boltz2 | 51 | 30 | multi-basin |

- opendde-abag: failing ensembles mostly sample several wrong basins (7/41 converged-wrong, 34/41 multi-basin).
- protenix-v2: failing ensembles mostly sample several wrong basins (18/56 converged-wrong, 38/56 multi-basin).
- boltz2: failing ensembles mostly sample several wrong basins (30/81 converged-wrong, 51/81 multi-basin).

