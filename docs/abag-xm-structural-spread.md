# AbAg-XM structural spread (spec 2.9)

Per-fold structural convergence of the 50-sample ensemble (mean pairwise TM over all 1225 sample pairs; sd in parens) split by success. Failing = no sample at DockQ >= 0.23. A LOW mean pairwise TM means the samples disagree structurally (several basins); HIGH means they converged on one pose. The 3 unscorable targets (9ly2/9ly3/9lz2) are excluded.

| generator | n fail | n succ | median mean-TM fail (sd) | median mean-TM succ (sd) | MWU p | Cliff's d |
|---|---|---|---|---|---|---|
| opendde-abag | 42 | 119 | 0.837 (0.094) | 0.976 (0.022) | 2.59e-07 | -0.54 |
| protenix-v2 | 57 | 104 | 0.765 (0.111) | 0.878 (0.083) | 6.26e-04 | -0.33 |
| boltz2 | 81 | 80 | 0.796 (0.103) | 0.912 (0.070) | 3.32e-05 | -0.38 |

## Failure class per generator (split at the generator median mean-TM)

| generator | multi-basin wrong (seeds might help) | converged-wrong (seeds won't help) | dominant class |
|---|---|---|---|
| opendde-abag | 34 | 8 | multi-basin |
| protenix-v2 | 38 | 19 | multi-basin |
| boltz2 | 51 | 30 | multi-basin |

- opendde-abag: failing ensembles mostly sample several wrong basins (8/42 converged-wrong, 34/42 multi-basin).
- protenix-v2: failing ensembles mostly sample several wrong basins (19/57 converged-wrong, 38/57 multi-basin).
- boltz2: failing ensembles mostly sample several wrong basins (30/81 converged-wrong, 51/81 multi-basin).

