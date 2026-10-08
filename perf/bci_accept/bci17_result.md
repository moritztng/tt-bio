# #17 at the reporter's own sample size: the measured result

Both arms, 8 trajectories, 288 tokens (hPDL1 115 aa + binder 173 aa), `campaign_seed=42`,
BindCraft 2 v1.0.1 filters, `design_dropout=false`, `mutate_steps=0` on both.
Reproduce the table with `paired_table.py host=<log> card=<log>`.

| arm | accepted | reached harden | reached final | where it ran |
|---|---|---|---|---|
| on-card | **0 of 8** | 2 of 8 | 0 of 8 | qb1 logical card 2 (p150a), AICLK 1350 MHz on 190 of 196 in-run samples, 14:14:53Z-15:52:49Z 2026-10-08 |
| host JAX | **0 of 8** | 4 of 8 | 2 of 8 | A100 80GB, vast 54920059, 22:43Z-23:09:28Z 2026-10-08 |

Same eight design hashes in the same order on both arms, drawn by construction from seed 42 and
binder length 173: fb272c068a88033f, f20e2293b3df8030, 72b0c1e76f0cb82a, 409b501e1bfd7f26,
06e36dad56ba7091, e45cc77e81d50b58, 0f635a2647f518c1, 83f5c78b531e9d8b.

Fisher exact, two-sided:

| comparison | p |
|---|---|
| accepted, card 0 of 8 against host 0 of 8 | 1.000 |
| the reporter's own 0 of 8 against 3 of 8 | 0.200 |
| reached harden, 2 of 8 against 4 of 8 | 0.608 |
| reached final, 0 of 8 against 2 of 8 | 0.467 |

At n=8 per arm the first split that reaches p<0.05 is 5 of 8 against 0 of 8. Eight trajectories can
refute the observation in the terms it was made in; they cannot establish equivalence, and the
reporter's own observation was never significant either.

The harden-stage i_pTM collapse to ~0.12 appears on **both** arms: card trajectories 2 and 4, host
trajectories 4 (0.20) and 6 (0.12). It is not specific to the on-card path.

Neither arm produced a `candidate` scope in `summary.csv`, so neither carries the `Target_pLDDT` /
`Interface_Residues` signature of #21: at 288 tokens with these filters no trajectory on either arm
survives to redesign, which is the stage #21 lives in.

## Per-stage i_pTM, paired by trajectory

```
per-stage i_pTM, paired by trajectory number. * = the stage rejected the trajectory, x = rejected with no reported value, blank = not reached yet.
 traj    stage      host      card
----------------------------------
    1   screen      0.69      0.41
    1   refine      0.12      0.6*
    1   anneal     0.29*          
    2   screen     0.81*      0.75
    2   refine                0.74
    2   anneal                 0.8
    2   harden               0.12*
    3   screen      0.84     0.57*
    3   refine      0.81          
    3   anneal      0.81          
    3   harden      0.62          
    4   screen      0.72      0.83
    4   refine      0.86      0.81
    4   anneal      0.86      0.73
    4   harden      0.2*     0.12*
    5   screen      0.78      0.86
    5   refine     0.15*     0.35*
    6   screen      0.86      0.84
    6   refine      0.85      0.86
    6   anneal      0.87     0.86*
    6   harden     0.12*          
    7   screen      0.81     0.75*
    7   refine     0.75*          
    8   screen      0.85      0.86
    8   refine      0.76     0.85*
    8   anneal      0.85          
    8   harden      0.85          

host: accepted 0 of 8
card: accepted 0 of 8

Acceptance at this budget resolves nothing on its own: Fisher two-sided on the reporter's own 0 of 8 against 3 of 8 is p=0.200. Read the harden rows.
```
