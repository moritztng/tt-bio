# grade: copy vs exact, normal mode

Provisional Blackhole A/A floor: tt-bio 0.12.1 exact, qb1 p150a card 1, seeds 101-103 (pfm-accuracy run, 2026-10-07/08),
re-graded with perf/spd/grade.py. The arch label reads wormhole_b0 only because these records were converted from
tt-bio results.json, which carries no arch; the folds ran on Blackhole. "copy" is the same folds under a second name,
so the PAIRED block is a self-check (all zeros), and only FLOOR is a measurement.

Records: 66 reps from 1 run dirs; failed/non-finite folds: 0

## wormhole_b0

Complexes 11; paired folds 33; paired seeds per complex: 28VJ 3, 9DBP 3, 9HL2 3, 9LLG 3, 9LV4 3, 9PCQ 3, 9TH6 3, 9TY2 3, 9W3L 3, 9W89 3, 9W8A 3

### FLOOR (A/A, exact seed vs seed, median / mean of |diff|)

| metric | n pairs | median | mean |
|---|---|---|---|
| dockq | 33 | 0.0117 | 0.0169 |
| lddt_ca | 33 | 0.0015 | 0.0015 |
| tm | 33 | 0.0035 | 0.0087 |
| lrmsd | 33 | 0.3319 | 3.9115 |
| irmsd | 33 | 0.0601 | 0.8903 |
| plddt | 33 | 0.0003 | 0.0006 |
| iptm | 33 | 0.0005 | 0.0019 |
| top-pose CA RMSD (A) | 33 | 0.816 | 2.357 |

### PAIRED (copy - exact, same seed; mean over complexes [95 % bootstrap CI])

| metric | mean delta | 95 % CI | better side |
|---|---|---|---|
| dockq | +0.0000 | [+0.0000, +0.0000] | higher |
| lddt_ca | +0.0000 | [+0.0000, +0.0000] | higher |
| tm | +0.0000 | [+0.0000, +0.0000] | higher |
| lrmsd | +0.0000 | [+0.0000, +0.0000] | lower |
| irmsd | +0.0000 | [+0.0000, +0.0000] | lower |
| plddt | +0.0000 | [+0.0000, +0.0000] | higher |
| iptm | +0.0000 | [+0.0000, +0.0000] | higher |

Same-seed top-pose deviation copy vs exact: median 0.000 A, max 0.000 A (n=33); A/A seed floor median 0.816 A; kill bar 0.6 A.
Docking success (DockQ >= 0.23, top pose): exact 24/33 = 72.7 %, copy 24/33 = 72.7 %.
Per-complex mean pLDDT change: 28VJ +0.000, 9DBP +0.000, 9HL2 +0.000, 9LLG +0.000, 9LV4 +0.000, 9PCQ +0.000, 9TH6 +0.000, 9TY2 +0.000, 9W3L +0.000, 9W89 +0.000, 9W8A +0.000

### VERDICT wormhole_b0: INSUFFICIENT: 3 paired seeds (< 4)

SUMMARY: wormhole_b0 INSUFFICIENT
