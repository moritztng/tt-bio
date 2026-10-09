# grade: exact vs exact, normal mode

Records: 44 reps from 11 run dirs; failed/non-finite folds: 0

## wormhole_b0

Complexes 11; paired folds 44; paired seeds per complex: 28VJ 4, 9DBP 4, 9HL2 4, 9LLG 4, 9LV4 4, 9PCQ 4, 9TH6 4, 9TY2 4, 9W3L 4, 9W89 4, 9W8A 4

### FLOOR (A/A, exact seed vs seed, median / mean of |diff|)

| metric | n pairs | median | mean |
|---|---|---|---|
| dockq | 66 | 0.0136 | 0.0550 |
| lddt_ca | 66 | 0.0008 | 0.0029 |
| tm | 66 | 0.0054 | 0.0143 |
| lrmsd | 66 | 0.5609 | 5.5916 |
| irmsd | 66 | 0.0684 | 1.9793 |
| plddt | 66 | 0.0002 | 0.0010 |
| iptm | 66 | 0.0008 | 0.0098 |
| top-pose CA RMSD (A) | 66 | 0.807 | 3.575 |

### PAIRED (exact - exact, same seed; mean over complexes [95 % bootstrap CI])

| metric | mean delta | 95 % CI | better side |
|---|---|---|---|
| dockq | +0.0000 | [+0.0000, +0.0000] | higher |
| lddt_ca | +0.0000 | [+0.0000, +0.0000] | higher |
| tm | +0.0000 | [+0.0000, +0.0000] | higher |
| lrmsd | +0.0000 | [+0.0000, +0.0000] | lower |
| irmsd | +0.0000 | [+0.0000, +0.0000] | lower |
| plddt | +0.0000 | [+0.0000, +0.0000] | higher |
| iptm | +0.0000 | [+0.0000, +0.0000] | higher |

Same-seed top-pose deviation exact vs exact: median 0.000 A, max 0.000 A (n=44); A/A seed floor median 0.807 A; kill bar 0.6 A.
Docking success (DockQ >= 0.23, top pose): exact 33/44 = 75.0 %, exact 33/44 = 75.0 %.
Per-complex mean pLDDT change: 28VJ +0.000, 9DBP +0.000, 9HL2 +0.000, 9LLG +0.000, 9LV4 +0.000, 9PCQ +0.000, 9TH6 +0.000, 9TY2 +0.000, 9W3L +0.000, 9W89 +0.000, 9W8A +0.000

### VERDICT wormhole_b0: PASS

SUMMARY: wormhole_b0 PASS
