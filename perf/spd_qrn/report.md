# grade: qr vs f0, normal mode

Records: 88 reps from 22 run dirs; failed/non-finite folds: 0

## wormhole_b0

Complexes 11; paired folds 44; paired seeds per complex: 28VJ 4, 9DBP 4, 9HL2 4, 9LLG 4, 9LV4 4, 9PCQ 4, 9TH6 4, 9TY2 4, 9W3L 4, 9W89 4, 9W8A 4

### FLOOR (A/A, f0 seed vs seed, median / mean of |diff|)

| metric | n pairs | median | mean |
|---|---|---|---|
| dockq | 66 | 0.0140 | 0.0661 |
| lddt_ca | 66 | 0.0029 | 0.0112 |
| tm | 66 | 0.0055 | 0.0270 |
| lrmsd | 66 | 1.1934 | 10.1958 |
| irmsd | 66 | 0.1030 | 3.0386 |
| plddt | 66 | 0.0044 | 0.0109 |
| iptm | 66 | 0.0086 | 0.0702 |
| top-pose CA RMSD (A) | 66 | 1.198 | 7.981 |

### PAIRED (qr - f0, same seed; mean over complexes [95 % bootstrap CI])

| metric | mean delta | 95 % CI | better side |
|---|---|---|---|
| dockq | -0.0043 | [-0.0152, +0.0069] | higher |
| lddt_ca | +0.0016 | [-0.0004, +0.0049] | higher |
| tm | +0.0135 | [-0.0022, +0.0346] | higher |
| lrmsd | +1.7714 | [+0.0954, +3.9203] | lower |
| irmsd | +0.3024 | [+0.0174, +0.7364] | lower |
| plddt | +0.0003 | [-0.0011, +0.0021] | higher |
| iptm | +0.0025 | [-0.0147, +0.0242] | higher |

Same-seed top-pose deviation qr vs f0: median 0.563 A, max 29.730 A (n=44); A/A seed floor median 1.198 A; kill bar 0.6 A.
Docking success (DockQ >= 0.23, top pose): f0 31/44 = 70.5 %, qr 31/44 = 70.5 %.
Per-complex mean pLDDT change: 28VJ +0.000, 9DBP -0.000, 9HL2 +0.001, 9LLG -0.000, 9LV4 +0.001, 9PCQ +0.000, 9TH6 -0.001, 9TY2 +0.000, 9W3L +0.007, 9W89 -0.005, 9W8A -0.001

### VERDICT wormhole_b0: FAIL: lrmsd worse, CI [+0.0954, +3.9203] excludes 0; irmsd worse, CI [+0.0174, +0.7364] excludes 0

SUMMARY: wormhole_b0 FAIL
