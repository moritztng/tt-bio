# grade: qr vs f0, fast mode

Records: 88 reps from 22 run dirs; failed/non-finite folds: 0

## wormhole_b0

Complexes 11; paired folds 44; paired seeds per complex: 28VJ 4, 9DBP 4, 9HL2 4, 9LLG 4, 9LV4 4, 9PCQ 4, 9TH6 4, 9TY2 4, 9W3L 4, 9W89 4, 9W8A 4

### FLOOR (A/A, f0 seed vs seed, median / mean of |diff|)

| metric | n pairs | median | mean |
|---|---|---|---|
| dockq | 66 | 0.0063 | 0.0104 |
| lddt_ca | 66 | 0.0020 | 0.0051 |
| tm | 66 | 0.0027 | 0.0111 |
| lrmsd | 66 | 0.3887 | 8.9426 |
| irmsd | 66 | 0.0422 | 2.4027 |
| plddt | 66 | 0.0014 | 0.0072 |
| iptm | 66 | 0.0055 | 0.0323 |
| top-pose CA RMSD (A) | 66 | 1.258 | 9.530 |

### PAIRED (qr - f0, same seed; mean over complexes [95 % bootstrap CI])

| metric | mean delta | 95 % CI | better side |
|---|---|---|---|
| dockq | -0.0003 | [-0.0025, +0.0020] | higher |
| lddt_ca | -0.0007 | [-0.0031, +0.0011] | higher |
| tm | +0.0025 | [-0.0007, +0.0066] | higher |
| lrmsd | -2.2585 | [-5.0426, -0.1671] | lower |
| irmsd | +0.3553 | [-0.0597, +0.8238] | lower |
| plddt | -0.0028 | [-0.0083, +0.0012] | higher |
| iptm | +0.0079 | [-0.0243, +0.0374] | higher |

Same-seed top-pose deviation qr vs f0: median 0.794 A, max 34.667 A (n=44); A/A seed floor median 1.258 A; kill bar 0.6 A.
Docking success (DockQ >= 0.23, top pose): f0 24/44 = 54.5 %, qr 24/44 = 54.5 %.
Per-complex mean pLDDT change: 28VJ -0.001, 9DBP +0.000, 9HL2 +0.000, 9LLG -0.000, 9LV4 +0.002, 9PCQ +0.001, 9TH6 -0.001, 9TY2 -0.000, 9W3L +0.005, 9W89 -0.026, 9W8A -0.011

### VERDICT wormhole_b0: PASS

SUMMARY: wormhole_b0 PASS
