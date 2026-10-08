Complexes 11; ttexact seeds 101,102,103; exact seeds 101,102,103,104,105; paired seeds 101,102,103.

## Top-ranked pose vs ground truth (mean over seeds; ttexact min-max in brackets)

| PDB | DockQ ttexact | DockQ exact | iRMSD ttexact | iRMSD exact | LRMSD ttexact | LRMSD exact | TM ttexact | TM exact | ipTM ttexact | ipTM exact |
|---|---|---|---|---|---|---|---|---|---|---|
| 28VJ | 0.669 [0.65-0.69] | 0.416 | 1.545 [1.49-1.61] | 9.275 | 3.762 [3.75-3.77] | 23.865 | 0.971 [0.97-0.97] | 0.831 | 0.759 [0.76-0.76] | 0.692 |
| 9DBP | 0.921 [0.92-0.92] | 0.924 | 0.640 [0.61-0.67] | 0.613 | 1.468 [1.23-1.67] | 1.532 | 0.991 [0.99-0.99] | 0.991 | 0.919 [0.92-0.92] | 0.919 |
| 9HL2 | 0.777 [0.77-0.78] | 0.790 | 1.495 [1.49-1.50] | 1.434 | 2.128 [2.01-2.26] | 2.222 | 0.909 [0.91-0.91] | 0.906 | 0.851 [0.85-0.85] | 0.851 |
| 9LLG | 0.868 [0.86-0.87] | 0.874 | 0.741 [0.73-0.75] | 0.742 | 1.613 [1.46-1.86] | 1.722 | 0.962 [0.95-0.97] | 0.963 | 0.849 [0.85-0.85] | 0.846 |
| 9LV4 | 0.792 [0.79-0.80] | 0.789 | 0.993 [0.99-1.00] | 1.001 | 1.635 [1.46-1.79] | 2.030 | 0.969 [0.97-0.97] | 0.968 | 0.924 [0.92-0.92] | 0.929 |
| 9PCQ | 0.369 [0.36-0.38] | 0.353 | 3.308 [3.24-3.37] | 3.792 | 19.345 [17.20-21.42] | 21.337 | 0.886 [0.88-0.89] | 0.885 | 0.708 [0.71-0.71] | 0.740 |
| 9TH6 | 0.006 [0.01-0.01] | 0.164 | 18.884 [18.79-18.97] | 17.409 | 75.175 [74.99-75.43] | 62.272 | 0.691 [0.69-0.69] | 0.728 | 0.261 [0.26-0.26] | 0.538 |
| 9TY2 | 0.748 [0.70-0.78] | 0.761 | 0.424 [0.31-0.64] | 0.318 | 10.168 [9.43-10.56] | 10.276 | 0.800 [0.78-0.82] | 0.801 | 0.934 [0.93-0.93] | 0.934 |
| 9W3L | 0.767 [0.75-0.78] | 0.610 | 0.949 [0.89-1.04] | 2.624 | 4.322 [3.71-4.85] | 12.786 | 0.952 [0.94-0.96] | 0.891 | 0.828 [0.83-0.83] | 0.744 |
| 9W89 | 0.043 [0.01-0.07] | 0.032 | 14.327 [9.57-23.12] | 13.861 | 41.729 [21.88-77.70] | 41.604 | 0.853 [0.83-0.87] | 0.843 | 0.253 [0.24-0.26] | 0.314 |
| 9W8A | 0.014 [0.01-0.01] | 0.531 | 13.165 [13.11-13.24] | 5.676 | 49.362 [49.23-49.58] | 21.718 | 0.827 [0.83-0.83] | 0.925 | 0.823 [0.82-0.82] | 0.881 |

## Success rate of the top-ranked pose (DockQ >= 0.23 / >= 0.49 / >= 0.80), all complex x seed runs

* ttexact: 72.7% acceptable, 63.6% medium, 21.2% high (n=33); mean DockQ 0.543
* exact: 74.5% acceptable, 65.5% medium, 25.5% high (n=55); mean DockQ 0.568

## Acceptable top-ranked pose per complex (seeds with DockQ >= 0.23) and discordance against the floor

| PDB | ttexact | exact | ttexact vs ttexact discordant pairs | exact vs ttexact same seed discordant |
|---|---|---|---|---|
| 28VJ | 3/3 | 3/5 | 0/3 | 2/3 |
| 9DBP | 3/3 | 5/5 | 0/3 | 0/3 |
| 9HL2 | 3/3 | 5/5 | 0/3 | 0/3 |
| 9LLG | 3/3 | 5/5 | 0/3 | 0/3 |
| 9LV4 | 3/3 | 5/5 | 0/3 | 0/3 |
| 9PCQ | 3/3 | 5/5 | 0/3 | 0/3 |
| 9TH6 | 0/3 | 1/5 | 0/3 | 1/3 |
| 9TY2 | 3/3 | 5/5 | 0/3 | 0/3 |
| 9W3L | 3/3 | 4/5 | 0/3 | 1/3 |
| 9W89 | 0/3 | 0/5 | 0/3 | 0/3 |
| 9W8A | 0/3 | 3/5 | 0/3 | 2/3 |

Discordance: ttexact vs ttexact 0/33 = 0.0%, exact vs ttexact same seed 6/33 = 18.2%. Same-seed flips: ttexact ok -> exact not 3, exact ok -> ttexact not 3; exact McNemar p = 1.00.

## FLOOR vs FAST, ground-truth metrics of the top-ranked pose (absolute difference, median / mean over complex x pair)

| metric | FLOOR ttexact(s) vs ttexact(s') | FAST exact(s) vs ttexact(s) | paired mean exact - ttexact [95 % CI] |
|---|---|---|---|
| dockq | 0.012 / 0.017 | 0.014 / 0.150 | +0.0137 [-0.1239, +0.1595] |
| irmsd | 0.060 / 0.890 | 0.179 / 3.453 | +0.1815 [-2.4921, +3.3268] |
| lrmsd | 0.332 / 3.912 | 1.023 / 11.024 | -0.7292 [-9.9215, +8.9676] |
| tm_complex | 0.003 / 0.009 | 0.005 / 0.051 | -0.0169 [-0.0711, +0.0278] |
| tm_binder | 0.003 / 0.004 | 0.004 / 0.009 | -0.0018 [-0.0088, +0.0051] |
| plddt | 0.023 / 0.060 | 0.493 / 0.863 | +0.2458 [-0.3568, +0.8333] |
| iptm | 0.000 / 0.002 | 0.009 / 0.080 | +0.0213 [-0.0460, +0.1147] |
| ranking_score | 0.000 / 0.002 | 0.010 / 0.067 | +0.0182 [-0.0388, +0.0954] |

## Pose-to-pose deviation of top-ranked models (median / max per complex)

| PDB | ttexact vs ttexact: DockQ | binder RMSD A | TM | exact vs ttexact same seed: DockQ | binder RMSD A | TM |
|---|---|---|---|---|---|---|
| 28VJ | 0.835 / min 0.822 | 1.40 / 1.55 | 0.992 | 0.034 / min 0.020 | 53.64 / 53.89 | 0.636 |
| 9DBP | 0.923 / min 0.920 | 1.60 / 1.60 | 0.993 | 0.966 / min 0.939 | 1.21 / 1.41 | 0.995 |
| 9HL2 | 0.924 / min 0.924 | 1.03 / 1.06 | 0.992 | 0.945 / min 0.883 | 1.14 / 2.19 | 0.986 |
| 9LLG | 0.985 / min 0.982 | 0.49 / 0.49 | 0.995 | 0.980 / min 0.849 | 0.31 / 0.65 | 0.994 |
| 9LV4 | 0.968 / min 0.961 | 0.93 / 1.05 | 0.998 | 0.976 / min 0.969 | 0.74 / 1.21 | 0.998 |
| 9PCQ | 0.620 / min 0.587 | 6.68 / 7.65 | 0.952 | 0.477 / min 0.460 | 15.21 / 17.53 | 0.916 |
| 9TH6 | 0.771 / min 0.616 | 3.38 / 4.94 | 0.959 | 0.008 / min 0.006 | 60.55 / 75.80 | 0.768 |
| 9TY2 | 0.946 / min 0.936 | 1.27 / 1.77 | 0.994 | 0.966 / min 0.961 | 1.32 / 1.35 | 0.996 |
| 9W3L | 0.869 / min 0.810 | 2.98 / 3.73 | 0.987 | 0.883 / min 0.023 | 4.08 / 42.23 | 0.978 |
| 9W89 | 0.006 / min 0.006 | 71.09 / 72.13 | 0.837 | 0.459 / min 0.011 | 8.28 / 53.32 | 0.940 |
| 9W8A | 0.631 / min 0.628 | 2.06 / 2.92 | 0.993 | 0.012 / min 0.012 | 49.18 / 49.58 | 0.830 |

Pooled binder RMSD between top poses: ttexact vs ttexact median 1.60 A (n=33), exact vs ttexact same seed median 2.17 A (n=33).

## Flags

* 28VJ: exact DockQ 0.02,0.02,0.63,0.70,0.70 outside ttexact range 0.65-0.69; exact-vs-ttexact binder deviation 53.89 A > largest ttexact-vs-ttexact 1.55 A
* 9HL2: exact-vs-ttexact binder deviation 2.19 A > largest ttexact-vs-ttexact 1.06 A
* 9LLG: exact-vs-ttexact binder deviation 0.65 A > largest ttexact-vs-ttexact 0.49 A
* 9LV4: exact-vs-ttexact binder deviation 1.21 A > largest ttexact-vs-ttexact 1.05 A
* 9PCQ: exact-vs-ttexact binder deviation 17.53 A > largest ttexact-vs-ttexact 7.65 A
* 9TH6: exact DockQ 0.80,0.01,0.00,0.01,0.01 outside ttexact range 0.01-0.01; exact-vs-ttexact binder deviation 75.80 A > largest ttexact-vs-ttexact 4.94 A
* 9W3L: exact DockQ 0.74,0.03,0.78,0.74,0.75 outside ttexact range 0.75-0.78; exact-vs-ttexact binder deviation 42.23 A > largest ttexact-vs-ttexact 3.73 A
* 9W8A: exact DockQ 0.86,0.89,0.01,0.01,0.87 outside ttexact range 0.01-0.01; exact-vs-ttexact binder deviation 49.58 A > largest ttexact-vs-ttexact 2.92 A
