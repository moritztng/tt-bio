# Boltz-2: what 200 sampling steps and 3 recycles are worth

Accuracy: panel of 10 targets x 13 settings, one Wormhole chip each (`sweep.py`, `results/*.json`). ARCH: WH.
Seconds: paired interleaved A/B, 5 reps x 6 arms in ONE process on card 7 of j10glx02, cdk2x2_512, ttnn 0.68.0, grid [8, 9]. ARCH: WH.
A/A floor on the incumbent arm: 19.42 % of its own median over all reps, 2.78 % after dropping its slowest rep. A ratio inside that has not been measured.

## Accuracy: CA lDDT against this target's own (200,3,seed 0) fold

```
        target   FLOOR             kind    200/3    150/3    100/3     75/3     50/3     25/3    200/2    100/2     50/2    200/1    100/1     50/1    200/0
 affinity_dhfr   99.91   protein+ligand  100.00    99.96    99.92    99.98    99.98    99.96   100.00    99.91    99.99   100.00    99.86!   99.92   100.00 
  affinity_fkg   99.59   protein+ligand  100.00    99.98    99.99    99.98   100.00   100.00   100.00    99.98   100.00   100.00   100.00   100.00   100.00 
 affinity_tryp  100.00   protein+ligand  100.00   100.00   100.00   100.00   100.00   100.00   100.00   100.00   100.00   100.00   100.00   100.00   100.00 
    cdk2x2_128   72.74          monomer  100.00    77.66    70.07!   69.80!   74.20    76.98    78.83    70.27!   76.53    71.40!   69.96!   73.38    76.26 
    cdk2x2_298   96.91          monomer  100.00    98.34    98.20    97.39    96.99    98.33    99.99    98.19    97.00    99.76    98.08    96.51!   98.53 
    cdk2x2_512   87.92  monomer-chimera  100.00    89.75    89.81    89.82    88.43    87.97    99.36    90.06    89.36    92.03    89.16    88.60    88.27 
           hsa   96.51          monomer  100.00    96.71    97.88    97.61    97.01    97.46    97.73    96.18!   96.79    94.28!   95.33!   95.47!   94.38!
      multimer   45.55          complex  100.00    47.06    49.47    48.67    47.38    49.60    45.06!   51.42    48.49    50.89    44.37!   46.28    45.05!
          prot   91.42          monomer  100.00    96.62    96.73    91.99    89.30!   94.70    99.67    94.00    87.49!   98.45    96.60    88.19!   86.69!
           ubq   98.63          monomer  100.00    98.99    98.65    99.73    98.90    99.19   100.00    98.68    98.89    99.76    98.70    98.70    98.46!
    misses /10                                -        0        1        1        1        0        1        2        1        2        4        3        4 
```
`!` = outside this target's own seed floor: worse than re-running production unchanged with a different seed.

## Accuracy: all-atom RMSD (Angstrom) against the same reference

```
        target   FLOOR             kind    200/3    150/3    100/3     75/3     50/3     25/3    200/2    100/2     50/2    200/1    100/1     50/1    200/0
 affinity_dhfr    0.67   protein+ligand    0.00     0.53     0.58     0.56     0.43     0.55     0.21     0.57     0.42     0.17     0.60     0.45     0.32 
  affinity_fkg    0.78   protein+ligand    0.00     0.57     0.51     0.49     0.56     0.59     0.07     0.54     0.53     0.17     0.56     0.52     0.29 
 affinity_tryp    0.39   protein+ligand    0.00     0.41!    0.38     0.39     0.34     0.38     0.05     0.39     0.35     0.05     0.39     0.35     0.17 
    cdk2x2_128    6.89          monomer    0.00     4.65     5.49     8.01!    4.36     4.47     3.02     5.39     4.27     6.08     5.24     4.65     5.34 
    cdk2x2_298    1.25          monomer    0.00     0.88     0.92     1.48!    1.09     0.69     0.27     0.95     1.01     0.56     0.99     1.12     0.65 
    cdk2x2_512   17.30  monomer-chimera    0.00    17.14    13.86    13.03    13.99    14.10     0.50    13.95    14.08     4.59    13.71    13.62    20.95!
           hsa    1.54          monomer    0.00     1.15     1.09     0.96     1.22     1.12     0.95     1.25     1.23     1.40     1.25     1.38     1.40 
      multimer   15.80          complex    0.00    11.80    12.04    10.02    12.28    10.03    10.79     9.45    11.00     9.68    14.02    12.48    13.52 
          prot    1.46          monomer    0.00     0.90     1.05     1.37     1.82!    1.02     0.44     1.19     1.94!    0.61     1.07     1.85!    2.07!
           ubq    1.23          monomer    0.00     1.11     1.52!    1.12     1.24!    1.06     0.36     1.50!    1.23     0.37     1.49!    1.24!    1.23!
    misses /10                                -        1        1        2        2        0        0        1        1        0        1        2        3 
```
Secondary. All-atom RMSD is superposition-based, so a hinge rotation saturates it while every domain stays identical; lDDT above is the primary read.

## The knee: below 25 steps

Separate run, 5 targets, its OWN reference and its own seed floor re-measured in the same process, so these columns are read against the same yardstick the panel used and not against the panel's numbers.

```
        target   FLOOR             kind    200/3     20/3     15/3     10/3      5/3
 affinity_dhfr   99.91   protein+ligand  100.00   100.00    99.99    32.74!    0.00!
  affinity_fkg   99.59   protein+ligand  100.00   100.00   100.00    35.38!    0.00!
 affinity_tryp  100.00   protein+ligand  100.00   100.00   100.00    29.35!    0.00!
    cdk2x2_298   96.91          monomer  100.00    98.41    98.51    33.16!    0.00!
          prot   91.42          monomer  100.00    94.01    97.27    30.85!    0.00!
     misses /5                                -        0        0        5        5 
```
```
        target   FLOOR             kind    200/3     20/3     15/3     10/3      5/3
 affinity_dhfr    0.67   protein+ligand    0.00     0.48     0.57     8.89!  566.39!
  affinity_fkg    0.78   protein+ligand    0.00     0.59     0.60     8.61!  553.54!
 affinity_tryp    0.39   protein+ligand    0.00     0.36     0.40!    8.29!  567.41!
    cdk2x2_298    1.25          monomer    0.00     0.83     0.73     9.03!  551.80!
          prot    1.46          monomer    0.00     1.07     0.85     9.60!  562.61!
     misses /5                                -        0        1        5        5 
```

## Seconds: measured, paired, same chip

```
      arm   n  median_s   ratio  removed_s  trim_ratio        load 1m   reps
    200/3   5    41.960  1.0000      0.000      1.0000          12-58   42.43 41.96 41.27 41.81 49.42
    100/3   5    36.903  1.1370      5.057      1.1376         13-161   36.90 36.70 37.84 113.88 36.73
     50/3   5    34.756  1.2073      7.204      1.2080         12-194   34.76 34.59 35.97 111.48 34.37
     25/3   5    33.554  1.2505      8.406      1.2496         11-167   33.55 33.48 33.63 52.90 33.33
    200/2   5    34.776  1.2066      7.184      1.2057          10-96   34.78 34.70 34.53 43.22 34.91
     50/2   5    28.338  1.4807     13.622      1.4838         10-159   28.11 28.34 29.32 74.02 28.01
```
`ratio` is the median over every rep, `trim_ratio` the median after each arm's slowest rep is dropped. They agree to well under the A/A floor, so the load step the `load 1m` column shows did not move the answer.

## The frontier

```
      arm  median_s   ratio  misses/10  lDDT median  worst RMSD_AA
    200/3    41.960  1.0000          -       100.00          0.000
    100/3    36.903  1.1370          1        98.04         13.861
     50/3    34.756  1.2073          1        97.00         13.994
     25/3    33.554  1.2505          0        97.90         14.097
    200/2    34.776  1.2066          1        99.83         10.788
     50/2    28.338  1.4807          1        96.90         14.082
```

