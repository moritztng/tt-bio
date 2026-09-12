# WAIT-HISTOGRAM — one settled 512 aa PairformerLayer, WH, whglx, 72 cores, 85.2936 ms span

Physical NOC coordinates. Row y=6 is the ethernet row and carries no worker core.

## Input-tile wait per core, ms of the block (CB-COMPUTE-WAIT-FRONT)

Flat. Gini 0.0128, max/median 1.019, top decile 9.95 % against a uniform 9.72 %.

```
             x=1     x=2     x=3     x=4     x=6     x=7     x=8     x=9
y=1        46.98   47.47   47.18   46.87   47.03   47.05   46.82   46.93
y=2        47.87   47.83   47.72   47.71   47.42   47.64   47.66   47.22
y=3        48.13   48.12   47.61   48.01   47.65   47.57   47.76   47.40
y=4        47.70   48.29   47.92   47.60   47.81   47.88   47.50   47.58
y=5        47.58   47.53   47.68   47.69   47.34   47.62   47.64   46.99
y=7        47.16   47.56   47.17   47.25   47.16   47.25   47.22   46.80
y=8        47.50   47.64   47.41   47.41   47.29   47.41   47.23   46.99
y=9        47.16   47.11   47.35   47.27   47.25   47.33   47.16   46.85
y=10       43.21   43.04   43.04   43.05   42.78   42.75   42.79   42.53
```

## Output-room wait per core, ms (CB-COMPUTE-RESERVE-BACK)

Not flat: a clean monotonic gradient down y, 11.05 ms at the top row to 5.72 at the bottom.
Spearman against y = -0.866. This is the only whole-block term that knows where a core sits.

```
             x=1     x=2     x=3     x=4     x=6     x=7     x=8     x=9
y=1         7.12   11.05   10.41    9.86    9.69    9.74    8.99    8.44
y=2        10.81   10.80   10.48    9.99    9.81    9.70    9.31    8.49
y=3        10.24   10.23    9.83    9.35    9.26    9.19    8.83    8.02
y=4         9.58    9.56    9.27    8.89    8.91    8.83    8.42    7.51
y=5         8.68    8.77    8.50    8.07    8.20    8.10    7.81    6.87
y=7         8.20    8.25    7.98    7.56    7.69    7.57    7.23    6.42
y=8         7.74    7.80    7.58    7.16    7.30    7.36    7.00    6.39
y=9         7.27    7.29    7.15    6.85    6.97    7.05    6.93    6.39
y=10        6.13    6.19    6.19    6.26    6.15    6.28    6.29    5.72
```

## Operand-chain wait per core, ms (B2Z2-IN0-CHAINWAIT, the instrumented log)

y=1 is the injector row and waits on nobody. Every other row waits 1.1-2.0 ms, and the last
hop pays 4.8. Spearman against y = 0.871, max/median 3.222.

```
             x=1     x=2     x=3     x=4     x=6     x=7     x=8     x=9
y=1         0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00
y=2         1.20    1.20    1.18    1.15    1.17    1.20    1.17    1.21
y=3         1.32    1.41    1.41    1.39    1.30    1.29    1.24    1.28
y=4         1.20    1.38    1.37    1.35    1.25    1.28    1.32    1.25
y=5         1.33    1.11    1.15    1.15    1.13    1.22    1.39    1.37
y=7         1.72    1.83    1.82    1.82    1.81    1.80    1.78    1.79
y=8         1.63    1.77    1.77    1.74    1.67    1.65    1.63    1.81
y=9         1.92    1.99    2.00    2.00    2.00    1.99    1.95    1.94
y=10        4.76    4.89    4.88    4.87    4.85    4.84    4.81    4.78
```

## Who finishes last

Times a core was the last of its op to finish, over 272 ops. Uniform would be 3.8, so the
straggler IS concentrated (Gini 0.5709) - it just does not cost much: 3.65 % of the grid.

```
  2,1:47  2,2:27  6,4:16  1,2:15  1,4:13  7,10:12  1,5:11  7,4:10  7,9:10  3,1:8
```

## The three bounds this map puts on reordering

| lever, assumed perfect and free | core-ms of 6141.1 | block |
|---|---|---|
| REBALANCE: every core waits as little as the luckiest core already does | 495.5 | **1.0878x** |
| OVERLAP: every core idle between kernels is given useful work | 401.5 | **1.0700x** |
| DECHAIN: multicast the operand instead of forwarding it core to core | 113.0 | **1.0187x** |
| all three, added, ignoring that they overlap | 1010.0 | **1.1968x** |
