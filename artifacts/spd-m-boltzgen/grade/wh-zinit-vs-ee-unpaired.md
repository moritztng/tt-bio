# WH device z_init (10-28 runs) vs TT_BIO_LEVERS=none, unpaired seed-cluster bootstrap

Base wh-ee0..7 (main e11e990dc, none). Test wh-z0..7 = 10-28 WH normal default set + device z_init (tree tb62b97915, .103); wh-y0,1,3 = same, fast mode. Read 2026-11-04 with perf/spd/bg_unpaired.py --expect 8. The stack is not today's shipped set; it is re-graded as a stack in gradewkz/gradewkzf (a3e477e10).

## design_iptm
```
base grade/wh-ee[0-7]: 8 runs, per-run means [0.163, 0.2, 0.185, 0.146, 0.161, 0.152, 0.156, 0.223]
grade/wh-z[0-7]: 8 runs, 64 designs, design_iptm +0.0222 [-0.0074, +0.0502], per-run [0.133, 0.209, 0.246, 0.214, 0.216, 0.182, 0.201, 0.161]
grade/wh-y[0-7]: 3 runs, 24 designs, design_iptm -0.0208 [-0.0445, +0.0018], per-run [0.135, 0.168, 0.154]
```
## design_ptm
```
base grade/wh-ee[0-7]: 8 runs, per-run means [0.757, 0.783, 0.717, 0.745, 0.697, 0.759, 0.747, 0.727]
grade/wh-z[0-7]: 8 runs, 64 designs, design_ptm +0.0193 [+0.0010, +0.0384], per-run [0.767, 0.767, 0.741, 0.757, 0.758, 0.762, 0.777, 0.757]
grade/wh-y[0-7]: 3 runs, 24 designs, design_ptm +0.0181 [-0.0008, +0.0372], per-run [0.763, 0.765, 0.75]
```
## complex_plddt
```
base grade/wh-ee[0-7]: 8 runs, per-run means [0.635, 0.637, 0.625, 0.638, 0.63, 0.641, 0.647, 0.631]
grade/wh-z[0-7]: 8 runs, 64 designs, complex_plddt +0.0016 [-0.0040, +0.0068], per-run [0.641, 0.642, 0.629, 0.638, 0.633, 0.639, 0.633, 0.641]
grade/wh-y[0-7]: 3 runs, 24 designs, complex_plddt +0.0076 [+0.0030, +0.0121], per-run [0.642, 0.644, 0.643]
```
