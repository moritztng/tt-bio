# The GPU catch-up rental

Raw per-cell reports from the rented boxes that filled the OpenBind-0, PXDesign and Nesso-1
GPU cells. One directory per GPU type. Each `gpu_<model>_prot512_<tag>.json` is one process:
1 cold fold discarded plus 3 warm, written by `scripts/gpu_vs_tt/gpu5_bench.py`; `gate_*` is
that cell's accuracy verdict on its own last warm structure.

**Every box runs an OpenFold3 control arm.** OpenBind-0 is the OpenFold3 runner on upstream
0.5.0 with a different checkpoint, its row is read beside the OpenFold3 row, and the published
OpenFold3 cell for that GPU is a number this box has to reproduce before its OpenBind number
means anything. Arms alternate order (`openfold3 openbind`, then `openbind openfold3`) so a
run-order effect shows up instead of hiding inside one arm.
