# Clock evidence audit

Run this CPU-only audit from the repository root:

```sh
python3 perf/c10_orchestrator/audit_clock.py
python3 -m unittest discover -s perf/c10_orchestrator -p 'test_*.py'
```

It validates the recorded clock aggregates and holder lists for eight folds per
arm, reproduces the historical two-endpoint fit and its sensitivity to using all
six folds, and calculates conditional work targets. It imports no device code.
Use `--out PATH` to save JSON; `--target-mhz` and `--target-s` change the projection.

`pinned_baseline.json` is an unchanged copy of
`perf/b2z2_aiclk_pin/out/mainab_qb2c1.json` from commit
`463da8e9bd326be091a008da7f886c3ecf5ece20` on
`origin/wk/b2z2-aiclk-default-decision`. That experiment measured commit
`0df13ad983e645722da958e176d22d96fa5067ee`. It records a 14.554 s median at
sampled minimum, mean and maximum 1350 MHz for the forced arm. This is historical
branch evidence, not a measurement of today's tree.

The historical fit uses the existing
`perf/b2z2_cell_recheck/out/cell_main_qb2c1_clock.json`. Its clocks vary during
folds, and three timed folds have foreign device holders. Neither artifact
retains raw sample timestamps, so continuous clock coverage and time-weighted
integration cannot be reconstructed. The script checks the aggregates actually
present; it cannot certify the original instrument.

The published `2.901 + 15355 / MHz` model does not identify removable host time
or a per-op cycle budget. Its work coefficient has units of MHz·seconds, or
Mcycles dimensionally. Reported target work and reductions are conditional
model calculations, not hardware cycle counts. `--baseline` and `--historical`
allow checking copies of these records, including intentionally invalid inputs.

The graph-traffic controls run without a device:

```sh
python3 -m pytest -q tests/test_perf_traffic.py
```

Both traffic counters share the terminal-output rule in
`perf/b2x_difflayer/itemize.py`. The saved 8192-cube bf16 matmul graph must count
402,653,184 bytes: two input reads and one output write. Metadata aliases add no
read; a consuming operation does. Internal scratch estimates remain separate in
`assumed_read_MB`, including opaque buffers in `opaque_buffer_assumed_read_MB`.
`traffic_model_version=2` distinguishes these counts from historical artifacts.
For fold analysis, use `perf/roof_arb/corrected_traffic.py` to retain its L1-output,
preallocated-output and partial-read corrections. A passing matmul control does
not certify unobserved kernel rereads, spills or whole-fold traffic.
