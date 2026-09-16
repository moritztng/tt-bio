# Bounded census host export

The isolated census harness now exports only operation metadata, with resource limits and failure checks, instead of generating a timing CSV and running Tracy's Python timing postprocessor.

[CPU results](RESULTS.md) reproduce all metadata bytes from three archived traces. The existing reducer also reproduces all 13 retained census windows, containing 1,514 programs. This completes the host-export prerequisite only. It establishes neither a full-fold cycle budget nor the campaign's 10-second goal. No production, model, kernel, dependency or default change was made. Model cycles saved: zero.

## What changes

`bounded_tracy.py` checks the inspected CLI and exporter hashes, then replaces the outer CLI's `generate_report` callback. The same installed CLI still captures the same model with the same profiler settings. Its inner `python3 -m tracy` command is unchanged except for `--check-exit-code`, which prevents reporting a failed model execution as a successful capture. The sum profiler remains enabled; there is no dependency on its disabled C++ summary.

`export_metadata.py` runs the existing tool with the exact message-export arguments used by Tracy:

```text
csvexport-release -m -s ; TRACE
```

The argument vector passes `;` literally, without a shell. The tool writes the complete message stream, including full/cache records and repeated calls. Nothing filters, renumbers or deduplicates it. The existing `reduce_census.host_ops` decoder validates the result in a bounded child process. Missing cache entries, malformed records, duplicate global call IDs and empty metadata fail. A successful native exporter exit and validation are both required before `tracy_ops_data.csv` appears; `report.json` must say `GO`.

The reducer's native counter mapping, raw operation IDs, source identities and graph/observer shape joins are unchanged. Cached shape labels remain stale where the original capture made them stale; they are never promoted to actual operand shapes. Both the original failure and its source hashes remain preserved at commit `6eb4dd6fa5eceab34f72a5e7ad382b8607554390`. The historical harnesses and manifest are also copied under `source/`.

## Bounds and failure behavior

| Resource | Default bound |
| --- | ---: |
| Trace admitted to export | 4 GiB |
| Each CPU child address space, including mappings and threads | 8 GiB |
| Each output file written by a child | 64 MiB |
| Each child CPU time | 300 seconds |
| Each child elapsed time | 600 seconds |
| Core dumps | disabled |

Export and validation run sequentially, so these memory limits do not add together. The supervisor hashes files in 1 MiB chunks. At most four child output files can coexist (metadata, exporter stderr, validation JSON, validation stderr), giving a 256 MiB scratch-file bound plus the small supervisor report. The child has no general directory access restriction; this disk bound follows the reviewed exporter/validator's output behavior, not a filesystem quota. Address-space limits are deliberately stronger than a sampled RSS alarm.

The file-size signal is reset before executing the exporter: Python otherwise ignores `SIGXFSZ`, which could let C stdio failures leave a successful exit with a valid-looking CSV prefix. Reaching the exact output cap is also rejected. Nonzero exits, timeout, malformed metadata and changed input fail with `STOP`, preserve diagnostics, delete the partial CSV and leave the input trace untouched. Existing output directories are refused, preserving previous runs. Supervisor interruption kills and reaps its own CPU child. No foreign process is signalled.

`csvexport` constructs `tracy::Worker(FileRead&)` with all events before it selects messages. **It still loads the entire trace.** The 2,919,007,827-byte census trace was not copied or tested. Small-trace success does not prove that trace fits 8 GiB, or any other chosen budget. An oversized trace is refused before starting the exporter; an expansion beyond the memory budget fails explicitly. Limits can be specified with `--trace-bytes`, `--output-bytes`, `--address-bytes`, `--cpu-seconds` and `--wall-seconds`; do not enlarge them on the basis of these small controls.

These are post-capture CPU limits. They do not bound model memory, raw-window capture, or Tracy recording itself. Full host traces are optional retained evidence after successful metadata extraction, not required reducer inputs. The report records their absolute paths, sizes and SHA256 without copying or deleting them. The device owner remains responsible for recording storage and for any explicit archival or deletion decision.

## CPU reproduction

Run from this checkout with Python 3.10 or newer. These commands do not import a TT runtime or open a device. Use a fresh `--out` each time.

```bash
python3 perf/c10_export_budget/verify_bundle.py
python3 -m unittest discover -s perf/c10_export_budget -p 'test_*.py' -v
python3 -m unittest discover -s perf/c10_burst_census -p 'test_*.py' -v
python3 perf/c10_export_budget/replay_controls.py \
  --csvexport perf/c10_export_budget/tools/csvexport-release \
  --runtime-dir perf/c10_export_budget/tools/runtime \
  --out perf/c10_export_budget/replay/new-control
```

The exporter binary is an existing build, not rebuilt here. `source_contract.json` pins its SHA256 and the inspected source files. `results/binary_dependencies.txt` and `results/runtime_dependencies.txt` show only CPU libraries; there is no TT runtime dependency. The file-based Worker constructor reads a trace rather than connecting to a device. `source/` retains the CLI, exporter, Worker and metadata decoder sources used for the audit.

On pc the qb2 binary needs newer glibc and libstdc++. The measurements use private copies of its loader and six CPU libraries, without changing system libraries. To reproduce that setup while the recorded build remains available:

```bash
mkdir -p perf/c10_export_budget/tools/runtime
scp qb2:/home/ttuser/tt-metal-k10/build/tools/profiler/bin/csvexport-release \
  perf/c10_export_budget/tools/
scp qb2:/lib64/ld-linux-x86-64.so.2 perf/c10_export_budget/tools/runtime/
scp qb2:/lib/x86_64-linux-gnu/libcapstone.so.4 \
  qb2:/lib/x86_64-linux-gnu/libtbb.so.12 \
  qb2:/lib/x86_64-linux-gnu/libstdc++.so.6 \
  qb2:/lib/x86_64-linux-gnu/libm.so.6 \
  qb2:/lib/x86_64-linux-gnu/libgcc_s.so.1 \
  qb2:/lib/x86_64-linux-gnu/libc.so.6 perf/c10_export_budget/tools/runtime/
```

These are read-only copies from the shared build and system library paths, never from a live worker's worktree. Library hashes are in every measured export record. If running on a compatible host, omit `--runtime-dir`. A changed binary is refused; revalidate source and controls before updating the contract. The copied executables and libraries are not committed.

The small inputs are the committed `c10_dm_control/raw` and `c10_burst_census/runs/smoke{1,2}/raw` traces and metadata archives. Full-census validation uses only its committed metadata and retained windows. The replay compares every metadata byte, ordered decoded record and complete window result. It also repackages the full archived metadata and clock under the limits, then feeds the new metadata archive to the window reducer. It reuses archived clock validity for the window comparison; it is not a new clock audit or hardware measurement.

## Next authorized device owner

Parent review and an explicit qb2 node-0 slot are required before invoking the census harness. Do not run these on pc. From the owner's checkout containing this change:

```bash
bash perf/c10_burst_census/run_census.sh NEW_CENSUS_NAME
```

The census retains normal benchlock, containment checks, node ownership, and 1350 MHz sampling during intervals. The census still uses 200 steps, three recycles, the fixed 35-row MSA, one sample and seed 0. No foreign signals, resets or model changes are authorized by this report. Separate bare/profiler evidence is required before any wall-performance claim.

After capture, bounded outputs are under `runs/NAME/tracy/metadata/`. Package them for the existing census reducer with:

```bash
python3 perf/c10_export_budget/archive_replay.py --run perf/c10_burst_census/runs/NEW_CENSUS_NAME
python3 perf/c10_burst_census/reduce_census.py --dir perf/c10_burst_census/runs/NEW_CENSUS_NAME
```

The packager requires a matching `GO` export, preserves all original files and writes `raw_manifest.json` last. It compresses only operation metadata and clock samples, with original/compressed SHA256. Compression uses the same CPU process limits, a 64 MiB metadata input cap and a 256 MiB clock input cap. Its two temporary archives each have a 64 MiB file limit; killed compression can leave bounded scratch files but cannot create a completion manifest. It refuses existing archives rather than overwriting them. The capture already retains its selected windows, graph/observer records, criterion and source records. The reducer independently verifies their original/compressed hashes and interval clocks.

Do not use the old DM `archive.py`, which expects the deliberately omitted timing CSV and `ops.csv`. The historical smoke harness/reducer is unchanged: it consumes the previously generated `ops.csv`. Adapting a new smoke capture is outside this census-only change; these CPU controls use its committed traces without running that harness.

Keep the full host trace where the export report records it if wanted; do not mistake its existence for device-cycle coverage. Raw data outside the original census's named windows was discarded with a ledger. Neither this exporter nor the retained host messages can reconstruct that missing full-fold cycle budget.
