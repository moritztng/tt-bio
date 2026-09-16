# Preserve cycles from bounded raw chunks

Reduce each closed profiler chunk synchronously into an immutable compressed JSONL
file. The directory is an append-only stream: one file per caller-supplied chunk
ID, with raw provenance, program endpoints, core/RISC sums and a completion footer.
This is a CPU prerequisite for a future capture, not a full-fold measurement.

Run from the repository root with Python's standard library:

```sh
python3 -m unittest discover -s perf/c10_cycle_stream -p 'test_*.py'
python3 perf/c10_cycle_stream/replay.py \
  --output-dir /tmp/c10-cycle-replay \
  --report /tmp/c10-cycle-replay.json
```

Use a fresh output directory for each reproduction. The replay checks all 13 census
windows, the calibrated raw/CSV known answers and seven published calibration
intervals. Its two additional reductions of the largest retained archive use the
same bytes and IDs; they are resource exercises, not new hardware observations.
See [replay_report.json](replay_report.json) for the measured CPU resources and
accepted comparisons. Existing evidence remains unchanged. On pc, the largest
retained raw chunk was 327,094,028 bytes; its three CPU reductions took
41.10–41.52 seconds and peaked at 469,040 KiB RSS. The 13 census windows produced
39,474,345 compressed output bytes from 478,607,197 raw bytes. These resource
measurements replay archives captured at sampled 1350 MHz; they are not fold
latencies. Synchronous reduction overhead still needs a device-capture assessment.

For the next device owner, call this after `ReadDeviceProfiler` has returned and
your existing drain has produced a closed, immutable raw chunk and byte/hash
receipt. Keep the raw file until this call succeeds and its output is backed up.
This helper neither opens a device nor changes the capture/drain policy:

```python
from perf.c10_cycle_stream.stream import append_chunk, verify_chunk

result = append_chunk(
    raw_chunk_path, cycle_output_directory,
    chunk_id="drain-000042",
    counter_scope="capture-UUID/boot-ID/process-ID/counter-epoch",
    expected_bytes=drain_raw_bytes,
    expected_sha256=drain_raw_sha256,
)
footer = verify_chunk(result["path"])
# Retain the raw chunk according to the capture's retention policy.
```

`expected_bytes` and `expected_sha256` describe the decompressed raw CSV, including
both header lines. Plain CSV and `.gz` inputs are supported. The CLI exposes the
same required arguments and resource limits; `--help` lists them. The synchronous
API runs a separate memory-limited CPU process and raises on any failure. A failed
reduction must stop the capture's discard path, not become a skipped chunk.

## Records and scope

Each `program` contains its actual raw `global_call_id`, `device`, `trace` and
`replay`, the accepted census `summary`, and every measured core/RISC's integer
start/end, residency, zone sums and unclassified remainder. Integer cycles never
pass through floating point. IDs are not assumed contiguous in native CSV order.
A reversible surrogate ID only adapts mixed identities to the existing calibrated
parser, which requires device 0 with no trace; it is never a published identity.

The full identity includes the caller's `counter_scope`. Use a new scope after a
process/counter reset; this tool cannot discover a reset from a CSV. Identical IDs
in different chunks remain separate observations, including exact duplicates.
Do not sum them as distinct executions. An endpoint pair split across chunks is
rejected; keep profiler drains at complete-program boundaries. This adapter does
not merge partial programs or coalesce repeated chunks.

`unreduced_marker` preserves every field of rows outside the calibrated kernel and
sum parser, including firmware markers and unknown RISCs. The footer counts every
marker type and labels unknown zones. Unknown zone totals on supported RISCs retain
their integer values without assigning a meaning to the zone. `unplaced_zone_sums`
preserves totals on RISCs with no kernel endpoints; the accepted reducer excludes
these from residency aggregates. Their residency remains unknown. A program with
only unsupported markers has a null summary, not zero cycles. Duplicate kernel or
sum markers fail; duplicate unsupported rows remain visible and counted.

The `complete` footer records decompressed and stored-input SHA256/bytes, row count,
per-device/trace/replay call-ID and tick ranges, parser source hashes, marker
coverage and payload digest. Min/max ranges do not assert continuity. The caller's
drain ledger must prove every chunk was presented and supply any expected counter
sets; a valid CSV missing whole programs cannot reveal those missing programs.
`verify_chunk` validates the entire gzip stream, record count and payload digest.
Consumers must verify a chunk before trusting its records.

Program extrema and per-core residency use the raw device counters. Zone sums are
accumulators, not elapsed time; their remainder is not issuing time. The legacy
aggregate subtracts all zone totals, while this adapter additionally refuses any
core/RISC whose totals exceed its own residency. Totals alone cannot prove zones
are disjoint. Do not add overlapping program spans as elapsed time, label gaps CPU
time, convert spans to fold seconds or infer synchronization/profiler perturbation.
The 1350 MHz CSV header is checked for compatibility with the calibration; it is
not telemetry. A future measurement still needs clock samples during its interval.
No shapes, FLOPs or measured-operation counts are inferred from host metadata.

## Resource and retention contract

Defaults are per chunk, independent of the number of previous chunks:

| Limit | Default |
| --- | ---: |
| Raw decompressed bytes, normalized scratch bytes | 512 MiB each |
| Raw rows | 4,000,000 |
| Program identities | 4,096 |
| Distinct operation/core/RISC keys | 600,000 |
| Marker types | 1,024 |
| Raw physical line / CSV field | 16 KiB |
| Output record | 16 MiB |
| Uncompressed output | 768 MiB |
| Compressed output | 256 MiB |
| Child address space (`RLIMIT_AS`, Linux) | 2,048 MiB |

Limits are CLI arguments or keyword arguments to `append_chunk`. Raising one does
not raise the others. Scratch uses at most the normalized-input and compressed
output caps, plus small buffering. A transient directory is created inside the
output directory. No whole-fold list is loaded. Peak RSS uses Linux `/proc/self/status` VmHWM after exec; the footer also records
`getrusage` separately because its watermark can include the parent before exec.
Each child exits before the next
synchronous call, releasing its resident data.

A chunk is staged, reduced with the unchanged
[calibrated parser](../c10_dm_control/analyze.py) and
[census arithmetic](../c10_burst_census/reduce_census.py), verified, fsynced and
published with an atomic no-overwrite hard link. The output directory is then
fsynced. A malformed/truncated input, provenance mismatch, resource limit or
endpoint error publishes nothing. An existing chunk ID is refused. The caller
keeps the raw receipt and reuses no ID until it has resolved any prior result.
If directory fsync fails after publication, treat the call as failed and retain
raw even though a complete file may exist. Process death can leave an uncommitted
`.cycle-*` scratch directory; consumers read only final `*.jsonl.gz` files.

This requires a filesystem supporting local hard links and directory fsync.
It provides bounded per-chunk resources, not an unlimited disk quota. Budget disk
for the entire compact stream and raw retention before capturing. Preserve raw on
failure and retain selected raw controls even after successful reduction. No raw
files are deleted by this tool.

The largest discarded chunk was 377,218,386 bytes. Replay of retained inputs does
not prove that discarded chunk's marker distribution, memory demand or completeness.
CPU replay also cannot price the extra synchronous capture overhead or establish
whole-fold calibration, a binding roof, a speedup or a ceiling. Parent review is
required before another device capture. Model defaults and production code are
unchanged.
