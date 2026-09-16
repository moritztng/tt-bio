# c10_cycle_throughput — full cycle retention at workload speed

Reduces one closed native profiler chunk into an immutable gzip JSONL archive, with the
same contract and the same published arithmetic as `perf/c10_cycle_stream`, fast enough
to retain a whole drain instead of a sample.

On the largest retained census window (327,094,028 raw bytes, 2,258,392 rows) the stream
child takes 41.10 s and this takes 2.17-2.25 s over four repeats: 145.7 MB/s measured from
the slowest repeat. Scaling that to the 80,895,977,723 bytes of the prior full drain
projects 555 s of wall and 555 s of serial CPU. That is a CPU planning projection from
replayed archives, not a hardware measurement and not a fold time.

No device is opened here and no model cycles are saved. The archives being replayed were
captured at a during-window sampled 1350 MHz; the CSV header frequency is provenance only.

## Using it

```python
from fast import append_chunk

result = append_chunk(
    raw_path,                      # the closed chunk, kept by the caller
    output_dir,
    chunk_id='DiffusionModule_0',  # [A-Za-z0-9_.-]{1,100}
    counter_scope='run-7-boot-a',  # names the process/boot/counter epoch
    expected_bytes=327_094_028,    # the drain's own receipt for the decompressed bytes
    expected_sha256='a5c7...7efb',
)
```

`append_chunk` is synchronous. It returns only after the archive has been reduced,
verified, fsynced and linked into place, and it never starts a background job. Errors
propagate; nothing is published unless every check passed. The raw input is never
modified or deleted, so the caller decides when to release it.

The call runs the reducer in a child process so `max_memory_mib` can be an address-space
limit, and that child runs `reduce.c` as a grandchild. Pass any of the limits below as
keyword arguments, `compact_firmware=False` to keep every firmware row verbatim, or
`native=False` to force the reference reducer. Same thing on the command line:

```
python3 fast.py RAW OUT --chunk-id ID --counter-scope SCOPE \
        --expected-bytes N --expected-sha256 HEX [--pure-python] [--no-compact-firmware]
```

`reduce.c` is compiled on demand into `.build/`, keyed by the source digest, using the
installed compiler against zlib and libcrypto. Nothing is rebuilt in tt-metal and no
dependency is upgraded.

## What the archive holds

One `chunk` record with the profiler header and the limits in force, one `program` record
per identity, one `unreduced_marker` record per row that was not reduced, and a `complete`
footer. Record order is not load-bearing; every record is self-describing and the footer
is always last.

A `program` record keeps the exact raw identity (global call id, device, trace, replay),
the accepted `reduce_census.span` summary, and one entry per core and RISC with its
integer start/end cycle, residency, per-zone cycles and unclassified cycles. Endpoints,
totals and sums stay integers at any magnitude, including above 2**53. Call ids recur
across cores, so an identity is never closed on an id transition.

Zone totals for a RISC that has no kernel endpoints anywhere in the program stay in
`unplaced_zone_sums` with their cycles and their core. Their residency is unknown and is
never written as zero. A program with no kernel endpoints at all is published with
`status: unsupported_no_kernel_endpoints` and a null summary, not dropped.

The footer carries the raw SHA256, byte count and row count, the stored input's own hash
and size, the reducer source hashes, the identity and counter ranges, the full marker
census and the peak RSS of both processes.

## Firmware bracket compaction

The stream child copied every `*-FW` `ZONE_START`/`ZONE_END` row into the archive as a
verbatim 15-field JSON record: 1,213,692 of them across the 13 census windows, 832,086 in
the largest one alone. Those rows are a recognized firmware marker, not an unsupported
payload, and for each `(RISC, zone, type)` combo their `timer_id`, `data`, `source line`,
`source file` and `meta data` are the same on every row. So the only per-row information
is identity, core and tick.

This reducer keeps that information where it belongs: `firmware_start_cycle` and
`firmware_end_cycle` on the core entry that already exists, plus the combo's
`constant_fields` in the footer. A core and RISC with a firmware bracket but no kernel
endpoints gets a `firmware_only_cores` entry instead.

Compaction is not allowed to lose anything, so it backs off row by row. If a row's five
constant columns differ from the combo's, the row is retained whole as well as placed, and
the combo's `constant_fields` becomes null. If a bracket already has that side filled, the
second row is retained whole and counted in `firmware_bracket_conflicts`. Any other marker
is retained whole as before. The raw input, with its hash and byte receipt in the footer,
remains the record for anything a reduction drops.

`replay.py` proves the round trip: for all 1,490,212 verbatim rows the stream child
published, it reads each row back out of the stream child's archive and checks that its
tick reappears as the firmware endpoint on the matching core entry and its other five
columns as the combo's `constant_fields`.

## Reducers and the fallback

`reduce.c` handles plain quote-free ASCII input, which is what the profiler emits. On a
double quote, a stray carriage return or any non-ASCII byte it refuses before writing
anything, because CPython's `csv` module is then the only authority on how the row splits,
and `reduce_chunk_python` runs instead at 25.9 MB/s. That reference reducer calls the
accepted `reduce_census.span` on the same structure `analyze.read_raw` builds, so it also
serves as the implementation the native payload is compared against.

`reduce.c` adds one check the reference does not have: a zero-padded call id or device is
refused rather than silently merged with its unpadded form.

## Limits

| limit | default | applies to |
| --- | --- | --- |
| `max_raw_bytes` | 512 MiB | decompressed input |
| `max_rows` | 4,000,000 | input rows |
| `max_operations` | 4,096 | distinct identities |
| `max_core_riscs` | 600,000 | distinct (identity, core, RISC) |
| `max_markers` | 1,024 | distinct (RISC, zone, type) |
| `max_line_bytes` | 16,384 | one input line |
| `max_output_bytes` | 768 MiB | uncompressed payload |
| `max_archive_bytes` | 256 MiB | published archive |
| `max_memory_mib` | 2,048 | address space per reducer process |

`reduce.c` additionally buffers rows it retains whole, bounded by `--max-verbatim-bytes`
(64 MiB); above that it hands the chunk to the streaming Python reducer.

## Reproducing

```
python3 -m pytest test_fast.py -q
python3 replay.py --output-dir DIR --report replay_report.json \
        --sustained-repeats 4 --python-repeats 2
```

`replay.py` reduces all 13 retained census windows and the calibration archive, and for
each one reproduces the accepted `read_raw`/`span` result live, the published census spans,
unions and window identity sets, and the 7 calibration intervals with their 241 published
operations through `analyze.reduce_op`. It then runs the native and reference reducers
against each other on identical bytes and repeats the largest window to get a sustained
rate. Repeating an archive is a repeated CPU reduction, not a new hardware execution.

## What this does not establish

The largest chunk the earlier capture discarded, 377,218,386 bytes, is gone. Nothing here
can show its marker mix, its resource demand, its completeness or that it would have
reduced at all, and the projection assumes the marker mix of the windows that were kept.

Capture and export overhead on hardware is unmeasured, the projection is linear in bytes
and says nothing about a fold, and no model cycle is saved by any of this.
