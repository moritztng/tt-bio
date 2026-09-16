"""Bounded synchronous CPU reduction of one closed native profiler chunk, at workload speed.

Same contract as perf/c10_cycle_stream/stream.py: one closed chunk in, one immutable
gzip JSONL archive out, nothing published until parse, endpoints, provenance, output
caps, gzip close and footer verification all succeed. Framing, output capping,
verification and publication are imported from that accepted module, not reimplemented.

Two changes carry the speed. The raw CSV is parsed once instead of twice, and a
recognized firmware bracket marker is retained as integer endpoints on the core entry
that already exists instead of as one verbatim JSON row copy. In reduce_chunk_python
the accepted span arithmetic is the accepted reducer itself: reduce_census.span is
called on the same structure analyze.read_raw builds, so no cycle arithmetic is
re-derived there.

reduce.c is a native accelerator for the same contract, used when the chunk is plain
quote-free ASCII, which is what the profiler emits. It refuses anything else and
reduce_chunk_python runs instead, so the Python reducer is both the reference the
native payload is compared against and the fallback. Measured on pc, the pure-Python
reducer runs 26 MB/s and the native one 240 MB/s on the same archives.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'c10_cycle_stream'))
from stream import (DEFAULTS, FIELDS, ROOT, RISCS, ZONES, Output, file_sha256, integer,
                    peak_rss_kib, read_raw, span, unchanged, verify_chunk)

NATIVE_SOURCE = Path(__file__).resolve().parent / 'reduce.c'
BUILD_DIR = Path(__file__).resolve().parent / '.build'
CC = os.environ.get('CC', 'cc')
HEADER_PREFIX = 'ARCH: blackhole, CHIP_FREQ[MHz]: 1350,'
BRACKET = ('ZONE_START', 'ZONE_END')
# The five columns a firmware bracket row carries beyond identity, core and tick.
CONSTANT_COLUMNS = ('timer_id', 'data', 'source line', 'source file', 'meta data')
DIGITS = re.compile('[0-9]+')
RETENTION_CONTRACT = (
    'Every raw row is either reduced into the accepted span arithmetic, retained as '
    'integer firmware endpoints on its core entry plus the constant_fields payload of '
    'its marker combo, or copied verbatim. The raw input is never deleted and its '
    'SHA256/byte/row receipt is in this footer, so any formatting this archive drops '
    'is recoverable from that input.')


def scan(path, limits):
    """Hash the decompressed raw bytes and bound size/rows/line length before parsing.

    Returns the ASCII verdict too: on an all-ASCII chunk str.isdigit accepts exactly
    the same strings as analyze.integer's [0-9]+ , so the per-row parse can use it.
    A chunk with any non-ASCII byte falls back to the regex validator.
    """
    h = hashlib.sha256()
    total = lines = longest = 0
    ascii_only = True
    carry = b''
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            total += len(block)
            if total > limits['max_raw_bytes']:
                raise ValueError('max_raw_bytes exceeded')
            h.update(block)
            if ascii_only and not block.isascii():
                ascii_only = False
            parts = (carry + block).split(b'\n')
            carry = parts.pop()
            if parts:
                longest = max(longest, max(map(len, parts)))
                lines += len(parts)
    if carry:
        raise ValueError('Truncated line: missing final newline')
    if longest + 1 > limits['max_line_bytes']:
        raise ValueError('max_line_bytes exceeded')
    if total == 0:
        raise ValueError('Empty chunk')
    return h.hexdigest(), total, lines, ascii_only


def append_chunk(raw_path, output_dir, *, chunk_id, counter_scope, expected_bytes,
                 expected_sha256, compact_firmware=True, native=True, **limits):
    """Reduce in a memory-limited child; return only after immutable output is fsynced.

    Caller owns closing/retaining raw_path and supplying the drain's byte/hash receipt.
    Errors propagate; never unlink raw input here. No TT imports or device operations.
    """
    unknown = set(limits) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'Unknown limits: {unknown}')
    cmd = [sys.executable, str(Path(__file__).resolve()), str(raw_path), str(output_dir),
           '--chunk-id', chunk_id, '--counter-scope', counter_scope,
           '--expected-bytes', str(expected_bytes), '--expected-sha256', expected_sha256]
    if not compact_firmware:
        cmd.append('--no-compact-firmware')
    if not native:
        cmd.append('--pure-python')
    for name, value in limits.items():
        cmd += ['--' + name.replace('_', '-'), str(value)]
    return json.loads(subprocess.check_output(cmd, text=True))


def check_request(chunk_id, counter_scope, expected_bytes, expected_sha256, limits):
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}', chunk_id) or chunk_id in ('.', '..'):
        raise ValueError('Invalid chunk_id')
    if not counter_scope or len(counter_scope) > 256:
        raise ValueError('counter_scope must identify process/boot/counter epoch')
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256) or expected_bytes <= 0:
        raise ValueError('Require closed raw byte/hash receipt')
    if any(v <= 0 for v in limits.values()):
        raise ValueError('Limits must be positive')
    if expected_bytes > limits['max_raw_bytes']:
        raise ValueError('max_raw_bytes exceeded by receipt')


def open_target(raw_path, output_dir, chunk_id, limits):
    raw_path, output_dir = Path(raw_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / (chunk_id + '.jsonl.gz')
    if final.exists():
        raise FileExistsError(final)
    input_stat = raw_path.stat()
    if input_stat.st_size > limits['max_raw_bytes']:
        raise ValueError('Stored input exceeds max_raw_bytes')
    return raw_path, output_dir, final, input_stat


def publish(out, final, output_dir, limits):
    """Verify the closed archive, fsync it and link it in without overwriting."""
    out.close()
    verify_chunk(out.path, max_output_bytes=limits['max_output_bytes'])
    with out.path.open('rb') as f:
        os.fsync(f.fileno())
    os.link(out.path, final)
    fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reduce_chunk_python(raw_path, output_dir, *, chunk_id, counter_scope, expected_bytes,
                        expected_sha256, limits, compact_firmware=True):
    """Reference reducer. Use append_chunk to isolate the address-space limit."""
    check_request(chunk_id, counter_scope, expected_bytes, expected_sha256, limits)
    raw_path, output_dir, final, input_stat = open_target(raw_path, output_dir, chunk_id, limits)
    raw_sha256, raw_bytes, raw_lines, ascii_only = scan(raw_path, limits)
    if raw_bytes != expected_bytes or raw_sha256 != expected_sha256:
        raise ValueError('Raw byte/hash receipt mismatch')
    if raw_lines - 2 > limits['max_rows']:
        raise ValueError('max_rows exceeded')
    csv.field_size_limit(limits['max_line_bytes'])

    identities = {}
    ops = []
    coverage = {}
    constants = {}
    payloads = {}
    ranges = {}
    core_riscs = set()
    rows = transitions = returns = 0
    firmware_pairs = firmware_conflicts = firmware_deviations = 0
    unplaced_count = unplaced_programs = unsupported_programs = 0

    with tempfile.TemporaryDirectory(prefix='.throughput-', dir=output_dir) as scratch:
        out = Output(Path(scratch) / 'chunk.gz', limits)
        try:
            with gzip.open(raw_path, 'rt', encoding='utf-8', newline='') if raw_path.suffix == '.gz' \
                    else raw_path.open('rt', encoding='utf-8', newline='') as source:
                header = next(source).rstrip('\r\n')
                # Keep the accepted calibrated timebase contract, never telemetry.
                if not header.startswith(HEADER_PREFIX):
                    raise ValueError('Unsupported profiler header')
                reader = csv.reader(source, skipinitialspace=True, strict=True)
                if next(reader) != FIELDS:
                    raise ValueError('Unexpected CSV columns')
                out.put(dict(kind='chunk', schema=2, chunk_id=chunk_id,
                             counter_scope=counter_scope, header=header, raw_fields=FIELDS,
                             limits=limits, input_path=str(raw_path.resolve()),
                             compact_firmware=bool(compact_firmware),
                             ascii_only_input=ascii_only, clock_telemetry=None))
                previous = None
                op = span_range = None
                previous_core_risc = None
                for row in reader:
                    rows += 1
                    try:
                        (device, x, y, risc, timer, tick, data, call, trace, replay,
                         zone, kind, line, filename, meta) = row
                    except ValueError:
                        raise ValueError(f'Malformed CSV row {rows}: {len(row)} columns') from None
                    if ascii_only:
                        if not (device.isdigit() and x.isdigit() and y.isdigit()
                                and timer.isdigit() and tick.isdigit() and data.isdigit()
                                and call.isdigit() and line.isdigit()):
                            raise ValueError(f'Not an integer field in row {rows}')
                    else:
                        for value in (device, x, y, timer, tick, data, call, line):
                            integer(value)
                    if trace or replay:
                        if bool(trace) != bool(replay):
                            raise ValueError('Incomplete trace/replay identity')
                        if not (DIGITS.fullmatch(trace) and DIGITS.fullmatch(replay)):
                            raise ValueError(f'Not an integer trace identity in row {rows}')
                    identity = (call, device, trace, replay)
                    if identity != previous:
                        transitions += 1
                        surrogate = identities.get(identity)
                        if surrogate is None:
                            if len(identities) >= limits['max_operations']:
                                raise ValueError('max_operations exceeded')
                            identities[identity] = len(identities)
                            ops.append(dict(kernels={}, sums={}, firmware={}))
                            surrogate = len(ops) - 1
                        else:
                            # An ID already allocated before this transition has recurred.
                            returns += 1
                        op = ops[surrogate]
                        previous = identity
                        domain = (device, trace, replay)
                        span_range = ranges.get(domain)
                        if span_range is None:
                            span_range = ranges[domain] = [call, call, None, None]
                        else:
                            # (length, digits) is a total order on unpadded digit strings,
                            # so call-ID extremes stay exact at any magnitude.
                            order = (len(call), call)
                            if order < (len(span_range[0]), span_range[0]):
                                span_range[0] = call
                            elif order > (len(span_range[1]), span_range[1]):
                                span_range[1] = call
                    core = (int(x), int(y))
                    key = (surrogate, core, risc)
                    if key != previous_core_risc:
                        previous_core_risc = key
                        core_riscs.add(key)
                        if len(core_riscs) > limits['max_core_riscs']:
                            raise ValueError('max_core_riscs exceeded')
                    marker = (risc, zone, kind)
                    count = coverage.get(marker)
                    if count is None:
                        if len(coverage) >= limits['max_markers']:
                            raise ValueError('max_markers exceeded')
                        coverage[marker] = 1
                        constants[marker] = payload = (timer, data, line, filename, meta)
                        payloads[marker] = payload
                    else:
                        coverage[marker] = count + 1
                    ticks = int(tick)
                    if span_range[2] is None:
                        span_range[2] = span_range[3] = ticks
                    elif ticks < span_range[2]:
                        span_range[2] = ticks
                    elif ticks > span_range[3]:
                        span_range[3] = ticks

                    if risc in RISCS:
                        if zone.endswith('-KERNEL'):
                            ends = op['kernels'].setdefault((core, risc), {})
                            if kind not in BRACKET or kind in ends:
                                raise ValueError(f'Duplicate/invalid kernel marker: {call} {(core, risc)}')
                            ends[kind] = ticks
                            continue
                        if kind == 'ZONE_TOTAL':
                            sums = op['sums']
                            sum_key = (core, risc, zone)
                            if sum_key in sums:
                                raise ValueError(f'Duplicate sum marker: {call} {sum_key}')
                            sums[sum_key] = int(data)
                            continue
                        if compact_firmware and zone.endswith('-FW') and kind in BRACKET:
                            slot = op['firmware'].setdefault((core, risc), {})
                            if kind in slot:
                                firmware_conflicts += 1
                            else:
                                slot[kind] = ticks
                                if (timer, data, line, filename, meta) != constants[marker]:
                                    # The combo is not constant after all: keep this row whole.
                                    firmware_deviations += 1
                                    payloads[marker] = None
                                    out.put(dict(kind='unreduced_marker', row=rows, values=row))
                                continue
                    out.put(dict(kind='unreduced_marker', row=rows, values=row))
            if not rows:
                raise ValueError('Empty chunk')
            if not unchanged(raw_path, input_stat):
                raise ValueError('Input changed during reduction')
            collisions = {}
            for (call, device, trace, replay), surrogate in identities.items():
                normalized = (int(call), int(device), trace, replay)
                if normalized in collisions:
                    raise ValueError(f'Ambiguous zero-padded identity encoding: {normalized}')
                collisions[normalized] = surrogate
            del core_riscs

            for (call, device, trace, replay), surrogate in identities.items():
                op = ops[surrogate]
                record = dict(kind='program', global_call_id=int(call), device=int(device),
                              trace=trace or None, replay=replay or None)
                # The accepted span reducer only visits RISCs with endpoints. Preserve
                # totals from entirely absent RISCs separately, without inventing
                # residency or silently assigning them to another thread.
                present_riscs = {r for c, r in op['kernels']}
                unplaced = [dict(core=list(c), risc=r, zone=z, cycles=op['sums'][c, r, z])
                            for c, r, z in op['sums'] if r not in present_riscs]
                record['unplaced_zone_sums'] = unplaced
                unplaced_count += len(unplaced)
                unplaced_programs += bool(unplaced)
                firmware = op['firmware']
                firmware_pairs += sum(len(v) == 2 for v in firmware.values())
                extra = sorted(k for k in firmware if k not in op['kernels'])
                record['firmware_only_cores'] = [
                    dict(core=list(c), risc=r,
                         firmware_start_cycle=firmware[c, r].get('ZONE_START'),
                         firmware_end_cycle=firmware[c, r].get('ZONE_END')) for c, r in extra]
                if not op['kernels']:
                    unsupported_programs += 1
                    out.put(dict(record, status='unsupported_no_kernel_endpoints', summary=None, cores=[]))
                    continue
                summary = span(op)  # The accepted calibration/census arithmetic.
                per_core = collections.defaultdict(dict)
                for (c, r, z), value in op['sums'].items():
                    per_core[c, r][z] = value
                cores = []
                for (c, r), ends in sorted(op['kernels'].items()):
                    zones = per_core.get((c, r), {})
                    resident = ends['ZONE_END'] - ends['ZONE_START']
                    if sum(zones.values()) > resident:
                        raise ValueError('Zones exceed own core/RISC residency')
                    bracket = firmware.get((c, r), {})
                    cores.append(dict(core=list(c), risc=r, start_cycle=ends['ZONE_START'],
                                      end_cycle=ends['ZONE_END'], resident_cycles=resident,
                                      zone_cycles=zones, unclassified_cycles=resident - sum(zones.values()),
                                      firmware_start_cycle=bracket.get('ZONE_START'),
                                      firmware_end_cycle=bracket.get('ZONE_END')))
                out.put(dict(record, status='reduced_with_unplaced_sums' if unplaced else 'reduced',
                             summary=summary, cores=cores))

            footer = dict(kind='complete', chunk_id=chunk_id, counter_scope=counter_scope,
                          header=header,
                          raw_sha256=raw_sha256, raw_bytes=raw_bytes, raw_rows=rows,
                          stored_input_sha256=file_sha256(raw_path), stored_input_bytes=input_stat.st_size,
                          reducer_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in
                              (Path(__file__), ROOT / 'perf/c10_cycle_stream/stream.py',
                               ROOT / 'perf/c10_dm_control/analyze.py',
                               ROOT / 'perf/c10_burst_census/reduce_census.py')},
                          programs=len(identities), records=out.records,
                          programs_without_endpoints=unsupported_programs,
                          programs_with_unplaced_sums=unplaced_programs, unplaced_sum_markers=unplaced_count,
                          payload_sha256=out.hash.hexdigest(), payload_bytes=out.bytes,
                          call_transitions=transitions, noncontiguous_returns=returns,
                          compact_firmware=bool(compact_firmware),
                          firmware_bracket_pairs=firmware_pairs,
                          firmware_bracket_conflicts=firmware_conflicts,
                          firmware_rows_retained_verbatim=firmware_deviations,
                          retention_contract=RETENTION_CONTRACT,
                          counter_ranges=[dict(device=int(d), trace=t or None, replay=p or None,
                                               min_call_id=int(v[0]), max_call_id=int(v[1]),
                                               min_tick=v[2], max_tick=v[3])
                                          for (d, t, p), v in sorted(ranges.items(), key=lambda kv: (int(kv[0][0]), kv[0][1], kv[0][2]))],
                          marker_coverage=[
                              marker_entry(r, z, k, n,
                                           payloads[r, z, k] if compact_firmware and z.endswith('-FW')
                                           and k in BRACKET and r in RISCS else None,
                                           compact_firmware)
                              for (r, z, k), n in sorted(coverage.items())],
                          peak_rss_kib=peak_rss_kib(),
                          rss_basis='Linux post-exec VmHWM, KiB',
                          rusage_maxrss_including_preexec_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if not unchanged(raw_path, input_stat):
                raise ValueError('Input changed during provenance hashing')
            out.put(footer, footer=True)
            # Atomic no-overwrite publication, including concurrent chunk-ID collision.
            publish(out, final, output_dir, limits)
            return dict(path=str(final.resolve()), reducer='python', raw_rows=rows, programs=len(identities),
                        raw_bytes=raw_bytes, output_bytes=out.bytes,
                        archive_bytes=final.stat().st_size, peak_rss_kib=peak_rss_kib(),
                        firmware_bracket_pairs=firmware_pairs,
                        firmware_rows_retained_verbatim=firmware_deviations,
                        firmware_bracket_conflicts=firmware_conflicts)
        finally:
            try:
                out.file.close()
            finally:
                out.stored.close()


class BlockOutput(Output):
    """Frame a native payload in blocks: hash, count records and cap bytes in bulk.

    The per-record work the accepted Output.put does is exactly what the payload
    size makes expensive, and all of it is bulk arithmetic over the same bytes.
    verify_chunk re-reads the closed archive record by record either way.
    """

    def __init__(self, path, limits):
        super().__init__(path, limits)
        self.partial = 0

    def put_block(self, data, max_record_bytes=16 * 1024**2):
        if self.bytes + len(data) > self.limits['max_output_bytes']:
            raise ValueError('max_output_bytes exceeded')
        pieces = data.split(b'\n')
        if self.partial + len(pieces[0]) > max_record_bytes:
            raise ValueError('16 MiB output record limit exceeded')
        if max(map(len, pieces[1:-1]), default=0) > max_record_bytes:
            raise ValueError('16 MiB output record limit exceeded')
        self.partial = len(pieces[-1]) if len(pieces) > 1 else self.partial + len(pieces[0])
        self.file.write(data)
        self.hash.update(data)
        self.records += len(pieces) - 1
        self.bytes += len(data)


def native_binary():
    """Compile reduce.c on demand; the cache key is the source digest."""
    source = NATIVE_SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    binary = BUILD_DIR / ('reduce-' + digest[:16])
    if not binary.exists():
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=BUILD_DIR) as work:
            staged = Path(work) / 'reduce'
            subprocess.run([CC, '-O2', '-Wall', '-Wextra', '-o', str(staged), str(NATIVE_SOURCE),
                            '-lz', '-lcrypto'], check=True)
            os.replace(staged, binary)
    return binary, digest


UNSUPPORTED_INPUT = 3


def reduce_chunk_native(raw_path, output_dir, *, chunk_id, counter_scope, expected_bytes,
                        expected_sha256, limits, compact_firmware=True):
    """Frame the native payload. Returns None when reduce.c refuses the input shape."""
    check_request(chunk_id, counter_scope, expected_bytes, expected_sha256, limits)
    raw_path, output_dir, final, input_stat = open_target(raw_path, output_dir, chunk_id, limits)
    binary, source_digest = native_binary()
    with tempfile.TemporaryDirectory(prefix='.throughput-', dir=output_dir) as scratch:
        stats_path = Path(scratch) / 'stats.json'
        cmd = [str(binary), str(raw_path), '--stats-path', str(stats_path),
               '--max-rows', str(limits['max_rows']), '--max-operations', str(limits['max_operations']),
               '--max-core-riscs', str(limits['max_core_riscs']), '--max-markers', str(limits['max_markers']),
               '--max-line-bytes', str(limits['max_line_bytes']), '--max-raw-bytes', str(limits['max_raw_bytes'])]
        if not compact_firmware:
            cmd.append('--no-compact-firmware')
        out = BlockOutput(Path(scratch) / 'chunk.gz', limits)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # The payload streams first: the chunk record carries the profiler header,
            # which reduce.c reports in its stats only once the input has been read.
            # Record order inside the archive is not load-bearing; every record is
            # self-describing and verify_chunk reads to the footer either way.
            while True:
                block = child.stdout.read(1 << 20)
                if not block:
                    break
                out.put_block(block)
            child.stdout.close()
            message = child.stderr.read().decode('utf-8', 'replace').strip()
            child.stderr.close()
            if child.wait() == UNSUPPORTED_INPUT:
                return None
            if child.returncode:
                raise ValueError(f'native reducer exit {child.returncode}: {message}')
            if out.partial:
                raise ValueError('native payload does not end on a record boundary')
            stats = json.loads(stats_path.read_text())
            if not unchanged(raw_path, input_stat):
                raise ValueError('Input changed during reduction')
            if stats['raw_bytes'] != expected_bytes or stats['raw_sha256'] != expected_sha256:
                raise ValueError('Raw byte/hash receipt mismatch')
            if stats['raw_rows'] > limits['max_rows']:
                raise ValueError('max_rows exceeded')
            if stats['raw_rows'] != sum(m['rows'] for m in stats['markers']):
                raise ValueError('Marker coverage does not account for every raw row')
            if out.records != stats['records']:
                raise ValueError('Native record count disagrees with the framed payload')
            if stats['programs'] > limits['max_operations'] or len(stats['markers']) > limits['max_markers']:
                raise ValueError('Native reducer exceeded an identity/marker cap')
            if stats['distinct_core_riscs'] > limits['max_core_riscs']:
                raise ValueError('max_core_riscs exceeded')
            if not stats['header'].startswith(HEADER_PREFIX):
                raise ValueError('Unsupported profiler header')
            out.put(dict(kind='chunk', schema=2, chunk_id=chunk_id, counter_scope=counter_scope,
                         header=stats['header'], raw_fields=FIELDS, limits=limits,
                         input_path=str(raw_path.resolve()), reducer='native',
                         ascii_only_input=stats['ascii_only_input'],
                         compact_firmware=bool(compact_firmware), clock_telemetry=None))
            footer = build_footer(
                stats, chunk_id=chunk_id, counter_scope=counter_scope, raw_path=raw_path,
                input_stat=input_stat, out=out, limits=limits, compact_firmware=compact_firmware,
                native=dict(source='perf/c10_cycle_throughput/reduce.c', source_sha256=source_digest,
                            binary_sha256=file_sha256(binary)))
            if not unchanged(raw_path, input_stat):
                raise ValueError('Input changed during provenance hashing')
            out.put(footer, footer=True)
            publish(out, final, output_dir, limits)
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            return dict(path=str(final.resolve()), reducer='native', raw_rows=stats['raw_rows'],
                        programs=stats['programs'], raw_bytes=stats['raw_bytes'],
                        output_bytes=out.bytes, archive_bytes=final.stat().st_size,
                        peak_rss_kib=peak_rss_kib(), native_peak_rss_kib=usage.ru_maxrss,
                        native_cpu_s=(usage.ru_utime + usage.ru_stime
                                      - before.ru_utime - before.ru_stime),
                        firmware_bracket_pairs=stats['firmware_bracket_pairs'],
                        firmware_rows_retained_verbatim=stats['firmware_rows_retained_verbatim'],
                        firmware_bracket_conflicts=stats['firmware_bracket_conflicts'])
        finally:
            try:
                out.file.close()
            finally:
                out.stored.close()


def marker_entry(risc, zone, kind, rows, payload, compact_firmware):
    """The stream child's arithmetic label, plus what this reducer did with the row."""
    return dict(risc=risc, zone=zone, type=kind, rows=rows,
                arithmetic=('kernel_endpoint' if zone.endswith('-KERNEL') and risc in RISCS else
                            'zone_sum' if kind == 'ZONE_TOTAL' and risc in RISCS else 'unreduced'),
                retention=('span_arithmetic' if (zone.endswith('-KERNEL') or kind == 'ZONE_TOTAL') and risc in RISCS
                           else 'firmware_endpoints' if compact_firmware and zone.endswith('-FW')
                           and kind in BRACKET and risc in RISCS else 'verbatim_record'),
                constant_fields=dict(zip(CONSTANT_COLUMNS, payload)) if payload else None,
                known_zone=zone in (*ZONES, 'CB-COMPUTE-WAIT-FRONT', 'CB-COMPUTE-RESERVE-BACK'))


def build_footer(stats, *, chunk_id, counter_scope, raw_path, input_stat, out, limits,
                 compact_firmware, native):
    sources = [Path(__file__), ROOT / 'perf/c10_cycle_stream/stream.py',
               ROOT / 'perf/c10_dm_control/analyze.py',
               ROOT / 'perf/c10_burst_census/reduce_census.py']
    return dict(kind='complete', chunk_id=chunk_id, counter_scope=counter_scope,
                raw_sha256=stats['raw_sha256'], raw_bytes=stats['raw_bytes'],
                raw_rows=stats['raw_rows'], stored_input_sha256=file_sha256(raw_path),
                stored_input_bytes=input_stat.st_size,
                reducer_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in sources},
                native_reducer=native,
                programs=stats['programs'], records=out.records,
                programs_without_endpoints=stats['programs_without_endpoints'],
                programs_with_unplaced_sums=stats['programs_with_unplaced_sums'],
                unplaced_sum_markers=stats['unplaced_sum_markers'],
                payload_sha256=out.hash.hexdigest(), payload_bytes=out.bytes,
                call_transitions=stats['call_transitions'],
                noncontiguous_returns=stats['noncontiguous_returns'],
                compact_firmware=bool(compact_firmware),
                firmware_bracket_pairs=stats['firmware_bracket_pairs'],
                firmware_bracket_conflicts=stats['firmware_bracket_conflicts'],
                firmware_rows_retained_verbatim=stats['firmware_rows_retained_verbatim'],
                retention_contract=RETENTION_CONTRACT,
                counter_ranges=sorted(stats['counter_ranges'],
                                      key=lambda r: (r['device'], r['trace'] or '', r['replay'] or '')),
                marker_coverage=sorted(
                    (marker_entry(m['risc'], m['zone'], m['type'], m['rows'],
                                  m['payload'] if m['route'] == 3 and m['payload_constant'] else None,
                                  compact_firmware) for m in stats['markers']),
                    key=lambda m: (m['risc'], m['zone'], m['type'])),
                peak_rss_kib=peak_rss_kib(),
                rss_basis='Linux post-exec VmHWM, KiB',
                rusage_maxrss_including_preexec_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                native_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                header=stats['header'])


def reduce_chunk(raw_path, output_dir, *, chunk_id, counter_scope, expected_bytes,
                 expected_sha256, limits, compact_firmware=True, native=True):
    """Reduce one closed chunk, natively when the input shape allows it."""
    if native:
        result = reduce_chunk_native(
            raw_path, output_dir, chunk_id=chunk_id, counter_scope=counter_scope,
            expected_bytes=expected_bytes, expected_sha256=expected_sha256,
            limits=limits, compact_firmware=compact_firmware)
        if result is not None:
            return result
    return reduce_chunk_python(
        raw_path, output_dir, chunk_id=chunk_id, counter_scope=counter_scope,
        expected_bytes=expected_bytes, expected_sha256=expected_sha256,
        limits=limits, compact_firmware=compact_firmware)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('raw_path', type=Path)
    ap.add_argument('output_dir', type=Path)
    ap.add_argument('--chunk-id', required=True)
    ap.add_argument('--counter-scope', required=True)
    ap.add_argument('--expected-bytes', type=int, required=True)
    ap.add_argument('--expected-sha256', required=True)
    ap.add_argument('--no-compact-firmware', dest='compact_firmware', action='store_false',
                    help='retain every firmware bracket row verbatim, as stream.py does')
    ap.add_argument('--pure-python', dest='native', action='store_false',
                    help='skip the native accelerator and run the reference reducer')
    for name, default in DEFAULTS.items():
        ap.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    args = vars(ap.parse_args())
    limits = {k: args.pop(k) for k in DEFAULTS}
    try:
        if limits['max_memory_mib'] <= 0:
            raise ValueError('Memory limit must be positive')
        resource.setrlimit(resource.RLIMIT_AS, (limits['max_memory_mib'] * 1024**2,) * 2)
        print(json.dumps(reduce_chunk(**args, limits=limits)))
    except (ValueError, OSError, EOFError, csv.Error, MemoryError, StopIteration) as exc:
        print(f'cycle-throughput STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
