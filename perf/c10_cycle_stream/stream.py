"""Bounded, synchronous CPU reduction of one closed native profiler chunk."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'perf/c10_burst_census'), str(ROOT / 'perf/c10_dm_control')]
from analyze import RISCS, ZONES, integer, read_raw
from reduce_census import span

FIELDS = ['PCIe slot', 'core_x', 'core_y', 'RISC processor type', 'timer_id',
          'time[cycles since reset]', 'data', 'run host ID', 'trace id',
          'trace id counter', 'zone name', 'type', 'source line', 'source file', 'meta data']
DEFAULTS = dict(max_raw_bytes=512 * 1024**2, max_rows=4_000_000,
                max_operations=4096, max_core_riscs=600_000, max_markers=1024,
                max_line_bytes=16384, max_output_bytes=768 * 1024**2,
                max_archive_bytes=256 * 1024**2, max_memory_mib=2048)


def peak_rss_kib():
    # getrusage's watermark can include the parent's pre-exec fork footprint.
    # Linux VmHWM is reset by exec, so it measures this reducer process image.
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1])
    raise RuntimeError('Linux VmHWM unavailable')


def unchanged(path, before):
    after = Path(path).stat()
    return all(getattr(before, k) == getattr(after, k)
               for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_chunk(path, *, max_output_bytes=DEFAULTS['max_output_bytes'], max_record_bytes=16 * 1024**2):
    """Read to gzip EOF/CRC and validate the completion footer with bounded memory."""
    h = hashlib.sha256()
    size = count = 0
    footer = None
    with gzip.open(path, 'rb') as f:
        while True:
            data = f.readline(max_record_bytes + 1)
            if not data:
                break
            if footer is not None or len(data) > max_record_bytes or not data.endswith(b'\n'):
                raise ValueError('Incomplete/oversize record or data after completion')
            size += len(data)
            if size > max_output_bytes:
                raise ValueError('Output verification byte cap exceeded')
            record = json.loads(data)
            if record['kind'] == 'complete':
                footer = record
                if (footer['payload_sha256'] != h.hexdigest() or footer['records'] != count
                        or footer['payload_bytes'] != size - len(data)):
                    raise ValueError('Output completion digest/count mismatch')
            else:
                h.update(data)
                count += 1
    if footer is None:
        raise ValueError('Missing completion footer')
    return footer


def encoded(value):
    return (json.dumps(value, separators=(',', ':'), ensure_ascii=True) + '\n').encode()


def append_chunk(raw_path, output_dir, *, chunk_id, counter_scope, expected_bytes,
                 expected_sha256, **limits):
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
    for name, value in limits.items():
        cmd += ['--' + name.replace('_', '-'), str(value)]
    return json.loads(subprocess.check_output(cmd, text=True))


class CappedFile(io.FileIO):
    def __init__(self, path, limit):
        super().__init__(path, 'w')
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise ValueError('max_archive_bytes exceeded')
        return super().write(data)


class Output:
    def __init__(self, path, limits):
        self.path, self.limits = path, limits
        self.stored = CappedFile(path, limits['max_archive_bytes'])
        try:
            self.file = gzip.GzipFile(filename='', mode='wb', fileobj=self.stored, compresslevel=1, mtime=0)
        except BaseException:
            self.stored.close()
            raise
        self.bytes = self.records = 0
        self.hash = hashlib.sha256()

    def put(self, record, *, footer=False):
        data = encoded(record)
        if len(data) > 16 * 1024**2:
            raise ValueError('16 MiB output record limit exceeded')
        if self.bytes + len(data) > self.limits['max_output_bytes']:
            raise ValueError('max_output_bytes exceeded')
        self.file.write(data)
        if not footer:
            self.hash.update(data)
            self.records += 1
        self.bytes += len(data)

    def close(self):
        self.file.close()
        self.stored.close()


def reduce_chunk(raw_path, output_dir, *, chunk_id, counter_scope, expected_bytes,
                 expected_sha256, limits):
    """Worker implementation. Use append_chunk to isolate the address-space limit."""
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
    raw_path, output_dir = Path(raw_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / (chunk_id + '.jsonl.gz')
    if final.exists():
        raise FileExistsError(final)
    input_stat = raw_path.stat()
    if input_stat.st_size > limits['max_raw_bytes']:
        raise ValueError('Stored input exceeds max_raw_bytes')
    raw_hash = hashlib.sha256()
    raw_bytes = rows = 0
    identities = {}
    core_riscs = set()
    coverage = collections.Counter()
    ranges = {}
    returns = transitions = 0
    unplaced_count = unplaced_programs = unsupported_programs = 0
    previous = None
    seen = set()
    csv.field_size_limit(limits['max_line_bytes'])

    # Both scratch files are bounded by explicit byte limits. Nothing is published
    # until parser, endpoints, provenance, output limits and gzip close all succeed.
    with tempfile.TemporaryDirectory(prefix='.cycle-', dir=output_dir) as scratch:
        scratch = Path(scratch)
        out = Output(scratch / 'chunk.gz', limits)
        try:
            opener = gzip.open if raw_path.suffix == '.gz' else open
            with opener(raw_path, 'rb') as source, (scratch / 'normalized.csv').open('w', newline='') as normalized:
                def lines():
                    nonlocal raw_bytes
                    while True:
                        data = source.readline(limits['max_line_bytes'] + 1)
                        if not data:
                            return
                        if len(data) > limits['max_line_bytes']:
                            raise ValueError('max_line_bytes exceeded')
                        if not data.endswith(b'\n'):
                            raise ValueError('Truncated line: missing final newline')
                        raw_bytes += len(data)
                        if raw_bytes > limits['max_raw_bytes']:
                            raise ValueError('max_raw_bytes exceeded')
                        raw_hash.update(data)
                        yield data.decode('utf-8')
                text = lines()
                header = next(text).rstrip('\r\n')
                # Keep the accepted calibrated timebase contract, never telemetry.
                if not header.startswith('ARCH: blackhole, CHIP_FREQ[MHz]: 1350,'):
                    raise ValueError('Unsupported profiler header')
                reader = csv.reader(text, skipinitialspace=True, strict=True)
                if next(reader) != FIELDS:
                    raise ValueError('Unexpected CSV columns')
                normalized.write(header + '\n')
                writer = csv.writer(normalized, lineterminator='\n')
                writer.writerow(FIELDS)
                out.put(dict(kind='chunk', schema=1, chunk_id=chunk_id,
                             counter_scope=counter_scope, header=header, raw_fields=FIELDS,
                             limits=limits, input_path=str(raw_path.resolve()),
                             clock_telemetry=None))
                for row in reader:
                    rows += 1
                    if rows > limits['max_rows']:
                        raise ValueError('max_rows exceeded')
                    if len(row) != len(FIELDS):
                        raise ValueError(f'Malformed CSV row {rows}: {len(row)} columns')
                    device, x, y, risc, timer, tick, data, call, trace, replay, zone, kind, line, filename, meta = row
                    for value in (device, x, y, timer, tick, data, call, line):
                        integer(value)
                    if trace:
                        integer(trace)
                    if replay:
                        integer(replay)
                    if bool(trace) != bool(replay):
                        raise ValueError('Incomplete trace/replay identity')
                    identity = (integer(call), integer(device), trace, replay)
                    if identity not in identities:
                        if len(identities) >= limits['max_operations']:
                            raise ValueError('max_operations exceeded')
                        identities[identity] = len(identities)
                    surrogate = identities[identity]
                    if previous != identity:
                        transitions += 1
                        # An ID already allocated before this transition has recurred.
                        if identity in seen:
                            returns += 1
                        seen.add(identity)
                        previous = identity
                    key = (surrogate, integer(x), integer(y), risc)
                    core_riscs.add(key)
                    if len(core_riscs) > limits['max_core_riscs']:
                        raise ValueError('max_core_riscs exceeded')
                    marker = (risc, zone, kind)
                    coverage[marker] += 1
                    if len(coverage) > limits['max_markers']:
                        raise ValueError('max_markers exceeded')
                    domain = (integer(device), trace, replay)
                    r = ranges.setdefault(domain, [integer(call), integer(call), integer(tick), integer(tick)])
                    r[:] = [min(r[0], integer(call)), max(r[1], integer(call)),
                            min(r[2], integer(tick)), max(r[3], integer(tick))]
                    kernel = zone.endswith('-KERNEL') and risc in RISCS
                    total = kind == 'ZONE_TOTAL' and risc in RISCS
                    if kernel or total:
                        copy = list(row)
                        # Only a reversible parser adapter: surrogate joins exact raw
                        # device/trace/replay identities; published IDs are never remapped.
                        copy[0], copy[7], copy[8], copy[9] = '0', str(surrogate), '', ''
                        writer.writerow(copy)
                    else:
                        out.put(dict(kind='unreduced_marker', row=rows, values=row))
                    if normalized.tell() > limits['max_raw_bytes']:
                        raise ValueError('Normalized scratch exceeds max_raw_bytes')
            if not rows:
                raise ValueError('Empty chunk')
            if raw_bytes != expected_bytes or raw_hash.hexdigest() != expected_sha256:
                raise ValueError('Raw byte/hash receipt mismatch')
            if not unchanged(raw_path, input_stat):
                raise ValueError('Input changed during reduction')
            del core_riscs
            _, raw, _ = read_raw(scratch / 'normalized.csv')
            for identity, surrogate in identities.items():
                op = raw.get(surrogate)
                record = dict(kind='program', global_call_id=identity[0], device=identity[1],
                              trace=identity[2] or None, replay=identity[3] or None)
                # The accepted span reducer only visits RISCs with endpoints. Preserve
                # totals from entirely absent RISCs separately, without inventing
                # residency or silently assigning them to another thread.
                present_riscs = {r for c, r in op['kernels']} if op else set()
                unplaced = [dict(core=list(c), risc=r, zone=z, cycles=v)
                            for (c,r,z),v in (op['sums'].items() if op else [])
                            if r not in present_riscs]
                record['unplaced_zone_sums'] = unplaced
                unplaced_count += len(unplaced)
                unplaced_programs += bool(unplaced)
                if not op or not op['kernels']:
                    unsupported_programs += 1
                    out.put(dict(record, status='unsupported_no_kernel_endpoints', summary=None, cores=[]))
                    continue
                summary = span(op)  # The accepted calibration/census arithmetic.
                cores = []
                per_core = collections.defaultdict(dict)
                for (core, risc, zone), value in op['sums'].items():
                    per_core[core, risc][zone] = value
                for (core, risc), ends in sorted(op['kernels'].items()):
                    zones = per_core.get((core, risc), {})
                    resident = ends['ZONE_END'] - ends['ZONE_START']
                    if sum(zones.values()) > resident:
                        raise ValueError('Zones exceed own core/RISC residency')
                    cores.append(dict(core=list(core), risc=risc, start_cycle=ends['ZONE_START'],
                                      end_cycle=ends['ZONE_END'], resident_cycles=resident,
                                      zone_cycles=zones, unclassified_cycles=resident-sum(zones.values())))
                out.put(dict(record, status='reduced_with_unplaced_sums' if unplaced else 'reduced', summary=summary, cores=cores))
            footer = dict(kind='complete', chunk_id=chunk_id, counter_scope=counter_scope,
                          raw_sha256=raw_hash.hexdigest(), raw_bytes=raw_bytes, raw_rows=rows,
                          stored_input_sha256=file_sha256(raw_path), stored_input_bytes=input_stat.st_size,
                          reducer_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in
                              (Path(__file__), ROOT / 'perf/c10_dm_control/analyze.py',
                               ROOT / 'perf/c10_burst_census/reduce_census.py')},
                          programs=len(identities), records=out.records,
                          programs_without_endpoints=unsupported_programs,
                          programs_with_unplaced_sums=unplaced_programs, unplaced_sum_markers=unplaced_count,
                          payload_sha256=out.hash.hexdigest(), payload_bytes=out.bytes,
                          call_transitions=transitions, noncontiguous_returns=returns,
                          counter_ranges=[dict(device=d, trace=t or None, replay=p or None,
                                               min_call_id=v[0], max_call_id=v[1],
                                               min_tick=v[2], max_tick=v[3])
                                          for (d,t,p),v in sorted(ranges.items())],
                          marker_coverage=[dict(risc=r, zone=z, type=k, rows=n,
                                                arithmetic=('kernel_endpoint' if z.endswith('-KERNEL') and r in RISCS else
                                                            'zone_sum' if k == 'ZONE_TOTAL' and r in RISCS else 'unreduced'),
                                                known_zone=z in (*ZONES, 'CB-COMPUTE-WAIT-FRONT', 'CB-COMPUTE-RESERVE-BACK'))
                                           for (r,z,k),n in sorted(coverage.items())],
                          peak_rss_kib=peak_rss_kib(),
                          rss_basis='Linux post-exec VmHWM, KiB',
                          rusage_maxrss_including_preexec_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if not unchanged(raw_path, input_stat):
                raise ValueError('Input changed during provenance hashing')
            out.put(footer, footer=True)
            out.close()
            verify_chunk(out.path, max_output_bytes=limits['max_output_bytes'])
            with out.path.open('rb') as f:
                os.fsync(f.fileno())
            # Atomic no-overwrite publication, including concurrent chunk-ID collision.
            os.link(out.path, final)
            fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            return dict(path=str(final.resolve()), raw_rows=rows, programs=len(identities),
                        raw_bytes=raw_bytes, output_bytes=out.bytes,
                        archive_bytes=final.stat().st_size, peak_rss_kib=peak_rss_kib())
        finally:
            try:
                out.file.close()
            finally:
                out.stored.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('raw_path', type=Path)
    ap.add_argument('output_dir', type=Path)
    ap.add_argument('--chunk-id', required=True)
    ap.add_argument('--counter-scope', required=True)
    ap.add_argument('--expected-bytes', type=int, required=True)
    ap.add_argument('--expected-sha256', required=True)
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
        print(f'cycle-stream STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
