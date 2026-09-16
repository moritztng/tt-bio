"""CPU archive controls, A/A reducer comparison and process resource measurement.

No device is opened and no clock is sampled. Repeating an archive is a repeated CPU
reduction of bytes captured earlier at a during-window sampled 1350 MHz, not a new
hardware execution.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import resource
import shutil
import statistics
import time

import fast
from fast import ROOT, RISCS, file_sha256, read_raw, span, verify_chunk
from analyze import reduce_op
from reduce_census import union_cycles

CENSUS = ROOT / 'perf/c10_burst_census/runs/census1'
CALIBRATION = ROOT / 'perf/c10_dm_control'
STREAM_ARCHIVES = Path('/home/moritz/.coworker/artifacts/c10-cycle-stream/verified')
FULL_DRAIN_BYTES = 80_895_977_723
LARGEST_DISCARDED_CHUNK_BYTES = 377_218_386
CPU_BUDGET_S = 20 * 60


def receipt(path):
    h = hashlib.sha256()
    size = 0
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            size += len(block)
            h.update(block)
    return size, h.hexdigest()


def subtree_cpu():
    """CPU charged to reaped descendants. append_chunk's child reaps the native one,
    so on its own exit the whole subtree lands here."""
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def run(path, out, name, scope, native=True):
    size, sha = receipt(path)
    before = subtree_cpu()
    start = time.monotonic()
    result = fast.append_chunk(path, out, chunk_id=name, counter_scope=scope,
                               expected_bytes=size, expected_sha256=sha, native=native)
    wall = time.monotonic() - start
    result['cpu_replay_wall_s'] = wall
    result['MB_per_s'] = result['raw_bytes'] / wall / 1e6
    result['subtree_cpu_s'] = subtree_cpu() - before
    result['raw_sha256'] = sha
    result['output_sha256'] = file_sha256(result['path'])
    footer = verify_chunk(result['path'])
    result['noncontiguous_returns'] = footer['noncontiguous_returns']
    result['call_transitions'] = footer['call_transitions']
    return result, footer


def records(path, kind=None):
    with gzip.open(path, 'rt') as f:
        for line in f:
            r = json.loads(line)
            if kind is None or r['kind'] == kind:
                yield r


def compare_accepted(path, result, published_ops=None, calibration_rows=None):
    """Reproduce the accepted read_raw/span reduction of the same input, live."""
    header, raw, zones = read_raw(path)
    seen = set()
    unplaced = verbatim = 0
    firmware = {}
    for r in records(result['path']):
        if r['kind'] == 'unreduced_marker':
            verbatim += 1
            continue
        if r['kind'] != 'program':
            continue
        i = r['global_call_id']
        assert i not in seen, (path, i, 'duplicate program record')
        seen.add(i)
        unplaced += len(r['unplaced_zone_sums'])
        present = {risc for c, risc in raw[i]['kernels']}
        assert r['unplaced_zone_sums'] == [dict(core=list(c), risc=risc, zone=z, cycles=v)
                                           for (c, risc, z), v in raw[i]['sums'].items()
                                           if risc not in present], (path, i, 'unplaced sums')
        assert r['summary'] == span(raw[i]), (path, i, 'calibrated span')
        assert r['device'] == 0 and r['trace'] is None and r['replay'] is None
        cores = {(tuple(c['core']), c['risc']): c for c in r['cores']}
        assert cores.keys() == raw[i]['kernels'].keys(), (path, i, 'core/RISC set')
        sums = collections.defaultdict(dict)
        for (c, risc, z), v in raw[i]['sums'].items():
            sums[c, risc][z] = v
        for key, ends in raw[i]['kernels'].items():
            assert cores[key]['start_cycle'] == ends['ZONE_START']
            assert cores[key]['end_cycle'] == ends['ZONE_END']
            assert cores[key]['zone_cycles'] == sums.get(key, {})
            assert (cores[key]['resident_cycles']
                    == ends['ZONE_END'] - ends['ZONE_START'])
            assert (cores[key]['unclassified_cycles'] == cores[key]['resident_cycles']
                    - sum(cores[key]['zone_cycles'].values()))
            firmware[i, key[0][0], key[0][1], key[1]] = (cores[key]['firmware_start_cycle'],
                                                         cores[key]['firmware_end_cycle'])
        for c in r['firmware_only_cores']:
            firmware[i, c['core'][0], c['core'][1], c['risc']] = (c['firmware_start_cycle'],
                                                                  c['firmware_end_cycle'])
        if published_ops and i in published_ops:
            accepted = published_ops[i]
            assert all(r['summary'][k] == accepted[k] for k in r['summary']), (path, i, 'published census')
        if calibration_rows:
            accepted = reduce_op(calibration_rows[i], raw[i])
            assert all(r['summary'][k] == accepted[k] for k in ('start_cycle', 'end_cycle', 'span_cycles'))
            for risc, t in accepted['threads'].items():
                observed = r['summary']['threads'][risc]
                assert observed['resident_core_cycles'] == t['resident_core_cycles_sum']
                assert observed['unclassified_core_cycles'] == t['unclassified_core_cycles_sum']
                assert all(observed['zone_core_cycles'].get(z, 0) == v
                           for z, v in t['zone_core_cycles_sum'].items())
    assert seen == raw.keys(), (path, 'program identity set')
    footer = verify_chunk(result['path'])
    assert footer['header'] == header, (path, 'profiler header provenance')
    actual = {(m['risc'], m['zone']): m['rows'] for m in footer['marker_coverage']
              if m['type'] == 'ZONE_TOTAL' and m['risc'] in RISCS}
    assert actual == zones, (path, 'ZONE_TOTAL census')
    assert sum(m['rows'] for m in footer['marker_coverage']) == footer['raw_rows']
    assert sum(m['rows'] for m in footer['marker_coverage']
               if m['retention'] == 'verbatim_record') == verbatim
    return dict(programs=len(seen), unplaced_zone_sums_preserved=unplaced,
                verbatim_records=verbatim,
                accepted_span_count_zone_core_endpoint_match=True), firmware


def compare_stream_child(name, result, firmware):
    """Every record and every verbatim row the stream child published must survive.

    The stream child kept each firmware bracket row as a verbatim JSON copy. Here the
    same row's tick has to reappear as the firmware endpoint on its core entry, and
    the row's remaining five columns as its marker combo's constant_fields.
    """
    reference = STREAM_ARCHIVES / (name + '.jsonl.gz')
    if not reference.exists():
        return None
    footer = verify_chunk(result['path'])
    constants = {(m['risc'], m['zone'], m['type']): m for m in footer['marker_coverage']}
    mine = {}
    for r in records(result['path'], 'program'):
        mine[r['global_call_id']] = r
    programs = markers = 0
    for r in records(reference):
        if r['kind'] == 'program':
            programs += 1
            m = mine[r['global_call_id']]
            for field in ('device', 'trace', 'replay', 'status', 'summary', 'unplaced_zone_sums'):
                assert m[field] == r[field], (name, r['global_call_id'], field)
            assert len(m['cores']) == len(r['cores'])
            for a, b in zip(m['cores'], r['cores']):
                for field in ('core', 'risc', 'start_cycle', 'end_cycle', 'resident_cycles',
                              'zone_cycles', 'unclassified_cycles'):
                    assert a[field] == b[field], (name, r['global_call_id'], field)
        elif r['kind'] == 'unreduced_marker':
            markers += 1
            v = r['values']
            risc, zone, kind = v[3], v[10], v[11]
            combo = constants[risc, zone, kind]
            assert combo['retention'] == 'firmware_endpoints', (name, risc, zone, kind)
            key = (int(v[7]), int(v[1]), int(v[2]), risc)
            start, end = firmware[key]
            assert int(v[5]) == (start if kind == 'ZONE_START' else end), (name, key, kind)
            assert combo['constant_fields'] == {
                'timer_id': v[4], 'data': v[6], 'source line': v[12],
                'source file': v[13], 'meta data': v[14]}, (name, risc, zone, kind)
    assert programs == len(mine), (name, 'program count')
    return dict(reference=str(reference), reference_programs=programs,
                reference_verbatim_rows_reconstructed=markers,
                reference_archive_bytes=reference.stat().st_size)


def compare_reducers(path, out, name):
    """A/A: the native reducer and the pure-Python reference on identical bytes."""
    size, sha = receipt(path)
    results = {}
    for native in (True, False):
        tag = f'{name}-aa-{"native" if native else "python"}'
        start = time.monotonic()
        r = fast.append_chunk(path, out, chunk_id=tag, counter_scope='archived-aa',
                              expected_bytes=size, expected_sha256=sha, native=native)
        r['cpu_replay_wall_s'] = time.monotonic() - start
        r['MB_per_s'] = r['raw_bytes'] / r['cpu_replay_wall_s'] / 1e6
        results['native' if native else 'python'] = r
    ordered = {}
    for tag, r in results.items():
        by_id = {}
        for rec in records(r['path'], 'program'):
            by_id[rec['global_call_id']] = rec
        ordered[tag] = by_id
    assert ordered['native'].keys() == ordered['python'].keys(), (name, 'A/A identity set')
    for i, a in ordered['native'].items():
        b = ordered['python'][i]
        for field in ('device', 'trace', 'replay', 'status', 'summary', 'unplaced_zone_sums',
                      'firmware_only_cores', 'cores'):
            assert a[field] == b[field], (name, i, 'A/A ' + field)
    fa, fb = verify_chunk(results['native']['path']), verify_chunk(results['python']['path'])
    for field in ('raw_sha256', 'raw_bytes', 'raw_rows', 'programs', 'records',
                  'programs_without_endpoints', 'programs_with_unplaced_sums',
                  'unplaced_sum_markers', 'call_transitions', 'noncontiguous_returns',
                  'firmware_bracket_pairs', 'firmware_bracket_conflicts',
                  'firmware_rows_retained_verbatim', 'counter_ranges', 'marker_coverage',
                  'header'):
        assert fa[field] == fb[field], (name, 'A/A footer ' + field)
    return dict(identical_program_records=len(ordered['native']),
                identical_footer_fields=True,
                native=dict(wall_s=results['native']['cpu_replay_wall_s'],
                            MB_per_s=results['native']['MB_per_s'],
                            peak_rss_kib=results['native']['peak_rss_kib'],
                            native_peak_rss_kib=results['native']['native_peak_rss_kib'],
                            archive_bytes=results['native']['archive_bytes']),
                python=dict(wall_s=results['python']['cpu_replay_wall_s'],
                            MB_per_s=results['python']['MB_per_s'],
                            peak_rss_kib=results['python']['peak_rss_kib'],
                            archive_bytes=results['python']['archive_bytes']))


def sustained(path, out, reps, native=True):
    """Interleaved repeats of the same archived bytes: a CPU-resource control only."""
    size, sha = receipt(path)
    runs = []
    for i in range(reps):
        tag = f'sustained-{"native" if native else "python"}-{i}'
        before = subtree_cpu()
        start = time.monotonic()
        r = fast.append_chunk(path, out, chunk_id=tag, counter_scope='archived-census1',
                              expected_bytes=size, expected_sha256=sha, native=native)
        wall = time.monotonic() - start
        cpu = subtree_cpu() - before
        assert verify_chunk(r['path'])['raw_sha256'] == sha
        runs.append(dict(repeat=i, wall_s=wall, MB_per_s=r['raw_bytes'] / wall / 1e6,
                         subtree_cpu_s=cpu,
                         peak_rss_kib=r['peak_rss_kib'],
                         native_peak_rss_kib=r.get('native_peak_rss_kib'),
                         archive_bytes=r['archive_bytes'], output_bytes=r['output_bytes'],
                         raw_bytes=r['raw_bytes'], raw_rows=r['raw_rows']))
        print(json.dumps(runs[-1]), flush=True)
    return runs


def projection(runs, label):
    """Full-workload CPU projection. A planning number, never a hardware measurement."""
    walls = [r['wall_s'] for r in runs]
    cpus = [r['subtree_cpu_s'] for r in runs]
    raw = runs[0]['raw_bytes']
    if min(cpus) <= 0:
        raise ValueError('subtree CPU accounting returned nothing; refusing to project')
    return dict(
        label=label, repeats=len(runs), measured_raw_bytes=raw, measured_rows=runs[0]['raw_rows'],
        wall_s=dict(min=min(walls), median=statistics.median(walls), max=max(walls)),
        subtree_cpu_s=dict(min=min(cpus), median=statistics.median(cpus), max=max(cpus)),
        MB_per_s_from_slowest_wall=raw / max(walls) / 1e6,
        MB_per_s_from_slowest_serial_cpu=raw / max(cpus) / 1e6,
        projected_full_drain_wall_s=FULL_DRAIN_BYTES / (raw / max(walls)),
        projected_full_drain_serial_cpu_s=FULL_DRAIN_BYTES / (raw / max(cpus)),
        projected_full_drain_archive_bytes=round(FULL_DRAIN_BYTES * runs[0]['archive_bytes'] / raw),
        projected_full_drain_payload_bytes=round(FULL_DRAIN_BYTES * runs[0]['output_bytes'] / raw),
        within_20_minute_budget=FULL_DRAIN_BYTES / (raw / max(cpus)) <= CPU_BUDGET_S,
        basis=('Scaled linearly from repeated CPU reduction of archived bytes on pc. The serial '
               'CPU figure charges the whole process subtree, so it holds without a second idle '
               'core; the wall figure keeps the Python framing and native reduction overlapped. '
               'This is a CPU planning projection, not a full-run or device latency measurement, '
               'and it assumes the marker mix of the retained windows.'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', required=True, type=Path)
    ap.add_argument('--report', required=True, type=Path)
    ap.add_argument('--sustained-repeats', type=int, default=4)
    ap.add_argument('--python-repeats', type=int, default=2)
    args = ap.parse_args()
    if args.sustained_repeats < 2:
        ap.error('sustained repeats must be at least 2')
    fast.native_binary()   # keep compilation out of every measurement below
    accepted = json.loads((CENSUS / 'analysis.json').read_text())
    cap = json.loads((CENSUS / 'out/capture.json').read_text())
    drains = {d['label']: d for d in cap['drains'] if d['retained']}
    report = dict(
        scope='CPU replay of archives captured at sampled 1350 MHz; no new hardware measurement',
        predicted_model_cycles_saved=0, measured_model_cycles_saved=0,
        census=[], calibration=None, aa=None, sustained=None,
        largest_discarded_chunk_bytes=LARGEST_DISCARDED_CHUNK_BYTES,
        limits=('Retained inputs and repeats cannot establish the discarded largest chunk marker '
                'mix, resource demand, completeness or successful reduction, and say nothing about '
                'capture overhead on hardware. No whole-fold budget, roof, ceiling or speedup.'))
    for w in accepted['windows']:
        name = w['case']
        p = CENSUS / 'out/windows' / (name + '.csv.gz')
        result, footer = run(p, args.output_dir, name, 'archived-census1')
        assert footer['raw_sha256'] == drains[name]['sha256']
        assert footer['raw_bytes'] == drains[name]['bytes']
        check, firmware = compare_accepted(p, result, {o['global_call_id']: o for o in w['ops']})
        window = next(x for x in cap['windows'] if x['case'] == name)
        expected_ids = set(range(window['device_counter_before'] - 3,
                                 window['device_counter_after'] + 3))
        ids, spans = [], []
        inner = {o['global_call_id'] for o in w['ops']}
        for r in records(result['path'], 'program'):
            ids.append(r['global_call_id'])
            if r['global_call_id'] in inner:
                spans.append(r['summary'])
        assert set(ids) == {i << 10 for i in expected_ids}, (name, 'identity window')
        assert len(spans) == w['invocations']
        assert sum(s['span_cycles'] for s in spans) == w['program_span_cycles_sum']
        assert union_cycles(spans) == w['program_union_cycles']
        result.update(check, accepted_inner_programs=w['invocations'],
                      accepted_inner_span_sum=w['program_span_cycles_sum'],
                      stream_child=compare_stream_child(name, result, firmware))
        report['census'].append(result)
        print(json.dumps(dict(case=name, MB_per_s=result['MB_per_s'],
                              raw_bytes=result['raw_bytes'], programs=result['programs'],
                              verbatim=result['verbatim_records'],
                              stream_child=result['stream_child'] and
                              result['stream_child']['reference_verbatim_rows_reconstructed'])),
              flush=True)
    p = CALIBRATION / 'raw/profile_log_device.csv.gz'
    result, footer = run(p, args.output_dir, 'calibration', 'archived-calibration')
    with (CALIBRATION / 'raw/ops.csv').open() as f:
        rows = {int(r['GLOBAL CALL COUNT']): r for r in csv.DictReader(f)}
    check, firmware = compare_accepted(p, result, calibration_rows=rows)
    result.update(check, stream_child=compare_stream_child('calibration', result, firmware))
    cal = json.loads((CALIBRATION / 'analysis.json').read_text())
    _, raw, _ = read_raw(p)
    for interval in cal['intervals']:
        for op in interval['ops']:
            assert reduce_op(rows[op['call_id']], raw[op['call_id']]) == op
    result['published_intervals'] = len(cal['intervals'])
    result['published_interval_ops'] = sum(len(i['ops']) for i in cal['intervals'])
    report['calibration'] = result
    del raw
    print(json.dumps(dict(case='calibration', MB_per_s=result['MB_per_s'],
                          intervals=result['published_intervals'],
                          interval_ops=result['published_interval_ops'])), flush=True)

    largest = max(report['census'], key=lambda r: r['raw_bytes'])
    name = Path(largest['path']).name.removesuffix('.jsonl.gz')
    archive = CENSUS / 'out/windows' / (name + '.csv.gz')
    report['aa'] = compare_reducers(archive, args.output_dir, name)
    print(json.dumps(report['aa'] | dict(case=name)), flush=True)
    native = sustained(archive, args.output_dir, args.sustained_repeats, native=True)
    python = sustained(archive, args.output_dir, args.python_repeats, native=False)
    report['sustained'] = dict(
        input=str(archive), synthetic_scope=('Same archived bytes and identities reduced again; '
                                             'zero new device observations'),
        native=native, python=python,
        native_projection=projection(native, 'native reducer'),
        python_projection=projection(python, 'pure-Python reference reducer'))
    budget = report['sustained']['native_projection']
    report['verdict'] = ('GO_CPU_PREREQUISITE_ONLY' if budget['within_20_minute_budget']
                         else 'STOP_CPU_BUDGET_NOT_MET')
    usage = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    report['harness_resources'] = dict(
        harness_peak_rss_kib=usage.ru_maxrss, child_peak_rss_kib=child.ru_maxrss,
        published_archive_bytes=sum(f.stat().st_size for f in args.output_dir.glob('*.jsonl.gz')),
        output_dir=str(args.output_dir.resolve()),
        free_bytes_on_output_filesystem=shutil.disk_usage(args.output_dir).free)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(verdict=report['verdict'], **{k: budget[k] for k in (
        'MB_per_s_from_slowest_wall', 'MB_per_s_from_slowest_serial_cpu',
        'projected_full_drain_wall_s', 'projected_full_drain_serial_cpu_s',
        'within_20_minute_budget')})), flush=True)
    print(args.report, flush=True)


if __name__ == '__main__':
    main()
