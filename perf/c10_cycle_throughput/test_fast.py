"""CPU tests for the throughput reducer. No device, no TT import, no network."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import fast
from fast import DEFAULTS, FIELDS, verify_chunk

HEADER = 'ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 120'
FW = {'BRISC': 'BRISC-FW', 'NCRISC': 'NCRISC-FW', 'TRISC_0': 'TRISC-FW',
      'TRISC_1': 'TRISC-FW', 'TRISC_2': 'TRISC-FW'}
KERNEL = {'BRISC': 'BRISC-KERNEL', 'NCRISC': 'NCRISC-KERNEL', 'TRISC_0': 'TRISC-KERNEL',
          'TRISC_1': 'TRISC-KERNEL', 'TRISC_2': 'TRISC-KERNEL'}


def row(call, risc, zone, kind, tick, *, x=1, y=1, data=0, device=0, trace='', replay='',
        timer=100, line=7, filename='/src/f.h', meta=''):
    return [str(device), str(x), str(y), risc, str(timer), str(tick), str(data), str(call),
            trace, replay, zone, kind, str(line), filename, meta]


def program(call, *, riscs=('BRISC',), cores=((1, 1),), start=1000, span=500, zones=(),
            firmware=True, **kw):
    out = []
    for x, y in cores:
        for risc in riscs:
            if firmware:
                out.append(row(call, risc, FW[risc], 'ZONE_START', start - 10, x=x, y=y,
                               timer=900, line=42, filename='/fw.cc', **kw))
            out.append(row(call, risc, KERNEL[risc], 'ZONE_START', start, x=x, y=y, **kw))
            for zone, value in zones:
                out.append(row(call, risc, zone, 'ZONE_TOTAL', start + 1, x=x, y=y, data=value, **kw))
            out.append(row(call, risc, KERNEL[risc], 'ZONE_END', start + span, x=x, y=y, **kw))
            if firmware:
                out.append(row(call, risc, FW[risc], 'ZONE_END', start + span + 10, x=x, y=y,
                               timer=900, line=42, filename='/fw.cc', **kw))
    return out


def write(path, rows, *, header=HEADER, columns=FIELDS, trailer='\n', gz=True):
    body = header + '\n' + ', '.join(columns) + '\n'
    body += ''.join(', '.join(r) + '\n' for r in rows)
    if trailer != '\n':
        body = body[:-1] + trailer
    data = body.encode()
    path.write_bytes(gzip.compress(data, 1) if gz else data)
    return len(data), hashlib.sha256(data).hexdigest()


def reduce(tmp, rows, *, name='c', native=True, limits=None, compact_firmware=True, **kw):
    raw = tmp / ('raw.csv.gz' if kw.get('gz', True) else 'raw.csv')
    size, sha = write(raw, rows, **kw)
    return fast.reduce_chunk(raw, tmp / 'out', chunk_id=name, counter_scope='test',
                             expected_bytes=size, expected_sha256=sha,
                             limits={**DEFAULTS, **(limits or {})},
                             compact_firmware=compact_firmware, native=native)


def payload(result, kind=None):
    with gzip.open(result['path'], 'rt') as f:
        return [r for r in map(json.loads, f) if kind is None or r['kind'] == kind]


def native_only(tmp, rows, **kw):
    """Run reduce.c alone; None means it refused the input shape."""
    raw = tmp / 'raw.csv.gz'
    size, sha = write(raw, rows, **kw)
    return fast.reduce_chunk_native(raw, tmp / 'out', chunk_id='c', counter_scope='t',
                                    expected_bytes=size, expected_sha256=sha,
                                    limits=dict(DEFAULTS))


@pytest.fixture
def tmp(tmp_path):
    (tmp_path / 'n').mkdir()
    (tmp_path / 'p').mkdir()
    return tmp_path


# ------------------------------------------------------------------ agreement
def run_pair(tmp, rows, **kw):
    a = reduce(tmp / 'n', rows, native=True, **kw)
    b = reduce(tmp / 'p', rows, native=False, **kw)
    assert a['reducer'] == 'native' and b['reducer'] == 'python'
    pa, pb = payload(a, 'program'), payload(b, 'program')
    assert pa == pb
    fa, fb = verify_chunk(a['path']), verify_chunk(b['path'])
    for field in ('raw_sha256', 'raw_bytes', 'raw_rows', 'programs', 'records', 'header',
                  'programs_without_endpoints', 'programs_with_unplaced_sums',
                  'unplaced_sum_markers', 'call_transitions', 'noncontiguous_returns',
                  'firmware_bracket_pairs', 'firmware_bracket_conflicts',
                  'firmware_rows_retained_verbatim', 'counter_ranges', 'marker_coverage'):
        assert fa[field] == fb[field], field
    assert [r for r in payload(a) if r['kind'] == 'unreduced_marker'] == \
           [r for r in payload(b) if r['kind'] == 'unreduced_marker']
    return a, fa


def test_reducers_agree_on_a_plain_program(tmp):
    a, footer = run_pair(tmp, program(64, zones=(('DM-SEM-WAIT', 40),)))
    (prog,) = payload(a, 'program')
    assert prog['global_call_id'] == 64 and prog['status'] == 'reduced'
    assert prog['summary']['span_cycles'] == 500
    assert prog['cores'][0]['zone_cycles'] == {'DM-SEM-WAIT': 40}
    assert prog['cores'][0]['unclassified_cycles'] == 460
    assert footer['firmware_bracket_pairs'] == 1


def test_firmware_bracket_is_retained_as_integer_endpoints(tmp):
    a, footer = run_pair(tmp, program(1, riscs=('BRISC', 'NCRISC')))
    (prog,) = payload(a, 'program')
    assert all(c['firmware_start_cycle'] == 990 and c['firmware_end_cycle'] == 1510
               for c in prog['cores'])
    assert not payload(a, 'unreduced_marker')
    combo = [m for m in footer['marker_coverage'] if m['zone'].endswith('-FW')]
    assert combo and all(m['retention'] == 'firmware_endpoints' for m in combo)
    assert all(m['constant_fields'] == {'timer_id': '900', 'data': '0', 'source line': '42',
                                        'source file': '/fw.cc', 'meta data': ''} for m in combo)
    assert all(m['arithmetic'] == 'unreduced' for m in combo)


def test_no_compaction_keeps_every_firmware_row_verbatim(tmp):
    a, footer = run_pair(tmp, program(1, riscs=('BRISC', 'NCRISC')), compact_firmware=False)
    assert len(payload(a, 'unreduced_marker')) == 4
    assert all(c['firmware_start_cycle'] is None for c in payload(a, 'program')[0]['cores'])
    assert footer['firmware_bracket_pairs'] == 0


def test_firmware_payload_deviation_is_retained_whole(tmp):
    rows = program(1, riscs=('BRISC', 'NCRISC'))
    rows.append(row(1, 'NCRISC', 'NCRISC-FW', 'ZONE_START', 900, x=2, y=2, timer=900, line=42,
                    filename='/fw.cc', meta='SOMETHING'))
    a, footer = run_pair(tmp, rows)
    assert footer['firmware_rows_retained_verbatim'] == 1
    (kept,) = payload(a, 'unreduced_marker')
    assert kept['values'][14] == 'SOMETHING'
    combos = {m['type']: m for m in footer['marker_coverage'] if m['zone'] == 'NCRISC-FW'}
    assert combos['ZONE_START']['constant_fields'] is None
    assert combos['ZONE_END']['constant_fields']['source file'] == '/fw.cc'
    # placed as an endpoint as well, so the tick is never only in the verbatim copy
    assert payload(a, 'program')[0]['firmware_only_cores'][0]['firmware_start_cycle'] == 900


def test_duplicate_firmware_bracket_keeps_the_second_row_whole(tmp):
    rows = program(1)
    rows.append(row(1, 'BRISC', 'BRISC-FW', 'ZONE_START', 991, timer=900, line=42, filename='/fw.cc'))
    a, footer = run_pair(tmp, rows)
    assert footer['firmware_bracket_conflicts'] == 1
    assert payload(a, 'unreduced_marker')[0]['values'][5] == '991'
    assert payload(a, 'program')[0]['cores'][0]['firmware_start_cycle'] == 990


def test_unrecognized_marker_stays_verbatim(tmp):
    rows = program(1) + [row(1, 'BRISC', 'SOME-NEW-ZONE', 'ZONE_START', 1200)]
    a, footer = run_pair(tmp, rows)
    (kept,) = payload(a, 'unreduced_marker')
    assert kept['values'][10] == 'SOME-NEW-ZONE' and kept['row'] == len(rows)
    combo = next(m for m in footer['marker_coverage'] if m['zone'] == 'SOME-NEW-ZONE')
    assert combo['retention'] == 'verbatim_record' and combo['arithmetic'] == 'unreduced'
    assert not combo['known_zone']


def test_non_risc_rows_stay_verbatim(tmp):
    rows = program(1) + [row(1, 'ERISC', 'ERISC-KERNEL', 'ZONE_START', 1200)]
    a, _ = run_pair(tmp, rows)
    assert payload(a, 'unreduced_marker')[0]['values'][3] == 'ERISC'


def test_exact_integers_above_2_to_the_53(tmp):
    big = 2**53 + 12345
    a, footer = run_pair(tmp, program(2**53 + 7, start=big, span=2**53))
    (prog,) = payload(a, 'program')
    assert prog['global_call_id'] == 2**53 + 7
    assert prog['summary']['start_cycle'] == big
    assert prog['summary']['span_cycles'] == 2**53
    assert footer['counter_ranges'][0]['max_call_id'] == 2**53 + 7


def test_ids_recur_across_cores_and_do_not_close_a_program(tmp):
    rows = (program(10, cores=((1, 1),)) + program(11) + program(10, cores=((2, 2),), start=4000))
    a, footer = run_pair(tmp, rows)
    by_id = {p['global_call_id']: p for p in payload(a, 'program')}
    assert len(by_id[10]['cores']) == 2
    assert footer['noncontiguous_returns'] == 1 and footer['call_transitions'] == 3


def test_mixed_device_trace_and_replay_identities(tmp):
    rows = (program(5) + program(5, device=1) + program(5, trace='3', replay='9'))
    a, footer = run_pair(tmp, rows)
    ids = {(p['global_call_id'], p['device'], p['trace'], p['replay']) for p in payload(a, 'program')}
    assert ids == {(5, 0, None, None), (5, 1, None, None), (5, 0, '3', '9')}
    assert len(footer['counter_ranges']) == 3


def test_unplaced_sums_keep_unknown_residency(tmp):
    rows = program(1) + [row(1, 'TRISC_2', 'CB-COMPUTE-RESERVE-BACK', 'ZONE_TOTAL', 1100, data=77)]
    a, footer = run_pair(tmp, rows)
    (prog,) = payload(a, 'program')
    assert prog['status'] == 'reduced_with_unplaced_sums'
    assert prog['unplaced_zone_sums'] == [
        dict(core=[1, 1], risc='TRISC_2', zone='CB-COMPUTE-RESERVE-BACK', cycles=77)]
    assert 'TRISC_2' not in prog['summary']['threads']
    assert footer['unplaced_sum_markers'] == 1 and footer['programs_with_unplaced_sums'] == 1


def test_program_without_kernel_endpoints_is_unsupported_not_zero(tmp):
    rows = [row(3, 'BRISC', 'DM-SEM-WAIT', 'ZONE_TOTAL', 10, data=5)]
    a, footer = run_pair(tmp, rows)
    (prog,) = payload(a, 'program')
    assert prog['status'] == 'unsupported_no_kernel_endpoints'
    assert prog['summary'] is None and prog['cores'] == []
    assert prog['unplaced_zone_sums'][0]['cycles'] == 5
    assert footer['programs_without_endpoints'] == 1


def test_firmware_only_core_is_kept_separately(tmp):
    rows = program(1) + [row(1, 'NCRISC', 'NCRISC-FW', 'ZONE_START', 800, x=9, y=9,
                             timer=900, line=42, filename='/fw.cc')]
    a, _ = run_pair(tmp, rows)
    (prog,) = payload(a, 'program')
    assert prog['firmware_only_cores'] == [
        dict(core=[9, 9], risc='NCRISC', firmware_start_cycle=800, firmware_end_cycle=None)]


def test_counter_ranges_and_marker_census_cover_every_row(tmp):
    rows = program(7, riscs=('BRISC', 'NCRISC'), zones=(('DM-SEM-WAIT', 3),))
    a, footer = run_pair(tmp, rows)
    assert sum(m['rows'] for m in footer['marker_coverage']) == footer['raw_rows'] == len(rows)
    (r,) = footer['counter_ranges']
    assert (r['min_call_id'], r['max_call_id'], r['min_tick'], r['max_tick']) == (7, 7, 990, 1510)


# ------------------------------------------------------------------- refusals
def expect_stop(tmp, rows, message, **kw):
    for native in (True, False):
        with pytest.raises((ValueError, KeyError, AssertionError)) as exc:
            reduce(tmp / ('n' if native else 'p'), rows, native=native, **kw)
        assert message.lower() in str(exc.value).lower(), (native, exc.value)


def test_malformed_row_is_refused(tmp):
    rows = program(1)
    rows.append(row(1, 'BRISC', 'BRISC-KERNEL', 'ZONE_START', 5)[:-1])
    expect_stop(tmp, rows, 'malformed csv row')


def test_truncated_final_line_is_refused(tmp):
    expect_stop(tmp, program(1), 'truncated line', trailer='')


def test_duplicate_kernel_marker_is_refused(tmp):
    rows = program(1) + [row(1, 'BRISC', 'BRISC-KERNEL', 'ZONE_START', 1001)]
    expect_stop(tmp, rows, 'duplicate/invalid kernel marker')


def test_kernel_zone_with_a_total_type_is_refused(tmp):
    rows = program(1) + [row(1, 'NCRISC', 'NCRISC-KERNEL', 'ZONE_TOTAL', 1001)]
    expect_stop(tmp, rows, 'duplicate/invalid kernel marker')


def test_duplicate_sum_marker_is_refused(tmp):
    rows = program(1, zones=(('DM-SEM-WAIT', 5),)) + [
        row(1, 'BRISC', 'DM-SEM-WAIT', 'ZONE_TOTAL', 1002, data=6)]
    expect_stop(tmp, rows, 'duplicate sum marker')


def test_reversed_endpoints_are_refused(tmp):
    expect_stop(tmp, [row(1, 'BRISC', 'BRISC-KERNEL', 'ZONE_START', 900),
                      row(1, 'BRISC', 'BRISC-KERNEL', 'ZONE_END', 800)],
                'missing/reversed endpoints')


def test_zone_sums_above_own_residency_are_refused(tmp):
    expect_stop(tmp, program(1, span=10, zones=(('DM-SEM-WAIT', 99),)), 'residency')


def test_sum_without_own_risc_endpoints_is_refused(tmp):
    rows = program(1, cores=((1, 1),)) + [
        row(1, 'BRISC', 'DM-SEM-WAIT', 'ZONE_TOTAL', 1005, x=4, y=4, data=3)]
    expect_stop(tmp, rows, 'sum without own risc endpoints')


def test_incomplete_trace_identity_is_refused(tmp):
    expect_stop(tmp, program(1, trace='4'), 'incomplete trace/replay identity')


def test_non_integer_field_is_refused(tmp):
    rows = program(1) + [row(1, 'BRISC', 'BRISC-KERNEL', 'ZONE_START', 'x')]
    expect_stop(tmp, rows, 'not an integer')


def test_bad_header_is_refused(tmp):
    expect_stop(tmp, program(1), 'unsupported profiler header',
                header='ARCH: blackhole, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 120')


def test_unexpected_columns_are_refused(tmp):
    expect_stop(tmp, program(1), 'unexpected csv columns', columns=FIELDS[:-1] + ['other'])


def test_empty_chunk_is_refused(tmp):
    expect_stop(tmp, [], 'empty chunk')


def test_receipt_mismatch_is_refused(tmp):
    raw = tmp / 'raw.csv.gz'
    size, sha = write(raw, program(1))
    for native in (True, False):
        with pytest.raises(ValueError, match='receipt'):
            fast.reduce_chunk(raw, tmp / ('n' if native else 'p') / 'o', chunk_id='c',
                              counter_scope='t', expected_bytes=size, expected_sha256='0' * 64,
                              limits=dict(DEFAULTS), native=native)
        with pytest.raises(ValueError, match='receipt'):
            fast.reduce_chunk(raw, tmp / ('n' if native else 'p') / 'o2', chunk_id='c',
                              counter_scope='t', expected_bytes=size + 1, expected_sha256=sha,
                              limits=dict(DEFAULTS), native=native)


def test_zero_padded_identity_is_refused(tmp):
    rows = [r for r in program(1)]
    rows.append(row('007', 'BRISC', 'BRISC-KERNEL', 'ZONE_START', 2000))
    rows.append(row('7', 'BRISC', 'BRISC-KERNEL', 'ZONE_END', 2100))
    for native in (True, False):
        with pytest.raises(ValueError, match='(?i)zero-padded'):
            reduce(tmp / ('n' if native else 'p'), rows, native=native)


@pytest.mark.parametrize('limit,value,message', [
    ('max_rows', 3, 'max_rows exceeded'),
    ('max_operations', 1, 'max_operations exceeded'),
    ('max_core_riscs', 1, 'max_core_riscs exceeded'),
    ('max_markers', 1, 'max_markers exceeded'),
    ('max_line_bytes', 40, 'max_line_bytes exceeded'),
    ('max_raw_bytes', 100, 'max_raw_bytes exceeded'),
    ('max_output_bytes', 200, 'max_output_bytes exceeded'),
    ('max_archive_bytes', 80, 'max_archive_bytes exceeded'),
])
def test_every_configured_limit_is_enforced(tmp, limit, value, message):
    rows = program(1, riscs=('BRISC', 'NCRISC'), cores=((1, 1), (2, 2))) + program(2)
    expect_stop(tmp, rows, message, limits={limit: value})


def test_negative_limits_are_refused(tmp):
    expect_stop(tmp, program(1), 'limits must be positive', limits={'max_rows': 0})


def test_memory_limit_makes_the_child_fail(tmp):
    raw = tmp / 'raw.csv.gz'
    size, sha = write(raw, program(1))
    with pytest.raises(subprocess.CalledProcessError):
        fast.append_chunk(raw, tmp / 'o', chunk_id='c', counter_scope='t', expected_bytes=size,
                          expected_sha256=sha, max_memory_mib=8)


def test_invalid_chunk_id_and_scope_are_refused(tmp):
    raw = tmp / 'raw.csv.gz'
    size, sha = write(raw, program(1))
    for bad, message in ((dict(chunk_id='../escape'), 'invalid chunk_id'),
                         (dict(chunk_id='.'), 'invalid chunk_id'),
                         (dict(counter_scope=''), 'counter_scope')):
        kw = dict(chunk_id='c', counter_scope='t', expected_bytes=size, expected_sha256=sha,
                  limits=dict(DEFAULTS))
        kw.update(bad)
        with pytest.raises(ValueError, match='(?i)' + message):
            fast.reduce_chunk(raw, tmp / 'o', **kw)


def test_existing_output_is_never_overwritten(tmp):
    rows = program(1)
    first = reduce(tmp / 'n', rows)
    before = Path(first['path']).read_bytes()
    with pytest.raises(FileExistsError):
        reduce(tmp / 'n', rows)
    assert Path(first['path']).read_bytes() == before


def test_nothing_is_published_when_reduction_fails(tmp):
    out = tmp / 'n' / 'out'
    with pytest.raises(ValueError):
        reduce(tmp / 'n', program(1, span=1, zones=(('DM-SEM-WAIT', 500),)))
    assert not list(out.glob('*.jsonl.gz'))
    assert not list(out.glob('.throughput-*'))


def test_footer_verification_catches_a_damaged_archive(tmp):
    result = reduce(tmp / 'n', program(1))
    lines = gzip.decompress(Path(result['path']).read_bytes()).split(b'\n')
    damaged = tmp / 'damaged.jsonl.gz'
    damaged.write_bytes(gzip.compress(b'\n'.join(lines[:-2]) + b'\n'))
    with pytest.raises(ValueError, match='Missing completion footer'):
        verify_chunk(damaged)
    tampered = json.loads(lines[-2])
    tampered['records'] += 1
    damaged.write_bytes(gzip.compress(b'\n'.join(lines[:-2]) + b'\n'
                                      + json.dumps(tampered).encode() + b'\n'))
    with pytest.raises(ValueError, match='digest/count mismatch'):
        verify_chunk(damaged)


def test_raw_input_survives_a_successful_reduction(tmp):
    raw = tmp / 'raw.csv.gz'
    size, sha = write(raw, program(1))
    before = raw.read_bytes()
    fast.reduce_chunk(raw, tmp / 'o', chunk_id='c', counter_scope='t', expected_bytes=size,
                      expected_sha256=sha, limits=dict(DEFAULTS))
    assert raw.read_bytes() == before


# --------------------------------------------------------- native fallbacks
def test_native_refuses_non_ascii_and_python_reduces_it(tmp):
    rows = program(1) + [row(1, 'BRISC', 'ZONE-µ', 'ZONE_START', 1200)]
    assert native_only(tmp / 'n', rows) is None
    result = reduce(tmp / 'p', rows)
    assert result['reducer'] == 'python'
    assert payload(result, 'unreduced_marker')[0]['values'][10] == 'ZONE-µ'


def test_native_refuses_a_quoted_field_and_python_reduces_it(tmp):
    rows = program(1) + [row(1, 'BRISC', 'Q', 'ZONE_START', 1200, meta='"a,b"')]
    assert native_only(tmp / 'n', rows) is None
    result = reduce(tmp / 'p', rows)
    assert result['reducer'] == 'python'
    assert payload(result, 'unreduced_marker')[0]['values'][14] == 'a,b'


def test_native_refuses_an_embedded_carriage_return(tmp):
    rows = program(1)
    rows[0] = list(rows[0])
    rows[0][14] = 'a\rb'
    assert native_only(tmp / 'n', rows) is None


def test_dispatcher_falls_back_transparently(tmp):
    rows = program(1) + [row(1, 'BRISC', 'ZONE-µ', 'ZONE_START', 1200)]
    assert reduce(tmp / 'n', rows, native=True)['reducer'] == 'python'


def test_native_binary_is_cached_by_source_digest(tmp):
    binary, digest = fast.native_binary()
    assert binary.exists() and binary.name.endswith(digest[:16])
    assert fast.native_binary() == (binary, digest)


def test_synchronous_api_returns_only_after_publication(tmp):
    raw = tmp / 'raw.csv.gz'
    size, sha = write(raw, program(1))
    result = fast.append_chunk(raw, tmp / 'o', chunk_id='c', counter_scope='t',
                               expected_bytes=size, expected_sha256=sha)
    path = Path(result['path'])
    assert path.exists() and verify_chunk(path)['records'] == result['programs'] + 1
    assert not list((tmp / 'o').glob('.throughput-*'))


def test_plain_uncompressed_input(tmp):
    for native in (True, False):
        result = reduce(tmp / ('n' if native else 'p'), program(1), native=native, gz=False)
        assert payload(result, 'program')[0]['summary']['span_cycles'] == 500
