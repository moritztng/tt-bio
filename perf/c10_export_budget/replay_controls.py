"""Reproduce known-answer CPU exports and all retained census window identities."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
import sys
from export_metadata import Budget, digest, export, limited, validate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def census_replay():
    sys.path.insert(0, str(HERE.parent / 'c10_burst_census'))
    from reduce_census import host_ops, reduce_window
    p = HERE.parent / 'c10_burst_census/runs/census1'
    metadata = Path(sys.argv[2]) if len(sys.argv) > 2 else p / 'raw/tracy_ops_data.csv.gz'
    ops, bad = host_ops(metadata)
    if bad:
        raise ValueError('Unparsed full-census metadata')
    capture = json.loads((p / 'out/capture.json').read_text())
    original = json.loads((p / 'analysis.json').read_text())
    rows = []
    for window, expected in zip(capture['windows'], original['windows'], strict=True):
        window = dict(window, clock=expected['clock'])
        result = json.loads(json.dumps(reduce_window(p, window, ops)))
        # timing_valid is added by the caller, outside the unchanged window reducer.
        wanted = {k: v for k, v in expected.items() if k != 'timing_valid'}
        if result != wanted:
            raise ValueError('Archived window changed: ' + window['case'])
        rows.append({'case': window['case'], 'invocations': result['invocations'],
                     'window_result_sha256': hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()})
    return {'verdict': 'GO', 'scope': 'existing retained windows only; clock coverage reused, no new device measurement',
            'calls': len(ops), 'metadata_archive': digest(metadata), 'windows': rows,
            'invocations': sum(w['invocations'] for w in rows),
            'original_analysis': digest(p / 'analysis.json')}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_census':
        print(json.dumps(census_replay(), indent=2))
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csvexport', type=Path, required=True)
    ap.add_argument('--runtime-dir', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'verdict': 'STOP', 'clock': 'CPU only; 1350 MHz campaign context, no hardware measurement',
              'predicted': 'same metadata/call identities without timing CSV; zero model cycles saved',
              'measured': [], 'large_trace_tested': False}
    try:
        for label, raw in [('dm_control', HERE.parent / 'c10_dm_control/raw'),
                           ('smoke1', HERE.parent / 'c10_burst_census/runs/smoke1/raw'),
                           ('smoke2', HERE.parent / 'c10_burst_census/runs/smoke2/raw')]:
            r = export(raw / 'tracy_profile_log_host.tracy', args.csvexport, args.out / label,
                       runtime_dir=args.runtime_dir)
            if r['verdict'] != 'GO':
                raise ValueError(label + ': ' + r['error'])
            expected = gzip.decompress((raw / 'tracy_ops_data.csv.gz').read_bytes())
            actual = (args.out / label / 'tracy_ops_data.csv').read_bytes()
            if actual != expected:
                raise ValueError('Metadata byte mismatch: ' + label)
            identity = validate(raw / 'tracy_ops_data.csv.gz')
            if identity['ordered_decoded_sha256'] != r['identity']['ordered_decoded_sha256']:
                raise ValueError('Decoded identity mismatch: ' + label)
            report['measured'].append({'case': label, 'byte_exact': True,
                                       'archive': digest(raw / 'tracy_ops_data.csv.gz'), **r})
        p = HERE.parent / 'c10_burst_census/runs/census1/raw/tracy_ops_data.csv.gz'
        r = limited([sys.executable, str(HERE / 'export_metadata.py'), '_validate', str(p)],
                    args.out / 'census_metadata.json', args.out / 'census_metadata.stderr', Budget())
        if r['returncode'] or r['timed_out']:
            raise ValueError('Full archived metadata validation failed')
        report['census_metadata'] = {'resource': r, 'identity': json.loads((args.out / 'census_metadata.json').read_text())}
        # Package the full archived metadata and clock, without claiming a new trace export.
        packed = args.out / 'packaging'
        (packed / 'tracy/metadata').mkdir(parents=True)
        (packed / 'out').mkdir()
        original = HERE.parent / 'c10_burst_census/runs/census1'
        for src, dst in [(original / 'raw/tracy_ops_data.csv.gz', packed / 'tracy/metadata/tracy_ops_data.csv'),
                         (original / 'out/clock.jsonl.gz', packed / 'out/clock.jsonl')]:
            with gzip.open(src, 'rb') as inp, dst.open('wb') as out:
                shutil.copyfileobj(inp, out, 1024**2)
        # This fixture report attests only the validated archived bytes above.
        (packed / 'tracy/metadata/report.json').write_text(json.dumps({
            'verdict': 'GO', 'scope': 'packaging control from validated archive; no large-trace export',
            'metadata': digest(packed / 'tracy/metadata/tracy_ops_data.csv')}))
        r = limited([sys.executable, str(HERE / 'archive_replay.py'), '_archive', str(packed)],
                    args.out / 'packaging.json', args.out / 'packaging.stderr', Budget())
        if r['returncode'] or r['timed_out']:
            raise ValueError('Metadata/clock packaging failed')
        packaged = json.loads((args.out / 'packaging.json').read_text())
        expected = json.loads((original / 'raw_manifest.json').read_text())
        for a, b in zip(packaged['files'], expected, strict=True):
            if (a['bytes'], a['sha256'], a['archive']) != (b['bytes'], b['sha256'], b['archive']):
                raise ValueError('Packaged input identity changed')
            h = hashlib.sha256()
            with gzip.open(packed / a['archive'], 'rb') as f:
                for block in iter(lambda: f.read(1024**2), b''):
                    h.update(block)
            if h.hexdigest() != b['sha256']:
                raise ValueError('Compressed archive changed source bytes')
        report['packaging'] = {'resource': r, 'result': packaged}
        r = limited([sys.executable, str(Path(__file__).resolve()), '_census',
                     str(packed / 'raw/tracy_ops_data.csv.gz')],
                    args.out / 'census_windows.json', args.out / 'census_windows.stderr', Budget())
        if r['returncode'] or r['timed_out']:
            raise ValueError('Full retained-window replay failed')
        report['census_windows'] = {'resource': r, 'result': json.loads((args.out / 'census_windows.json').read_text())}
        report['verdict'] = 'GO'
    finally:
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'verdict': report['verdict'], 'report': str(args.out / 'report.json')}))


if __name__ == '__main__':
    main()
