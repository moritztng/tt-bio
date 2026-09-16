"""Package bounded metadata and clock inputs for the unchanged census reducer."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
from export_metadata import Budget, digest, limited


def archive(run, max_clock_bytes=256*1024**2):
    run = Path(run).resolve()
    report_path = run / 'tracy/metadata/report.json'
    if report_path.stat().st_size > 1024**2:
        raise ValueError('Export report exceeds 1 MiB')
    report = json.loads(report_path.read_text())
    metadata = run / 'tracy/metadata/tracy_ops_data.csv'
    if report['verdict'] != 'GO' or digest(metadata) != report['metadata']:
        raise ValueError('Metadata lacks a matching successful export report')
    sources = [(metadata, 'raw/tracy_ops_data.csv.gz', 64*1024**2),
               (run / 'out/clock.jsonl', 'out/clock.jsonl.gz', max_clock_bytes)]
    manifest = run / 'raw_manifest.json'
    for source, target, cap in sources:
        if source.stat().st_size > cap:
            raise ValueError('Archive input exceeds bound: ' + str(source))
        if (run / target).exists():
            raise FileExistsError(run / target)
    if manifest.exists():
        raise FileExistsError(manifest)
    published = []
    with tempfile.TemporaryDirectory(prefix='host-archive-', dir=run) as temp:
        temp = Path(temp)
        rows = []
        try:
            for i, (source, target, cap) in enumerate(sources):
                before = digest(source)
                with source.open('rb') as inp, (temp / str(i)).open('wb') as dest:
                    with gzip.GzipFile(filename='', mode='wb', fileobj=dest, mtime=0) as out:
                        while block := inp.read(1024**2):
                            out.write(block)
                if digest(source) != before:
                    raise ValueError('Archive source changed: ' + str(source))
                rows.append({'original': before['path'], 'bytes': before['bytes'],
                             'sha256': before['sha256'], 'archive': target,
                             'archive_sha256': digest(temp / str(i))['sha256']})
            for i, row in enumerate(rows):
                target = run / row['archive']
                target.parent.mkdir(exist_ok=True)
                os.link(temp / str(i), target)  # refuse races/overwrites
                published.append(target)
            # The manifest is the completion marker, written last.
            (temp / 'manifest').write_text(json.dumps(rows, indent=2) + '\n')
            os.link(temp / 'manifest', manifest)
        except BaseException:
            for p in published:
                p.unlink()
            raise
    return {'verdict': 'GO', 'manifest': digest(manifest), 'files': rows}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_archive':
        print(json.dumps(archive(Path(sys.argv[2])), indent=2))
        return 0
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    # All compression runs under the same CPU/memory/file/time caps as export.
    result = limited([sys.executable, str(Path(__file__).resolve()), '_archive', str(args.run)],
                     args.run / 'host_archive.json', args.run / 'host_archive.stderr', Budget())
    print(json.dumps(result, indent=2))
    return 0 if result['returncode'] == 0 and not result['timed_out'] else 2


if __name__ == '__main__':
    sys.exit(main())
