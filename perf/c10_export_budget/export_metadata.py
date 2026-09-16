"""Bounded CPU-only Tracy message export. Never imports Tracy or a TT runtime."""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / 'source_contract.json').read_text())


@dataclass(frozen=True)
class Budget:
    trace_bytes: int = 4 * 1024**3
    output_bytes: int = 64 * 1024**2
    address_bytes: int = 8 * 1024**3
    cpu_seconds: int = 300
    wall_seconds: float = 600

    def check(self):
        if any(v <= 0 for v in asdict(self).values()):
            raise ValueError('All resource limits must be positive')


def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def limited(command, stdout, stderr, budget):
    """Limits apply only to this CPU child, never to the model/capture process."""
    budget.check()
    command = [sys.executable, str(Path(__file__).resolve()), '_exec',
               str(budget.address_bytes), str(budget.output_bytes),
               str(budget.cpu_seconds), *map(str, command)]
    started = time.monotonic()
    with open(stdout, 'xb') as out, open(stderr, 'xb') as err:
        proc = subprocess.Popen(command, stdout=out, stderr=err)
        timed_out = False
        def interrupted(signum, frame):
            raise InterruptedError(f'CPU supervisor interrupted by signal {signum}')
        previous_term = signal.signal(signal.SIGTERM, interrupted)
        try:
            while True:
                pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    break
                if time.monotonic() - started > budget.wall_seconds:
                    timed_out = True
                    proc.kill()
                    _, status, usage = os.wait4(proc.pid, 0)
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    break
                time.sleep(0.02)
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            if proc.returncode is None:
                proc.kill()
                proc.wait()
    return {'command': command, 'returncode': proc.returncode, 'timed_out': timed_out,
            'elapsed_seconds': time.monotonic() - started,
            'max_rss_KiB': usage.ru_maxrss, 'user_seconds': usage.ru_utime,
            'system_seconds': usage.ru_stime, 'stdout_bytes': Path(stdout).stat().st_size,
            'stderr_bytes': Path(stderr).stat().st_size}


def validate(path):
    # Reuse the exact census decoder. No new encoding or shape interpretation.
    sys.path.insert(0, str(HERE.parent / 'c10_burst_census'))
    from reduce_census import host_ops
    ops, bad = host_ops(path)
    if bad or not ops:
        raise ValueError(f'Incomplete operation metadata: {len(ops)} calls, {len(bad)} unparsed messages')
    # A streaming hash of ordered decoded records preserves multiplicity and all fields.
    h = hashlib.sha256()
    for key, value in ops.items():
        h.update(json.dumps([key, value], sort_keys=True, separators=(',', ':')).encode())
        h.update(b'\n')
    return {'calls': len(ops), 'unparsed_messages': len(bad),
            'ordered_decoded_sha256': h.hexdigest(), 'metadata': digest(path)}


def export(trace, binary, output, budget=Budget(), runtime_dir=None):
    """Create a fresh result directory; metadata is usable only with verdict GO."""
    budget.check()
    trace, binary, output = map(Path, (trace, binary, output))
    output.mkdir(parents=True, exist_ok=False)
    report = {'verdict': 'STOP', 'scope': 'CPU metadata export only; zero model cycles saved',
              'budget': asdict(budget), 'trace_retention': 'Input retained unchanged, including on failure',
              'trace': {'path': str(trace.resolve())}}
    part = output / 'metadata.partial'
    try:
        size = trace.stat().st_size
        report['trace']['bytes'] = size
        if size <= 0 or size > budget.trace_bytes:
            raise ValueError(f'Trace size {size} outside 1..{budget.trace_bytes}; export not started')
        report['trace'] = digest(trace)
        report['binary'] = digest(binary)
        if report['binary']['sha256'] != CONTRACT['binary_sha256']:
            raise ValueError('Unreviewed csvexport binary; source/tool contract must be revalidated')
        prefix = []
        if runtime_dir:
            runtime_dir = Path(runtime_dir).resolve()
            report['runtime'] = [digest(p) for p in sorted(runtime_dir.iterdir()) if p.is_file()]
            prefix = [str(runtime_dir / 'ld-linux-x86-64.so.2'), '--library-path', str(runtime_dir)]
        command = [*prefix, str(binary.resolve()), '-m', '-s', ';', str(trace.resolve())]
        report['export'] = limited(command, part, output / 'export.stderr', budget)
        if (report['export']['returncode'] or report['export']['timed_out']
                or report['export']['stdout_bytes'] >= budget.output_bytes):
            raise ValueError('Metadata exporter failed or exceeded a resource limit; see export.stderr')
        # Run the existing in-memory decoder in a separately bounded process too.
        report['validation'] = limited(
            [sys.executable, str(Path(__file__).resolve()), '_validate', str(part)],
            output / 'validation.json', output / 'validation.stderr', budget)
        if report['validation']['returncode'] or report['validation']['timed_out']:
            raise ValueError('Metadata validation failed or exceeded a resource limit; see validation.stderr')
        report['identity'] = json.loads((output / 'validation.json').read_text())
        if digest(trace) != report['trace']:
            raise ValueError('Trace changed while exporting')
        final = output / 'tracy_ops_data.csv'
        part.rename(final)
        report['metadata'] = digest(final)
        report['verdict'] = 'GO'
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        # A partial CSV never survives under a name accepted by the reducer.
        part.unlink(missing_ok=True)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def add_budget_args(parser):
    for name, default in asdict(Budget()).items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)


def budget_from(args):
    return Budget(**{name: getattr(args, name) for name in asdict(Budget())})


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_exec':
        address, output, cpu = map(int, sys.argv[2:5])
        resource.setrlimit(resource.RLIMIT_AS, (address, address))
        resource.setrlimit(resource.RLIMIT_FSIZE, (output, output))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        os.execv(sys.argv[5], sys.argv[5:])
    if len(sys.argv) > 1 and sys.argv[1] == '_validate':
        print(json.dumps(validate(Path(sys.argv[2]))))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--csvexport', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path)
    add_budget_args(parser)
    args = parser.parse_args()
    report = export(args.trace, args.csvexport, args.out, budget_from(args), args.runtime_dir)
    print(json.dumps(report, indent=2))
    return 0 if report['verdict'] == 'GO' else 2


if __name__ == '__main__':
    sys.exit(main())
