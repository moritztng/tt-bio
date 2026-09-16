"""Floor screen: the fastest per-row loop pure CPython can run on this archive.

Does strictly less than a correct reducer (no output, no provenance, no per-program
arithmetic, no caps) so its rate is an upper bound on any pure-Python reducer.
"""
from __future__ import annotations
import csv, gzip, sys, time
from pathlib import Path

RISCS = ('BRISC', 'NCRISC', 'TRISC_0', 'TRISC_1', 'TRISC_2')


def screen(path):
    rows = 0
    identities = {}
    kernels = {}
    sums = {}
    coverage = {}
    with gzip.open(path, 'rt', newline='') as f:
        next(f)
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        prev_key = None
        surrogate = -1
        for row in reader:
            rows += 1
            device, x, y, risc, timer, tick, data, call, trace, replay, zone, kind, line, filename, meta = row
            if not (call.isdigit() and tick.isdigit() and data.isdigit()
                    and x.isdigit() and y.isdigit() and device.isdigit() and timer.isdigit()
                    and line.isdigit()):
                raise ValueError('non-integer field')
            ident = (call, device, trace, replay)
            if ident != prev_key:
                prev_key = ident
                surrogate = identities.setdefault(ident, len(identities))
            marker = (risc, zone, kind)
            coverage[marker] = coverage.get(marker, 0) + 1
            if kind == 'ZONE_TOTAL' and risc in RISCS:
                sums[surrogate, x, y, risc, zone] = int(data)
            elif risc in RISCS and (zone.endswith('-KERNEL') or zone.endswith('-FW')):
                kernels[surrogate, x, y, risc, zone, kind] = int(tick)
    return rows, len(identities), len(kernels), len(sums), len(coverage)


def main():
    for path in sys.argv[1:]:
        size = 0
        with gzip.open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                size += len(b)
        for rep in range(2):
            t = time.monotonic()
            rows, ops, ke, sm, cov = screen(path)
            dt = time.monotonic() - t
            print(f'{Path(path).name} rep{rep} rows={rows} bytes={size} '
                  f'{dt:.3f}s {size/dt/1e6:.2f}MB/s {rows/dt:,.0f}rows/s '
                  f'ops={ops} endpoints={ke} sums={sm} markers={cov}', flush=True)


if __name__ == '__main__':
    main()
