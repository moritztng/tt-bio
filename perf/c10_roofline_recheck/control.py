#!/usr/bin/env python3
"""Fail closed if the existing roofline counters miss an exact dense-matmul control.

No model changes and no roof or fold-time claims. Capture one warmed 8192-cube matmul,
record sysfs AICLK samples inside its capture, then require exact FLOPs and compulsory
bytes. On failure, exit 2 and do not proceed to a fold or publish a roofline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import types
import select

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'perf/roof_budget'))
sys.path.insert(0, str(ROOT / 'perf/b2x_difflayer'))
CLOCK_REF = '5c60137c2:tt_bio/aiclk.py'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    initial_holders = device_holders()
    if initial_holders:
        raise RuntimeError(f'Quiet-board precondition failed: {initial_holders}')
    if os.environ.get('TT_BIO_AICLK') != '1350':
        raise ValueError('This control requires TT_BIO_AICLK=1350')

    # Reuse the clock holder verbatim, without changing model files or installing it.
    source = subprocess.check_output(['git', 'show', CLOCK_REF], cwd=ROOT)
    clock = types.ModuleType('c10_clock')
    exec(compile(source, CLOCK_REF, 'exec'), clock.__dict__)

    import torch
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    import tt_bio.tenstorrent as T
    import exec_flops
    from real_traffic import counts
    sys.path.insert(0, str(ROOT / 'perf/roof_arb'))
    from corrected_traffic import counts as corrected_counts
    from itemize import itemize

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    if not str(Path(ttnn.__file__).resolve()).startswith('/home/ttuser/tt-metal-k10/'):
        raise RuntimeError(f'Wrong loaded ttnn: {ttnn.__file__}')
    dev = T.get_device()
    sampler = None
    sample_path = args.out / 'clock.jsonl'
    result = {}
    try:
        clock.engage('blackhole')
        nodes = clock.status()['nodes']
        if nodes != [0]:
            raise RuntimeError(f'Expected only physical node 0, got {nodes}')
        sampler = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), '--clock-worker', str(sample_path)],
            stdin=subprocess.PIPE, text=True)
        n = 8192
        x = ttnn.from_torch(torch.ones(n, n, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y = ttnn.from_torch(torch.ones(n, n, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)

        def call():
            return ttnn.matmul(x, y, compute_kernel_config=config,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

        for _ in range(3):
            z = call()
            ttnn.synchronize_device(dev)
            ttnn.deallocate(z)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        start = time.monotonic_ns()
        z = call()
        ttnn.synchronize_device(dev)
        end = time.monotonic_ns()
        graph = ttnn.graph.end_graph_capture()
        graph = json.loads(graph) if isinstance(graph, str) else graph
        (args.out / 'matmul_graph.json').write_text(json.dumps(graph, indent=2) + '\n')
        sampler.communicate('stop\n', timeout=10)
        samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
        errors = [s['error'] for s in samples if 'error' in s]
        foreign = [s for s in samples if s.get('foreign_holders')]
        if foreign:
            raise RuntimeError(f'Quiet-board condition lost: {foreign}')

        during = [s for s in samples if s['read_start_ns'] >= start and s['read_end_ns'] <= end]
        if not during or errors or any(not 1200 <= s['MHz'] <= 1400 for s in during):
            raise RuntimeError(f'Invalid during-capture clock: {during}, errors={errors}')

        centers = [(s['read_start_ns'] + s['read_end_ns']) // 2 for s in during]
        gaps = [b-a for a,b in zip([start]+centers, centers+[end])]
        clock_coverage = {
            'samples': len(during), 'first_offset_ns': centers[0]-start,
            'last_offset_ns': end-centers[-1], 'max_gap_ns': max(gaps),
            'sample_span_fraction': (centers[-1]-centers[0])/(end-start),
            'protocol': 'Independent Python process, 1ms poll delay; read brackets wholly inside synchronized host-monotonic interval.',
            'node': '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk',
        }
        if len(during) < 3 or max(gaps) > 10_000_000 or any(s['MHz'] != 1350 for s in during):
            raise RuntimeError(f'Missing 1350 MHz clock coverage: {clock_coverage}')
        corrected = corrected_counts({'nodes': graph})
        flops = exec_flops.totals(graph)
        traffic = counts({'nodes': graph})
        ops, buffers = itemize({'nodes': graph})
        expected_flops = 2 * n**3
        expected_bytes = 3 * n*n*2
        observed_bytes = round(traffic['real_MB'] * 1e6)
        result = {
            'host': socket.gethostname(), 'physical_card': 0,
            'repo_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'ttnn_module': ttnn.__file__, 'ttnn_version': getattr(ttnn, '__version__', None),
            'metal_source_head': Path('/home/ttuser/tt-metal-k10/.git/HEAD').read_text().strip(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'clock_coverage': clock_coverage,
            'corrected_byte_counter': corrected,
            'corrected_bytes_exact': round(corrected['real_MB'] * 1e6) == expected_bytes,
            'loaded_shared_libraries': sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'tt-metal' in line}),
            'flags': {k:v for k,v in os.environ.items() if k.startswith(('TT_', 'TTNN_', 'PYTHONPATH', 'LD_LIBRARY_PATH'))},
            'clock_holder_ref': CLOCK_REF, 'clock_holder_sha256': hashlib.sha256(source).hexdigest(),
            'clock_pin_MHz': 1350, 'clock_hold': clock.status(),
            'capture_start_monotonic_ns': start, 'capture_end_monotonic_ns': end,
            'during_capture_clock': during, 'clock_errors': errors,
            'clock_min_MHz': min(s['MHz'] for s in during),
            'clock_max_MHz': max(s['MHz'] for s in during),
            'clock_mean_MHz': sum(s['MHz'] for s in during) / len(during),
            'shape': {'M': n, 'N': n, 'K': n}, 'dtype': 'bf16',
            'expected_flops': expected_flops, 'observed_flops': flops['matmul_padded'],
            'expected_min_bytes': expected_bytes, 'observed_bytes': observed_bytes,
            'bytes_error': observed_bytes - expected_bytes,
            'byte_ratio': observed_bytes / expected_bytes,
            'flops_exact': flops['matmul_padded'] == expected_flops,
            'bytes_exact': observed_bytes == expected_bytes,
            'flop_counter': flops, 'byte_counter': traffic, 'ops': ops, 'buffers': buffers,
            'device_cycles': None,
            'timing_note': 'Capture boundaries identify clock samples only; no device timing or roof measured.',
        }
        result['verdict'] = 'PASS' if result['flops_exact'] and result['bytes_exact'] and result['corrected_bytes_exact'] else 'INSTRUMENT_UNFIT'
        (args.out / 'control.json').write_text(json.dumps(result, indent=2) + '\n')
        ttnn.deallocate(z)
        ttnn.deallocate(x)
        ttnn.deallocate(y)
        print(json.dumps({k: result[k] for k in (
            'verdict', 'expected_flops', 'observed_flops', 'expected_min_bytes',
            'observed_bytes', 'byte_ratio', 'clock_pin_MHz', 'clock_min_MHz',
            'clock_max_MHz', 'clock_mean_MHz')}, indent=2), flush=True)
    except BaseException as error:
        result.update({
            'verdict': 'STOP', 'error': repr(error),
            'host': socket.gethostname(), 'physical_card': 0,
            'clock_target_MHz': 1350,
            'scope': 'Failed control; no roof, floor, throughput or model timing claim.',
        })
        (args.out / 'control.json').write_text(json.dumps(result, indent=2) + '\n')
        raise
    finally:
        if sampler and sampler.poll() is None:
            sampler.communicate('stop\n', timeout=10)
        clock.release()
        T.cleanup()
    return 0 if result.get('verdict') == 'PASS' else 2


def device_holders(exclude=()):
    holders = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in exclude:
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith('/dev/tenstorrent/'):
                    holders.append({'pid': int(proc.name), 'node': target})
        except (OSError, PermissionError):
            continue
    return holders


def clock_worker(path):
    clk = Path('/sys/class/tenstorrent/tenstorrent!0/tt_aiclk')
    owner = os.getppid()
    last_scan = 0
    with Path(path).open('w') as out:
        while not select.select([sys.stdin], [], [], 0.001)[0]:
            before = time.monotonic_ns()
            try:
                value = int(clk.read_text())
                row = {'read_start_ns': before, 'read_end_ns': time.monotonic_ns(), 'MHz': value}
            except (OSError, ValueError) as error:
                row = {'error': str(error)}
            if before - last_scan > 100_000_000:
                row['foreign_holders'] = device_holders(exclude=(owner, os.getpid()))
                row['holders_checked_ns'] = time.monotonic_ns()
                last_scan = before
            out.write(json.dumps(row) + '\n')
            out.flush()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--clock-worker':
        clock_worker(sys.argv[2])
    else:
        raise SystemExit(main())
