"""Run CPU controls and write a labeled synthetic capture; never imports ttnn."""
import argparse
import io
import json
from pathlib import Path
import sys
import unittest

from perf.c10_generic_identity.observer import Capture
from perf.c10_generic_identity import test_observer as controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromModule(controls)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (args.out / 'tests.txt').write_text(log.getvalue())
    capture = args.out / 'synthetic.jsonl'
    kernel = controls.KernelDescriptor('void kernel_main() { /* CPU fixture, never compiled */ }')
    kernel.source_type = controls.SourceType.SOURCE_CODE
    pd = controls.ProgramDescriptor([kernel])
    tensors = [controls.Tensor(), controls.Tensor()]
    binding = controls.binding(lambda *a, **kw: tensors[-1])
    with Capture(binding, capture, config={'evidence': 'CPU doubles only; no installed binding or device validation'}) as observer:
        assert binding.generic_op(tensors, pd) is tensors[-1]
        kernel.runtime_args.pairs[0][1][0] = 4096
        kernel.common_runtime_args[0] = 8192
        assert binding.generic_op(io_tensors=tensors, program_descriptor=pd) is tensors[-1]
    calls = [r for r in controls.load_records(capture) if r['kind'] == 'call']
    report = {'evidence': 'CPU controls only', 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'success': result.wasSuccessful(), 'synthetic_calls': len(calls),
              'same_descriptor_object': calls[0]['descriptor_object_id'] == calls[1]['descriptor_object_id'],
              'same_double_cache_hash': calls[0]['binding_cache_hash'] == calls[1]['binding_cache_hash'],
              'different_execution_identity': calls[0]['execution_sha256'] != calls[1]['execution_sha256'],
              'observer_errors': observer.errors, 'live_binding_compatibility': 'unclaimed',
              'live_device_smoke': 'owed to device worker', 'performance_measurement': False,
              'clock': 'CPU only; 1350 MHz target unmeasured; cycles unmeasured'}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
