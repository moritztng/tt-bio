"""Device-owner entry point: unchanged Tracy capture, bounded metadata-only reporting.

Do not invoke this script on a CPU-only worker. export_metadata.py is CPU-only.
"""
import argparse
import importlib
from pathlib import Path
import sys
from export_metadata import CONTRACT, add_budget_args, budget_from, digest, export


def install_report_hook(cli, budget):
    def metadata_report(outputFolder, binFolder, nameAppend, childCalls,
                        collect_noc_traces=False, device_analysis_types=()):
        if nameAppend or childCalls or collect_noc_traces or device_analysis_types:
            raise ValueError('Census metadata-only reporting does not support extra report analyses')
        logs = Path(outputFolder) / '.logs'
        result = export(logs / 'tracy_profile_log_host.tracy',
                        Path(binFolder) / 'csvexport-release',
                        Path(outputFolder) / 'metadata', budget)
        if result['verdict'] != 'GO':
            raise RuntimeError('Bounded host export STOP: ' + result['error'])
    cli.generate_report = metadata_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metal-root', type=Path, required=True)
    add_budget_args(parser)
    parser.add_argument('tracy_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    budget = budget_from(args)
    budget.check()
    # Verify the inspected CLI before it can launch a device process.
    for relative, recorded in [('tools/tracy/__main__.py', 'tracy_main.py'),
                               ('tools/tracy/__init__.py', 'tracy_init.py'),
                               ('tt_metal/third_party/tracy/csvexport/src/csvexport.cpp', 'csvexport.cpp')]:
        if digest(args.metal_root / relative)['sha256'] != CONTRACT['sources'][recorded]:
            raise ValueError('Tracy source changed; revalidate before capture: ' + relative)
    if digest(args.metal_root / 'build/tools/profiler/bin/csvexport-release')['sha256'] != CONTRACT['binary_sha256']:
        raise ValueError('csvexport binary changed; revalidate before capture')
    argv = args.tracy_args
    if argv and argv[0] == '--':
        argv = argv[1:]
    if '-r' not in argv or '--check-exit-code' not in argv:
        raise ValueError('Require -r and --check-exit-code to capture and reject failed model runs')
    cli = importlib.import_module('tracy.__main__')
    if Path(cli.__file__).resolve() != (args.metal_root / 'tools/tracy/__main__.py').resolve():
        raise ValueError('Wrong imported Tracy module')
    install_report_hook(cli, budget)
    # The upstream CLI removes -r for its inner python3 -m tracy process.
    # Its capture, flags and model execution are untouched. Only the outer report is replaced.
    sys.argv = [str(Path(__file__).resolve()), *argv]
    cli.main()


if __name__ == '__main__':
    main()
