"""CPU controls only; mock Tracy namespaces never import a device runtime."""
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from export_metadata import Budget, CONTRACT, export, limited, validate
from bounded_tracy import install_report_hook

HERE = Path(__file__).resolve().parent


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name)
        self.trace = self.p / 'trace'
        self.trace.write_bytes(b'control')

    def fake_exporter(self, body):
        p = self.p / 'exporter'
        p.write_text('#!' + sys.executable + '\n' + body)
        p.chmod(0o755)
        return p

    def run_fake(self, body, budget=Budget()):
        exe = self.fake_exporter(body)
        with patch.dict(CONTRACT, binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()):
            return export(self.trace, exe, self.p / 'out', budget)

    def assert_stop(self, report):
        self.assertEqual(report['verdict'], 'STOP')
        self.assertFalse((self.p / 'out/tracy_ops_data.csv').exists())
        self.assertFalse((self.p / 'out/metadata.partial').exists())
        self.assertEqual(self.trace.read_bytes(), b'control')

    def test_input_limit_before_child_or_output(self):
        r = export(self.trace, self.p / 'absent', self.p / 'out', replace(Budget(), trace_bytes=2))
        self.assert_stop(r)
        self.assertNotIn('export', r)

    def test_unreviewed_tool_refused(self):
        exe = self.fake_exporter('raise RuntimeError("must not execute")')
        self.assert_stop(export(self.trace, exe, self.p / 'out'))

    def test_disk_limit_cannot_publish_a_prefix(self):
        r = self.run_fake('import os\nwhile True: os.write(1, b"x"*4096)\n', replace(Budget(), output_bytes=8192))
        self.assert_stop(r)
        self.assertLessEqual(r['export']['stdout_bytes'], 8192)
        self.assertNotEqual(r['export']['returncode'], 0)

    def test_address_limit(self):
        r = self.run_fake('x = bytearray(1024**3)\n', replace(Budget(), address_bytes=128*1024**2))
        self.assert_stop(r)
        self.assertNotEqual(r['export']['returncode'], 0)

    def test_wall_timeout_reaps_owned_child(self):
        r = self.run_fake('import time\ntime.sleep(30)\n', replace(Budget(), wall_seconds=0.2))
        self.assert_stop(r)
        self.assertTrue(r['export']['timed_out'])

    def test_cpu_limit(self):
        r = self.run_fake('while True: pass\n', replace(Budget(), cpu_seconds=1))
        self.assert_stop(r)
        self.assertNotEqual(r['export']['returncode'], 0)

    def test_nonzero_exit_not_coverage(self):
        self.assert_stop(self.run_fake('print("MessageName;total_ns")\nraise SystemExit(3)\n'))

    def test_empty_or_malformed_metadata_not_coverage(self):
        self.assert_stop(self.run_fake('print("MessageName;total_ns")\n'))

    def test_no_overwrite(self):
        out = self.p / 'out'
        out.mkdir()
        (out / 'evidence').write_text('keep')
        with self.assertRaises(FileExistsError):
            export(self.trace, self.p / 'absent', out)
        self.assertEqual((out / 'evidence').read_text(), 'keep')

    def test_duplicate_and_missing_cache_rejected(self):
        sample = HERE.parent / 'c10_burst_census/runs/smoke2/raw/tracy_ops_data.csv.gz'
        text = gzip.decompress(sample.read_bytes()).decode()
        p = self.p / 'messages.csv'
        # Duplicate the complete stream (without its header).
        p.write_text(text + text.split('\n', 1)[1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate(p)
        p.write_text('MessageName;total_ns\n`TT_DNN_DEVICE_OP: Op,123,0,true,1024`;1\n')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            validate(p)

    def test_report_hook_never_calls_old_report_or_process_ops(self):
        cli = types.SimpleNamespace(generate_report=lambda *a: self.fail('old report invoked'))
        with patch('bounded_tracy.export', return_value={'verdict': 'GO'}) as call:
            install_report_hook(cli, Budget())
            cli.generate_report(self.p, self.p / 'bin', None, None, False, [])
            self.assertEqual(call.call_args.args[0], self.p / '.logs/tracy_profile_log_host.tracy')
            self.assertEqual(call.call_args.args[2], self.p / 'metadata')
        with patch('bounded_tracy.export', return_value={'verdict': 'STOP', 'error': 'budget'}):
            with self.assertRaisesRegex(RuntimeError, 'STOP'):
                cli.generate_report(self.p, self.p, None, None)

    def test_archived_cli_calls_only_replacement_after_capture(self):
        import importlib.util
        import os
        fake = types.ModuleType('tracy')
        fake.sys, fake.os = sys, os
        fake.logger = types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)
        fake.PROFILER_ARTIFACTS_DIR = self.p
        fake.PROFILER_BIN_DIR = self.p / 'bin'
        fake.split_comma_list = lambda *a: None
        fake.get_available_port = lambda: '8086'
        capture = types.SimpleNamespace(communicate=lambda **k: None)
        fake.run_report_setup = lambda *a: (True, capture)
        fake.generate_report = lambda *a: self.fail('unbounded report called')
        fake.signal = types.SimpleNamespace(SIGINT=2, SIGTERM=15, signal=lambda *a: None)
        with patch('subprocess.Popen') as popen:
            popen.return_value.returncode = 0
            fake.subprocess = subprocess
            with patch.dict(sys.modules, tracy=fake), patch.dict(os.environ):
                spec = importlib.util.spec_from_file_location('reviewed_cli', HERE / 'source/tracy_main.py')
                cli = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cli)
                install_report_hook(cli, Budget())
                argv = ['bounded_tracy.py', '-r', '-p', '--check-exit-code',
                        '--enable-sum-profiling', '--dump-device-data-mid-run',
                        '--disable-device-data-push-to-tracy', '--op-support-count', '8192',
                        '--', 'census.py', '--out', 'output']
                with patch.object(sys, 'argv', argv), patch('bounded_tracy.export', return_value={'verdict': 'GO'}) as call:
                    cli.main()
                    call.assert_called_once()
                inner = popen.call_args.args[0][0]
                self.assertNotIn(' -r ', inner)
                self.assertIn('python3 -m tracy -p', inner)
                env = popen.call_args.kwargs['env']
                for name in ['TTNN_OP_PROFILER', 'TT_METAL_PROFILER_TRACE_TRACKING',
                             'TT_METAL_DEVICE_PROFILER', 'TT_METAL_PROFILER_SUM',
                             'TT_METAL_PROFILER_MID_RUN_DUMP', 'TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY']:
                    self.assertEqual(env[name], '1')
                self.assertEqual(env['TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT'], '8192')

    def test_archive_metadata_roundtrip_and_no_overwrite(self):
        from archive_replay import archive
        from export_metadata import digest
        raw = self.p / 'tracy/metadata'
        raw.mkdir(parents=True)
        metadata = raw / 'tracy_ops_data.csv'
        metadata.write_bytes(b'MessageName;total_ns\n')
        (raw / 'report.json').write_text(json.dumps({'verdict': 'GO', 'metadata': digest(metadata)}))
        (self.p / 'out').mkdir()
        (self.p / 'out/clock.jsonl').write_bytes(b'{"clock":1350}\n')
        r = archive(self.p)
        self.assertEqual(r['verdict'], 'GO')
        for row in r['files']:
            self.assertEqual(gzip.decompress((self.p / row['archive']).read_bytes()), Path(row['original']).read_bytes())
        with self.assertRaises(FileExistsError):
            archive(self.p)

    def test_archive_rejects_changed_metadata(self):
        from archive_replay import archive
        raw = self.p / 'tracy/metadata'
        raw.mkdir(parents=True)
        (raw / 'tracy_ops_data.csv').write_bytes(b'changed')
        (raw / 'report.json').write_text(json.dumps({'verdict': 'GO', 'metadata': {}}))
        with self.assertRaisesRegex(ValueError, 'matching'):
            archive(self.p)
        self.assertFalse((self.p / 'raw_manifest.json').exists())

    def test_harness_preserves_capture_flags(self):
        import shlex
        for current, old in [('run_census.sh', 'run_census.failed.sh')]:
            before = (HERE / 'source' / old).read_text()
            after = (HERE.parent / 'c10_burst_census' / current).read_text()
            before_args = shlex.split(before.split('python3 -m tracy ', 1)[1].replace('\\\n', ' '))
            after_args = shlex.split(after.split('--metal-root "$TT_METAL_HOME" -- ', 1)[1].replace('\\\n', ' '))
            after_args.remove('--check-exit-code')
            self.assertEqual(before_args, after_args)


if __name__ == '__main__':
    unittest.main()
