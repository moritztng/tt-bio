"""Stopping the gate stops its folds.

`_run_fold` starts each fold in its own session so a hang can be killed as a group, which also
hides the fold from the SIGINT that stops the gate. On 2026-10-09 a stopped size-ladder leg left
its census fold on .107 chip 17 running with ppid 1, holding the device after the chip's flock
had been released to the next job. Host-only: the "fold" is a sleep.
"""
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_sigint_to_the_gate_reaches_the_fold(tmp_path):
    pytest.importorskip("torch")                # release_gate imports it at module level
    pidfile = tmp_path / "fold.pid"
    gate = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("rg", {str(REPO / "scripts" / "release_gate.py")!r})
        rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)
        rg._release_driver_device = lambda: None
        rg._run_fold([sys.executable, "-c",
                      "import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(600)"],
                     600)
    """)], cwd=REPO)
    for _ in range(600):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.1)
    fold = int(pidfile.read_text())
    gate.send_signal(signal.SIGINT)
    gate.wait(timeout=180)
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(fold, 0)
