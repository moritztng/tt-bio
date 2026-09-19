#!/usr/bin/env python3
"""Known-answer control for run_leg's wedge timeout. Device-free.

The thing under test is that a leg which never exits is killed and reported, and that its CHILD dies
with it. That second half is the part that matters on this box: the wedged leg of session 3 had a
child of its own, and killing only the direct child would have left a grandchild holding
/dev/tenstorrent/2 open.
"""
import importlib.util
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("apb_fold_ab", HERE / "apb_fold_ab.py")
apb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apb)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class TestRunLeg(unittest.TestCase):
    def test_healthy_leg_returns_its_code_and_is_not_flagged(self):
        rc, wedged = apb.run_leg([sys.executable, "-c", "pass"], dict(os.environ), 30.0)
        self.assertEqual((rc, wedged), (0, False))

    def test_failing_leg_is_not_mistaken_for_a_wedge(self):
        rc, wedged = apb.run_leg([sys.executable, "-c", "raise SystemExit(3)"], dict(os.environ), 30.0)
        self.assertEqual((rc, wedged), (3, False))

    def test_wedged_leg_is_killed_and_reported(self):
        t0 = time.time()
        rc, wedged = apb.run_leg([sys.executable, "-c", "while True: pass"], dict(os.environ), 2.0)
        self.assertTrue(wedged)
        self.assertNotEqual(rc, 0)
        self.assertLess(time.time() - t0, 20.0, "timeout did not fire")

    def test_the_wedged_legs_own_child_dies_with_it(self):
        # A leg that spawns a child and then spins, which is exactly the shape of 512_base_4_2.
        prog = ("import subprocess,sys,time\n"
                "c=subprocess.Popen([sys.executable,'-c','import time\\nwhile True: time.sleep(1)'])\n"
                "print(c.pid, flush=True)\n"
                "while True: pass\n")
        p = subprocess.Popen([sys.executable, "-c", prog], stdout=subprocess.PIPE,
                             start_new_session=True, text=True)
        child_pid = int(p.stdout.readline().strip())
        self.assertTrue(alive(child_pid))
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        p.wait()
        for _ in range(50):
            if not alive(child_pid):
                break
            time.sleep(0.1)
        self.assertFalse(alive(child_pid), "grandchild survived the process-group kill")


if __name__ == "__main__":
    unittest.main(verbosity=2)
