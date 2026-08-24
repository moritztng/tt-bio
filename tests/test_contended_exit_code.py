"""Proof that device contention gets its own exit code, so a gate cannot score it as accuracy.

Device-free. A leg that cannot open the card waits out TT_BIO_LEASE_TIMEOUT and, before this,
exited 1 -- indistinguishable from the model being wrong. Every release-gate arm renders a
non-zero rc as its own verdict ("missed the ground-truth floor", "drifted run to run"), so
eleven v0.7.0 gate legs across two passes read as model defects while not one of them executed a
device instruction. See state/nesso1-device-parity-crash-diagnose.md.

Run: python3 tests/test_contended_exit_code.py, or as part of the release suite via pytest.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from tt_bio.device_lease import CONTENDED_EXIT_CODE, DeviceInUseError


def test_reserved_code_is_ex_tempfail_and_distinct():
    """75 is sysexits EX_TEMPFAIL. It must not collide with the parent-death guard's 70, nor
    with the 0/1 a normal PASS/FAIL uses, or the whole signal is lost."""
    assert CONTENDED_EXIT_CODE == 75
    assert CONTENDED_EXIT_CODE not in (0, 1, 70)


def test_cli_maps_contention_to_the_code_and_leaves_real_bugs_at_1():
    import click
    from click.testing import CliRunner

    from tt_bio.main import _Cli

    @click.group(cls=_Cli)
    def g():
        pass

    @g.command()
    def contended():
        raise DeviceInUseError("physical card 1 on h is in use by worker:x (pid 9)")

    @g.command()
    def broken():
        raise RuntimeError("a real bug")

    res = CliRunner().invoke(g, ["contended"])
    assert res.exit_code == CONTENDED_EXIT_CODE, res.exit_code
    assert "device contention" in res.output

    # The whole point is the DISTINCTION: a real failure must stay a real failure.
    assert CliRunner().invoke(g, ["broken"]).exit_code == 1


def test_device_parity_maps_contention_to_the_code():
    """The nesso1 gate arm's harness, which is where this was found."""
    script = os.path.join(REPO, "scripts", "nesso1_port", "device_parity.py")
    src = open(script).read()
    assert "CONTENDED_EXIT_CODE" in src and "DeviceInUseError" in src, \
        "device_parity.py must map a contended device open to the reserved code"
    # Exercise the mapping without a card: a stand-in main() that raises what get_device would.
    stub = (
        "import sys; sys.path.insert(0, %r)\n"
        "from tt_bio.device_lease import CONTENDED_EXIT_CODE, DeviceInUseError\n"
        "def main():\n"
        "    raise DeviceInUseError('card 1 in use by worker:x')\n"
        "try:\n"
        "    raise SystemExit(main())\n"
        "except DeviceInUseError:\n"
        "    raise SystemExit(CONTENDED_EXIT_CODE)\n" % REPO
    )
    rc = subprocess.run([sys.executable, "-c", stub]).returncode
    assert rc == CONTENDED_EXIT_CODE, rc


def test_release_gate_records_contended_legs_and_only_those():
    import release_gate as rg

    rg._CONTENDED.clear()
    assert rg._run_fold([sys.executable, "-c", f"raise SystemExit({CONTENDED_EXIT_CODE})"], 60) \
        == (CONTENDED_EXIT_CODE, False)
    assert rg._CONTENDED, "a contended leg must be recorded for the not-a-verdict report"

    before = len(rg._CONTENDED)
    assert rg._run_fold([sys.executable, "-c", "raise SystemExit(1)"], 60) == (1, False)
    assert rg._run_fold([sys.executable, "-c", "pass"], 60) == (0, False)
    assert len(rg._CONTENDED) == before, "only exit 75 is contention; 0 and 1 mean what they say"
    rg._CONTENDED.clear()


def test_release_gate_labels_both_command_shapes():
    """The gate spawns two shapes: `-m tt_bio.main <sub>` and a bare script path."""
    import release_gate as rg

    class _Done:
        def wait(self, timeout=None):
            return CONTENDED_EXIT_CODE

    real_popen = subprocess.Popen
    subprocess.Popen = lambda *a, **k: _Done()
    try:
        rg._CONTENDED.clear()
        rg._run_fold([sys.executable, "-m", "tt_bio.main", "predict", "x.yaml"], 60)
        rg._run_fold([sys.executable, "/a/b/device_parity.py", "--fixture", "t"], 60)
        assert rg._CONTENDED == ["tt-bio predict", "device_parity.py"], rg._CONTENDED
    finally:
        subprocess.Popen = real_popen
        rg._CONTENDED.clear()


def _dead(name, exitcode):
    """Stand-in for a multiprocessing.Process that has already exited."""
    class _P:
        pass
    proc = _P()
    proc.name, proc.exitcode = name, exitcode
    proc.is_alive = lambda: False
    return proc


def test_predict_reports_a_contended_fan_out_as_contention_not_a_run_failure():
    """The other half of the bug. The workers die in CHILD processes, so the parent only ever
    saw exit codes -- it raised a plain RuntimeError and the CLI exited 1. Six of the eleven
    v0.7.0 legs reached the gate table through this line, as `every local worker exited ...
    (SpawnProcess-1 exit 0)`, and were rendered as accuracy misses."""
    from tt_bio.main import _stream_run

    class _Client:
        def events(self, run_id, after):
            return {"events": [], "status": "running"}

    def run(procs):
        return _stream_run(_Client(), "r1", total=1, n_workers=len(procs), debug=True,
                           log=False, results_path=None, struct_dir=None, model="m",
                           local_procs=procs)

    # A co-tenant took the card: contention, and the CLI group turns this into exit 75.
    try:
        run([_dead("SpawnProcess-1", CONTENDED_EXIT_CODE)])
    except DeviceInUseError as exc:
        assert "leased by another process" in str(exc), exc
    else:
        raise AssertionError("a contended fan-out must raise DeviceInUseError")

    # Any other dead-worker cause stays a run failure, and must NOT claim to know why.
    try:
        run([_dead("SpawnProcess-1", 1)])
    except DeviceInUseError:
        raise AssertionError("exit 1 is not contention")
    except RuntimeError as exc:
        assert "traceback above says why" in str(exc), exc


def test_worker_device_open_exits_on_the_code_only_for_contention():
    """run_worker_loop used to `return` on a failed device open, i.e. exit 0 -- a worker that
    never opened a chip was indistinguishable from one that finished its work. Device-free:
    tt_bio.tenstorrent is stubbed, so no card is touched and ttnn is never imported."""
    stub = (
        "import sys, types; sys.path.insert(0, %r)\n"
        "err = sys.argv[1]\n"
        "from tt_bio.device_lease import DeviceInUseError\n"
        "m = types.ModuleType('tt_bio.tenstorrent')\n"
        "def get_device():\n"
        "    raise (DeviceInUseError('card 1 in use by worker:x (pid 9)')\n"
        "           if err == 'contended' else RuntimeError('chip failed to come up'))\n"
        "m.get_device = get_device\n"
        "sys.modules['tt_bio.tenstorrent'] = m\n"
        "from tt_bio.worker import run_worker_loop\n"
        "run_worker_loop('http://127.0.0.1:1/nope', {\n"
        "    'worker_id': 'w0', 'device_id': 0, 'host': 'h', 'label': 'w0',\n"
        "    'accelerator': 'tenstorrent', 'visible_devices': '0'}, debug=True)\n"
    ) % REPO
    contended = subprocess.run([sys.executable, "-c", stub, "contended"],
                               capture_output=True, text=True)
    assert contended.returncode == CONTENDED_EXIT_CODE, (contended.returncode, contended.stderr)
    # A wedged or badly-brought-up chip is a real failure, not a retriable co-tenant.
    other = subprocess.run([sys.executable, "-c", stub, "wedged"],
                           capture_output=True, text=True)
    assert other.returncode == 1, (other.returncode, other.stderr)


if __name__ == "__main__":
    test_reserved_code_is_ex_tempfail_and_distinct()
    test_cli_maps_contention_to_the_code_and_leaves_real_bugs_at_1()
    test_device_parity_maps_contention_to_the_code()
    test_release_gate_records_contended_legs_and_only_those()
    test_release_gate_labels_both_command_shapes()
    test_predict_reports_a_contended_fan_out_as_contention_not_a_run_failure()
    test_worker_device_open_exits_on_the_code_only_for_contention()
    print("ALL CONTENDED-EXIT-CODE TESTS PASSED")
