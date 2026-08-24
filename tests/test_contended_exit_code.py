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


if __name__ == "__main__":
    test_reserved_code_is_ex_tempfail_and_distinct()
    test_cli_maps_contention_to_the_code_and_leaves_real_bugs_at_1()
    test_device_parity_maps_contention_to_the_code()
    test_release_gate_records_contended_legs_and_only_those()
    test_release_gate_labels_both_command_shapes()
    print("ALL CONTENDED-EXIT-CODE TESTS PASSED")
