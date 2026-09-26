"""`launch.sh` must RELEASE its lease when the run ends.

This is the defect that made the test worth writing: the script ended in `exec python ...`, and
`exec` replaces the shell, which takes its EXIT trap with it. So the lease file survived every
run and the next reader of `state/leases/` saw a card leased to a pid that had exited. On this
fleet that is not cosmetic -- `fleet.sh:remote_busy_raw` reads those files, and a card held by a
dead lease is a card nobody takes.

The assertion that `launch.sh` has no `exec` line is only half of it. A test that greps for a
keyword passes just as well on a script that never released the lease for some OTHER reason, so
the second half RUNS both shapes and shows the mechanism: the `exec` control leaks the file and
the shape `launch.sh` now has does not. The control has to fail, or the subject proves nothing.
"""
import pathlib
import re
import subprocess

LAUNCH = pathlib.Path(__file__).resolve().parents[1] / "perf" / "bcx_bwbytes" / "launch.sh"

_LEAKS = """#!/bin/bash
L=$1
echo held > "$L"
trap 'rm -f "$L"' EXIT INT TERM
exec /bin/true
"""

_RELEASES = """#!/bin/bash
L=$1
echo held > "$L"
cleanup() { rm -f "$L"; }
trap cleanup EXIT
/bin/true &
CHILD=$!
wait "$CHILD"
"""


def _run(tmp_path, body, name):
    script = tmp_path / f"{name}.sh"
    script.write_text(body)
    lease = tmp_path / f"{name}.lease"
    subprocess.run(["bash", str(script), str(lease)], check=True)
    return lease.exists()


def test_exec_control_leaks_the_lease(tmp_path):
    """The control: the shape `launch.sh` USED to have, which must still leak."""
    assert _run(tmp_path, _LEAKS, "leaks") is True


def test_child_and_wait_shape_releases_the_lease(tmp_path):
    """The subject's shape, run rather than read."""
    assert _run(tmp_path, _RELEASES, "releases") is False


def test_launch_sh_does_not_exec_away_its_trap():
    text = LAUNCH.read_text()
    execs = [ln for ln in text.splitlines() if re.match(r"\s*exec\s+\S", ln)]
    assert execs == [], f"launch.sh execs away its EXIT trap: {execs}"


def test_launch_sh_waits_on_a_child_and_still_traps():
    text = LAUNCH.read_text()
    assert "trap cleanup EXIT" in text
    assert re.search(r'wait "\$CHILD"', text), "no wait on the python child"
    # SIGKILL on a device holder leaves the card unopenable and the next open hard-hangs the host.
    assert "kill -KILL" not in text and "kill -9" not in text
