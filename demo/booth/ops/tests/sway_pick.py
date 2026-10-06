"""watchdog.booth_sway picks the sway under session/session.sh, whatever its place in pgrep's list.
Two fake "sway" processes (copies of sleep): a decoy started first (lower pid, no session loop)
and one under a script called session.sh. Control: the old rule, first pid listed, picks the decoy.
Sends no signal to anything. Run with any python3."""
import os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from watchdog import booth_sway

d = Path(tempfile.mkdtemp(prefix="sway_pick."))
shutil.copy("/bin/sleep", d / "sway")
(d / "session.sh").write_text(f"#!/bin/sh\n{d}/sway 30\n")
(d / "session.sh").chmod(0o755)
decoy = subprocess.Popen([str(d / "sway"), "30"])
time.sleep(0.2)
loop = subprocess.Popen([str(d / "session.sh")])
time.sleep(0.5)
try:
    mine = [p for p in subprocess.run(["pgrep", "-x", "sway"], capture_output=True, text=True).stdout.split()
            if Path(f"/proc/{p}/exe").resolve() == (d / "sway").resolve()]
    under = [p for p in mine if p != str(decoy.pid)][0]
    ok = booth_sway(mine) == int(under)
    control = int(mine[0]) == int(under)
    print(f"candidates {mine}: decoy {decoy.pid}, under session.sh {under}")
    print(f"booth_sway -> {booth_sway(mine)}: {'PASS' if ok else 'FAIL'}")
    print(f"control, first listed -> {mine[0]}: {'picks the decoy, as expected' if not control else 'UNEXPECTED'}")
    rc = 0 if ok and not control else 1
finally:
    for p in (decoy, loop):
        p.terminate()
    subprocess.run(["pkill", "-TERM", "-f", f"{d}/sway"])
    shutil.rmtree(d, ignore_errors=True)
sys.exit(rc)
