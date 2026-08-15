#!/usr/bin/env python3
"""Read TTBridge's call counters out of a LIVE relion_refine_mpi, without stopping it.

TTBridge::report() only prints at exit, so the no-op trap of the state doc's parity gate cannot be
cleared until an arm has finished. A refinement is hours long and an arm that turns out to have been
declining the whole time is hours wasted. This reads the counters straight out of the running
process instead, so the trap clears in the first minutes.

Same mechanism as harvest_residual.py: resolve the PIE load base from /proc/PID/maps, resolve the
anonymous-namespace symbols with nm rather than hard-coding addresses (they move on every rebuild),
read /proc/PID/mem. Read-only, no ptrace attach, no stop. Needs sudo because yama/ptrace_scope is 1.

  sudo -n python3 harvest_counters.py $(pgrep -f relion_refine_mpi)
"""
import re
import struct
import subprocess
import sys

EXE = "/home/ttuser/relion-scratch/relion/build-fine/bin/relion_refine_mpi"
NAMES = ["g_handled", "g_declined", "g_fine_handled", "g_fine_declined"]

nm = subprocess.run(["nm", "-C", EXE], capture_output=True, text=True).stdout
OFF = {}
for line in nm.splitlines():
    m = re.match(r"^([0-9a-f]+) . \(anonymous namespace\)::(\w+)$", line)
    if m and m.group(2) in NAMES:
        OFF[m.group(2)] = int(m.group(1), 16)
missing = [k for k in NAMES if k not in OFF]
if missing:
    raise SystemExit(f"symbols not in {EXE}: {missing}")


def base_of(pid):
    for line in open(f"/proc/{pid}/maps"):
        if line.rstrip().endswith(EXE):
            return int(line.split("-")[0], 16)
    raise SystemExit(f"exe mapping not found for {pid}")


tot = {k: 0 for k in NAMES}
for pid in sys.argv[1:]:
    b = base_of(pid)
    vals = {}
    with open(f"/proc/{pid}/mem", "rb") as m:
        for k in NAMES:
            m.seek(b + OFF[k])
            vals[k] = struct.unpack("<q", m.read(8))[0]
            tot[k] += vals[k]
    print(pid, vals)
print("TOTAL", tot)
