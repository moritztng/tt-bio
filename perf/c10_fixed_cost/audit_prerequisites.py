"""Read-only evidence for C10's quiet-board measurement prerequisite."""
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HOLDERS = """
import json, os
from pathlib import Path
rows=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    nodes=set()
    try:
        for fd in (p/'fd').iterdir():
            try: target=os.readlink(fd)
            except OSError: continue
            if target.startswith('/dev/tenstorrent/'): nodes.add(target)
        if not nodes: continue
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        env=dict(e.split('=',1) for e in (p/'environ').read_text().split(chr(0)) if '=' in e)
        rows.append({'pid':int(p.name),'nodes':sorted(nodes),
                     'state':stat[0], 'utime_ticks':int(stat[11]), 'stime_ticks':int(stat[12]),
                     'starttime_ticks':int(stat[19]),
                     'argv':(p/'cmdline').read_text().replace(chr(0),' '),
                     'env':{k:env.get(k) for k in ('TT_VISIBLE_DEVICES','TT_BIO_LEASE_CARDS','TT_BIO_LEASE_HOLDER')}})
    except (OSError,PermissionError): continue
print(json.dumps(rows))
"""

def main():
    result = {
        "host":socket.gethostname(), "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "containment":subprocess.check_output(["systemctl","is-active","qb2-endpoint-containment.service"],text=True).strip(),
        "module_srcversion":Path("/sys/module/tenstorrent/srcversion").read_text().strip(),
        "cpu_ticks_per_second":os.sysconf("SC_CLK_TCK"), "snapshots":[],
        "scope":"Read-only prerequisite audit. No device opened, model run, roof or floor measured.",
        "clock_note":"Node 0 idle telemetry is not during-fold coverage and is not a performance measurement.",
        "observer_pid":os.getpid(),
    }
    for i in range(3):
        snap={"host_monotonic_ns":time.monotonic_ns(),
              "utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "loadavg":os.getloadavg(),
              "holders":json.loads(subprocess.check_output(["sudo","-n","python3","-c",HOLDERS],text=True))}
        before=time.monotonic_ns()
        try:
            snap["idle_card0_clock"]={"read_start_ns":before,
                "MHz":int(Path("/sys/class/tenstorrent/tenstorrent!0/tt_aiclk").read_text()),
                "read_end_ns":time.monotonic_ns()}
        except (OSError,ValueError) as e:
            snap["idle_card0_clock"]={"error":repr(e)}
        result["snapshots"].append(snap)
        if i<2: time.sleep(5)
    result["assigned_node_available"]=not any("/dev/tenstorrent/0" in h["nodes"] for s in result["snapshots"] for h in s["holders"])
    result["mapping"]={x.name:str(x.resolve()) for x in Path("/sys/class/tenstorrent").iterdir()}
    result["quiet_board_available"]=not any(s["holders"] for s in result["snapshots"])
    result["verdict"]="QUIET" if result["quiet_board_available"] else "PREREQUISITE_UNAVAILABLE"
    result["probe_elapsed_ns"]=time.monotonic_ns()-result["snapshots"][0]["host_monotonic_ns"]
    out=ROOT/"perf/c10_dm_control/preflight.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()

