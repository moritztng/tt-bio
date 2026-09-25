"""Run one command and measure it the way JapanFold's two predict watchdogs do.

    python perf/mgx_opendde_watchdog/silence.py <out.json> -- <cmd ...>

The platform kills a predict on either of two clocks (aiand-bio japanfold/jobs.py): the whole
job's wall past catalog.max_runtime_s, and max_stall_s (600 s) with no growth of the job's log.
The second is invisible to a harness that captures output and writes it at the end, so this one
reads the child's merged stdout/stderr as it arrives and records when each chunk landed. Its
longest gap is the silence the stall watchdog would see. AICLK and load/nproc are sampled DURING
the run (perf/clocksample.py); the raw output goes to <out.json>.log.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clocksample import during  # noqa: E402

out, cmd = Path(sys.argv[1]), sys.argv[sys.argv.index("--") + 1:]
arrivals, gap, gap_end = [], 0.0, None
t0 = time.time()
with during(period=10.0) as clk, open(f"{out}.log", "wb") as log:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    last = t0
    while chunk := os.read(p.stdout.fileno(), 65536):
        now = time.time()
        if now - last > gap:
            gap, gap_end = now - last, now - t0
        last = now
        arrivals.append(round(now - t0, 1))
        log.write(chunk)
        log.flush()
    rc = p.wait()
end = time.time()
if end - last > gap:   # silence between the last byte and exit counts too
    gap, gap_end = end - last, end - t0
loads = sorted(round(x, 3) for x in clk.load)
meta = {"cmd": cmd, "rc": rc, "wall_s": round(end - t0, 1),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "max_silence_s": round(gap, 1), "max_silence_ends_at_s": round(gap_end or 0, 1),
        "chunks": len(arrivals), "aiclk": clk.summary().get(0),
        "load_per_nproc": ({"max": loads[-1], "median": loads[len(loads) // 2], "n": len(loads)}
                           if loads else None),
        "nproc": os.cpu_count()}
out.write_text(json.dumps(meta, indent=1) + "\n")
Path(f"{out}.arrivals.json").write_text(json.dumps(arrivals) + "\n")
print(json.dumps(meta), flush=True)
sys.exit(rc)
