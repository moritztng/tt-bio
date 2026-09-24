"""Run one command on the pinned chip with AICLK and 1-min load sampled DURING it.

    python perf/mgx_wide_seq/run.py <out.json> -- <cmd ...>

Writes {cmd, rc, wall_s, aiclk{min,median,max,n}, load{max,median,n}} to <out.json>.meta.json.
AICLK comes from perf/clocksample.py via release_gate's own wrapper, so it is attributed to the
one granted card exactly as a ladder fold's is.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import release_gate as rg  # noqa: E402

out, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
loads, stop = [], threading.Event()


def _load():
    n = os.cpu_count()
    while not stop.wait(5.0):
        loads.append(round(os.getloadavg()[0] / n, 3))


th = threading.Thread(target=_load, daemon=True)
th.start()
t0 = time.time()
with rg._clock_during() as clk:
    rc = subprocess.call(cmd, cwd=ROOT)
stop.set()
ls = sorted(loads)
meta = {"cmd": cmd, "rc": rc, "wall_s": round(time.time() - t0, 1),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "aiclk": rg._aiclk_cell(clk),
        "load_per_nproc": {"max": ls[-1], "median": ls[len(ls) // 2], "n": len(ls)} if ls else None}
Path(out + ".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
print(json.dumps(meta), flush=True)
sys.exit(rc)
