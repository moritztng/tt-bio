#!/usr/bin/env python3
"""Per-card AICLK + host contention, sampled every 30 s for the size-ladder arm.

Two things this exists to record, because neither can be reconstructed afterwards:

  AICLK, DURING the fold. On this box the governor alone swings a 512 aa fold 1.27-1.41x, so a
  ladder timed at an unrecorded clock is not a measurement. Sampled PER CARD -- a snapshot that
  greps the first AICLK it finds reports whichever card tt-smi happened to list first, and on this
  host three cards idle at 800 while one folds at 1350.

  Host contention. benchlock samples load ONCE, at acquisition, and is blind to a co-tenant that
  arrives afterwards; on 2026-09-20 this arm acquired at loadavg 2.72 and had a 565 % job land on
  it 12 seconds later. size-ladder scores RUNTIME against scaling exponents and runtime_s carries a
  size-independent host term, so contention inflates the small rungs most and pushes the exponent
  down. A red judged without this record cannot be told from a co-tenant.

qb2 has 16 cores. Nothing here kills or throttles anything: other rows' jobs are their work.
"""
import json, os, subprocess, time

OUT = os.environ.get("CONTENTION_OUT",
                     "/home/ttuser/pvx_arms/gateland_sizeladder_contention.jsonl")
TT = os.path.expanduser("~/.local/bin/tt-smi")


def aiclk():
    try:
        d = json.loads(subprocess.check_output([TT, "-s"], text=True, timeout=60))
    except Exception as e:
        return {"error": str(e)[:80]}
    out = {}
    for i, c in enumerate(d.get("device_info", [])):
        out[str(i)] = c.get("telemetry", {}).get("aiclk")
    return out


def top(n=4):
    try:
        raw = subprocess.check_output(
            ["ps", "-eo", "pid,pcpu,comm,args", "--sort=-pcpu"], text=True, timeout=30)
    except Exception:
        return []
    rows = []
    for line in raw.splitlines()[1:n + 1]:
        f = line.split(None, 3)
        if len(f) == 4 and float(f[1]) >= 10.0:
            rows.append({"pid": int(f[0]), "pcpu": float(f[1]), "cmd": f[3][:110]})
    return rows


while True:
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "loadavg": os.getloadavg(), "ncpu": os.cpu_count(),
           "aiclk": aiclk(), "top": top()}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    time.sleep(30)
