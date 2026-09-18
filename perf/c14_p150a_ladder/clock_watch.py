"""Sample AICLK + host load into a jsonl while a long record pass runs beside it.

The size-ladder record is hours of folds in a subprocess chain, so the clock cannot be
sampled from inside it the way perf/clocksample.py does for a single timed region. This
runs as its own process for the whole campaign and stops when the sentinel file goes away.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from perf.clocksample import TT_SMI  # noqa: E402
import subprocess  # noqa: E402

out_path, sentinel = sys.argv[1], sys.argv[2]
period = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
with open(out_path, "a", buffering=1) as fh:
    while os.path.exists(sentinel):
        rec = {"t": round(time.time(), 1), "load1": os.getloadavg()[0]}
        try:
            r = subprocess.run([TT_SMI, "-s"], capture_output=True, text=True, timeout=40)
            d = json.loads(r.stdout)
            rec["aiclk"] = [dev.get("telemetry", {}).get("aiclk")
                            for dev in d.get("device_info", [])]
            rec["power"] = [dev.get("telemetry", {}).get("power")
                            for dev in d.get("device_info", [])]
        except Exception as exc:
            rec["err"] = str(exc)[:120]
        fh.write(json.dumps(rec) + "\n")
        time.sleep(period)
