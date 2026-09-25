"""Sample AICLK on one card every 2 s into a jsonl, so a fold's clock can be read DURING it."""
import json, subprocess, sys, time
card = int(sys.argv[1]); out = sys.argv[2]
while True:
    try:
        d = json.loads(subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"],
                                      capture_output=True, text=True, timeout=20).stdout)
        t = d["device_info"][card]["telemetry"]
        rec = {"t": time.time(), "aiclk": int(str(t["aiclk"]).strip()),
               "power": float(str(t["power"]).strip()),
               "temp": float(str(t["asic_temperature"]).strip())}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    time.sleep(2)
