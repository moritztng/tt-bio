"""Sample AICLK on pc every 2 s into a jsonl, so a fold's clock can be read DURING it.

pc reads the clock from the tt-kmd sysfs node instead of `tt-smi -s`. The node is the
driver's own telemetry, so it costs no BAR traffic and does not open the device, which
`tt-smi -s` does. The two agree at idle (both 800 MHz, checked before and after the run).
"""
import json, sys, time
out = sys.argv[1]
CLK = "/sys/class/tenstorrent/tenstorrent!0/tt_aiclk"
PWR = "/sys/class/hwmon/hwmon2/power1_input"   # microwatts
TMP = "/sys/class/hwmon/hwmon2/temp1_input"    # millidegrees C
def rd(p):
    with open(p) as f:
        return f.read().strip()
while True:
    try:
        rec = {"t": time.time(), "aiclk": int(rd(CLK)),
               "power": int(rd(PWR)) / 1e6, "temp": int(rd(TMP)) / 1e3}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    time.sleep(2)
