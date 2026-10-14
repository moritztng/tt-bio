"""tel.py <out.csv> [period_s]: sample every Blackhole chip's AICLK (sysfs tt_aiclk) and hwmon in/curr/power/temp
inputs into one CSV until SIGTERM. Reads sysfs only, opens no device. spd-nocdeadlock2 slot test."""
import glob, os, re, signal, sys, time
out, period = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
chips = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
src = []
for n in chips:
    base = f"/sys/class/tenstorrent/tenstorrent!{n}"
    src.append((f"c{n}_aiclk", f"{base}/tt_aiclk"))
    for f in sorted(glob.glob(f"{base}/device/hwmon/hwmon*/*_input")):
        if re.match(r"(in|curr|power|temp)\d+_input$", os.path.basename(f)):
            src.append((f"c{n}_{os.path.basename(f)[:-6]}", f))
def rd(p):
    try:
        return open(p).read().split()[0]
    except Exception:
        return ""
with open(out, "w", buffering=1) as fh:
    fh.write("t," + ",".join(k for k, _ in src) + "\n")
    while True:
        fh.write(f"{time.time():.2f}," + ",".join(rd(p) for _, p in src) + "\n")
        time.sleep(period)
