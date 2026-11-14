"""Sample one Tenstorrent chip's AICLK, power, current, vcore and temperature at a fixed rate until killed.

pw7 showed the BH 6B forward ~11 % slower back to back than with 150 ms of rest between forwards, while the 1 Hz
AICLK sampler of run.sh read 1350 throughout. This reads the same sysfs nodes ~50x a second, so a throttle that
lives inside a forward shows up. The chip is the /dev/tenstorrent/N node; its hwmon is the one on the same PCI device.

    python perf/spd_m_esmc/telem.py --node 1 --hz 50 --out telem.tsv     (columns: epoch MHz W A mV C)
"""
import argparse
import signal
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--node", type=int, required=True)
ap.add_argument("--hz", type=float, default=50)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tt = Path(f"/sys/class/tenstorrent/tenstorrent!{a.node}")
pci = (tt / "device").resolve()
hw = next(h for h in Path("/sys/class/hwmon").iterdir() if (h / "device").resolve() == pci)
nodes = [tt / "tt_aiclk"] + [hw / f for f in ("power1_input", "curr1_input", "in0_input", "temp1_input")]
scale = [1, 1e-6, 1e-3, 1, 1e-3]
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
dt = 1 / a.hz
with open(a.out, "w", buffering=1) as f:
    t = time.time()
    while True:
        vals = [float(n.read_text()) * s for n, s in zip(nodes, scale)]
        f.write(f"{time.time():.3f}\t" + "\t".join(f"{v:g}" for v in vals) + "\n")
        t += dt
        time.sleep(max(0.0, t - time.time()))
