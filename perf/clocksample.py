"""AICLK sampling, shared by every perf harness in this tree.

A number without a clock is not a measurement, and on Blackhole the AICLK SETS the fold
time -- at 512 aa, 800 MHz reads 21.90 s where the 1350 burst reads 14.69 s. So the clock
has to be sampled DURING the work, on a thread, never read once before it.

Extracted from perf/hallgrad/memprobe.py, which had the only copy. Two harnesses needing
the same sampler is a shared-code problem, not a copy-paste one: an instrument that drifts
between rows makes their numbers incomparable.
"""

import json
import subprocess
import threading
import time

TT_SMI = "/home/ttuser/.local/bin/tt-smi"


def sample_aiclk(stop, out, *, period=2.0):
    """Append AICLK samples per device index into ``out`` until ``stop`` is set.

    With TT_VISIBLE_DEVICES exported, tt-smi honours it and reports the granted chip as
    index 0, so index 0 here is the granted card rather than UMD 0.
    """
    while not stop.is_set():
        try:
            r = subprocess.run([TT_SMI, "-s"], capture_output=True, text=True, timeout=25)
            d = json.loads(r.stdout)
            for i, dev in enumerate(d.get("device_info", [])):
                clk = dev.get("telemetry", {}).get("aiclk")
                if clk is not None:
                    out.setdefault(i, []).append(int(clk))
        except Exception:
            pass
        time.sleep(period)


class during:
    """Context manager: sample the clock for the duration, then report it.

    ``with during() as clk: ...`` then ``clk.summary()`` gives the min/max/median and the
    sample count per device, which is what every perf line in this tree has to carry.
    """

    def __init__(self, period=2.0):
        self.clocks = {}
        self._stop = threading.Event()
        self._period = period

    def __enter__(self):
        self._t = threading.Thread(target=sample_aiclk,
                                   args=(self._stop, self.clocks),
                                   kwargs={"period": self._period}, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=30)
        return False

    def summary(self):
        out = {}
        for i, xs in self.clocks.items():
            if xs:
                s = sorted(xs)
                out[i] = {"min": s[0], "max": s[-1], "median": s[len(s) // 2],
                          "n": len(s)}
        return out

    def line(self, dev=0):
        s = self.summary().get(dev)
        if not s:
            return "CLOCK: NOT SAMPLED -- treat every timing here as unclocked"
        return (f"CLOCK: dev{dev} min {s['min']} / median {s['median']} / max {s['max']} MHz "
                f"over {s['n']} samples, polled DURING the work")
