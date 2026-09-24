"""AICLK sampling, shared by every perf harness in this tree.

A number without a clock is not a measurement, and on Blackhole the AICLK SETS the fold
time -- at 512 aa, 800 MHz reads 21.90 s where the 1350 burst reads 14.69 s. So the clock
has to be sampled DURING the work, on a thread, never read once before it.

Extracted from perf/hallgrad/memprobe.py, which had the only copy. Two harnesses needing
the same sampler is a shared-code problem, not a copy-paste one: an instrument that drifts
between rows makes their numbers incomparable.
"""

import json
import os
import shutil
import subprocess
import threading
import time


def _tt_smi():
    """The first tt-smi that exists. `TT_SMI` overrides; then PATH, ~/.local/bin (the Blackhole
    boxes) and qb1's old absolute path. On whglx it is /usr/local/bin, where a path hardcoded to
    the last one silently sampled nothing and every Galaxy size-ladder cell had no clock."""
    for c in (os.environ.get("TT_SMI"), shutil.which("tt-smi"),
              os.path.expanduser("~/.local/bin/tt-smi"), "/home/ttuser/.local/bin/tt-smi"):
        if c and os.path.exists(c):
            return c
    return "tt-smi"


TT_SMI = _tt_smi()


def sample_aiclk(stop, out, *, period=2.0, load=None):
    """Append AICLK samples per device index into ``out`` until ``stop`` is set.

    With TT_VISIBLE_DEVICES exported, tt-smi honours it and reports the granted chip as
    index 0, so index 0 here is the granted card rather than UMD 0.

    ``load``, if given, collects the host's 1-min loadavg / nproc at the same instants. On a
    shared host the scheduler times the fold as much as the chip does (whglx ran at 8.5x during
    the MGX re-record), so a runtime needs its load beside it just as it needs its clock.
    """
    while not stop.is_set():
        if load is not None:
            load.append(os.getloadavg()[0] / (os.cpu_count() or 1))
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
        self.load = []
        self._stop = threading.Event()
        self._period = period

    def __enter__(self):
        self._t = threading.Thread(target=sample_aiclk,
                                   args=(self._stop, self.clocks),
                                   kwargs={"period": self._period, "load": self.load},
                                   daemon=True)
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
