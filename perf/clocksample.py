"""AICLK sampling, shared by every perf harness in this tree.

A number without a clock is not a measurement, and on Blackhole the AICLK SETS the fold
time -- at 512 aa, 800 MHz reads 21.90 s where the 1350 burst reads 14.69 s. So the clock
has to be sampled DURING the work, on a thread, never read once before it.

Extracted from perf/hallgrad/memprobe.py, which had the only copy. Two harnesses needing
the same sampler is a shared-code problem, not a copy-paste one: an instrument that drifts
between rows makes their numbers incomparable.
"""

import glob
import json
import os
import shutil
import subprocess
import threading
import time

from tt_bio.runtime import aiclk_reading, umd_index_to_dev_node, visible_device_indices


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


#: Below this the clock, not the code, sets the time, so a number taken there is an artifact.
#: The floor is per arch and has to be, because the two architectures have different ceilings:
#: a Blackhole p150a/p300c bursts to 1350 and a contended one falls to 800, while a Wormhole
#: Galaxy chip's ceiling IS 1000 (it reads 1000 throughout a fold and 500 idle). Blackhole's
#: 1200 applied to Wormhole marks every honest Galaxy measurement `clock_artifact`, which is
#: how a whole architecture's perf numbers get read as regressions.
CLOCK_FLOOR_MHZ = {"blackhole": 1200, "wormhole_b0": 1000}
#: What to assume when the arch cannot be read (no card, no sysfs): the stricter floor.
CLOCK_FLOOR_DEFAULT_MHZ = 1200


def clock_floor_mhz(arch=None):
    """The AICLK below which a measurement on this card is an artifact, not a result."""
    if arch is None:
        try:
            from tt_bio.tenstorrent import arch_name
            arch = arch_name()
        except Exception:
            arch = None
    return CLOCK_FLOOR_MHZ.get(arch, CLOCK_FLOOR_DEFAULT_MHZ)


def is_clock_artifact(mhz, arch=None):
    """True when ``mhz`` is below this arch's floor. ``None`` (nothing sampled) is not a result
    either, but the callers differ on whether they discard it, so they keep deciding that."""
    return mhz is not None and mhz < clock_floor_mhz(arch)


def _sysfs_reading():
    """(AICLK MHz, board watts) per visible chip from the driver's sysfs telemetry, or None.

    `tt-smi -s` scans every chip on the box. On a 32-chip Wormhole Galaxy with its other chips
    serving it took over 40 s, past the 25 s timeout, so from 2026-11-01 05:14Z every MOL1 gate
    round on .111 came back unclocked. The same eight clocks read from sysfs in 11 ms.
    """
    nodes = umd_index_to_dev_node()
    vis = os.environ.get("TT_VISIBLE_DEVICES", "")
    rows = []
    for k in visible_device_indices(vis) if vis.strip() else sorted(nodes):
        d = f"/sys/class/tenstorrent/tenstorrent!{nodes.get(k)}"
        try:
            with open(f"{d}/tt_aiclk") as f:
                clk = int(f.read())
        except (OSError, ValueError):
            return None
        w = None
        for hw in glob.glob(f"{d}/device/hwmon/hwmon*/power1_input"):
            with open(hw) as f:
                w = int(f.read()) / 1e6  # microwatts
        rows.append((clk, w))
    return rows or None


def _tt_smi_reading():
    """(AICLK MHz, board watts) per chip tt-smi lists; it honours TT_VISIBLE_DEVICES."""
    r = subprocess.run([TT_SMI, "-s"], capture_output=True, text=True, timeout=25)
    tel = [dev.get("telemetry", {}) for dev in json.loads(r.stdout).get("device_info", [])]
    return [(t.get("aiclk"), None if t.get("power") is None else float(str(t["power"]).strip()))
            for t in tel]


def sample_aiclk(stop, out, *, period=2.0, load=None, power=None):
    """Append AICLK samples per device index into ``out`` until ``stop`` is set.

    Index k is the k-th chip of TT_VISIBLE_DEVICES (every chip when unset), so index 0 is the
    granted card rather than UMD 0. The reading comes from sysfs where the driver publishes it
    and from `tt-smi -s` otherwise.

    ``load``, if given, collects the host's 1-min loadavg / nproc at the same instants. On a
    shared host the scheduler times the fold as much as the chip does (whglx ran at 8.5x during
    the MGX re-record), so a runtime needs its load beside it just as it needs its clock.

    ``power``, if given, collects board power in watts per device index from the same reading.
    """
    while not stop.is_set():
        if load is not None:
            load.append(os.getloadavg()[0] / (os.cpu_count() or 1))
        try:
            for i, (clk, w) in enumerate(_sysfs_reading() or _tt_smi_reading()):
                clk = None if clk is None else aiclk_reading(int(clk))
                if clk is not None:
                    out.setdefault(i, []).append(clk)
                if power is not None and w is not None:
                    power.setdefault(i, []).append(w)
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
        self.power = {}
        self._stop = threading.Event()
        self._period = period

    def __enter__(self):
        self._t = threading.Thread(target=sample_aiclk,
                                   args=(self._stop, self.clocks),
                                   kwargs={"period": self._period, "load": self.load, "power": self.power},
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
