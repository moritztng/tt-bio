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


# --- sysfs sampler -------------------------------------------------------------------
#
# `tt-smi -s` snapshots EVERY chip and blocks on each one, so on a four-card host with three
# busy it takes seconds per sample and can hang outright. It is also the wrong instrument for
# a shared box: it reports the chips of other tenants, and `during` above indexes them by
# position, which is only the granted card because TT_VISIBLE_DEVICES happens to reorder them.
#
# sysfs answers the same question for one chip with a file read. The trap it replaces is worse
# than the cost: the lease card number and the /dev/tenstorrent node number are NOT the same on
# qb1 (lease card 1 is node 2), so indexing `tenstorrent!N` by the lease number reads an idle
# neighbours clock while the card under the fold throttles. Resolve the node by its PCI BDF.


# --- sysfs sampler -------------------------------------------------------------------
#
# `tt-smi -s` snapshots EVERY chip and blocks on each one, so on a four-card host with three
# busy it costs seconds per sample and can hang outright. It is also the wrong instrument for
# a shared box: it reports other tenants' chips and `during` above indexes them by position,
# which is only the granted card because TT_VISIBLE_DEVICES happens to reorder them.
#
# sysfs answers the same question for one chip with a file read. The trap it removes is worse
# than the cost it saves: the lease card number and the /dev/tenstorrent node number are NOT
# the same on qb1 (lease card 1 is node 2), so indexing `tenstorrent!N` by the lease number
# reads an idle neighbour's clock while the card under the fold throttles. Resolve by PCI BDF.

_SENTINEL = 0xFFFFFFFF


def node_for_bdf(bdf):
    """The /sys/class/tenstorrent node whose PCI device is `bdf`, or None."""
    import glob
    for n in glob.glob("/sys/class/tenstorrent/tenstorrent!*"):
        try:
            if os.path.basename(os.path.realpath(os.path.join(n, "device"))) == bdf:
                return n
        except OSError:
            pass
    return None


def bdf_for_node(node):
    """The PCI BDF behind a /sys/class/tenstorrent node path, or None."""
    try:
        return os.path.basename(os.path.realpath(os.path.join(node, "device")))
    except OSError:
        return None


class sysfs_during:
    """Sample ONE chip's AICLK from sysfs for the duration of a block.

    `bdf` names the card, so this cannot read a neighbour by mistake. Pass `node` instead
    when the caller already resolved it.

    A dead ARC answers `tt_aiclk` with 4294967295 without raising, and averaging that
    manufactures a clock for a chip that has none. Sentinels are counted separately and never
    enter the median; if every sample is one, `summary()` reports no reading rather than a
    number.
    """

    def __init__(self, bdf=None, node=None, period=0.5):
        self.node = node or (node_for_bdf(bdf) if bdf else None)
        self.bdf = bdf or (bdf_for_node(self.node) if self.node else None)
        self.samples = []
        self.sentinels = 0
        self.load = []
        self._stop = threading.Event()
        self._period = period

    def _run(self):
        path = os.path.join(self.node, "tt_aiclk") if self.node else None
        while not self._stop.is_set():
            self.load.append(os.getloadavg()[0] / (os.cpu_count() or 1))
            if path:
                try:
                    with open(path) as fh:
                        v = int(fh.read().strip())
                except (OSError, ValueError):
                    v = None
                if v == _SENTINEL:
                    self.sentinels += 1
                elif v is not None:
                    self.samples.append(v)
            time.sleep(self._period)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)
        return False

    def summary(self):
        s = sorted(self.samples)
        out = {"node": self.node, "bdf": self.bdf, "n": len(s),
               "sentinels": self.sentinels,
               "load_median": (round(sorted(self.load)[len(self.load) // 2], 2)
                               if self.load else None)}
        if s:
            out.update(min=s[0], max=s[-1], median=s[len(s) // 2])
        return out

    def line(self, dev=0):
        # `dev` is accepted and ignored: this sampler already holds exactly one chip, named by
        # its BDF. The argument is here so a harness can swap `during` for this one unchanged.
        s = self.summary()
        if not s["n"]:
            return ("CLOCK: NO READING from {} ({} sentinels) -- every timing here is "
                    "unclocked".format(s["bdf"], s["sentinels"]))
        return ("CLOCK: {} min {} / median {} / max {} MHz over {} samples DURING "
                "({} sentinels rejected), host load/core median {}".format(
                    s["bdf"], s["min"], s["median"], s["max"], s["n"],
                    s["sentinels"], s["load_median"]))
