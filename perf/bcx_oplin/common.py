"""Shared by the bcx-oplin scripts: the two arms of `ops.linear`, and an AICLK sampler.

The "off" arm rebinds `tt_bio.ops._via2d` to plain `fn(x)`. `ops.linear` looks `_via2d` up as
a module global on every call, so the rebind reaches every caller of `ops.linear`, and nothing
else: `autograd` holds its own reference, and inference never goes through it.
"""
import contextlib
import math
import os
import statistics
import threading
import time

import tt_bio.ops as ops

_ON = ops._via2d


def _off(x, fn, kw=None):
    return fn(x)


@contextlib.contextmanager
def arm(on: bool):
    prev = ops._via2d
    ops._via2d = _ON if on else _off
    try:
        yield
    finally:
        ops._via2d = prev


class Census:
    """Every `ops.linear` call that reaches `_via2d`: shape, whether it collapsed, and why not."""

    def __init__(self):
        self.rows = {}

    def __enter__(self):
        import ttnn
        inner = ops._via2d

        def counted(x, fn, kw=None):
            s = tuple(int(d) for d in x.shape)
            mc = (kw or {}).get("memory_config")
            why = ("rank<=2" if len(s) <= 2 else
                   "batch of one" if math.prod(s[:-2]) <= 1 else
                   "rows%32" if s[-2] % ttnn.TILE_SIZE else
                   "not TILE" if x.layout != ttnn.TILE_LAYOUT else
                   "sharded x" if x.is_sharded() else
                   "program_config" if (kw or {}).get("program_config") is not None else
                   "sharded out" if (mc is not None and mc.is_sharded()) else "collapsed")
            k = (s, why)
            self.rows[k] = self.rows.get(k, 0) + 1
            return inner(x, fn, kw)

        self._prev = ops._via2d
        ops._via2d = counted
        return self

    def __exit__(self, *exc):
        ops._via2d = self._prev

    def summary(self):
        out = {}
        for (s, why), n in self.rows.items():
            out.setdefault(why, []).append({"shape": list(s), "calls": n})
        return {why: sorted(v, key=lambda r: -r["calls"]) for why, v in out.items()}


def sysfs_node(visible=None):
    """The sysfs node of the card TT_VISIBLE_DEVICES names.

    TT_VISIBLE_DEVICES counts cards in PCI bus order, as UMD and tt-smi do; the kernel's
    /dev/tenstorrent/N does not. On qb1 index 0 is 0000:01:00.0, which is node 1: reading
    node 0 sampled an idle card at 800 MHz while the fold ran at 1350.
    """
    visible = visible or os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    root = "/sys/class/tenstorrent"
    nodes = sorted(os.listdir(root),
                   key=lambda n: os.path.basename(os.path.realpath(f"{root}/{n}/device")))
    node = nodes[int(visible)]
    return f"{root}/{node}", os.path.basename(os.path.realpath(f"{root}/{node}/device"))


class Clock:
    """AICLK from sysfs every `dt` s while the block runs, off the card TT_VISIBLE_DEVICES names."""

    def __init__(self, dt=0.25):
        node, self.pci = sysfs_node()
        self.path = f"{node}/tt_aiclk"
        self.dt, self.samples = dt, []

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(int(open(self.path).read().strip()))
            except Exception:
                pass
            self._stop.wait(self.dt)

    def __enter__(self):
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def stats(self):
        s = self.samples
        return dict(n=len(s), median=statistics.median(s) if s else None,
                    min=min(s) if s else None, max=max(s) if s else None, pci=self.pci)
