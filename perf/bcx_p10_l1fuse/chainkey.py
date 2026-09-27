#!/usr/bin/env python3
"""bcx-p10-l1fuse leg 1: the round keyed by producer -> consumer EDGE, not by op.

Every census this campaign has taken keys a call by what it is: its family, its op name, its
shape, its program config. None of them can see the thing wave 12 is about, which is a tensor
that op A writes to DRAM and op B reads straight back out of DRAM with nothing in between that
needs it there. That edge is the unit of a residency decision and it belongs to neither op, so a
per-op table cannot rank it.

So this widens `OpTimer` a fourth time, on a different axis. The second key stays the op name
(`bcx-p10-calls`'s A_verb, so the totals reconcile) and what is added is a side table of EDGES.
For every ttnn call the timer records which tensors it produced; when a later call consumes one,
the pair is written down with the bytes, the call distance and both label contexts.

IDENTITY IS THE WHOLE INSTRUMENT AND `id()` IS THE WRONG ONE. CPython reuses the id of a freed
object, a ttnn tensor is freed constantly inside a block, and `ttnn.Tensor` on this build cannot
be weak-referenced, so an id map cannot tell a live tensor from a freed one whose address was
reused and would manufacture edges between two unrelated tensors. The census keys on the DEVICE
BUFFER ADDRESS instead, and that key is exact rather than a better guess: two LIVE tensors can
never share a device address, so overwriting the entry for an address every time a call produces
a tensor there means a consumer always resolves to the tensor it is actually holding. A stale
entry can only be read by consuming a tensor that has been freed, which cannot happen.

A tensor with no device buffer -- a host tensor, or one whose address the build declines to
report -- is counted in `skipped` and takes part in no edge, so the table under-reports rather
than inventing. An aliasing op (`reshape`, `experimental.view`) returns a new tensor over the same
buffer and therefore becomes that address's producer, which is what a chain wants: the consumer
reads the view, not the thing the view was made from.
"""
from __future__ import annotations

import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_devmap import devmap as D          # noqa: E402


class ChainOpTimer(D.OpTimer):
    """`OpTimer` plus the dataflow graph between the calls it already times.

    The edge bookkeeping runs in `verb_key`, which the base class calls OUTSIDE the timed
    region, so the graph costs the measurement nothing. That is the same property that let
    `bcx-p10-mmlay` widen the key for free.
    """

    def __init__(self, ttnn, device):
        super().__init__(ttnn, device)
        self.seq = 0
        #: device buffer address -> (seq, verb, tag, bytes, desc)
        self.live = {}
        self.edges = collections.Counter()
        self.edge_bytes = collections.Counter()
        self.edge_gap = collections.Counter()
        self.edge_adj = collections.Counter()
        #: (producer verb, tag, desc) -> how many consuming calls that producer's output saw
        self.consumers = collections.Counter()
        #: tensors that took part in no edge because they carry no readable device address
        self.skipped = collections.Counter()

    # ---------------------------------------------------------------- identity

    @staticmethod
    def _desc(t):
        try:
            shape = "x".join(str(int(d)) for d in t.padded_shape)
        except Exception:
            try:
                shape = "x".join(str(int(d)) for d in t.shape)
            except Exception:
                shape = "?"
        try:
            dt = str(t.dtype).rsplit(".", 1)[-1]
        except Exception:
            dt = "?"
        try:
            buf = str(t.memory_config().buffer_type).rsplit(".", 1)[-1]
        except Exception:
            buf = "?"
        return shape, dt, buf

    @staticmethod
    def _addr(t):
        """The device buffer address, or None for anything that does not have one."""
        try:
            a = int(t.buffer_address())
        except Exception:
            return None
        return a or None

    # ---------------------------------------------------------------- the hook

    def verb_key(self, path, args, kwargs, out):
        ins, outs = [], []
        self._walk(args, ins)
        self._walk(kwargs, ins)
        self._walk(out, outs)

        self.seq += 1
        seq, tag = self.seq, D._tag()

        seen = set()
        for t in ins:
            key = self._addr(t)
            if key is None:
                self.skipped["in||" + path] += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            rec = self.live.get(key)
            if rec is None:
                continue
            pseq, pverb, ptag, pbytes, pdesc = rec
            if pseq == seq:                   # an in-place op reading its own destination
                continue
            ek = (pverb, ptag, path, tag) + pdesc
            self.edges[ek] += 1
            self.edge_bytes[ek] += pbytes
            self.edge_gap[ek] += seq - pseq
            if seq - pseq == 1:
                self.edge_adj[ek] += 1
            self.consumers[(pverb, ptag) + pdesc] += 1

        for t in outs:
            key = self._addr(t)
            if key is None:
                self.skipped["out||" + path] += 1
                continue
            if key in seen:                   # an in-place op returning its own input
                continue
            self.live[key] = (seq, path, tag, D._nbytes(t), self._desc(t))

        return path

    # ---------------------------------------------------------------- reporting

    def take(self):
        snap = super().take()
        snap["edges"] = {"||".join(k): v for k, v in self.edges.items()}
        snap["edge_bytes"] = {"||".join(k): v for k, v in self.edge_bytes.items()}
        snap["edge_gap"] = {"||".join(k): v for k, v in self.edge_gap.items()}
        snap["edge_adj"] = {"||".join(k): v for k, v in self.edge_adj.items()}
        snap["consumers"] = {"||".join(k): v for k, v in self.consumers.items()}
        snap["skipped"] = dict(self.skipped)
        snap["calls_seen"] = self.seq
        for c in (self.edges, self.edge_bytes, self.edge_gap, self.edge_adj, self.consumers,
                  self.skipped):
            c.clear()
        # The producer map is NOT cleared: a tensor can outlive the window that made it, and
        # dropping it would lose the edge that reads it in the next one.
        return snap
