"""Attribute a taped backward's seconds to the op that taped each node.

`of3t-perf10` sized OpenFold3's trunk backward at 21.304 s, 34.2 % of the 62.237 s training
step, over 5,127 tape nodes, and nobody had ever asked what those nodes ARE. This answers it
without a profiler build and without a second harness: the label is already on the node.

`autograd._tape` stores a closure built by the op's own `make()`, so `fn.__qualname__` reads
`_taped_linear.<locals>.make.<locals>.bw` and the text before the first `.<locals>` is the op
that taped it. `_Node.fn` is a plain slot, so wrapping is an assignment -- the REAL
`autograd._backward` still runs the walk, the ordering, the fan-in and the retirement. An
instrument that reimplements the loop it measures is measuring its own copy.

Three things the naive version gets wrong, each of which costs the whole table:

* **A checkpoint segment's cost is in its `group`, not in its nodes.** A checkpointed
  `PairformerLayer` publishes two outputs whose closures only deposit a gradient into a slot;
  `backward` then fires the group once and THAT recomputes the forward and replays an inner
  tape. Bill the nodes only and 79 % of the wall lands nowhere.
* **The inner tape is invisible to the outer walk.** `checkpoint._recompute` calls the module
  global `autograd.backward`, so patching that name catches every nested replay. Nodes are
  tagged with the depth they fired at, which is what separates the trunk's own ops from the
  segment interiors.
* **Host time is not device time.** ttnn is asynchronous, so an unsynced closure is billed its
  ENQUEUE and the device work lands on whichever later node happens to block. `sync=True`
  drains after every closure, which attributes the device work exactly and serialises what the
  fast arm overlaps. Run both: the unsynced arm owns the total, the synced arm owns the split.
"""

from __future__ import annotations

import time
from collections import defaultdict


def op_of(fn) -> str:
    """The op that taped `fn`, from the closure's qualified name."""
    q = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "?")
    head = q.split(".<locals>", 1)[0]
    return head.rsplit(".", 1)[-1] or q


class NodeProf:
    """Times every tape node's closure, by op, at the depth it fired.

    `install` must be called with the same roots `backward` is about to get, because the wrap
    walks `_reverse_topo` and a node the walk does not reach is a node the backward will not
    fire either.
    """

    def __init__(self, ag, sync=None):
        self.ag = ag
        self._sync = sync
        self.rows = defaultdict(lambda: {"calls": 0, "self_s": 0.0, "total_s": 0.0})
        self.depth = 0
        self._child = [0.0]
        self.wrapped = 0
        self.groups = 0
        self._seen_group = {}
        self._orig_backward = None
        self.nested_calls = 0

    # -- billing ----------------------------------------------------------------------
    def _timed(self, fn, label):
        def run(*a, **k):
            self._child.append(0.0)
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                if self._sync is not None:
                    self._sync()
                el = time.perf_counter() - t0
                kids = self._child.pop()
                self._child[-1] += el
                r = self.rows[label]
                r["calls"] += 1
                r["total_s"] += el
                r["self_s"] += el - kids
        run.__qualname__ = getattr(fn, "__qualname__", label)
        return run

    # -- wrapping ---------------------------------------------------------------------
    def wrap(self, roots):
        """Wrap every node closure and every checkpoint group reachable from `roots`."""
        order = self.ag._reverse_topo(roots)
        d = self.depth
        for t in order:
            n = t.node
            if n is None or getattr(n.fn, "_np_wrapped", False):
                continue
            n.fn = self._timed(n.fn, f"d{d}:{op_of(n.fn)}")
            n.fn._np_wrapped = True
            self.wrapped += 1
            g = n.group
            if g is not None and not getattr(g, "_np_wrapped", False):
                key = id(g)
                if key not in self._seen_group:
                    w = self._timed(g, f"d{d}:checkpoint_segment")
                    w._np_wrapped = True
                    self._seen_group[key] = w
                    self.groups += 1
                n.group = self._seen_group[key]
            elif g is not None:
                n.group = self._seen_group.get(id(g), g)
        return len(order)

    # -- the nested hook --------------------------------------------------------------
    def __enter__(self):
        ag = self.ag
        self._orig_backward = ag.backward

        def backward(roots, seeds=None):
            # A segment replay arrives here from `checkpoint._recompute`. Its nodes are fresh
            # every fire, so they are wrapped every time rather than once. The wrap runs at
            # the depth of the CALL, before the increment, so the trunk's own tape is d0 and
            # a segment interior is d1.
            if self.depth:
                self.nested_calls += 1
            self.wrap([roots] if isinstance(roots, ag.Tensor) else list(roots))
            self.depth += 1
            try:
                return self._orig_backward(roots, seeds)
            finally:
                self.depth -= 1

        ag.backward = backward
        return self

    def __exit__(self, *exc):
        self.ag.backward = self._orig_backward
        return False

    # -- reporting --------------------------------------------------------------------
    def table(self):
        rows = [{"op": k, **v} for k, v in self.rows.items()]
        for r in rows:
            r["self_s"] = round(r["self_s"], 4)
            r["total_s"] = round(r["total_s"], 4)
        rows.sort(key=lambda r: -r["self_s"])
        return rows

    def by_depth(self):
        out = defaultdict(lambda: {"calls": 0, "self_s": 0.0})
        for k, v in self.rows.items():
            d = k.split(":", 1)[0]
            out[d]["calls"] += v["calls"]
            out[d]["self_s"] += v["self_s"]
        return {k: {"calls": v["calls"], "self_s": round(v["self_s"], 4)}
                for k, v in sorted(out.items())}

    def report(self, wall_s=None):
        billed = sum(v["self_s"] for v in self.rows.values())
        out = {"nodes_wrapped": self.wrapped, "groups_wrapped": self.groups,
               "nested_backwards": self.nested_calls,
               "synced_per_node": self._sync is not None,
               "billed_self_s": round(billed, 3),
               "by_depth": self.by_depth(), "ops": self.table()}
        if wall_s is not None:
            out["wall_s"] = round(wall_s, 3)
            out["unbilled_s"] = round(wall_s - billed, 3)
        return out
