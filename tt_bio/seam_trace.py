"""Replay BindCraft 2's taped device seams from captured ttnn traces.

A taped seam's device time is fixed, but issuing it is thousands of Python ttnn calls, and ttnn
dispatch holds the GIL. With several trajectories interleaved on one card
(`tt_bio.duotraj`), one trajectory's enqueue and every other trajectory's host work exclude
each other, and that, not the card, is what bounds the round (`state/perf10/bcx-p10-gatesplit.md`).
A replayed trace issues the same device work as one command.

Per (trajectory, stack) this keeps a fixed set of device buffers: the inputs, the cotangent
seeds, the outputs, the gradients and a BANK the size of the stack's tape. Per (trajectory,
stack, checkpoint) it captures two traces:

    forward   inputs -> taped forward -> tape copied into the bank, outputs into fixed buffers
    backward  bank copied back over the tape -> backward -> gradients into fixed buffers

The tape crosses other work between the two (the recycles, the other stacks, the other
trajectories), and at its captured addresses anything may overwrite it once the capture is over.
The bank is what carries it: it is allocated before any capture, so no capture ever puts
anything there.

Correctness rests on one rule: nothing that must outlive a gate hold may be allocated after the
first capture, because it can land in a capture's freed addresses and a later replay overwrites
it. So a trajectory's first round runs eager and builds its buffers, every checkpoint in the pool
is loaded before the first capture, and after it `persistent()` refuses: an eager taped seam, a
new mask, a trunk load. The harness starts every trajectory's second round together, which is
the shape this needs.

Off unless ``TT_BIO_TRACE_SEAMS`` is set. Measured bit-equal to eager on the Evoformer forward
and backward (`perf/bcx_p10_trace/results/leg{1,2}_*.json`).
"""
from __future__ import annotations

import gc

from tt_bio.envflags import env_flag

STATS = {"captures": 0, "replays_fwd": 0, "replays_bwd": 0, "eager_fwd": 0, "eager_bwd": 0,
         "leaked_after_capture": 0}

_captured = False
#: Every stack's traced face in this process, so the first capture can prime and capture all.
_ALL: list = []


def enabled() -> bool:
    return env_flag("TT_BIO_TRACE_SEAMS", False)


def persistent(what: str) -> None:
    """Refuse a long-lived device allocation once a trace exists; see the module docstring."""
    if _captured:
        raise RuntimeError(
            f"{what} would allocate device memory that outlives a gate hold after the first "
            f"seam trace was captured, where a replay can overwrite it. Under "
            f"TT_BIO_TRACE_SEAMS every trajectory must finish its first round before any "
            f"trajectory starts its second.")


def _tape_of(roots):
    """Every device value the backward can read, once each, in a fixed order."""
    from tt_bio.autograd import _reverse_topo
    seen, out = set(), []
    for t in _reverse_topo(roots):
        v = t.value
        if v is None or not v.is_allocated():
            continue
        a = v.buffer_address()
        if a not in seen:
            seen.add(a)
            out.append(v)
    return out


def _live(device):
    import ttnn
    return {b.address for b in ttnn._ttnn.reports.get_buffers(device)}


class _Fixed:
    """One trajectory's buffers for one stack, shared by every checkpoint's traces."""

    def __init__(self):
        self.inputs = self.outputs = self.seeds = self.bank = self.grads = None
        self.shapes = None
        self.traces: dict = {}
        self.fn = None              # the stack's taped forward, fn(trunk, leaves) -> roots
        self.pending = None         # ("eager", leaves, roots) | ("traced", key)


class SeamTraces:
    """The traced face of one stack (Evoformer, extra-MSA or template).

    `forward(slot, trunk, xs, fn)` and `backward(slot, trunk, gs)` run inside the stack's
    `duotraj.card` hold. `xs` and `gs` are padded host tensors; `fn(trunk, leaves) -> roots` is the
    stack's taped forward on device leaves; it must take the trunk as its first argument and hold
    nothing checkpoint-specific, because the first capture replays it on every checkpoint. Both return padded float host tensors.
    """

    def __init__(self, name: str, pool):
        self.name, self.pool = name, pool
        self._fixed: dict[str, _Fixed] = {}
        _ALL.append(self)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _host(trunk, t, shape):
        import torch
        ttnn = trunk.ttnn
        return ttnn.from_torch(t.detach().reshape(list(shape)).to(torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    def _prepare_all(self):
        """Everything persistent, then every trace, before the first replay.

        A checkpoint's first forward does one-time setup, some of it a host write, and a host
        write inside a capture is a TT_FATAL; setup that allocates after a capture is the hazard
        the module docstring names. So every stack runs eager once on every checkpoint, and only
        then is anything captured: every (trajectory, stack, checkpoint) at once.
        """
        trunks = []
        for name in self.pool.names:
            trunk = self.pool._load(name)
            trunk.opm_device()
            trunks.append(trunk)
        for st in _ALL:
            for slot, fx in st._fixed.items():
                if fx.grads is None:
                    persistent(f"{st.name}: trajectory {slot!r} is still in its eager round")
                for trunk in trunks:
                    st._prime(fx, trunk)
        for st in _ALL:
            for fx in st._fixed.values():
                for trunk in trunks:
                    st._capture(fx, trunk)

    @staticmethod
    def _prime(fx, trunk):
        ttnn, ag = trunk.ttnn, trunk.ag
        leaves = [ag.Tensor(ttnn.clone(i), requires_grad=True) for i in fx.inputs]
        with trunk.taped.tape():
            roots = fx.fn(trunk, leaves)
        ag.release_pins()
        ag.backward(roots, [ttnn.clone(s) for s in fx.seeds])
        ag.release_pins()
        ttnn.synchronize_device(trunk.device)
        del leaves, roots
        gc.collect()

    # ------------------------------------------------------------------ forward

    def forward(self, slot, trunk, xs, fn):
        import torch  # noqa: F401
        ttnn = trunk.ttnn
        fx = self._fixed.get(slot)
        if fx is None or fx.grads is None:
            return self._eager_forward(slot, trunk, xs, fn)
        if [tuple(x.shape) for x in xs] != fx.shapes:
            raise ValueError(f"{self.name}: trajectory {slot!r} changed shape from {fx.shapes} "
                             f"to {[tuple(x.shape) for x in xs]}; its traces are static")
        key = id(trunk)
        if key not in fx.traces and not _captured:
            self._prepare_all()
        if key not in fx.traces:
            raise RuntimeError(f"{self.name}: no trace for this checkpoint on {slot!r}; every "
                               f"checkpoint in the pool is captured at the first replay")
        for x, buf in zip(xs, fx.inputs):
            ttnn.copy_host_to_device_tensor(self._host(trunk, x, buf.shape), buf)
        ttnn.execute_trace(trunk.device, fx.traces[key][0], cq_id=0, blocking=False)
        fx.pending = ("traced", key)
        STATS["replays_fwd"] += 1
        return [torch_of(trunk, o, x.shape) for o, x in zip(fx.outputs, xs)]

    def _eager_forward(self, slot, trunk, xs, fn):
        persistent(f"{self.name}: an eager taped forward on trajectory {slot!r}")
        ttnn = trunk.ttnn
        leaves = [trunk.leaf(x) for x in xs]
        with trunk.taped.tape():
            roots = fn(trunk, leaves)
        trunk.ag.release_pins()
        fx = self._fixed.get(slot)
        if fx is None:
            fx = self._fixed[slot] = _Fixed()
            fx.shapes = [tuple(x.shape) for x in xs]
            fx.inputs = [trunk.up(x) for x in xs]
            fx.outputs = [ttnn.clone(r.value) for r in roots]
            fx.seeds = [ttnn.clone(r.value) for r in roots]
            tape = _tape_of(roots)
            fx.bank = [ttnn.clone(v) for v in tape]
            # Every program a trace issues compiles now: a program load is a device write and a
            # write inside a capture is a TT_FATAL.
            for v, b in zip(tape, fx.bank):
                ttnn.copy(v, b)
                ttnn.copy(b, v)
            for r, o in zip(roots, fx.outputs):
                ttnn.copy(r.value, o)
            for t in fx.inputs + fx.seeds:
                ttnn.deallocate(ttnn.clone(t))
        fx.fn = fn
        fx.pending = ("eager", leaves, roots)
        STATS["eager_fwd"] += 1
        return [torch_of(trunk, r.value, x.shape) for r, x in zip(roots, xs)]

    # ------------------------------------------------------------------ backward

    def backward(self, slot, trunk, gs):
        ttnn = trunk.ttnn
        fx = self._fixed[slot]
        pending, fx.pending = fx.pending, None
        if pending is None:
            raise RuntimeError(f"{self.name}: backward on {slot!r} with no forward pending")
        if pending[0] == "eager":
            _, leaves, roots = pending
            trunk.ag.backward(roots, [trunk.seed(g, r) for g, r in zip(gs, roots)])
            if fx.grads is None:
                fx.grads = [ttnn.clone(leaf.grad) for leaf in leaves]
                for leaf, g in zip(leaves, fx.grads):
                    ttnn.copy(leaf.grad, g)
            out = [trunk.grad(leaf, tuple(x.shape)) for leaf, x in zip(leaves, gs)]
            trunk.ag.release_pins()
            STATS["eager_bwd"] += 1
            return out
        for g, buf in zip(gs, fx.seeds):
            ttnn.copy_host_to_device_tensor(self._host(trunk, g, buf.shape), buf)
        ttnn.execute_trace(trunk.device, fx.traces[pending[1]][1], cq_id=0, blocking=False)
        STATS["replays_bwd"] += 1
        return [torch_of(trunk, b, g.shape) for b, g in zip(fx.grads, gs)]

    # ------------------------------------------------------------------ capture

    def _capture(self, fx, trunk):
        global _captured
        ttnn, ag, dev = trunk.ttnn, trunk.ag, trunk.device
        from tt_bio import tenstorrent
        tenstorrent.require_trace_region(f"TT_BIO_TRACE_SEAMS ({self.name})")
        ttnn.synchronize_device(dev)
        before = _live(dev)
        f = ttnn.begin_trace_capture(dev, cq_id=0)
        leaves = [ag.Tensor(ttnn.clone(i), requires_grad=True) for i in fx.inputs]
        with trunk.taped.tape():
            roots = fx.fn(trunk, leaves)
        ag.release_pins()
        tape = _tape_of(roots)
        if [tuple(v.shape) for v in tape] != [tuple(b.shape) for b in fx.bank]:
            ttnn.end_trace_capture(dev, f, cq_id=0)
            raise RuntimeError(f"{self.name}: the captured tape does not line up with the bank")
        for v, b in zip(tape, fx.bank):
            ttnn.copy(v, b)
        for r, o in zip(roots, fx.outputs):
            ttnn.copy(r.value, o)
        ttnn.end_trace_capture(dev, f, cq_id=0)
        b = ttnn.begin_trace_capture(dev, cq_id=0)
        for v, bk in zip(tape, fx.bank):
            ttnn.copy(bk, v)
        ag.backward(roots, [ttnn.clone(s) for s in fx.seeds])
        for leaf, g in zip(leaves, fx.grads):
            ttnn.copy(leaf.grad, g)
        ag.release_pins()
        ttnn.end_trace_capture(dev, b, cq_id=0)
        _captured = True
        del leaves, roots, tape
        gc.collect()
        STATS["leaked_after_capture"] += len(_live(dev) - before)
        fx.traces[id(trunk)] = (f, b)
        STATS["captures"] += 1


def torch_of(trunk, t, shape):
    """A fixed buffer's value on host, float, at the stack's padded shape."""
    return trunk.down(t, tuple(shape))
