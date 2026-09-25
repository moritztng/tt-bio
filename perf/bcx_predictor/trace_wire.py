#!/usr/bin/env python3
"""The loop's taped step as two captured traces, one pair per input shape.

`bcx-trace` measured the scheme and left it unwired: a captured forward and a captured backward,
with BindCraft 2's JAX tail running between them on the host. This is the wiring. It knows
nothing about AF2 -- the caller hands it the body to capture -- and it does nothing unless a
caller asks for it.

What a capture needs that an eager call does not:

* **Persistent buffers.** A trace is a command stream over fixed addresses, so the inputs, the
  cotangents, the primal outputs and the gradients all live in buffers allocated before capture.
  A step writes its inputs into them and reads its outputs out of them.
* **No host write inside the captured region.** `autograd.DEVICE_ZEROS` takes every zero a
  backward builds on the host off the host; this module turns it on. The masks are uploaded once
  per shape before capture. The loop runs no extra-MSA block on device, so the per-call OPM
  constant upload `bcx-trace` had to hoist is not on this path.
* **A tape that outlives the callback.** The captured backward reads the addresses the captured
  forward writes, so the tape is held as long as the shape is, not dropped per step.
* **Nothing allocated later may land in the trace's scratch.** Every intermediate the capture
  allocated and freed is an address the replay still writes. With the Evoformer alone on the
  card nothing else was live across a replay, so it did not show; with the extra-MSA stack on
  device too, its tape is allocated after the capture, lives across two replays, and came back
  overwritten (`perf/bcx_tracewire/bisect_xmsa_*`). So the capture runs inside a fence: DRAM is
  filled with ballast except for one contiguous hole sized from the warm pass's peak, the
  capture happens in the hole, what the capture left free in it is filled, and the ballast is
  released. From then on no allocation can reach the hole.

One shape at a time: a new shape releases the previous shape's traces and tape, because a
trajectory holds one padded length and the next one draws a different binder.
"""
from __future__ import annotations

import time

import torch


class _Shape:
    """Everything one captured (forward, backward) pair owns."""

    def __init__(self, key):
        self.key = key
        self.tid_f = self.tid_b = None
        self.bufs = None       # leaf input buffers, written per step
        self.seeds = None      # cotangent buffers, written per step
        self.prim = None       # what the forward trace writes out
        self.gout = None       # what the backward trace writes out
        self.leaves = self.roots = None   # held so the tape's addresses stay valid
        self.fence = None      # buffers that occupy the rest of the trace's scratch
        self.capture_s = 0.0
        self.replays = [0, 0]


class TraceWire:
    """Capture once per shape, then replay. `stats()` is the row's evidence."""

    def __init__(self, dev, keep: int = 1):
        self.dev, self.keep = dev, keep
        self._shapes: dict = {}
        self._order: list = []
        self.captures: list = []
        # Replay wall split four ways, summed over every replay. Everything but `wait` is the
        # host's, so their sum is the host floor this capture leaves behind.
        self.seg = {"write": 0.0, "enqueue": 0.0, "wait": 0.0, "read": 0.0}
        self.l1_foreign: list = []
        dev.ag.DEVICE_ZEROS = True

    # ------------------------------------------------------------------ the fence

    #: Ballast granularity, per bank. The hole is a run of adjacent chunks, so this is also the
    #: precision the hole is sized to.
    CHUNK = 64 << 20

    def _mem(self, kind: str = "DRAM"):
        ttnn = self.dev.ttnn
        return ttnn.get_memory_view(self.dev.device, getattr(ttnn.BufferType, kind))

    def _l1(self) -> int:
        return int(self._mem("L1").total_bytes_allocated_per_bank)

    def _hold(self, per_bank: int, banks: int):
        """One DRAM buffer taking ``per_bank`` bytes in every bank: a row-major bfloat16 tensor
        whose rows are pages, dealt round-robin, so each bank gets the same number of them."""
        ttnn = self.dev.ttnn
        page = min(per_bank, 1 << 20)
        return ttnn.empty([banks * (per_bank // page), page // 2], dtype=ttnn.bfloat16,
                          layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev.device,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _fill(self, chunk: int = 0) -> list:
        """Take every free DRAM byte, in ``chunk``-sized buffers where one fits."""
        held = []
        while True:
            mv = self._mem()
            big = int(mv.largest_contiguous_bytes_free_per_bank)
            want = min(big, chunk) if chunk and big >= chunk else big
            want -= want % 64
            if want <= 0:
                return held
            held.append(self._hold(want, int(mv.num_banks)))

    def _peak(self, run, stride: int = 32) -> int:
        """DRAM allocated per bank at its highest during ``run``, sampled after every
        ``stride``-th ttnn op.

        An allocation only happens inside an op, so sampling after ops finds the peak.
        Sampling before each `ttnn.deallocate` instead read low: much of a tape is released by
        dropping the last reference, which calls no deallocate, and the fence it sized was
        8 MB short of the capture. Every op would be exact and costs 480 s at n=224 (the
        memory view builds a block table per call); a stride misses at most the ops between
        two samples, which the fence's margin covers, and an undersized fence fails at
        capture as an out-of-memory, never as a wrong replay."""
        from ttnn import decorators as D
        peak = [int(self._mem().total_bytes_allocated_per_bank)]
        real = {c: c.__call__ for c in (D.FastOperation, D.Operation)}
        n = [0]

        def sampled(real_call):
            def call(op, *a, **k):
                out = real_call(op, *a, **k)
                n[0] += 1
                if n[0] % stride == 0:
                    peak[0] = max(peak[0], int(self._mem().total_bytes_allocated_per_bank))
                return out
            return call

        for c, f in real.items():
            c.__call__ = sampled(f)
        try:
            run()
        finally:
            for c, f in real.items():
                c.__call__ = f
        return peak[0]

    def _open_fence(self, need: int) -> list:
        """Ballast everywhere except one contiguous hole of at least ``need`` bytes per bank."""
        ballast = self._fill(self.CHUNK)
        chunks = sorted((t for t in ballast if self._per_bank(t) == self.CHUNK),
                        key=lambda t: t.buffer_address())
        k = -(-need // self.CHUNK)
        run: list = []
        for t in chunks:
            if run and t.buffer_address() != run[-1].buffer_address() + self.CHUNK:
                run = []
            run.append(t)
            if len(run) == k:
                break
        if len(run) < k:
            for t in ballast:
                self.dev.ttnn.deallocate(t)
            raise RuntimeError(f"no contiguous {k * self.CHUNK >> 20} MiB per bank for the "
                               f"trace's scratch; the capture would share addresses with "
                               f"whatever is allocated after it")
        ids = {id(t) for t in run}
        for t in run:
            self.dev.ttnn.deallocate(t)
        return [t for t in ballast if id(t) not in ids]

    def _per_bank(self, t) -> int:
        return int(t.shape[0]) * int(t.shape[1]) * 2 // int(self._mem().num_banks)

    # ------------------------------------------------------------------ device plumbing

    def _write(self, t, dst) -> None:
        """Host tensor into an already-allocated device buffer: no allocation, no host build
        inside a timed replay beyond the one upload the eager path also does."""
        ttnn = self.dev.ttnn
        h = ttnn.from_torch(t.detach().reshape([int(d) for d in dst.shape]).to(torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.copy_host_to_device_tensor(h, dst)

    def _sync(self):
        self.dev.ttnn.synchronize_device(self.dev.device)

    def _out(self, tensors, shapes):
        return [self.dev.down(t, s) for t, s in zip(tensors, shapes)]

    # ------------------------------------------------------------------ capture

    def _capture(self, key, ins, body) -> _Shape:
        """Warm, capture the forward, capture the backward, then replay the forward once.

        A capture records; it does not run. The primal buffers hold nothing until the first
        replay, so that replay happens here and the calling step gets real outputs.
        """
        dev, ttnn, ag = self.dev, self.dev.ttnn, self.dev.ag
        sh = _Shape(key)
        t0 = time.time()
        sh.bufs = [dev.up(t) for t in ins]

        def once():
            leaves = [ag.Tensor(ttnn.clone(b), requires_grad=True) for b in sh.bufs]
            with dev.tt.tape():
                roots = list(body(*leaves))
            return leaves, roots

        # Warm: every program compiled and in the program cache before the recorder runs, and
        # the cotangent buffers allocated against the roots the body actually returns. The
        # second pass is the one the capture repeats, so its DRAM peak sizes the fence.
        def warm(first):
            leaves, roots = once()
            if first:
                sh.seeds = [dev.seed(torch.zeros([int(d) for d in r.value.shape]), r)
                            for r in roots]
            ag.backward(roots, [ttnn.clone(s) for s in sh.seeds])
            ag.release_pins()

        warm(True)
        self._sync()
        base = int(self._mem().total_bytes_allocated_per_bank)
        rise = self._peak(lambda: warm(False)) - base
        self._sync()
        # A quarter over the measured rise: the capture repeats the warm pass's allocations,
        # but into a smaller space, so it can fragment where the warm pass did not.
        mv = self._mem()
        print(f"[trace_wire] warm rise {rise >> 20} MiB/bank, free {int(mv.total_bytes_free_per_bank) >> 20}"
              f" MiB/bank, largest {int(mv.largest_contiguous_bytes_free_per_bank) >> 20}", flush=True)
        ballast = self._open_fence(rise + rise // 4 + self.CHUNK)

        sh.tid_f = ttnn.begin_trace_capture(dev.device, cq_id=0)
        try:
            leaves, roots = once()
            # The readout is part of the trace: the backward frees the roots, and the primal
            # has to survive the host round trip that runs between the two replays.
            sh.prim = [ttnn.clone(r.value) for r in roots]
        finally:
            ttnn.end_trace_capture(dev.device, sh.tid_f, cq_id=0)
        self._sync()

        sh.tid_b = ttnn.begin_trace_capture(dev.device, cq_id=0)
        try:
            # No `release_pins` and no clone of the gradients. The pins are what keep the
            # addresses this trace was captured against, and every replay writes the same
            # `leaf.grad` buffers, so the tape is held until the shape changes -- which is the
            # lifetime `bcx-trace` named. `ttnn.clone` of a gradient refuses inside a capture.
            ag.backward(roots, [ttnn.clone(s) for s in sh.seeds])
            sh.gout = [lf.grad for lf in leaves]
        finally:
            ttnn.end_trace_capture(dev.device, sh.tid_b, cq_id=0)
        self._sync()
        sh.fence = self._fill()
        for t in ballast:
            ttnn.deallocate(t)
        self.fence_mb = {"warm_rise": rise >> 20,
                         "filled_after_capture": sum(self._per_bank(t) for t in sh.fence) >> 20,
                         "free_after": int(self._mem().total_bytes_free_per_bank) >> 20}
        if any(g is None for g in sh.gout):
            raise RuntimeError("a leaf took no gradient in the captured backward; the trace "
                               "would replay a gradient that is never written")
        self._sync()
        sh.leaves, sh.roots = leaves, roots
        sh.l1_own = self._l1()
        sh.capture_s = time.time() - t0

        self._evict()
        self._shapes[key] = sh
        self._order.append(key)
        self.captures.append({"key": str(key), "capture_s": round(sh.capture_s, 2),
                              "fence_mb_per_bank": self.fence_mb})
        print(f"[trace_wire] captured {key} in {sh.capture_s:.1f} s", flush=True)
        ttnn.execute_trace(dev.device, sh.tid_f, cq_id=0, blocking=False)
        self._sync()
        sh.replays[0] += 1
        return sh

    def _evict(self) -> None:
        while len(self._order) >= max(self.keep, 1):
            old = self._shapes.pop(self._order.pop(0))
            for tid in (old.tid_f, old.tid_b):
                if tid is not None:
                    self.dev.ttnn.release_trace(self.dev.device, tid)
            for t in old.fence or ():
                self.dev.ttnn.deallocate(t)
            old.leaves = old.roots = old.prim = old.gout = old.bufs = old.seeds = None
            old.fence = None

    # ------------------------------------------------------------------ the two verbs

    def forward(self, key, ins, body, out_shapes):
        """Leaf inputs in as host tensors, the roots out as host tensors."""
        sh = self._shapes.get(key)
        if sh is None:
            sh = self._capture(key, ins, body)
        else:
            self._replay(sh, 0, ins, sh.bufs)
        t = time.perf_counter()
        out = self._out(sh.prim, out_shapes)
        self.seg["read"] += time.perf_counter() - t
        return out

    def backward(self, key, cotangents, out_shapes):
        """Cotangents in, leaf gradients out, all host tensors."""
        sh = self._shapes[key]
        self._replay(sh, 1, cotangents, sh.seeds)
        t = time.perf_counter()
        out = self._out(sh.gout, out_shapes)
        self.seg["read"] += time.perf_counter() - t
        return out

    def _replay(self, sh, which, ins, dsts) -> None:
        t0 = time.perf_counter()
        for t, dst in zip(ins, dsts):
            self._write(t, dst)
        # L1 has no fence: a tensor held in L1 across this replay by anything but the capture
        # sits where the captured program's own L1 intermediates may land. Record it.
        extra = self._l1() - sh.l1_own
        if extra:
            self.l1_foreign.append({"replay": sh.replays[which], "which": which,
                                    "bytes_per_bank": extra})
        t1 = time.perf_counter()
        self.dev.ttnn.execute_trace(self.dev.device, (sh.tid_f, sh.tid_b)[which], cq_id=0,
                                    blocking=False)
        t2 = time.perf_counter()
        self._sync()
        t3 = time.perf_counter()
        sh.replays[which] += 1
        for k, dt in zip(("write", "enqueue", "wait"), (t1 - t0, t2 - t1, t3 - t2)):
            self.seg[k] += dt

    def stats(self) -> dict:
        return {"captures": self.captures,
                "replays": {str(k): s.replays for k, s in self._shapes.items()},
                "capture_s_total": round(sum(c["capture_s"] for c in self.captures), 2),
                "seg_s": {k: round(v, 4) for k, v in self.seg.items()},
                "l1_foreign": self.l1_foreign}


#: A capture is its command stream, not its tensors: `bcx-trace` read metal's own refusal at
#: 435 MiB (n=128) and 510 MiB (n=256) for the whole 4+48 step. 768 MiB carries the pair of
#: traces this module captures with room above n=256.
REGION_MB = 768


def open_traced_device(region_mb: int = REGION_MB):
    """Open the card with a trace region. Must run before anything else opens the device."""
    from tt_bio import tenstorrent as T
    T.TRACE_REGIONS["bcx_evoformer"] = {"blackhole": region_mb << 20,
                                        "wormhole_b0": region_mb << 20}
    return T.get_device(trace="bcx_evoformer")
