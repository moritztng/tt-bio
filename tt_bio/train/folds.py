"""Device weights as views of upstream parameters: the gradient of a load-time fold.

A serving model folds at load. RFD4 multiplies `norm_cond.weight` into both AdaLN projections, scales
`to_bias^T` by `pair_norm.weight` and sqrt(head_dim), splits `to_qkv` and the transition input by output
column, concatenates the trimul in-projections, and keeps a second packed copy of the pair transition for
its fused kernel. A gradient taken on those device tensors names no upstream parameter, and training in
the folded space is a different model: Muon on three slices of `to_qkv` is not Muon on `to_qkv`, a product
of two parameters trains as one, and a duplicate copy goes stale for inference.

So optimizer state stays in upstream parameter space and the device weights stay derived. `Folds` builds
the model with the upstream state dict as float32 leaves that want a gradient and records the torch
source of every `ttnn.from_torch` made while it does. Each source carries the fold as a torch graph, so
the fold's VJP is torch's own backward through it, whatever shape the fold has: a closure, a raw
`from_torch` outside `Module.torch_to_tt`, or a lazy upload during the first forward (record that forward
too). Two device copies of one parameter sum. A weight uploaded from a tensor no leaf feeds is a constant
(`constants`: RFD4's trimul LayerNorm biases, which upstream builds with `bias=False`); a weight with no
recorded upload at all (made on the device, or outside the recording) is an orphan (`orphans`), and both are
named so a caller can refuse them, never zeroed.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Mapping

import torch

from .bridge import HostParam

__all__ = ["Folds"]


def _plain_transpose(s) -> bool:
    """`s` is a 2-D transpose of a leaf, or of slices of one: its VJP is pure layout, no reduction."""
    f = s.grad_fn
    if s.dim() != 2 or type(f).__name__ != "TBackward0":
        return False
    f = f.next_functions[0][0]
    while type(f).__name__ == "SliceBackward0":
        f = f.next_functions[0][0]
    return type(f).__name__ == "AccumulateGrad"


def _read_back(g, s):
    """A device gradient as a host tensor for source `s`. Most of the fold VJP is transposing gradients back
    (`W.t()` uploads); torch would do that as a strided copy on the host. For a plain transpose the card
    transposes instead and autograd gets a transposed view, so the host only accumulates contiguously. The
    bits go through as uint32: a float32 tile transpose rounds to ~TF32, a uint32 one moves data only. A
    transpose under a product (an AdaLN `norm_cond` fold) keeps the host path: its broadcast sum reduces in
    memory order, so a contiguous grad would round differently."""
    import ttnn
    if not (g.dtype == ttnn.float32 and _plain_transpose(s) and all(d == 1 for d in tuple(g.shape)[:-2])):
        return ttnn.to_torch(g)
    u = ttnn.transpose(ttnn.bitcast(g, ttnn.uint32), -2, -1)
    t = ttnn.bitcast(u, ttnn.float32)
    ttnn.deallocate(u)
    return ttnn.to_torch(t).reshape(s.shape[1], s.shape[0]).t()


class Folds:
    def __init__(self, state: Mapping[str, torch.Tensor]):
        self.leaves = {k: v.detach().to(torch.float32).clone().requires_grad_(True)
                       for k, v in state.items() if torch.is_tensor(v) and v.is_floating_point()}
        self.sources = {}  # id(device handle) -> (handle, torch source); holding the handle keeps the id live
        self.consts = {}   # id(device handle) -> handle uploaded from a tensor no leaf feeds

    @contextmanager
    def recording(self, device_pack: bool = False):
        """Yields the leaf state dict to build from; every `ttnn.from_torch` of a tensor that wants a
        gradient is recorded until exit. The weight is uploaded from the detached source, so the build is
        the serving build bit for bit.

        ``device_pack`` packs bfloat8_b weights on the card (float32 upload, then typecast) instead of on
        the host, which is most of a per-step rebuild: host packing runs at ~110 Melem/s and does not
        thread. Not bit-identical: the card rounds a tie half away from zero where the host rounds it to
        even, so ~4e-6 of elements differ by one ulp and none lands further from the float32 source."""
        import ttnn
        from .. import tenstorrent as tt
        if getattr(tt, "_weight_cache", None) is not None:
            raise RuntimeError("Folds.recording under the shared weight cache: a cached tile has no torch source")
        real = ttnn.from_torch

        def upload(x, *a, **k):
            if (not a and k.get("dtype") == ttnn.bfloat16 and k.get("layout") == ttnn.TILE_LAYOUT
                    and k.get("device") is not None and x.dim() >= 2 and x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0):
                # Tilize on the card: the host tilize in from_torch is most of a rebuild (~10 ms vs ~1 ms per
                # 2.4 M elements at 2 CPUs). The bf16 rounding stays torch's, and tilize only moves data, so the
                # weight is the same bits (perf/r4t_perf/fold/tilt.py).
                r = real(x.to(torch.bfloat16), **{**k, "layout": ttnn.ROW_MAJOR_LAYOUT})
                t = ttnn.to_layout(r, ttnn.TILE_LAYOUT, memory_config=k.get("memory_config"))
                ttnn.deallocate(r)
                return t
            if not (device_pack and not a and k.get("dtype") == ttnn.bfloat8_b
                    and k.get("device") is not None and k.get("layout") == ttnn.TILE_LAYOUT):
                return real(x, *a, **k)
            t32 = real(x.to(torch.float32), **{**k, "dtype": ttnn.float32})
            t = ttnn.typecast(t32, ttnn.bfloat8_b)
            ttnn.deallocate(t32)
            return t

        def rec(x, *a, **k):
            if not torch.is_tensor(x):
                return real(x, *a, **k)
            t = upload(x.detach(), *a, **k)
            if x.requires_grad:
                self.sources[id(t)] = (t, x)
            else:
                self.consts[id(t)] = t
            return t

        ttnn.from_torch = rec
        try:
            yield dict(self.leaves)
        finally:
            ttnn.from_torch = real

    def source(self, handle):
        hit = self.sources.get(id(getattr(handle, "value", handle)))
        return None if hit is None or hit[0] is not getattr(handle, "value", handle) else hit[1]

    def constants(self, named: Mapping[str, object]) -> list:
        """Names among `named` (`{name: device handle or tape leaf}`) uploaded from no upstream leaf."""
        return sorted(n for n, h in named.items() if self.consts.get(id(getattr(h, "value", h))) is getattr(h, "value", h))

    def orphans(self, named: Mapping[str, object]) -> list:
        """Names among `named` with no recorded upload at all."""
        return sorted(n for n, h in named.items() if self.source(h) is None and n not in set(self.constants({n: h})))

    def unreached(self) -> list:
        """Upstream leaves no recorded source depends on: parameters the device model never reads."""
        srcs = [s for _, s in self.sources.values()]
        names = list(self.leaves)
        if not srcs:
            return sorted(names)
        got = torch.autograd.grad([s.sum() for s in srcs], [self.leaves[n] for n in names],
                                  allow_unused=True, retain_graph=True)
        return sorted(n for n, g in zip(names, got) if g is None)

    def backward(self, grads: Iterable) -> None:
        """Accumulate into the upstream leaves' `.grad` from `(device handle, gradient)` pairs, the
        gradient a torch or a device tensor. A handle with no source raises."""
        import ttnn
        srcs, gs = [], []
        for h, g in grads:
            s = self.source(h)
            if s is None:
                raise KeyError(f"no fold source for a device weight of shape {tuple(h.shape)}")
            if not torch.is_tensor(g):
                g = _read_back(g, s)
            srcs.append(s)
            gs.append(g.to(s.dtype).reshape(s.shape))
        if srcs:
            torch.autograd.backward(srcs, gs, retain_graph=True)

    def tape_backward(self) -> list:
        """`backward` from the tape: every recorded weight whose tape leaf holds a gradient. Returns the
        handles with none (a duplicate the taped forward never reads, or frozen this step).

        The device gradients are consumed: once summed into the upstream leaves they are freed and the
        tape leaves cleared, so a rebuild does not strand one float32 copy of the model per step."""
        import ttnn
        from .. import autograd as ag
        pairs, idle, used = [], [], []
        for h, _ in self.sources.values():
            leaf = ag.parameter_for(h)
            g = None if leaf is None else leaf.grad
            (idle if g is None else pairs).append(h if g is None else (h, g))
            if g is not None:
                used.append(leaf)
        self.backward(pairs)
        for leaf in used:
            g, leaf.grad = leaf.grad, None
            if g is not None and not torch.is_tensor(g):
                ttnn.deallocate(g)
        return idle

    def params(self, prefix: str = "") -> dict:
        """`{upstream name: HostParam}`, what the optimizer owns. Writing one does not move a device
        weight: record a rebuild after the step."""
        return {prefix + n: HostParam(p) for n, p in self.leaves.items()}
