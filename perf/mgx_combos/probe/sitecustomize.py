"""Residency probe for a combination fold: which DRAM buffers are live at each stage boundary.

On the PYTHONPATH only when run.sh is given PROBE=<file>. `tt-bio predict` folds in a spawned
worker, and CPython imports sitecustomize in every interpreter, so the hook reaches the process
that opens the device. After `tt_bio.boltz2` loads, the stage entry points below are wrapped to
append one line per call to PROBE: DRAM allocated and largest free block per bank, then every
live DRAM buffer >= 16 MiB grouped by size. Buffers carry no names, so a size is the handle; a
pair tensor is n*n*c*2 bytes and is recognisable from that alone. Reading the allocator drains
the pipeline, so a probed fold's wall time means nothing.
"""
import functools
import importlib.abc
import os
import sys
from collections import Counter

OUT = os.environ.get("PROBE")
STAGES = {"tt_bio.boltz2": [("DiffusionConditioning", "forward"), ("AtomDiffusion", "sample"),
                            ("ConfidenceModule", "forward")]}


def _dump(tag):
    try:
        import ttnn
        import tt_bio.tenstorrent as tt
        dev = getattr(tt, "_device", None)
        if dev is None:
            return
        v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        sizes = Counter(b.max_size_per_bank * v.num_banks
                        for b in ttnn._ttnn.reports.get_buffers([dev])
                        if b.buffer_type == ttnn.BufferType.DRAM)
        big = sorted(((s, k) for s, k in sizes.items() if s >= 16 << 20), reverse=True)
        line = (f"{tag}: alloc/bank {v.total_bytes_allocated_per_bank / 2**20:.1f} MiB "
                f"largest_free {v.largest_contiguous_bytes_free_per_bank / 2**20:.1f} MiB "
                f"buffers {sum(sizes.values())} | "
                + " ".join(f"{k}x{s / 2**20:.1f}" for s, k in big)
                + f" | small {sum(s * k for s, k in sizes.items() if s < 16 << 20) / 2**20:.1f} MiB")
    except Exception as exc:                      # never let the probe fail a fold
        line = f"{tag}: probe error {type(exc).__name__}: {exc}"
    with open(OUT, "a") as fh:
        fh.write(line + "\n")


def _wrap(qual, fn):
    @functools.wraps(fn)
    def w(*a, **kw):
        _dump(f"{qual} enter")
        r = fn(*a, **kw)
        _dump(f"{qual} exit")
        return r
    return w


class _After(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in STAGES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        run = spec.loader.exec_module

        def exec_module(mod):
            run(mod)
            for cls, meth in STAGES[name]:
                c = getattr(mod, cls)
                setattr(c, meth, _wrap(f"{cls}.{meth}", getattr(c, meth)))
        spec.loader.exec_module = exec_module
        return spec


if OUT:
    import importlib.util
    sys.meta_path.insert(0, _After())
