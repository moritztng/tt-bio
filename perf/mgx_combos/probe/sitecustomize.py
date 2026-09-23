"""Residency probe for a combination fold: which DRAM buffers are live at each stage boundary.

On the PYTHONPATH only when run.sh is given PROBE=<file>. `tt-bio predict` folds in a spawned
worker, and CPython imports sitecustomize in every interpreter, so the hook reaches the process
that opens the device. After `tt_bio.boltz2` loads, the stage entry points below are wrapped to
append one line per call to PROBE: DRAM allocated and largest free block per bank, then every
live DRAM buffer >= 16 MiB grouped by size. Buffers carry no names, so a size is the handle; a
pair tensor is n*n*c*2 bytes and is recognisable from that alone. Reading the allocator drains
the pipeline, so a probed fold's wall time means nothing.

HASH=<file> instead logs a sha1 of every tensor a boltz2 stage takes and returns (trunk inputs,
each recycle's s/z, the conditioning, the sampled coordinates, the confidence outputs), so two
runs of one input can be diffed stage by stage to find where they first part. Reading a device
tensor back synchronises the device, which can hide a race; a probed run that stops diverging
is itself a result.
"""
import functools
import importlib.abc
import os
import sys
from collections import Counter

OUT = os.environ.get("PROBE")
HASH = os.environ.get("HASH")
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


HASH_STAGES = {"tt_bio.tenstorrent": [("TrunkModule", "forward"), ("TrunkModule", "_iteration")],
               "tt_bio.boltz2": [("DiffusionConditioning", "forward_atoms"), ("AtomDiffusion", "sample"),
                                 ("ConfidenceModule", "forward")]}
_calls = Counter()


def _digests(x, name, out):
    import hashlib
    import torch
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            _digests(x[k], f"{name}.{k}", out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _digests(v, f"{name}[{i}]", out)
    elif isinstance(x, torch.Tensor):
        t = x.detach().cpu().contiguous().reshape(-1)
        out.append(f"{name}={hashlib.sha1(t.view(torch.uint8).numpy().tobytes()).hexdigest()[:10]}")
    elif type(x).__name__ == "Tensor" and type(x).__module__.startswith("ttnn"):
        import ttnn
        _digests(ttnn.to_torch(x), name, out)


def _hash_wrap(qual, fn):
    @functools.wraps(fn)
    def w(*a, **kw):
        n = _calls[qual]
        _calls[qual] += 1
        ins = []
        if qual == "TrunkModule.forward":        # s_inputs, s_init, z_init, feats: host inputs, before any device work
            _digests(a[1:5], "in", ins)
        r = fn(*a, **kw)
        outs = []
        try:
            _digests(r, "out", outs)
        except Exception as exc:                  # never let the probe fail a fold
            outs.append(f"hash error {type(exc).__name__}: {exc}")
        with open(HASH, "a") as fh:
            fh.write(f"{qual}#{n} " + " ".join(ins + outs) + "\n")
        return r
    return w


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
                setattr(c, meth, WRAP(f"{cls}.{meth}", getattr(c, meth)))
        spec.loader.exec_module = exec_module
        return spec


if HASH:
    STAGES, WRAP = HASH_STAGES, _hash_wrap
else:
    WRAP = _wrap
if OUT or HASH:
    import importlib.util
    sys.meta_path.insert(0, _After())
