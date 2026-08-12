#!/usr/bin/env python3
"""Dump the DRAM block layout at the moment a large allocation is about to be made.

`largest free block` in a ttnn OOM says the heap is fragmented but not by what. This wraps the
allocating ops, and when one is about to request at least `TT_BIO_LAYOUT_DUMP_BYTES` (default
2 GiB) it calls `ttnn.dump_device_memory_state` first, so the block table is written while the
refusal's exact allocator state is still live. The request then proceeds normally: on the
abag-xm 9j4c cell it throws, and the dump is the last thing on disk.

`dump_device_memory_state(device, prefix)` writes its CSVs under the ttnn reports root; the
prefix carries the call index so successive dumps do not overwrite each other.

    TT_VISIBLE_DEVICES=26 python3 scripts/dram_layout_probe.py predict target.yaml --model ...
"""
import os
import sys
import traceback

import ttnn

DUMP_BYTES = int(os.environ.get("TT_BIO_LAYOUT_DUMP_BYTES", 2 << 30))
SINK = os.environ.get("TT_BIO_ALLOC_TRACE_FILE", "/tmp/dram_layout_probe.txt")
TRACED_OPS = ("layer_norm", "from_torch", "concat", "clone", "typecast")
_ITEMSIZE = {ttnn.bfloat16: 2, ttnn.float32: 4}
_n = [0]


def _emit(line):
    with open(SINK, "a") as fh:
        fh.write(line + "\n")
    print(line, file=sys.stderr, flush=True)


def _tensors(args):
    for a in args:
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            yield a
        elif isinstance(a, (list, tuple)):
            yield from _tensors(a)


def _nbytes(t):
    try:
        n = 1
        for d in t.shape:
            n *= int(d)
        return n * _ITEMSIZE.get(t.dtype, 2)
    except Exception:
        return 0


def _dump(tag):
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    lcf = mv.largest_contiguous_bytes_free_per_bank
    if isinstance(lcf, (list, tuple)):
        lcf = min(lcf)
    _emit(f"[LAYOUT] {tag}: banks={mv.num_banks} "
          f"per_bank_total={mv.total_bytes_per_bank} free={mv.total_bytes_free_per_bank} "
          f"largest_free={lcf} "
          f"({lcf / 2 ** 20:.1f} MiB of {mv.total_bytes_free_per_bank / 2 ** 20:.1f} MiB free)")
    try:
        ttnn.dump_device_memory_state(dev, f"{tag}_")
        _emit(f"[LAYOUT] {tag}: dump_device_memory_state written")
    except Exception as e:                      # a diagnostic must never break the fold
        _emit(f"[LAYOUT] {tag}: dump failed: {e}")


def _wrap(name):
    orig = getattr(ttnn, name)

    def traced(*args, **kwargs):
        big = max([_nbytes(t) for t in _tensors(args)] or [0])
        if big >= DUMP_BYTES:
            _n[0] += 1
            tag = f"{name}{_n[0]}"
            shapes = ", ".join(f"{tuple(t.shape)}:{t.dtype}" for t in _tensors(args))
            _emit(f"[LAYOUT] about to run ttnn.{name}({shapes}) -> {big / 2 ** 30:.3f} GiB")
            for frame in traceback.format_stack()[:-1]:
                if "tt_bio" in frame:
                    _emit("[LAYOUT]     " + " | ".join(p.strip() for p in frame.splitlines()))
            _dump(tag)
        return orig(*args, **kwargs)

    setattr(ttnn, name, traced)


for _op in TRACED_OPS:
    _wrap(_op)

def _bracket_seam():
    """Also dump either side of the residue -> structural-token seam.

    The refusal dump alone says which blocks split the free space but not when they were
    allocated. Bracketing the expander dates them to the trunk or to the expander, which is
    what decides where a fix can move them. Wrapping the method here keeps the probe out of
    the model code.
    """
    from tt_bio.opendde import OpenDDE
    orig = OpenDDE.expand_and_refine

    def wrapped(self, *a, **kw):
        _dump("seam_pre_expander")
        try:
            return orig(self, *a, **kw)
        finally:
            _dump("seam_post_refiner")

    OpenDDE.expand_and_refine = wrapped


from tt_bio.main import cli  # noqa: E402  (import after the patch, so tt_bio sees it)

_bracket_seam()

if __name__ == "__main__":
    sys.exit(cli(standalone_mode=True))
