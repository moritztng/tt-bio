#!/usr/bin/env python3
"""Name the Python call site of a large device allocation, then run the normal CLI.

A ttnn OOM prints a C++ backtrace that stops at the pybind boundary: it names the op
(`ttnn::layer_norm`) and the refused byte count, but not which of the twenty-odd
`ttnn.layer_norm` calls in tt_bio issued it. Guessing from the byte count alone is how
`abag-xm-panel-complete-164` spent five chip-hours laddering `max_parallel_samples` on a
buffer whose size never depended on it.

This wraps the ttnn ops that can allocate a full-size pair tensor and, for every call whose
largest input is at or above `TT_BIO_ALLOC_TRACE_BYTES` (default 1 GiB), prints the input
shapes and the tt_bio frames of the stack. Nothing else is patched and every argument is
forwarded unchanged, so a traced fold produces the same numbers as an untraced one.

    TT_VISIBLE_DEVICES=2 python3 scripts/pair_alloc_trace.py predict target.yaml --model ...
"""
import os
import sys
import traceback

import ttnn

THRESHOLD = int(os.environ.get("TT_BIO_ALLOC_TRACE_BYTES", 1 << 30))
TRACED_OPS = ("layer_norm", "concat", "linear", "typecast", "clone", "add", "permute")
_ITEMSIZE = {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.uint32: 4, ttnn.int32: 4}

# A file and not stdout, for the reason `tenstorrent.dram_peak` documents: `tt-bio predict`
# folds in a spawned worker whose stdout the live-progress view owns and drops when it is
# not a TTY, so a printed trace is invisible exactly when it is being collected.
SINK = os.environ.get("TT_BIO_ALLOC_TRACE_FILE")


def _emit(line):
    if SINK:
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


def _wrap(name):
    orig = getattr(ttnn, name)

    def traced(*args, **kwargs):
        seen = [(t, _nbytes(t)) for t in _tensors(args)]
        big = max([b for _, b in seen] or [0])
        if big >= THRESHOLD:
            shapes = ", ".join(f"{tuple(t.shape)}:{t.dtype}" for t, _ in seen)
            _emit(f"[ALLOC] ttnn.{name}({shapes}) largest input {big / 2 ** 30:.3f} GiB")
            for frame in traceback.format_stack()[:-1]:
                if "tt_bio" in frame:
                    _emit("[ALLOC]     " + " | ".join(p.strip() for p in frame.splitlines()))
        return orig(*args, **kwargs)

    setattr(ttnn, name, traced)


for _op in TRACED_OPS:
    _wrap(_op)

from tt_bio.main import cli  # noqa: E402  (import after the patch, so tt_bio sees it)

if __name__ == "__main__":
    sys.exit(cli(standalone_mode=True))
